"""Load and clean Source 2 (master/mission sheet): known-value corrections and
duplicate-CH-number resolution, ahead of mission segmentation."""
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE2_PATH = REPO_ROOT / "data" / "raw" / "Published Space Capabilities Catalogue_Cleaned.xlsx"

NAME_COL = "Organisation Name"
CH_COL = "CH No. (full)"
CH_SHORT_COL = "CH No."
URL_COL = "Beauhurst URL"

# Manually verified fixes to specific known-bad values in Source 2.
KNOWN_CORRECTIONS = [
    {
        "name": "GeoData Institute",
        "field": CH_COL,
        "new_value": pd.NA,
        "also_field": CH_SHORT_COL,
        "also_new_value": pd.NA,
        "reason": (
            "GeoData Institute has no genuine Companies House number of its "
            "own (University of Southampton entity, not an independently "
            "registered company) — was incorrectly carrying RC000668 "
            "(University of Southampton's charity number). Nulled; falls "
            "back to URL/name-based matching."
        ),
    },
    {
        "name": "ISVR Consulting",
        "field": CH_COL,
        "new_value": "14701170",
        "also_field": CH_SHORT_COL,
        "also_new_value": "14701170",
        "reason": (
            "ISVR Consulting Limited's correct Companies House number is "
            "14701170. Was incorrectly carrying RC000668 (University of "
            "Southampton's charity number)."
        ),
    },
]

LEGAL_SUFFIXES = re.compile(
    r"\b(ltd|limited|llp|plc|inc|the|group|holdings|uk)\b", re.IGNORECASE
)


def load_source2(path: Path = SOURCE2_PATH) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=0)


def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""
    name = LEGAL_SUFFIXES.sub("", name.lower())
    return re.sub(r"[^a-z0-9]", "", name)


def apply_known_corrections(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    log_rows = []
    for correction in KNOWN_CORRECTIONS:
        mask = df[NAME_COL] == correction["name"]
        if mask.sum() == 0:
            continue
        for field, new_value in (
            (correction["field"], correction["new_value"]),
            (correction.get("also_field"), correction.get("also_new_value")),
        ):
            if field is None:
                continue
            old_values = df.loc[mask, field].tolist()
            df.loc[mask, field] = new_value
            for old_value in old_values:
                log_rows.append(
                    {
                        NAME_COL: correction["name"],
                        "field_corrected": field,
                        "old_value": old_value,
                        "new_value": new_value,
                        "reason": correction["reason"],
                    }
                )
    corrections_log = pd.DataFrame(log_rows)
    return df, corrections_log


def resolve_duplicate_ch_numbers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two rows are the same company only if they share a CH number AND a
    normalised name. A shared CH number with genuinely different names is
    logged as a data-quality anomaly, not collapsed into one identity —
    their financials and mission assignments are never merged."""
    df = df.copy()
    df["_normalized_name"] = df[NAME_COL].apply(normalize_name)

    has_ch = df[CH_COL].notna()
    # NaN CH numbers are mutually "equal" under duplicated(), so AND with
    # has_ch to zero those out rather than partial-assigning into a
    # pre-allocated bool Series (which trips pandas' incompatible-dtype
    # FutureWarning when the duplicated() result isn't plain numpy bool).
    dup_ch = df[CH_COL].duplicated(keep=False) & has_ch

    same_name_in_group = df.groupby(df[CH_COL].where(dup_ch))["_normalized_name"].transform(
        lambda names: names.nunique() == 1
    )
    is_true_duplicate = dup_ch & same_name_in_group.fillna(False)
    is_shared_ch_anomaly = dup_ch & ~is_true_duplicate

    df["is_true_duplicate"] = is_true_duplicate
    df["is_shared_ch_anomaly"] = is_shared_ch_anomaly
    df = df.drop(columns=["_normalized_name"])

    log_rows = []
    for flag_col, reason in (
        ("is_true_duplicate", "duplicate_company_record: same CH number and same normalised name"),
        ("is_shared_ch_anomaly", "shared_ch_number_anomaly: same CH number but genuinely different company names — treated as separate entities, not merged"),
    ):
        flagged = df[df[flag_col]]
        for _, row in flagged.sort_values(CH_COL).iterrows():
            log_rows.append(
                {
                    NAME_COL: row[NAME_COL],
                    URL_COL: row[URL_COL],
                    CH_COL: row[CH_COL],
                    "reason": reason,
                }
            )
    dedup_log = pd.DataFrame(log_rows)
    return df, dedup_log


def prepare_source2(path: Path = SOURCE2_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_source2(path)
    df, corrections_log = apply_known_corrections(df)
    df, dedup_log = resolve_duplicate_ch_numbers(df)

    quality_log = pd.concat(
        [corrections_log.assign(log_type="correction"), dedup_log.assign(log_type="duplicate_ch")],
        ignore_index=True,
    )
    return df, quality_log


if __name__ == "__main__":
    cleaned, log = prepare_source2()
    print(f"Rows: {len(cleaned)}")
    print(f"True duplicates (excluded pending review): {int(cleaned['is_true_duplicate'].sum())}")
    print(f"Shared-CH anomalies (logged, not excluded): {int(cleaned['is_shared_ch_anomaly'].sum())}")
    print()
    print(log.to_string(index=False))
