"""Split the Source 2 space-company dataset into mission-specific populations."""
from pathlib import Path

import pandas as pd

from src.data_prep import CH_COL, COMPANY_ID_COL, NAME_COL, URL_COL, prepare_source2

REPO_ROOT = Path(__file__).resolve().parents[1]
MAPPING_PATH = REPO_ROOT / "data" / "mission_mapping.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "processed"

VALUE_STREAM_COL = "Value Stream"
MISSION_COL = "Mission"
SKY_UK_ERROR_VALUE = "Sky UK"

REAL_MISSIONS = ["ACE", "Beyond Earth", "Resilient Earth"]

# Manually verified: specific companies whose presence in the space-
# capabilities catalogue itself looks like a curation error, independent of
# any turnover value — excluded from mission mapping entirely (same
# treatment as SKY_UK_ERROR_VALUE), not folded into Cross-cutting, pending
# whoever curated Source 2 confirming or correcting the categorisation.
# Keyed by company_id (data_prep.make_company_id), not raw CH number, for
# the same collision-safety reason company_id exists everywhere else.
KNOWN_MISCATEGORIZED_COMPANIES = {
    "ch_10993763_volanteglobal": (
        "Not a genuine space-sector company: SIC Code 1 = 70100, LinkedIn "
        "Industry = 'Insurance', LinkedIn Specialties = 'Insurance, "
        "Underwriting' (Source 2) — independently corroborated by Source 1's "
        "own description ('Volante provides reinsurance and insurance "
        "products to clients across various industries') and matching SIC "
        "Code 70100. CH number (10993763) and Beauhurst URL are identical "
        "between Source 1 (Financial Statement 1 - Turnover = £19,474,210, "
        "FY2024) and Source 2 (observed turnover -£3,757,048) — confirmed "
        "the same legal entity, not a name collision, but an insurance/"
        "reinsurance holding group has no evident space-sector business, "
        "regardless of which turnover figure would apply. User decision: "
        "exclude entirely rather than guess which figure (or which mission) "
        "would be defensible."
    ),
}


def load_mapping(path: Path = MAPPING_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def segment_missions(df: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`df` must already carry `is_true_duplicate` / `is_shared_ch_anomaly`
    from data_prep.prepare_source2(). A shared CH number alone never excludes
    a row from training — only a true duplicate (same CH *and* same
    normalised name) does."""
    merged = df.merge(mapping, on=VALUE_STREAM_COL, how="left").copy()
    is_sky_uk_error = merged[VALUE_STREAM_COL] == SKY_UK_ERROR_VALUE
    is_known_miscategorized = merged[COMPANY_ID_COL].isin(KNOWN_MISCATEGORIZED_COMPANIES)
    is_unmapped = merged[MISSION_COL].isna() & ~is_sky_uk_error & ~is_known_miscategorized
    merged.loc[is_sky_uk_error | is_known_miscategorized, MISSION_COL] = pd.NA

    reasons = pd.Series("", index=merged.index)
    reasons += is_sky_uk_error.map(
        {True: "data_entry_error: Value Stream is company's own name; ", False: ""}
    )
    reasons += merged[COMPANY_ID_COL].map(
        lambda cid: f"known_miscategorized_company: {KNOWN_MISCATEGORIZED_COMPANIES[cid]}; "
        if cid in KNOWN_MISCATEGORIZED_COMPANIES
        else ""
    )
    reasons += is_unmapped.map(lambda x: "unmapped_value_stream; " if x else "")
    reasons += merged["is_true_duplicate"].map(
        lambda x: "duplicate_company_record; " if x else ""
    )
    merged["exclusion_reason"] = reasons.str.rstrip("; ")

    merged["training_eligible"] = (
        merged[MISSION_COL].notna()
        & (merged[MISSION_COL] != "Cross-cutting")
        & ~merged["is_true_duplicate"]
    )
    merged["sample_weight"] = 1.0

    mission_log = merged.loc[
        is_sky_uk_error | is_known_miscategorized | is_unmapped,
        [NAME_COL, URL_COL, CH_COL, VALUE_STREAM_COL, MISSION_COL, "exclusion_reason"],
    ].copy()
    mission_log["log_type"] = "mission_mapping"
    mission_log = mission_log.rename(columns={"exclusion_reason": "reason"})

    return merged, mission_log


def summarise(segmented: pd.DataFrame) -> pd.DataFrame:
    groups = list(REAL_MISSIONS) + ["Cross-cutting"]
    rows = []
    for mission in groups:
        subset = segmented[segmented[MISSION_COL] == mission]
        rows.append(
            {
                "Mission": mission,
                "total_companies": len(subset),
                "training_eligible": int(subset["training_eligible"].sum()),
                "excluded_true_duplicate": int(subset["is_true_duplicate"].sum()),
                "shared_ch_anomaly_not_excluded": int(
                    (subset["is_shared_ch_anomaly"] & subset["training_eligible"]).sum()
                ),
            }
        )
    excluded = segmented[segmented[MISSION_COL].isna()]
    rows.append(
        {
            "Mission": "(excluded: Sky UK / miscategorized / unmapped)",
            "total_companies": len(excluded),
            "training_eligible": 0,
            "excluded_true_duplicate": 0,
            "shared_ch_anomaly_not_excluded": 0,
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    prepped, prep_quality_log = prepare_source2()
    mapping = load_mapping()
    segmented, mission_log = segment_missions(prepped, mapping)

    quality_log = pd.concat([prep_quality_log, mission_log], ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    segmented.to_csv(OUTPUT_DIR / "space_companies_segmented.csv", index=False)
    quality_log.to_csv(OUTPUT_DIR / "mission_segmentation_quality_log.csv", index=False)

    summary = summarise(segmented)
    print(summary.to_string(index=False))
    print(f"\nTotal companies: {len(segmented)}")
    print(f"Quality log rows: {len(quality_log)}")
    print(f"\nWrote {OUTPUT_DIR / 'space_companies_segmented.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'mission_segmentation_quality_log.csv'}")


if __name__ == "__main__":
    main()
