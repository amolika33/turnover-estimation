"""Split the Source 2 space-company dataset into mission-specific populations."""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE2_PATH = REPO_ROOT / "data" / "raw" / "Published Space Capabilities Catalogue_Cleaned.xlsx"
MAPPING_PATH = REPO_ROOT / "data" / "mission_mapping.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "processed"

NAME_COL = "Organisation Name"
URL_COL = "Beauhurst URL"
CH_COL = "CH No. (full)"
VALUE_STREAM_COL = "Value Stream"
MISSION_COL = "Mission"
SKY_UK_ERROR_VALUE = "Sky UK"

REAL_MISSIONS = ["ACE", "Beyond Earth", "Resilient Earth"]


def load_source2(path: Path = SOURCE2_PATH) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=0)


def load_mapping(path: Path = MAPPING_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def segment_missions(df: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = df.merge(mapping, on=VALUE_STREAM_COL, how="left").copy()
    is_sky_uk_error = merged[VALUE_STREAM_COL] == SKY_UK_ERROR_VALUE
    is_unmapped = merged[MISSION_COL].isna() & ~is_sky_uk_error
    merged.loc[is_sky_uk_error, MISSION_COL] = pd.NA

    is_duplicate_ch = merged[CH_COL].duplicated(keep=False)
    ch_mission_conflict = (
        merged.groupby(CH_COL)[MISSION_COL].transform("nunique") > 1
    ) & is_duplicate_ch

    reasons = pd.Series("", index=merged.index)
    reasons += is_sky_uk_error.map(
        {True: "data_entry_error: Value Stream is company's own name; ", False: ""}
    )
    reasons += is_unmapped.map(lambda x: "unmapped_value_stream; " if x else "")
    reasons += is_duplicate_ch.map(lambda x: "duplicate_ch_number; " if x else "")
    merged["exclusion_reason"] = reasons.str.rstrip("; ")

    merged["training_eligible"] = (
        merged[MISSION_COL].notna()
        & (merged[MISSION_COL] != "Cross-cutting")
        & ~is_duplicate_ch
    )
    merged["sample_weight"] = 1.0

    quality_log = merged.loc[
        is_sky_uk_error | is_unmapped | is_duplicate_ch,
        [NAME_COL, URL_COL, CH_COL, VALUE_STREAM_COL, MISSION_COL, "exclusion_reason"],
    ].copy()
    quality_log["ch_group_mission_conflict"] = ch_mission_conflict.loc[quality_log.index]
    quality_log = quality_log.sort_values(CH_COL)

    return merged, quality_log


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
                "excluded_duplicate_ch": int(
                    (~subset["training_eligible"] & subset["exclusion_reason"].str.contains("duplicate_ch_number")).sum()
                ),
            }
        )
    excluded = segmented[segmented[MISSION_COL].isna()]
    rows.append(
        {
            "Mission": "(excluded: Sky UK / unmapped)",
            "total_companies": len(excluded),
            "training_eligible": 0,
            "excluded_duplicate_ch": 0,
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    df = load_source2()
    mapping = load_mapping()
    segmented, quality_log = segment_missions(df, mapping)

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
