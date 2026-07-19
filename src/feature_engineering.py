"""Build company characteristics, financial indicators, and categorical
features for the labelled panel. Composite indicators and commercial-activity
features (grants, funding, accelerators — Source 1 only, not merged yet) are
deferred to a later pass. Nothing here may be derived from `total_turnover`:
that population's inference companies have no turnover history at all, so a
turnover-derived feature would be unusable for exactly the rows that need
predicting, and it would leak the target besides."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_prep import CH_COL, COMPANY_ID_COL, NAME_COL, URL_COL, prepare_source2
from src.mission_segmentation import MISSION_COL, load_mapping, segment_missions
from src.sample_construction import ID_COLS, YEARS, construct_samples

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "processed"

STATIC_COLS = {
    "Founded": "founded_year",
    "SIC Code 1": "sic_code_1",
    "LinkedIn Industry": "linkedin_industry",
    "Value Stream": "value_stream",
}

# Merge key: company_id (see data_prep.make_company_id), not the old
# name+URL+CH composite — a single guaranteed-non-null column, robust to
# GeoData-Institute-style nulled CH numbers and shared-CH-number anomalies.
MERGE_KEY = [COMPANY_ID_COL]
IDENTITY_COLS = [COMPANY_ID_COL, NAME_COL, URL_COL, CH_COL]

# Columns considered and deliberately excluded, with reasons.
DROPPED_COLUMNS = {
    "Average Turnover Growth": "derived from Total Turnover — forbidden by the no-target-leakage rule",
    "Turnover Growth Rate (OECD)": "derived from Total Turnover — forbidden by the no-target-leakage rule",
    "Latest 3 Years: Growth Rate (OECD: 20%)": "derived from Total Turnover (verified: matches Turnover Growth Rate (OECD) / a multi-year turnover CAGR for the rest) — forbidden",
    "Latest 3 Years: Growth Rate (OECD-Esq: 10%)": "same as above — turnover-derived",
    "Average Average": "verified formula = mean(Average Employee Growth, Average Turnover Growth, Sector CAGR) — includes a turnover-derived term, so excluded",
    "Sector CAGR": "constant (4.58) across all 1,225 companies — zero variance, no signal",
    "Average Employee Growth": "point-in-time snapshot as of export date, not year-indexed — attaching it to every historical panel row (2013-2025) would misrepresent growth rates from a decade before the company necessarily had that trajectory. Revisit as an inference-time-only feature.",
    "Employee Growth Rate (OECD)": "same temporal-mismatch reason as Average Employee Growth",
    "LinkedIn Specialties (Keywords)": "free text, ~unique per company (1,020 unique values / 1,173 non-null rows) — needs NLP/keyword extraction, not a direct/derived feature; natural fit for the planned buzzword-similarity logic (composite indicators / adjacent & cross-cutting mission assignment), not this pass",
    "Company Size / Size (Power BI) / Size (LinkedIn)": "static snapshots; superseded by the year-indexed `Size {year}` columns, which give a properly time-varying company_size per panel row instead",
    "SIC Code 2-4": "sparse (315/148/78 non-null out of 1,225) secondary/tertiary classifications — SIC Code 1 alone kept for this pass",
    "Filing Date (year)": "not built this pass — a filing-timeliness feature (e.g. lag vs. accounting year end) is a reasonable future financial indicator, not included yet",
}

FEATURE_COLUMNS = [
    "year",
    "company_age_years",
    "total_employees",
    "employee_count_source",
    "balance_sheet_total_assets",
    "total_export_revenue",
    "assets_per_employee",
    "export_revenue_per_employee",
    "company_size",
    "sic_code_1",
    "linkedin_industry",
    "value_stream",
]


def _melt_year_indexed(segmented_df: pd.DataFrame, prefix_fmt: str, out_col: str) -> pd.DataFrame:
    cols = {prefix_fmt.format(year=y): y for y in YEARS if prefix_fmt.format(year=y) in segmented_df.columns}
    long = segmented_df[MERGE_KEY + list(cols)].melt(
        id_vars=MERGE_KEY, var_name="_col", value_name=out_col
    )
    long["year"] = long["_col"].map(cols)
    return long.drop(columns="_col")


def add_features(panel: pd.DataFrame, segmented_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = panel.merge(
        segmented_df[MERGE_KEY + list(STATIC_COLS)].rename(columns=STATIC_COLS),
        on=MERGE_KEY,
        how="left",
    )

    size_long = _melt_year_indexed(segmented_df, "Size {year}", "company_size")
    df = df.merge(size_long, on=MERGE_KEY + ["year"], how="left")

    df["company_age_years"] = df["year"] - df["founded_year"]

    with np.errstate(divide="ignore", invalid="ignore"):
        df["assets_per_employee"] = df["balance_sheet_total_assets"] / df["total_employees"]
        df["export_revenue_per_employee"] = df["total_export_revenue"] / df["total_employees"]
    df["assets_per_employee"] = df["assets_per_employee"].replace([np.inf, -np.inf], np.nan)
    df["export_revenue_per_employee"] = df["export_revenue_per_employee"].replace(
        [np.inf, -np.inf], np.nan
    )

    is_age_anomaly = df["company_age_years"] < 0
    age_log = df.loc[
        is_age_anomaly, IDENTITY_COLS + ["year", "founded_year", "company_age_years"]
    ].copy()
    age_log["reason"] = "negative_company_age: turnover recorded in a year before Founded"
    age_log = age_log.rename(columns={"company_age_years": "original_company_age_years"})
    df.loc[is_age_anomaly, "company_age_years"] = np.nan

    return df, age_log


def build_features(segmented_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_all, _, _ = construct_samples(segmented_df)
    return add_features(panel_all, segmented_df)


def main() -> None:
    prepped, _ = prepare_source2()
    mapping = load_mapping()
    segmented, _ = segment_missions(prepped, mapping)

    features, age_log = build_features(segmented)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "labelled_features.csv"
    features.to_csv(out_path, index=False)
    log_path = OUTPUT_DIR / "feature_engineering_quality_log.csv"
    age_log.to_csv(log_path, index=False)

    print("Feature columns (%d):" % len(FEATURE_COLUMNS))
    for c in FEATURE_COLUMNS:
        non_null = features[c].notna().sum()
        print(f"  {c:<32} non-null: {non_null}/{len(features)}")

    print(f"\nID/metadata columns (not features): {IDENTITY_COLS + [MISSION_COL, 'sample_weight', 'population_type']}")
    print("Target (never a feature): total_turnover")

    if len(age_log):
        print(
            f"\nData quality flag: {len(age_log)} rows across "
            f"{age_log[NAME_COL].nunique()} companies had a negative "
            "company_age_years (turnover recorded in a year before Founded). "
            "company_age_years nulled for those rows only (other features/"
            f"turnover kept); logged to {log_path}."
        )

    print(f"\nDropped columns ({len(DROPPED_COLUMNS)}):")
    for col, reason in DROPPED_COLUMNS.items():
        print(f"  {col}: {reason}")

    print(f"\nWrote {out_path} ({len(features)} rows, {len(features.columns)} columns)")


if __name__ == "__main__":
    main()
