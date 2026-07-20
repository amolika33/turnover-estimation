"""Build company characteristics, financial indicators, and categorical
features for the labelled panel. Composite indicators and commercial-activity
features (grants, funding, accelerators — Source 1 only, not merged yet) are
deferred to a later pass. Nothing here may be derived from `total_turnover`:
that population's inference companies have no turnover history at all, so a
turnover-derived feature would be unusable for exactly the rows that need
predicting, and it would leak the target besides.

TODO (adjacent-company integration, not yet built — see
ADJACENT_DATA_REQUIREMENTS.md): add `population_type` (already stamped
"space" on every panel row by sample_construction.build_long_panel, ready
to take "adjacent" once that data is merged in) as a categorical feature
here, alongside `sic_code_1`/`value_stream`. The point isn't just carrying
the label through — it's letting the model *learn* a systematic
space-vs-adjacent adjustment (a coefficient/split on population_type)
rather than population_type only ever acting through `sample_weight`
(ADJACENT_SAMPLE_WEIGHT in model_bakeoff.py). Weighting alone can
down-rank adjacent rows' influence on the fit; it can't let the model
represent "adjacent companies of this profile tend to report turnover
differently than space companies of this profile," which a feature can."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_prep import CH_COL, COMPANY_ID_COL, NAME_COL, URL_COL, prepare_source2
from src.mission_segmentation import MISSION_COL, load_mapping, segment_missions
from src.sample_construction import ID_COLS, YEARS, construct_samples

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "processed"
SOURCE3_PATH = REPO_ROOT / "data" / "raw" / "beauhurst_company_export_20260720-092535.csv.xlsx"

STATIC_COLS = {
    "Founded": "founded_year",
    "SIC Code 1": "sic_code_1",
    "Value Stream": "value_stream",
}

# Merge key: company_id (see data_prep.make_company_id), not the old
# name+URL+CH composite — a single guaranteed-non-null column, robust to
# GeoData-Institute-style nulled CH numbers and shared-CH-number anomalies.
MERGE_KEY = [COMPANY_ID_COL]
IDENTITY_COLS = [COMPANY_ID_COL, NAME_COL, URL_COL, CH_COL]

# Source 3 (grants/accelerator/funding enrichment, same 1,372-company
# universe as Source 1, joined by Beauhurst URL — it has no CH number of
# its own, so it can't be tagged with company_id directly; see
# _source3_url_to_company_id). 13 boolean signal columns exist in the file
# (not 14) — 8 are brought in directly; 5 are excluded, see
# DROPPED_COLUMNS. "Growth signals - Accelerator" and "Innovation signals -
# Academic spinout" were dropped from here specifically because they're
# ~100% redundant with the derived has_attended_accelerator/
# is_academic_spinout below (1 disagreement out of 1,372 rows for
# accelerator, 0 for spinout) — the derived versions are kept since
# they're clearer to explain (built from an explicit date/name slot, not
# an opaque platform flag).
SOURCE3_SAFE_BOOLEAN_SIGNALS = {
    "Growth signals - Equity fundraising": "signal_equity_fundraising",
    "Growth signals - Debt fundraising": "signal_debt_fundraising",
    "Growth signals - MBO/MBI": "signal_mbo_mbi",
    "Growth signals - Acquired": "signal_acquired",
    "Growth signals - Made acquisition": "signal_made_acquisition",
    "Growth signals - IPO": "signal_ipo",
    "Innovation signals - R&D grant": "signal_rd_grant",
    "Innovation signals - Patent": "signal_patent",
}

SOURCE3_ACCELERATOR_NAME_COLS = [f"Accelerator Attendances {i} - Accelerator Name" for i in range(1, 6)]
SOURCE3_SPINOUT_NAME_COLS = [f"Academic Spinout Events {i} - Academic Institution Name" for i in range(1, 3)]

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
    "Growth signals - 10% scaleup / 20% scaleup": "ambiguous turnover-derivation risk (user decision): Beauhurst's own 'scaleup' methodology is typically based on the OECD high-growth-enterprise definition (>=10%/20% p.a. average growth in employees OR turnover over 3 years). Verified NOT a re-export of Source 2's own turnover-growth columns (only 14-17% agreement — different computation/time window), but can't be confirmed independent of turnover, so excluded per the project's absolute no-turnover-derivation rule.",
    "Growth signals - High growth list": "same growth-classification ambiguity as the scaleup flags (~7-9% agreement with our own turnover/employee growth columns — still not confirmed independent of turnover).",
    "IPO market capitalisations (converted to GBP)": "too sparse (19/1,372 non-null) and not clearly useful for this pass",
    "Accelerator Attendances N - Accelerator Name / entry-exit dates": "raw free-text names not brought in as features; dates used internally only to derive has_attended_accelerator/accelerator_count, not exposed as raw date features",
    "Academic Spinout Events N - Academic Institution Name / date": "same as accelerator names — used internally to derive is_academic_spinout only",
    "Growth signals - Accelerator": "~100% redundant with derived has_attended_accelerator (1 disagreement out of 1,372 rows) — kept the derived version, it's clearer to explain (built from an explicit date/name slot, not an opaque platform flag)",
    "Innovation signals - Academic spinout": "~100% redundant with derived is_academic_spinout (0 disagreements out of 1,372 rows) — same reasoning as Growth signals - Accelerator",
    "LinkedIn Industry": "raw, externally-scraped LinkedIn classification, not part of the project's own deliberate mission/industry taxonomy — Value Stream and SIC Code 1 already cover company categorisation more reliably (curated for this project specifically) and overlap heavily with what LinkedIn Industry captures. Also one of the two high-cardinality columns (158 categories) that caused the original linear-model numerical instability (Linear Regression/Ridge/Elastic Net blowing up to 1e83+, see model_bakeoff.py's module docstring) before min_frequency bucketing was added — removing it outright is more robust than continuing to rely on that bucketing to contain it.",
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
    "value_stream",
] + list(SOURCE3_SAFE_BOOLEAN_SIGNALS.values()) + [
    "has_attended_accelerator",
    "accelerator_count",
    "is_academic_spinout",
    "grants_count",
    "grants_total_amount",
    "grant_recency_years",
    "fundraising_count",
    "fundraising_total_amount",
    "fundraising_recency_years",
]


def _melt_year_indexed(segmented_df: pd.DataFrame, prefix_fmt: str, out_col: str) -> pd.DataFrame:
    cols = {prefix_fmt.format(year=y): y for y in YEARS if prefix_fmt.format(year=y) in segmented_df.columns}
    long = segmented_df[MERGE_KEY + list(cols)].melt(
        id_vars=MERGE_KEY, var_name="_col", value_name=out_col
    )
    long["year"] = long["_col"].map(cols)
    return long.drop(columns="_col")


def _normalize_url(url):
    if pd.isna(url):
        return None
    return str(url).strip().lower().rstrip("/")


def load_source3(path: Path = SOURCE3_PATH) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=0)


def _clean_currency(series: pd.Series) -> pd.Series:
    """Grants/Fundraisings amount columns mix real numbers with a literal
    "(no value)" string sentinel, forcing object dtype — coerce to numeric,
    treating the sentinel as missing rather than 0 (0 would misrepresent an
    unknown amount as a confirmed zero)."""
    return pd.to_numeric(series.replace("(no value)", pd.NA), errors="coerce")


def build_source3_features(src3: pd.DataFrame) -> pd.DataFrame:
    df = src3.copy()
    out = pd.DataFrame({"_url_norm": df[URL_COL].apply(_normalize_url)})

    for src_col, out_col in SOURCE3_SAFE_BOOLEAN_SIGNALS.items():
        out[out_col] = df[src_col].astype(int)

    out["has_attended_accelerator"] = df[SOURCE3_ACCELERATOR_NAME_COLS].notna().any(axis=1).astype(int)
    out["accelerator_count"] = df[SOURCE3_ACCELERATOR_NAME_COLS].notna().sum(axis=1)
    out["is_academic_spinout"] = df[SOURCE3_SPINOUT_NAME_COLS].notna().any(axis=1).astype(int)

    out["grants_count"] = df["Grants - Number of grants received by the company"]
    out["grants_total_amount"] = _clean_currency(
        df["Grants - Total amount received by the company through grants (GBP)"]
    )
    out["_grants_latest_date"] = df["Grants - Date of the company's latest grant"]

    out["fundraising_count"] = df["Fundraisings - Number of fundraisings completed by the company"]
    out["fundraising_total_amount"] = _clean_currency(
        df["Fundraisings - Total amount received by the company through fundraisings (GBP)"]
    )
    out["_fundraising_latest_date"] = df["Fundraisings - Date of the company's latest fundraising"]

    return out


def _source3_url_to_company_id(segmented_df: pd.DataFrame) -> pd.DataFrame:
    """Source 3 has no CH number of its own, so it can't be tagged with
    company_id directly — join by normalised Beauhurst URL instead, same
    approach as the original Source1/Source2 join (DATA_SCHEMA.md). Known
    limitation: segmented_df has 6 rows where two different company_ids
    share the same Beauhurst URL (the shared-CH-number-anomaly cases) —
    drop_duplicates here means one of those company_ids silently gets no
    Source 3 features rather than both getting the same (possibly wrong)
    ones. Affects at most 6 companies; not resolved further this pass."""
    lookup = segmented_df[[URL_COL, COMPANY_ID_COL]].copy()
    lookup["_url_norm"] = lookup[URL_COL].apply(_normalize_url)
    lookup = lookup.dropna(subset=["_url_norm"]).drop_duplicates(subset="_url_norm")
    return lookup[["_url_norm", COMPANY_ID_COL]]


def merge_source3_features(df: pd.DataFrame, segmented_df: pd.DataFrame) -> pd.DataFrame:
    """Shared by add_features (labelled panel) and predict.py's
    add_prediction_features (inference population) — same Source 3 join +
    recency logic either way. `df` must already have a `year` column and
    `company_id`."""
    src3 = load_source3()
    src3_features = build_source3_features(src3)
    url_lookup = _source3_url_to_company_id(segmented_df)
    src3_features = src3_features.merge(url_lookup, on="_url_norm", how="inner").drop(columns="_url_norm")
    df = df.merge(src3_features, on=COMPANY_ID_COL, how="left")

    # Recency relative to each row's year, same pattern as
    # company_age_years — but unlike founded_year (a true static fact),
    # "latest grant/fundraising date" is a cumulative-to-export-date
    # snapshot: a grant received in 2023 shouldn't appear "recent" (or
    # exist at all) for a company's 2015 row. Rather than emit a negative
    # recency (which would leak that a future grant is coming), null it
    # out for any row whose year predates the latest event.
    df["grant_recency_years"] = df["year"] - df["_grants_latest_date"].dt.year
    df.loc[df["grant_recency_years"] < 0, "grant_recency_years"] = np.nan
    df["fundraising_recency_years"] = df["year"] - df["_fundraising_latest_date"].dt.year
    df.loc[df["fundraising_recency_years"] < 0, "fundraising_recency_years"] = np.nan
    df = df.drop(columns=["_grants_latest_date", "_fundraising_latest_date"])
    return df


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

    df = merge_source3_features(df, segmented_df)

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
