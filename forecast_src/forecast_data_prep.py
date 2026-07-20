"""Validate the two required inputs to the 2030 forecasting pipeline
(FORECASTING_METHODOLOGY.md sec 2-3): the completed company baseline
(turnover-estimation's `assemble.py` output, read as-is — not built here)
and the historical company-year panel (built here, by
build_historical_panel_source, since no existing turnover-estimation
output covers all 4 mission groups — see that function's docstring).

Column names in the source CSVs don't match the methodology's spec names
(e.g. "Organisation Name" vs `company_name`, "year" vs `accounting_year`) —
*_COLUMN_MAP below renames them explicitly rather than requiring every
downstream forecast_src module to know both naming schemes.

Some spec-required fields have no equivalent anywhere in the
turnover-estimation pipeline's output (current_assets, liabilities,
funding_raised as a per-year figure, baseline_lower/baseline_upper
prediction intervals). These are never fabricated — `find_missing_fields`
reports them explicitly so downstream modules (feature engineering in
particular) know a feature can't be built, rather than silently training on
a placeholder.

Each of the five checks in sec 3.1-3.5 follows the same shape: rows that
fail are pulled into a quality-log entry (never silently dropped without a
trace — same convention as `src/data_prep.py` and `src/mission_segmentation.py`),
and the "clean" frame returned is what the next stage should actually use.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_prep import COMPANY_ID_COL, prepare_source2
from src.feature_engineering import STATIC_COLS, _melt_year_indexed
from src.mission_segmentation import MISSION_COL, REAL_MISSIONS, load_mapping, segment_missions
from src.sample_construction import YEARS, build_long_panel, split_labelled_inference

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"

# mission_segmentation.py's own spelling for the fourth group (its
# `training_eligible` check uses this literal directly, no exported
# constant exists) — matched here rather than "Cross-Cutting" (the
# methodology's canonical spelling, applied downstream by check_mission).
CROSS_CUTTING_MISSION_VALUE = "Cross-cutting"

# Defaults: the completed baseline is turnover-estimation's assemble.py
# output. The historical panel is built fresh by build_historical_panel_source
# below, NOT read from labelled_features.csv — that file only covers ACE/
# Beyond Earth/Resilient Earth (mission_segmentation.py's training_eligible
# flag deliberately excludes Cross-cutting from ML feature engineering,
# since those companies never train any mission model). Forecasting needs
# Cross-cutting's observed turnover history too — the 108 companies with
# real history are forecastable with company-level growth models
# (persistence, CAGR) that don't need a mission-specific model at all, so
# they don't need labelled_features.csv's richer ML feature set either.
COMPLETED_BASELINE_PATH = DATA_DIR / "final_completed_dataset.csv"
HISTORICAL_PANEL_SOURCE_PATH = DATA_DIR / "forecast_full_historical_panel.csv"
HISTORICAL_PANEL_PATH = HISTORICAL_PANEL_SOURCE_PATH

# Sec 3.4: "outside the supported historical range" — reuse the estimation
# pipeline's own YEARS range rather than a second hardcoded copy that could
# drift out of sync with it (the same class of bug fixed in predict.py's
# Source3 merge earlier in this project).
SUPPORTED_YEAR_RANGE = (min(YEARS), max(YEARS))

# Sec 3.5. Canonical spelling per the methodology doc; the source data spells
# the fourth value "Cross-cutting" (lowercase c) — check_mission normalizes
# case-insensitively rather than treating that as an invalid value.
VALID_MISSIONS = ["ACE", "Beyond Earth", "Resilient Earth", "Cross-Cutting"]
PRIMARY_FIT_MISSIONS = ["ACE", "Beyond Earth", "Resilient Earth"]

REQUIRED_BASELINE_FIELDS = [
    "company_id",
    "company_name",
    "mission",
    "baseline_year",
    "baseline_turnover",
    "turnover_source",
    "baseline_lower",
    "baseline_upper",
    "baseline_reliability",
]

BASELINE_COLUMN_MAP = {
    "company_id": "company_id",
    "Organisation Name": "company_name",
    "Mission": "mission",
    "year": "baseline_year",
    "turnover_value": "baseline_turnover",
    "turnover_source": "turnover_source",
    "reliability": "baseline_reliability",
}

# assemble.py's turnover_source has 4 values, not the spec's 2:
# "observed" and "predicted" both carry a numeric baseline_turnover and map
# cleanly (predicted -> estimated, matching predict.py's own reliability
# framing). "cross_cutting_unmodelled" and "no_model_insufficient_data" both
# carry NO baseline_turnover at all (confirmed: 312 rows null baseline_turnover
# == 197 + 115) — there is no "estimated" value to report for these, so they
# are left unmapped (turnover_source -> NaN) rather than forced into either
# spec category; normalize_turnover_source logs every one of these rows.
TURNOVER_SOURCE_MAP = {"observed": "observed", "predicted": "estimated"}

REQUIRED_PANEL_FIELDS = [
    "company_id",
    "company_name",
    "mission",
    "accounting_year",
    "turnover",
    "employees",
    "total_assets",
    "current_assets",
    "liabilities",
    "funding_raised",
    "export_revenue",
    "company_age",
    "company_size",
    "value_stream",
]

PANEL_COLUMN_MAP = {
    "company_id": "company_id",
    "Organisation Name": "company_name",
    "Mission": "mission",
    "year": "accounting_year",
    "total_turnover": "turnover",
    "total_employees": "employees",
    "balance_sheet_total_assets": "total_assets",
    "total_export_revenue": "export_revenue",
    "company_age_years": "company_age",
    "company_size": "company_size",
    "value_stream": "value_stream",
}
# current_assets, liabilities: no equivalent column anywhere in the
# turnover-estimation output (only a single combined balance_sheet_total_assets
# figure exists, no asset/liability breakdown was ever extracted from Source 1/2).
# funding_raised: fundraising_total_amount exists in labelled_features.csv but
# is cumulative-to-date as of the Beauhurst export date, not a per-accounting-
# year figure — aliasing it to funding_raised would misrepresent every
# historical row as if that company's full lifetime funding total applied
# retroactively to each of its earlier years, which sec 2.2 explicitly
# prohibits ("current company values must not be attached retrospectively to
# earlier years"). Left absent rather than aliased; see find_missing_fields.


def build_historical_panel_source() -> pd.DataFrame:
    """Builds the historical company-year panel fresh, covering all 4 mission
    groups (ACE, Beyond Earth, Resilient Earth, Cross-Cutting) — not the
    labelled_features.csv companies-in-training-eligible-missions-only
    population. Reuses sample_construction's split_labelled_inference/
    build_long_panel (both already mission-agnostic, just never called on
    anything but REAL_MISSIONS before this) and feature_engineering's
    STATIC_COLS/_melt_year_indexed for company_age/value_stream/sic_code_1/
    company_size — the same static-column join add_features does for the ML
    panel, without any of the grants/accelerator/Source3 features, which
    Cross-cutting's company-level growth models (persistence, CAGR) don't
    need and which haven't been verified to cover the Cross-cutting
    population anyway."""
    prepped, _ = prepare_source2()
    mapping = load_mapping()
    segmented, _ = segment_missions(prepped, mapping)
    # Same true-duplicate exclusion every other module in this pipeline
    # applies (0 rows affected currently, kept for consistency — see
    # data_prep.py's resolve_duplicate_ch_numbers).
    segmented = segmented[~segmented["is_true_duplicate"]]

    groups = REAL_MISSIONS + [CROSS_CUTTING_MISSION_VALUE]
    panels = []
    for group in groups:
        group_df = segmented[segmented[MISSION_COL] == group]
        labelled, _inference = split_labelled_inference(group_df)
        panels.append(build_long_panel(labelled))
    full_panel = pd.concat(panels, ignore_index=True)

    static = segmented[[COMPANY_ID_COL] + list(STATIC_COLS)].rename(columns=STATIC_COLS)
    full_panel = full_panel.merge(static, on=COMPANY_ID_COL, how="left")

    size_long = _melt_year_indexed(segmented, "Size {year}", "company_size")
    full_panel = full_panel.merge(size_long, on=[COMPANY_ID_COL, "year"], how="left")

    full_panel["company_age_years"] = full_panel["year"] - full_panel["founded_year"]
    full_panel.loc[full_panel["company_age_years"] < 0, "company_age_years"] = np.nan

    return full_panel


def find_missing_fields(df: pd.DataFrame, required_fields: list[str], dataset_name: str) -> pd.DataFrame:
    missing = [f for f in required_fields if f not in df.columns]
    return pd.DataFrame({"dataset": dataset_name, "missing_field": missing})


def check_company_id(df: pd.DataFrame, id_col: str = "company_id") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sec 3.1: rows without a stable, non-blank company_id are excluded from
    the returned clean frame and retained only in the exception log."""
    is_invalid = df[id_col].isna() | (df[id_col].astype(str).str.strip() == "")
    exceptions = df[is_invalid].copy()
    exceptions["exclusion_reason"] = "missing_or_blank_company_id"
    return df[~is_invalid].copy(), exceptions


def check_duplicates(df: pd.DataFrame, subset: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sec 3.2. The spec's resolution order is (1) most reliable source, (2)
    most recently updated record, (3) manual review — neither a source-
    reliability ranking nor a last-updated timestamp exists anywhere in the
    current pipeline's output, so every duplicate group falls straight
    through to (3): flagged in full (all rows in the group, not just the
    dropped ones) and the first occurrence is kept so downstream stages
    still get exactly one row per key. Untested against real duplicates as
    of this run (none exist in either input) — exercised only once real
    duplicate company-year rows appear."""
    dup_mask = df.duplicated(subset=subset, keep=False)
    duplicates = df[dup_mask].copy()
    if duplicates.empty:
        return df, duplicates
    duplicates["exclusion_reason"] = "duplicate_key_flagged_for_manual_review_no_reliability_or_timestamp_signal"
    deduped = df.drop_duplicates(subset=subset, keep="first")
    return deduped, duplicates


def check_turnover(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sec 3.3: non-numeric/negative/non-finite turnover is converted to
    missing and logged — the row itself is kept (unlike 3.1/3.2/3.4/3.5,
    this check never drops a row)."""
    numeric = pd.to_numeric(df[col], errors="coerce")
    was_non_numeric = df[col].notna() & numeric.isna()
    is_negative = numeric.notna() & (numeric < 0)
    is_non_finite = numeric.notna() & ~np.isfinite(numeric)

    reasons = pd.Series("", index=df.index, dtype="object")
    reasons[was_non_numeric] = "non_numeric"
    reasons[is_negative] = "negative"
    reasons[is_non_finite] = "non_finite"
    invalid_mask = reasons != ""

    cleaned = df.copy()
    cleaned[col] = numeric
    cleaned.loc[invalid_mask, col] = np.nan

    log = df[invalid_mask].copy()
    log["exclusion_reason"] = "turnover_" + reasons[invalid_mask]
    return cleaned, log


def check_years(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sec 3.4: accounting_year/baseline_year must be a four-digit integer
    within the pipeline's supported historical range; rows outside it are
    removed (the spec's "removed or reviewed" — removed here, logged either
    way)."""
    numeric = pd.to_numeric(df[col], errors="coerce")
    is_integer_valued = numeric.notna() & (numeric == numeric.round())
    is_four_digit = is_integer_valued & (numeric >= 1000) & (numeric <= 9999)
    in_range = numeric.between(*SUPPORTED_YEAR_RANGE)
    valid_mask = is_four_digit & in_range

    log = df[~valid_mask].copy()
    log["exclusion_reason"] = np.where(
        ~is_four_digit[~valid_mask], "year_not_four_digit_integer", "year_outside_supported_range"
    )
    return df[valid_mask].copy(), log


def check_mission(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sec 3.5: mission must resolve (case-insensitively) to one of the 4
    valid values. Rows that don't (including a genuinely null mission, e.g.
    the upstream "Sky UK" data-entry error already excluded from mission
    mapping in mission_segmentation.py) are excluded and logged — never
    guessed. Cross-Cutting is a VALID value here (only flagged separately as
    primary_model_eligible=False downstream), consistent with the spec's
    "retained in the final output but excluded from primary model fitting"."""
    raw = df[col]
    canonical_lookup = {m.lower(): m for m in VALID_MISSIONS}
    mapped = raw.astype("string").str.strip().str.lower().map(canonical_lookup)
    valid_mask = mapped.notna()

    cleaned = df[valid_mask].copy()
    cleaned[col] = mapped[valid_mask].to_numpy()
    cleaned["primary_model_eligible"] = cleaned[col].isin(PRIMARY_FIT_MISSIONS)

    log = df[~valid_mask].copy()
    log["exclusion_reason"] = np.where(raw[~valid_mask].isna(), "missing_mission", "unrecognized_mission_value")
    return cleaned, log


def normalize_turnover_source(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baseline-only. Maps the pipeline's 4-value turnover_source down to the
    spec's observed/estimated, leaving it null (not fabricated) for rows with
    no baseline_turnover value at all. Doesn't drop rows — this isn't a
    sec 3 hard check, just the observed/estimated framing sec 2.1 requires."""
    df = df.copy()
    has_value = df["baseline_turnover"].notna()
    mapped = df["turnover_source"].map(TURNOVER_SOURCE_MAP)

    df["turnover_source_raw"] = df["turnover_source"]
    df["turnover_source"] = mapped.where(has_value, pd.NA)

    unresolved = has_value & mapped.isna()
    log = df[unresolved].copy()
    log["exclusion_reason"] = "turnover_source_value_not_in_observed_estimated_map"
    return df, log


def validate_completed_baseline(path: Path = COMPLETED_BASELINE_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path)
    df = raw.rename(columns=BASELINE_COLUMN_MAP)
    field_gaps = find_missing_fields(df, REQUIRED_BASELINE_FIELDS, "completed_baseline")

    logs = []
    df, log = check_company_id(df)
    logs.append(log.assign(check="company_id"))
    df, log = check_duplicates(df, subset=["company_id"])
    logs.append(log.assign(check="duplicate_company_id"))
    df, log = check_turnover(df, "baseline_turnover")
    logs.append(log.assign(check="baseline_turnover"))
    df, log = check_years(df, "baseline_year")
    logs.append(log.assign(check="baseline_year"))
    df, log = check_mission(df, "mission")
    logs.append(log.assign(check="mission"))
    df, log = normalize_turnover_source(df)
    logs.append(log.assign(check="turnover_source"))

    quality_log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    return df, quality_log, field_gaps


def validate_historical_panel(path: Path = HISTORICAL_PANEL_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path)
    df = raw.rename(columns=PANEL_COLUMN_MAP)
    field_gaps = find_missing_fields(df, REQUIRED_PANEL_FIELDS, "historical_panel")

    logs = []
    df, log = check_company_id(df)
    logs.append(log.assign(check="company_id"))
    df, log = check_duplicates(df, subset=["company_id", "accounting_year"])
    logs.append(log.assign(check="duplicate_company_year"))
    df, log = check_turnover(df, "turnover")
    logs.append(log.assign(check="turnover"))
    df, log = check_years(df, "accounting_year")
    logs.append(log.assign(check="accounting_year"))
    df, log = check_mission(df, "mission")
    logs.append(log.assign(check="mission"))

    quality_log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    return df, quality_log, field_gaps


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Completed baseline ===")
    baseline_raw = pd.read_csv(COMPLETED_BASELINE_PATH)
    baseline, baseline_log, baseline_gaps = validate_completed_baseline()
    print(f"Input: {COMPLETED_BASELINE_PATH.relative_to(REPO_ROOT)} ({len(baseline_raw)} rows)")
    print(f"Clean rows: {len(baseline)} ({len(baseline_raw) - len(baseline)} excluded)")
    if len(baseline_log):
        print(baseline_log.groupby(["check", "exclusion_reason"]).size().to_string())
    if len(baseline_gaps):
        print(f"Missing required fields (not fabricated): {baseline_gaps['missing_field'].tolist()}")
    print(f"turnover_source (post-normalization): {baseline['turnover_source'].value_counts(dropna=False).to_dict()}")
    print(f"mission distribution: {baseline['mission'].value_counts(dropna=False).to_dict()}")

    print("\n=== Historical company-year panel ===")
    panel_source = build_historical_panel_source()
    panel_source.to_csv(HISTORICAL_PANEL_SOURCE_PATH, index=False)
    print(f"Built {HISTORICAL_PANEL_SOURCE_PATH.relative_to(REPO_ROOT)} ({len(panel_source)} rows, 4 mission groups incl. Cross-Cutting)")

    panel_raw = pd.read_csv(HISTORICAL_PANEL_PATH)
    panel, panel_log, panel_gaps = validate_historical_panel()
    print(f"Input: {HISTORICAL_PANEL_PATH.relative_to(REPO_ROOT)} ({len(panel_raw)} rows)")
    print(f"Clean rows: {len(panel)} ({len(panel_raw) - len(panel)} excluded)")
    if len(panel_log):
        print(panel_log.groupby(["check", "exclusion_reason"]).size().to_string())
    if len(panel_gaps):
        print(f"Missing required fields (not fabricated): {panel_gaps['missing_field'].tolist()}")
    print(f"mission distribution: {panel['mission'].value_counts(dropna=False).to_dict()}")

    baseline.to_csv(DATA_DIR / "forecast_baseline_validated.csv", index=False)
    baseline_log.to_csv(DATA_DIR / "forecast_baseline_quality_log.csv", index=False)
    panel.to_csv(DATA_DIR / "forecast_panel_validated.csv", index=False)
    panel_log.to_csv(DATA_DIR / "forecast_panel_quality_log.csv", index=False)
    pd.concat([baseline_gaps, panel_gaps], ignore_index=True).to_csv(DATA_DIR / "forecast_field_gaps.csv", index=False)

    print(f"\nWrote {DATA_DIR / 'forecast_baseline_validated.csv'}")
    print(f"Wrote {DATA_DIR / 'forecast_baseline_quality_log.csv'}")
    print(f"Wrote {DATA_DIR / 'forecast_panel_validated.csv'}")
    print(f"Wrote {DATA_DIR / 'forecast_panel_quality_log.csv'}")
    print(f"Wrote {DATA_DIR / 'forecast_field_gaps.csv'}")


if __name__ == "__main__":
    main()
