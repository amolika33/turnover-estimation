"""Builds the sec 7.1-7.8 feature set (FORECASTING_METHODOLOGY.md) for every
row of the ordered historical panel, then subsets to forecast_sample_
construction.py's training rows for this module's own output.

`build_engineered_panel` is deliberately generic (operates on the WHOLE
ordered panel, one row per (company, year), not just training rows) —
forecast_recursive.py will need the identical feature computation at
inference time later (a company's own baseline year is a "forecast origin"
row too, just without a known future target), so the feature logic is
written once here and imported, not reimplemented — same "no duplicated
logic to drift out of sync" lesson as src/feature_engineering.py's
merge_source3_features.

Sec 7's opening line ("Only information available at or before year t may
be used") is enforced structurally: every feature below is a lag, rolling-
to-current, or expanding-to-current computation via groupby(company_id) on
data already sorted by accounting_year — none of it can see a future row.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_src.forecast_data_prep import DATA_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]
ORDERED_PANEL_PATH = DATA_DIR / "forecast_ordered_panel.csv"
TRAINING_SAMPLES_PATH = DATA_DIR / "forecast_training_samples.csv"
BASELINE_VALIDATED_PATH = DATA_DIR / "forecast_baseline_validated.csv"

GROWTH_ROLLING_WINDOW = 3

# Sec 7.6 required fields with no equivalent anywhere in this project's data
# (already logged once by forecast_data_prep.py's find_missing_fields against
# the historical panel — repeated here since 7.7's derived ratios that need
# them as a denominator/numerator are skipped entirely below, not fabricated
# as all-null columns).
MISSING_COMPANY_FIELDS = ["current_assets", "liabilities", "funding_raised"]
# 7.7 ratios that can't be built because of the above:
#   liabilities_to_assets (needs liabilities), current_assets_to_assets
#   (needs current_assets), funding_per_employee (needs funding_raised).

# 7.8's "binary missingness indicators alongside continuous variables" —
# scoped to the features most likely to actually be missing and most
# decision-relevant, not literally every column (would be excessive: most
# columns here are either always-present by construction, e.g. turnover, or
# already have their own explicit availability flag, e.g. employee_count_source).
MISSINGNESS_FLAG_SOURCE_COLS = ["turnover_lag_1", "turnover_lag_2", "turnover_lag_3", "total_assets", "employees", "export_revenue"]


def add_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Sec 7.1."""
    df = panel.copy()
    grouped = df.groupby("company_id")["turnover"]
    df["turnover_t"] = df["turnover"]
    df["turnover_lag_1"] = grouped.shift(1)
    df["turnover_lag_2"] = grouped.shift(2)
    df["turnover_lag_3"] = grouped.shift(3)
    return df


def add_log_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Sec 7.2. Every value fed to log1p here is guaranteed non-negative by
    forecast_data_prep.py's sec 3.3 check + invalid-turnover-history policy
    (a negative anywhere in a company's history removes that company
    entirely, not just the offending row) — asserted, not silently trusted."""
    df = panel.copy()
    for col in ["turnover_t", "turnover_lag_1", "turnover_lag_2", "turnover_lag_3"]:
        negative = df[col].notna() & (df[col] < 0)
        if negative.any():
            raise ValueError(
                f"{negative.sum()} negative values in {col} — should be impossible after "
                "forecast_data_prep.py's invalid-turnover-history exclusion; investigate before proceeding."
            )
    df["log_turnover_t"] = np.log1p(df["turnover_t"])
    df["log_turnover_lag_1"] = np.log1p(df["turnover_lag_1"])
    df["log_turnover_lag_2"] = np.log1p(df["turnover_lag_2"])
    return df


def add_growth_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Sec 7.3. growth_volatility/rolling_growth_std_3 (7.4) and
    log_growth_3y_mean/rolling_growth_mean_3 (7.4) are the same underlying
    quantity under two names in the spec (7.3 and 7.4 don't give a
    distinguishing formula) — computed once here, aliased to both column
    names rather than risking two silently-different implementations."""
    df = panel.copy()
    df["log_growth_1y"] = df["log_turnover_t"] - df["log_turnover_lag_1"]

    grouped_growth = df.groupby("company_id")["log_growth_1y"]
    rolling_mean_3 = grouped_growth.transform(lambda s: s.rolling(GROWTH_ROLLING_WINDOW, min_periods=1).mean())
    rolling_std_3 = grouped_growth.transform(lambda s: s.rolling(GROWTH_ROLLING_WINDOW, min_periods=1).std())
    df["log_growth_2y_mean"] = grouped_growth.transform(lambda s: s.rolling(2, min_periods=1).mean())
    df["log_growth_3y_mean"] = rolling_mean_3
    df["rolling_growth_mean_3"] = rolling_mean_3
    df["growth_volatility"] = rolling_std_3
    df["rolling_growth_std_3"] = rolling_std_3

    df["growth_acceleration"] = grouped_growth.transform(lambda s: s - s.shift(1))

    is_positive = (df["log_growth_1y"] > 0).astype(float).where(df["log_growth_1y"].notna())
    is_negative = (df["log_growth_1y"] < 0).astype(float).where(df["log_growth_1y"].notna())
    df["positive_growth_count"] = is_positive.groupby(df["company_id"]).transform(lambda s: s.expanding().sum())
    df["negative_growth_count"] = is_negative.groupby(df["company_id"]).transform(lambda s: s.expanding().sum())

    return df


def add_rolling_turnover_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Sec 7.4 (turnover-level rolling stats; growth-level ones are in
    add_growth_features above). Windows include the current row — "the
    current row may be included because it is available at the forecast
    origin" (sec 7.4) — future rows can't leak in since the source series is
    the ordered, groupby'd, shift/rolling-only view already used everywhere else."""
    df = panel.copy()
    grouped = df.groupby("company_id")["turnover"]
    df["rolling_turnover_mean_2"] = grouped.transform(lambda s: s.rolling(2, min_periods=1).mean())
    df["rolling_turnover_mean_3"] = grouped.transform(lambda s: s.rolling(GROWTH_ROLLING_WINDOW, min_periods=1).mean())
    df["rolling_turnover_median_3"] = grouped.transform(lambda s: s.rolling(GROWTH_ROLLING_WINDOW, min_periods=1).median())
    df["historical_turnover_max"] = grouped.transform(lambda s: s.expanding().max())
    df["historical_turnover_min"] = grouped.transform(lambda s: s.expanding().min())
    return df


def _consecutive_length_ending_here(years: pd.Series) -> pd.Series:
    """Per-row version of forecast_panel_construction.py's company-level
    consecutive_history_length: for each row, the length of the unbroken
    run of consecutive years ending AT THAT row specifically (not the
    company's overall latest year) — sec 7.5 needs this as of each forecast
    origin, not as of "today"."""
    years_arr = years.to_numpy()
    lengths = np.ones(len(years_arr), dtype=int)
    for i in range(1, len(years_arr)):
        if years_arr[i] - years_arr[i - 1] == 1:
            lengths[i] = lengths[i - 1] + 1
    return pd.Series(lengths)


def add_history_quality_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Sec 7.5. Distinct from forecast_panel_construction.py's company-level
    (one value per company, relative to that company's own latest year)
    history properties — these are per-forecast-origin-row, "as of year t"
    versions of the same ideas."""
    df = panel.copy()
    grouped = df.groupby("company_id")

    df["n_prior_turnover_values"] = grouped.cumcount()
    df["first_turnover_year_to_date"] = grouped["accounting_year"].transform("min")
    df["history_span_years"] = df["accounting_year"] - df["first_turnover_year_to_date"]
    df["years_since_previous_turnover"] = grouped["accounting_year"].diff()

    consecutive = []
    for _, g in grouped:
        consecutive.append(_consecutive_length_ending_here(g["accounting_year"]).set_axis(g.index))
    df["consecutive_history_length_to_date"] = pd.concat(consecutive).sort_index()

    df["has_growth_history"] = df["n_prior_turnover_values"] >= 1
    df["has_three_year_history"] = df["n_prior_turnover_values"] >= 2

    df = df.drop(columns=["first_turnover_year_to_date"])
    return df


def add_derived_financial_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Sec 7.7. liabilities_to_assets/current_assets_to_assets/
    funding_per_employee are skipped entirely (not emitted as all-null
    columns) — see MISSING_COMPANY_FIELDS. export_share is defined as
    export_revenue / turnover (fraction of turnover from exports) — the
    spec doesn't give a formula, this is the only interpretation that
    doesn't need a field we don't have."""
    df = panel.copy()

    def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = numerator / denominator
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        ratio = ratio.where(denominator != 0)
        return ratio

    df["assets_per_employee"] = safe_ratio(df["total_assets"], df["employees"])
    df["export_share"] = safe_ratio(df["export_revenue"], df["turnover"])

    grouped = df.groupby("company_id")
    employees_lag1 = grouped["employees"].shift(1)
    assets_lag1 = grouped["total_assets"].shift(1)
    df["employee_growth"] = safe_ratio(df["employees"] - employees_lag1, employees_lag1)
    df["asset_growth"] = safe_ratio(df["total_assets"] - assets_lag1, assets_lag1)

    return df


def add_data_quality_features(panel: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Sec 7.8. `estimation_reliability`/`estimation_reliability_reason` are
    this module's answer to the spec's "out-of-distribution indicators
    generated during the estimation framework" bullet — genuinely available
    (assemble.py's baseline_reliability/reliability_reason columns), merged
    in per company rather than left as a gap. missing_feature_count/
    missing_feature_proportion are computed last, over the columns listed
    in feature_cols (passed in by build_engineered_panel once the full set
    is known)."""
    df = panel.merge(
        baseline[["company_id", "turnover_source", "baseline_reliability", "reliability_reason"]].rename(
            columns={
                "turnover_source": "estimation_turnover_source",
                "baseline_reliability": "estimation_reliability",
                "reliability_reason": "estimation_reliability_reason",
            }
        ),
        on="company_id",
        how="left",
    )

    for col in MISSINGNESS_FLAG_SOURCE_COLS:
        df[f"is_missing_{col}"] = df[col].isna()

    df["has_employee_data"] = df["employees"].notna()
    df["has_financial_statement_data"] = df["total_assets"].notna()

    return df


def finalize_missing_feature_summary(panel: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = panel.copy()
    df["missing_feature_count"] = df[feature_cols].isna().sum(axis=1)
    df["missing_feature_proportion"] = df["missing_feature_count"] / len(feature_cols)
    return df


def build_engineered_panel(ordered_panel: pd.DataFrame, baseline: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Runs sec 7.1-7.8 on the FULL ordered panel (every company-year row,
    not just training rows) — the shared function forecast_recursive.py will
    reuse at inference time. Returns (engineered_panel, feature_cols) where
    feature_cols is every column this function added (used both for
    missing_feature_count here and, later, as the model's predictor list)."""
    before_cols = set(ordered_panel.columns)

    df = add_lag_features(ordered_panel)
    df = add_log_features(df)
    df = add_growth_features(df)
    df = add_rolling_turnover_features(df)
    df = add_history_quality_features(df)
    df = add_derived_financial_features(df)
    df = add_data_quality_features(df, baseline)

    feature_cols = [c for c in df.columns if c not in before_cols and not c.startswith("estimation_")]
    df = finalize_missing_feature_summary(df, feature_cols)
    feature_cols += ["missing_feature_count", "missing_feature_proportion"]

    return df, feature_cols


def main() -> None:
    for path in (ORDERED_PANEL_PATH, TRAINING_SAMPLES_PATH, BASELINE_VALIDATED_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}: run the earlier forecast_src build-order steps first.")

    ordered_panel = pd.read_csv(ORDERED_PANEL_PATH)
    training_samples = pd.read_csv(TRAINING_SAMPLES_PATH)
    baseline = pd.read_csv(BASELINE_VALIDATED_PATH)

    print(f"Missing company fields (sec 7.6/7.7, not fabricated): {MISSING_COMPANY_FIELDS}")
    print("Ratios skipped as a result: liabilities_to_assets, current_assets_to_assets, funding_per_employee\n")

    engineered_panel, feature_cols = build_engineered_panel(ordered_panel, baseline)

    key_cols = ["company_id", "accounting_year"]
    new_cols = [c for c in engineered_panel.columns if c not in ordered_panel.columns]
    training_features = training_samples.merge(engineered_panel[key_cols + new_cols], on=key_cols, how="left")

    print(f"Engineered panel: {len(engineered_panel)} rows, {len(feature_cols)} feature columns")
    print(f"Training features: {len(training_features)} rows (should match forecast_training_samples.csv's {len(training_samples)})")
    assert len(training_features) == len(training_samples), "Row count changed during feature merge — investigate."

    print("\nNon-null coverage for a sample of features:")
    sample_cols = [
        "turnover_lag_1", "log_growth_1y", "growth_volatility", "rolling_turnover_mean_3",
        "n_prior_turnover_values", "consecutive_history_length_to_date", "assets_per_employee",
        "export_share", "estimation_reliability", "missing_feature_count",
    ]
    for c in sample_cols:
        non_null = training_features[c].notna().sum()
        print(f"  {c:<35} non-null: {non_null}/{len(training_features)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "forecast_training_features.csv"
    training_features.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(training_features)} rows, {len(training_features.columns)} columns)")


if __name__ == "__main__":
    main()
