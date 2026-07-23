"""Combines the real historical panel + forecast_recursive.py's recursive
predictions into the final 2030 outputs (build order step 8).

Two outputs, not one, because forecast_reporting.py (build order step 9,
not yet built) needs both shapes:

- `forecast_full_trajectories.csv`: long format, one row per (company_id,
  year), spanning each company's earliest REAL year (or its estimated
  baseline year, for the 749 companies with zero real history) through
  2030. `data_type` distinguishes "observed" (real Companies House data),
  "estimated_baseline" (the turnover-ESTIMATION pipeline's own prediction,
  used as this pipeline's starting point), and "predicted" (this
  project's own recursive forecast). forecast_reporting.py's planned
  gazelle criteria (3 CONSECUTIVE years of sustained growth) need to walk
  every year-over-year transition, including real pre-baseline history —
  a company-level summary alone can't answer that.
- `forecast_2030_summary.csv`: one row per company (the "final
  company-level 2030 trajectories" the build order literally names) —
  baseline info, the 2030 prediction, the overall growth multiple and its
  annualized-rate equivalent, and a summary of how much of the projected
  horizon was spent "growing" vs "stable". Reliability/evidence-group
  context carries straight through from forecast_baseline_validated.csv
  and forecast_evidence_groups.csv, not recomputed.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_src.forecast_data_prep import DATA_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]
ORDERED_PANEL_PATH = DATA_DIR / "forecast_ordered_panel.csv"
BASELINE_VALIDATED_PATH = DATA_DIR / "forecast_baseline_validated.csv"
TRAJECTORIES_PATH = DATA_DIR / "forecast_trajectories.csv"
EVIDENCE_GROUPS_PATH = DATA_DIR / "forecast_evidence_groups.csv"

FORECAST_END_YEAR = 2030


def build_full_trajectories(real_panel: pd.DataFrame, predicted: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    observed = real_panel[["company_id", "mission", "accounting_year", "turnover"]].copy()
    observed["data_type"] = "observed"

    existing_keys = set(zip(observed["company_id"], observed["accounting_year"]))
    needs_baseline_row = baseline[
        ~baseline.apply(lambda r: (r["company_id"], r["baseline_year"]) in existing_keys, axis=1)
    ]
    baseline_rows = needs_baseline_row[["company_id", "mission", "baseline_year", "baseline_turnover"]].rename(
        columns={"baseline_year": "accounting_year", "baseline_turnover": "turnover"}
    )
    baseline_rows["data_type"] = "estimated_baseline"

    predicted_cols = ["company_id", "mission", "accounting_year", "turnover", "model_used", "growth_classification"]
    # turnover_lower/turnover_upper (forecast_prediction_intervals.py) are
    # optional here — carried through when present so downstream consumers
    # (the dashboard) get the confidence band without forecast_assemble.py
    # needing to know how they were computed, but this module must still
    # work if intervals haven't been generated yet.
    interval_cols = [c for c in ["turnover_lower", "turnover_upper"] if c in predicted.columns]
    predicted_rows = predicted[predicted_cols + interval_cols].copy()
    predicted_rows["data_type"] = "predicted"

    full = pd.concat([observed, baseline_rows, predicted_rows], ignore_index=True)
    return full.sort_values(["company_id", "accounting_year"]).reset_index(drop=True)


def build_2030_summary(baseline: pd.DataFrame, full_trajectories: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    final_2030 = full_trajectories[full_trajectories["accounting_year"] == FORECAST_END_YEAR][
        ["company_id", "turnover"]
    ].rename(columns={"turnover": "turnover_2030"})

    summary = baseline.merge(final_2030, on="company_id", how="left")
    summary["growth_multiple_2030"] = summary["turnover_2030"] / summary["baseline_turnover"]

    years_to_2030 = (FORECAST_END_YEAR - summary["baseline_year"]).replace(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        summary["annualized_growth_rate_to_2030"] = summary["growth_multiple_2030"] ** (1 / years_to_2030) - 1

    if len(predicted):
        class_counts = predicted.groupby("company_id")["growth_classification"].value_counts().unstack(fill_value=0)
        for col in ["growing", "stable"]:
            if col not in class_counts.columns:
                class_counts[col] = 0
        class_counts = class_counts.rename(columns={"growing": "n_years_growing", "stable": "n_years_stable"})
        final_step_model = predicted.sort_values("accounting_year").groupby("company_id")["model_used"].last()
        final_step_model = final_step_model.rename("final_step_model_used")

        summary = summary.merge(class_counts[["n_years_growing", "n_years_stable"]], on="company_id", how="left")
        summary = summary.merge(final_step_model, on="company_id", how="left")
    else:
        summary["n_years_growing"] = 0
        summary["n_years_stable"] = 0
        summary["final_step_model_used"] = pd.NA

    summary["n_years_growing"] = summary["n_years_growing"].fillna(0).astype(int)
    summary["n_years_stable"] = summary["n_years_stable"].fillna(0).astype(int)
    summary["ever_classified_growing"] = summary["n_years_growing"] > 0

    return summary


def validate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Same "trust but verify" convention as src/predict.py's
    validate_predictions — forecast_recursive.py already guarantees
    finite/non-negative turnover_2030, but re-checked here rather than
    assumed, since this is the artifact other stages will build on."""
    is_missing = summary["turnover_2030"].isna()
    is_invalid = ~is_missing & (~np.isfinite(summary["turnover_2030"]) | (summary["turnover_2030"] < 0))
    if is_missing.any() or is_invalid.any():
        bad = summary[is_missing | is_invalid]
        log_path = DATA_DIR / "forecast_2030_summary_invalid.csv"
        bad.to_csv(log_path, index=False)
        print(f"WARNING: {len(bad)} companies with missing/invalid turnover_2030, wrote {log_path}")
    return summary


def enforce_one_company_per_row(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dup_mask = df["company_id"].duplicated(keep=False)
    if not dup_mask.any():
        return df, df.iloc[0:0].copy()
    return df[~dup_mask].copy(), df[dup_mask].copy()


def main() -> None:
    for path in (ORDERED_PANEL_PATH, BASELINE_VALIDATED_PATH, TRAJECTORIES_PATH, EVIDENCE_GROUPS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}: run the earlier forecast_src build-order steps first.")

    real_panel = pd.read_csv(ORDERED_PANEL_PATH)
    baseline = pd.read_csv(BASELINE_VALIDATED_PATH)
    predicted = pd.read_csv(TRAJECTORIES_PATH)
    evidence_groups = pd.read_csv(EVIDENCE_GROUPS_PATH)[["company_id", "forecast_evidence_group"]]

    full_trajectories = build_full_trajectories(real_panel, predicted, baseline)
    summary = build_2030_summary(baseline, full_trajectories, predicted)
    summary = summary.merge(evidence_groups, on="company_id", how="left")
    summary = validate_summary(summary)

    summary, duplicates = enforce_one_company_per_row(summary)
    if len(duplicates):
        dup_path = DATA_DIR / "forecast_assemble_duplicate_company_id.csv"
        duplicates.to_csv(dup_path, index=False)
        print(f"WARNING: {len(duplicates)} duplicate company_id rows removed, wrote {dup_path}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    trajectories_path = DATA_DIR / "forecast_full_trajectories.csv"
    summary_path = DATA_DIR / "forecast_2030_summary.csv"
    full_trajectories.to_csv(trajectories_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Full trajectories: {len(full_trajectories)} rows, {full_trajectories['company_id'].nunique()} companies")
    print(f"data_type breakdown:\n{full_trajectories['data_type'].value_counts().to_string()}")
    print(f"\n2030 summary: {len(summary)} companies")
    print(f"turnover_2030 coverage: {summary['turnover_2030'].notna().sum()} / {len(summary)}")
    print(f"\nGrowth multiple distribution:\n{summary['growth_multiple_2030'].describe().to_string()}")
    print(f"\nEvidence group distribution:\n{summary['forecast_evidence_group'].value_counts().to_string()}")
    print(f"Ever classified growing: {int(summary['ever_classified_growing'].sum())} / {len(summary)}")

    print(f"\nWrote {trajectories_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
