"""Reshape the validated historical company-year panel into one ordered time
series per company, and compute the company-history properties
FORECASTING_METHODOLOGY.md sec 4 lists (n_turnover_years,
first_turnover_year, latest_turnover_year, history_span,
consecutive_history_length, years_since_latest_turnover).

Reads forecast_data_prep.py's persisted output (forecast_panel_validated.csv)
rather than re-deriving it from Source 2 — same reason model_selection.py
reads model_bakeoff.py's saved CSVs instead of re-running the bake-off:
avoids redundant recompute and guarantees this module sees exactly the
validated/exclusion-applied panel everyone else is looking at. Run
`python -m forecast_src.forecast_data_prep` first if that file is missing
or stale.

Every row in the validated panel already has a non-null, non-negative,
finite `turnover` (sample_construction.build_long_panel only ever emits a
row for a company-year with an observed turnover value in the first place,
and forecast_data_prep.py's sec 3.3 check + invalid-turnover-history policy
remove the rest) — so `n_turnover_years` here is exactly each company's row
count in this panel, not a separate missing-data count.

Sec 4's closing line ("A company may contribute several training rows, but
only where a valid future target exists") describes sec 5's shift-based
target construction, not this module — that's forecast_sample_construction.py,
built next. This module only computes the ordered series + history
properties; it doesn't create training rows or targets.
"""
from pathlib import Path

import pandas as pd

from forecast_src.forecast_data_prep import DATA_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_VALIDATED_PATH = DATA_DIR / "forecast_panel_validated.csv"

ORDER_COLS = ["company_id", "accounting_year"]

# ASSUMPTIONS not pinned down by the methodology text — documented here
# since they affect every downstream forecast_src module:
#
# - history_span: latest_turnover_year - first_turnover_year (a calendar
#   span in years, allowing for gaps). Distinct from n_turnover_years,
#   which counts only years actually observed — a company with turnover in
#   2013 and 2020 only has history_span=7 but n_turnover_years=2.
#
# - consecutive_history_length: the length of the unbroken run of
#   consecutive years ENDING AT the company's latest_turnover_year, not the
#   longest unbroken run anywhere in its history. Chosen because sec 5's
#   one-year-ahead target construction (target_year == accounting_year + 1)
#   can only ever chain off an unbroken run reaching the most recent data —
#   an unbroken run buried earlier in a company's history with a gap after
#   it is not usable for forecasting forward from the baseline year. A
#   company with years [2013, 2014, 2018, 2019, 2020] has
#   consecutive_history_length=3 (2018-2020), not 2 (2013-2014), even
#   though both runs have the same length here — the tie is broken toward
#   the run that reaches latest_turnover_year.
#
# - years_since_latest_turnover: computed relative to the most recent
#   accounting_year seen ANYWHERE in the panel (across all companies), not
#   "today" — same reference-point convention as src/assemble.py's
#   turnover_age_years/turnover_is_stale (STALE_THRESHOLD_YEARS), reused
#   here for consistency rather than inventing a second convention for
#   "how old is this data" in the same project.


def build_ordered_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Sec 4's `company_history = panel.sort_values(["company_id",
    "accounting_year"])`, applied to the whole panel at once rather than
    per-company (equivalent result, one sort)."""
    return panel.sort_values(ORDER_COLS).reset_index(drop=True)


def compute_history_properties(ordered_panel: pd.DataFrame) -> pd.DataFrame:
    grouped = ordered_panel.groupby("company_id")["accounting_year"]
    properties = grouped.agg(
        n_turnover_years="count",
        first_turnover_year="min",
        latest_turnover_year="max",
    )
    properties["history_span"] = properties["latest_turnover_year"] - properties["first_turnover_year"]

    dataset_latest_year = ordered_panel["accounting_year"].max()
    properties["years_since_latest_turnover"] = dataset_latest_year - properties["latest_turnover_year"]

    properties["consecutive_history_length"] = grouped.apply(_longest_run_ending_at_latest)

    return properties.reset_index()


def _longest_run_ending_at_latest(years: pd.Series) -> int:
    sorted_years = sorted(years.unique())
    run_length = 1
    for i in range(len(sorted_years) - 1, 0, -1):
        if sorted_years[i] - sorted_years[i - 1] == 1:
            run_length += 1
        else:
            break
    return run_length


def build_company_history(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (ordered_panel_with_history_properties, history_properties)
    — the properties are both attached back onto every row of the ordered
    panel (so forecast_sample_construction.py's evidence-group assignment
    can use them per row without a second join) and returned standalone
    (one row per company, the natural shape for a summary report)."""
    ordered = build_ordered_panel(panel)
    properties = compute_history_properties(ordered)
    ordered_with_properties = ordered.merge(properties, on="company_id", how="left")
    return ordered_with_properties, properties


def main() -> None:
    if not PANEL_VALIDATED_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PANEL_VALIDATED_PATH}: run `python -m forecast_src.forecast_data_prep` first."
        )
    panel = pd.read_csv(PANEL_VALIDATED_PATH)

    ordered_with_properties, properties = build_company_history(panel)

    print(f"Input: {PANEL_VALIDATED_PATH.relative_to(REPO_ROOT)} ({len(panel)} rows, {panel['company_id'].nunique()} companies)")
    print(f"\nHistory-property summary across {len(properties)} companies:")
    print(properties[["n_turnover_years", "history_span", "consecutive_history_length", "years_since_latest_turnover"]].describe().to_string())

    print("\nn_turnover_years distribution:")
    print(properties["n_turnover_years"].value_counts().sort_index().to_string())

    gapped = properties[properties["consecutive_history_length"] < properties["n_turnover_years"]]
    print(f"\nCompanies with at least one gap in their history (consecutive_history_length < n_turnover_years): {len(gapped)}")

    stale = properties[properties["years_since_latest_turnover"] > 0]
    print(f"Companies whose latest turnover year is behind the dataset's most recent year: {len(stale)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ordered_path = DATA_DIR / "forecast_ordered_panel.csv"
    properties_path = DATA_DIR / "forecast_company_history_properties.csv"
    ordered_with_properties.to_csv(ordered_path, index=False)
    properties.to_csv(properties_path, index=False)

    print(f"\nWrote {ordered_path} ({len(ordered_with_properties)} rows)")
    print(f"Wrote {properties_path} ({len(properties)} rows)")


if __name__ == "__main__":
    main()
