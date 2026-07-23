"""Builds the one-year-ahead supervised-learning sample
(FORECASTING_METHODOLOGY.md sec 5) and assigns each company one historical
evidence group A-D (sec 6).

Reads forecast_panel_construction.py's ordered output
(forecast_ordered_panel.csv) and forecast_data_prep.py's validated baseline
(forecast_baseline_validated.csv) — same "read the persisted upstream
output, don't re-derive it" convention as forecast_panel_construction.py
itself and src/model_selection.py. Run
`python -m forecast_src.forecast_panel_construction` first if
forecast_ordered_panel.csv is missing or stale.
"""
from pathlib import Path

import pandas as pd

from forecast_src.forecast_data_prep import DATA_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]
ORDERED_PANEL_PATH = DATA_DIR / "forecast_ordered_panel.csv"
BASELINE_VALIDATED_PATH = DATA_DIR / "forecast_baseline_validated.csv"

EVIDENCE_GROUP_COL = "forecast_evidence_group"


def build_training_samples(ordered_panel: pd.DataFrame) -> pd.DataFrame:
    """Sec 5's shift-based target construction. Requires `ordered_panel` to
    already be sorted by [company_id, accounting_year] (forecast_panel_
    construction.build_ordered_panel does this) — shift(-1) within a
    groupby only means "next accounting year" if the rows are in
    chronological order to begin with.

    "turnover_t is available / turnover_t_plus_1 is available / company_id
    is valid / mission is valid" (sec 5) are all already guaranteed by this
    point: every row in the validated panel has a non-null turnover,
    non-null company_id, and a mission that passed check_mission — the only
    condition this function still has to enforce itself is
    target_year == accounting_year + 1, which by construction also implies
    target_turnover_next_year is non-null (shift(-1) landing on a real
    row's turnover, since every row has one)."""
    df = ordered_panel.copy()
    grouped = df.groupby("company_id")
    df["target_turnover_next_year"] = grouped["turnover"].shift(-1)
    df["target_year"] = grouped["accounting_year"].shift(-1)

    is_consecutive = df["target_year"] == df["accounting_year"] + 1
    samples = df[is_consecutive].copy()
    samples["target_year"] = samples["target_year"].astype(int)
    return samples.reset_index(drop=True)


def assign_evidence_groups(baseline: pd.DataFrame, history_properties: pd.DataFrame) -> pd.DataFrame:
    """Sec 6, computed over the FULL baseline population (1,222 companies),
    not just the ones with a training row — Group D companies specifically
    have zero training rows (no history to shift a target from at all) but
    still need a group label downstream (reliability reporting, sec 8's
    recursive forecast needs to know a company's group even when it never
    trained on anything).

    n_turnover_years comes from forecast_panel_construction.py's per-company
    history properties, left-joined here — a company absent from the panel
    entirely (never had an observed turnover year) gets n_turnover_years=0
    via fillna, not NaN, so the A/B/C/D comparisons below don't need a
    separate null-check branch."""
    merged = baseline.merge(
        history_properties[["company_id", "n_turnover_years"]], on="company_id", how="left"
    )
    merged["n_turnover_years"] = merged["n_turnover_years"].fillna(0).astype(int)

    n = merged["n_turnover_years"]
    is_group_a = n >= 3
    is_group_b = n == 2
    is_group_c = n == 1
    is_group_d = (n == 0) & (merged["turnover_source"] == "estimated")

    merged[EVIDENCE_GROUP_COL] = pd.NA
    merged.loc[is_group_a, EVIDENCE_GROUP_COL] = "A"
    merged.loc[is_group_b, EVIDENCE_GROUP_COL] = "B"
    merged.loc[is_group_c, EVIDENCE_GROUP_COL] = "C"
    merged.loc[is_group_d, EVIDENCE_GROUP_COL] = "D"

    # Not covered by sec 6's literal A/B/C/D definitions: n_turnover_years==0
    # AND turnover_source=="observed" — a company with zero panel rows but
    # an "observed" baseline shouldn't be possible (both derive from the
    # same non-null-Total-Turnover check on Source 2), but forecast_data_prep's
    # baseline/panel checks run independently, so a company could pass one
    # and fail the other. Flagged rather than silently forced into a group.
    is_unclassified = merged[EVIDENCE_GROUP_COL].isna()
    merged.loc[is_unclassified, EVIDENCE_GROUP_COL] = "UNCLASSIFIED"

    return merged, merged[is_unclassified]


def main() -> None:
    if not ORDERED_PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {ORDERED_PANEL_PATH}: run `python -m forecast_src.forecast_panel_construction` first."
        )
    if not BASELINE_VALIDATED_PATH.exists():
        raise FileNotFoundError(
            f"Missing {BASELINE_VALIDATED_PATH}: run `python -m forecast_src.forecast_data_prep` first."
        )

    ordered_panel = pd.read_csv(ORDERED_PANEL_PATH)
    baseline = pd.read_csv(BASELINE_VALIDATED_PATH)
    history_properties = ordered_panel.drop_duplicates(subset="company_id")[
        ["company_id", "n_turnover_years", "first_turnover_year", "latest_turnover_year"]
    ]

    print("=== Training samples (sec 5) ===")
    samples = build_training_samples(ordered_panel)
    print(f"Input: {len(ordered_panel)} panel rows, {ordered_panel['company_id'].nunique()} companies")
    print(f"Training rows (target_year == accounting_year + 1): {len(samples)}")
    print(f"Companies contributing at least one training row: {samples['company_id'].nunique()}")
    dropped_no_next_year = ordered_panel['company_id'].nunique() - samples['company_id'].nunique()
    print(f"Companies with panel rows but NO training row (single year, or gap breaks every transition): {dropped_no_next_year}")

    print("\n=== Evidence groups (sec 6) ===")
    with_groups, unclassified = assign_evidence_groups(baseline, history_properties)
    print(f"Baseline population: {len(baseline)} companies")
    print(with_groups[EVIDENCE_GROUP_COL].value_counts().to_string())
    if len(unclassified):
        print(f"\nWARNING: {len(unclassified)} companies didn't fit sec 6's A/B/C/D definitions (n_turnover_years=0 but turnover_source='observed') — see forecast_evidence_group_anomalies.csv")

    samples = samples.merge(with_groups[["company_id", EVIDENCE_GROUP_COL]], on="company_id", how="left")
    print("\nTraining rows by evidence group:")
    print(samples[EVIDENCE_GROUP_COL].value_counts().to_string())

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    samples_path = DATA_DIR / "forecast_training_samples.csv"
    groups_path = DATA_DIR / "forecast_evidence_groups.csv"
    samples.to_csv(samples_path, index=False)
    with_groups.to_csv(groups_path, index=False)
    if len(unclassified):
        unclassified.to_csv(DATA_DIR / "forecast_evidence_group_anomalies.csv", index=False)

    print(f"\nWrote {samples_path} ({len(samples)} rows)")
    print(f"Wrote {groups_path} ({len(with_groups)} rows)")


if __name__ == "__main__":
    main()
