"""Business-facing outputs from the completed 2030 forecast (build order
step 9): £10M-by-2030 crossings, two-tier gazelle/high-growth companies,
their £50M intersection, and a turnover-independent operational-scaling
signal. Reads forecast_assemble.py's outputs, doesn't recompute anything
already built there.

Every output carries the same context columns so a reader can immediately
judge how much to trust a flag, not treat every row as equally confident:
  - forecast_evidence_group: 3+/2/1/0 real turnover years (A/B/C/D)
  - baseline_turnover_source: observed (real data) vs estimated (the
    turnover-ESTIMATION pipeline's own prediction, itself just a starting
    point for this pipeline, not ground truth)
  - n_real_years: count of real (data_type="observed") turnover years
  - evidence_gate_triggered: True if forecast_recursive.py's growth_signal
    said "growing" at some step but got downgraded to Persistence for
    insufficient evidence (MIN_EVIDENCE_GROUP_FOR_TREND_CONTINUATION) —
    a company that hit this gate had LESS trend-continuation applied to
    it than its raw signal wanted, which matters for judging a flag here
  - growth_decay_applied: True if any step was ever routed through
    Ridge/CAGR at all (decay applies unconditionally whenever that
    happens) — a company that never triggered this had every year decided
    by Persistence, i.e. its whole 2030 trajectory is "no growth assumed"

GAZELLE DEFINITION (both tiers): walks each company's FULL trajectory
(forecast_full_trajectories.csv — real history AND predicted years
together, not just baseline-vs-2030) for the longest run of CONSECUTIVE
calendar years where that year's own one-year log-growth (relative to the
immediately preceding calendar year — a gap year breaks the run, same
"no non-consecutive transitions" rule forecast_sample_construction.py's
sec 5 established) meets the tier's threshold. GAZELLE_CONSECUTIVE_YEARS=3
matches the OECD high-growth-enterprise definition (already the project's
stated rationale in PROJECT_NOTES.md) — "3 consecutive years" here means 3
qualifying YEAR-OVER-YEAR TRANSITIONS in a row (the OECD's own convention:
a 3-year growth period is 3 successive 1-year transitions, not 4 flat
data points bracketing 3 gaps). Thresholds reuse forecast_recursive.py's
GROWTH_THRESHOLD construction exactly (log1p(0.10)/log1p(0.20)) rather
than inventing a second growth-rate formula for the same underlying idea.

OPERATIONAL SCALING (new, independent of turnover): sec 7.7's
employee_growth/asset_growth (log-difference, same construction as
turnover's log_growth_1y after their earlier reformulation — see PROJECT_NOTES.md)
checked with the identical 10%/20%-for-3+-years logic, computed from REAL
years only (these were never forecast forward — forecast_recursive.py
holds employees/total_assets at their last known value for all future
years, so a predicted-year employee_growth is trivially 0 and would be
meaningless to include here). Requires 3+ REAL years of the RELEVANT
covariate specifically (employees for employee_growth, total_assets for
asset_growth) — not forecast_evidence_group, which is about TURNOVER
history depth and is a different measurement axis; a company could have
thin turnover evidence but 5 real years of filed employee counts, or vice
versa, and this flag should track its own evidence, not turnover's.

CREDIBILITY GATE (new, applied to the £50M intersection output only): a
company is only flagged as a genuine £50M candidate if it has
forecast_evidence_group=="A" (3+ real turnover years) AND at least £1M
turnover in one of its 5 most recent REAL reporting years. Rationale: even
Group A (this project's own "trustworthy" tier) includes companies whose
real history is 3 small, volatile early-stage years (SaxaVord Spaceport,
Infleqtion, Map of Agriculture from the growth-decay investigation) —
evidence group alone doesn't rule out "genuinely real, but a tiny base
company nowhere near £50M-scale operations." Companies that would
otherwise cross £50M but fail this gate are NOT excluded from the output —
excluding them would look like they were never considered, when the
point is exactly the opposite: they were considered and explicitly found
not to meet the credibility bar. `credibility_status` says which.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_src.forecast_data_prep import DATA_DIR
from forecast_src.forecast_feature_engineering import build_engineered_panel

REPO_ROOT = Path(__file__).resolve().parents[1]
ORDERED_PANEL_PATH = DATA_DIR / "forecast_ordered_panel.csv"
BASELINE_VALIDATED_PATH = DATA_DIR / "forecast_baseline_validated.csv"
FULL_TRAJECTORIES_PATH = DATA_DIR / "forecast_full_trajectories.csv"
SUMMARY_PATH = DATA_DIR / "forecast_2030_summary.csv"
TRAJECTORIES_PATH = DATA_DIR / "forecast_trajectories.csv"

FORECAST_END_YEAR = 2030
TEN_MILLION = 10_000_000
FIFTY_MILLION = 50_000_000
ONE_MILLION = 1_000_000

GAZELLE_CONSECUTIVE_YEARS = 3
GAZELLE_THRESHOLDS = {"10pct": np.log1p(0.10), "20pct": np.log1p(0.20)}

CREDIBILITY_MIN_EVIDENCE_GROUP = "A"
CREDIBILITY_MIN_RECENT_TURNOVER = ONE_MILLION
CREDIBILITY_RECENT_YEARS_WINDOW = 5


def longest_consecutive_qualifying_run(years: np.ndarray, values: np.ndarray, threshold: float) -> int:
    """`years`/`values` must already be sorted by year ascending (one row
    per year, no duplicate years). Returns the longest run of consecutive
    calendar years where the log-growth from the immediately preceding
    year meets `threshold` — a gap year (or a transition below threshold)
    resets the run, matching the "no non-consecutive transitions" rule
    used throughout this project."""
    best = current = 0
    for i in range(1, len(years)):
        is_consecutive_year = years[i] - years[i - 1] == 1
        if is_consecutive_year:
            growth = np.log1p(values[i]) - np.log1p(values[i - 1])
            qualifies = growth >= threshold
        else:
            qualifies = False
        current = current + 1 if qualifies else 0
        best = max(best, current)
    return best


def compute_gazelle_runs(full_trajectories: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for company_id, g in full_trajectories.sort_values("accounting_year").groupby("company_id"):
        years = g["accounting_year"].to_numpy()
        turnover = g["turnover"].to_numpy()
        rows.append(
            {
                "company_id": company_id,
                "longest_run_10pct": longest_consecutive_qualifying_run(years, turnover, GAZELLE_THRESHOLDS["10pct"]),
                "longest_run_20pct": longest_consecutive_qualifying_run(years, turnover, GAZELLE_THRESHOLDS["20pct"]),
            }
        )
    runs = pd.DataFrame(rows)
    runs["gazelle_10pct"] = runs["longest_run_10pct"] >= GAZELLE_CONSECUTIVE_YEARS
    runs["gazelle_20pct"] = runs["longest_run_20pct"] >= GAZELLE_CONSECUTIVE_YEARS
    return runs


def longest_consecutive_qualifying_run_from_values(years: np.ndarray, metric_values: np.ndarray, threshold: float) -> int:
    """Like longest_consecutive_qualifying_run, but `metric_values[i]` is
    ALREADY a precomputed one-year log-growth rate (employee_growth/
    asset_growth), not a raw level to be differenced. Checked against the
    FULL per-company year sequence (not pre-filtered to non-null rows) so
    calendar adjacency is verified against real years, not against
    whichever rows happen to have a non-null metric — a metric that's
    null for an in-between year must break the run here exactly like a
    genuine gap year would, not be silently skipped over."""
    best = current = 0
    for i in range(1, len(years)):
        is_consecutive_year = years[i] - years[i - 1] == 1
        qualifies = is_consecutive_year and pd.notna(metric_values[i]) and metric_values[i] >= threshold
        current = current + 1 if qualifies else 0
        best = max(best, current)
    return best


def compute_operational_scaling(real_panel: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    engineered, _ = build_engineered_panel(real_panel, baseline)

    rows = []
    for company_id, g in engineered.sort_values("accounting_year").groupby("company_id"):
        years = g["accounting_year"].to_numpy()
        row = {"company_id": company_id}
        for metric in ["employee_growth", "asset_growth"]:
            metric_values = g[metric].to_numpy()
            n_real = int(g[metric].notna().sum())
            row[f"n_real_years_{metric}"] = n_real
            if n_real < GAZELLE_CONSECUTIVE_YEARS:
                row[f"{metric}_10pct"] = False
                row[f"{metric}_20pct"] = False
                continue
            row[f"{metric}_10pct"] = (
                longest_consecutive_qualifying_run_from_values(years, metric_values, GAZELLE_THRESHOLDS["10pct"])
                >= GAZELLE_CONSECUTIVE_YEARS
            )
            row[f"{metric}_20pct"] = (
                longest_consecutive_qualifying_run_from_values(years, metric_values, GAZELLE_THRESHOLDS["20pct"])
                >= GAZELLE_CONSECUTIVE_YEARS
            )
        rows.append(row)

    scaling = pd.DataFrame(rows)
    scaling["operational_scaling_10pct"] = scaling["employee_growth_10pct"] | scaling["asset_growth_10pct"]
    scaling["operational_scaling_20pct"] = scaling["employee_growth_20pct"] | scaling["asset_growth_20pct"]
    return scaling


def compute_context_columns(full_trajectories: pd.DataFrame, trajectories: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    n_real_years = (
        full_trajectories[full_trajectories["data_type"] == "observed"].groupby("company_id").size().rename("n_real_years")
    )

    if len(trajectories):
        evidence_gate_triggered = (
            trajectories.assign(triggered=(trajectories["growth_signal"] == "growing") & (trajectories["growth_classification"] == "stable"))
            .groupby("company_id")["triggered"]
            .any()
            .rename("evidence_gate_triggered")
        )
    else:
        evidence_gate_triggered = pd.Series(name="evidence_gate_triggered", dtype=bool)

    context = summary[["company_id", "forecast_evidence_group", "turnover_source", "ever_classified_growing"]].copy()
    context = context.rename(columns={"turnover_source": "baseline_turnover_source", "ever_classified_growing": "growth_decay_applied"})
    context = context.merge(n_real_years, on="company_id", how="left")
    context = context.merge(evidence_gate_triggered, on="company_id", how="left")
    context["n_real_years"] = context["n_real_years"].fillna(0).astype(int)
    context["evidence_gate_triggered"] = context["evidence_gate_triggered"].fillna(False)
    return context


def compute_credibility_gate(full_trajectories: pd.DataFrame, evidence_group: pd.Series) -> pd.Series:
    observed = full_trajectories[full_trajectories["data_type"] == "observed"].sort_values(
        ["company_id", "accounting_year"], ascending=[True, False]
    )
    recent = observed.groupby("company_id").head(CREDIBILITY_RECENT_YEARS_WINDOW)
    max_recent_turnover = recent.groupby("company_id")["turnover"].max()

    has_evidence = evidence_group == CREDIBILITY_MIN_EVIDENCE_GROUP
    has_recent_scale = max_recent_turnover.reindex(evidence_group.index).fillna(0) >= CREDIBILITY_MIN_RECENT_TURNOVER
    return has_evidence & has_recent_scale


def main() -> None:
    for path in (ORDERED_PANEL_PATH, BASELINE_VALIDATED_PATH, FULL_TRAJECTORIES_PATH, SUMMARY_PATH, TRAJECTORIES_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}: run the earlier forecast_src build-order steps first.")

    real_panel = pd.read_csv(ORDERED_PANEL_PATH)
    baseline = pd.read_csv(BASELINE_VALIDATED_PATH)
    full_trajectories = pd.read_csv(FULL_TRAJECTORIES_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    trajectories = pd.read_csv(TRAJECTORIES_PATH)

    context = compute_context_columns(full_trajectories, trajectories, summary)
    base = summary[["company_id", "company_name", "mission", "baseline_year", "baseline_turnover", "turnover_2030"]].merge(
        context, on="company_id", how="left"
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. £10M-by-2030 crossings (from below only) ---
    crossings = base[(base["baseline_turnover"] < TEN_MILLION) & (base["turnover_2030"] >= TEN_MILLION)].copy()
    crossings_path = DATA_DIR / "forecast_10m_crossings.csv"
    crossings.to_csv(crossings_path, index=False)

    # --- 2. Gazelle tiers (walks the FULL trajectory, not just baseline-vs-2030) ---
    gazelle_runs = compute_gazelle_runs(full_trajectories)
    gazelle_10pct = base.merge(gazelle_runs, on="company_id", how="left")
    gazelle_10pct = gazelle_10pct[gazelle_10pct["gazelle_10pct"]]
    gazelle_20pct = base.merge(gazelle_runs, on="company_id", how="left")
    gazelle_20pct = gazelle_20pct[gazelle_20pct["gazelle_20pct"]]
    gazelle_10_path = DATA_DIR / "forecast_gazelle_10pct.csv"
    gazelle_20_path = DATA_DIR / "forecast_gazelle_20pct.csv"
    gazelle_10pct.to_csv(gazelle_10_path, index=False)
    gazelle_20pct.to_csv(gazelle_20_path, index=False)

    # --- 3. £50M intersection + credibility gate ---
    gazelle_either = base.merge(gazelle_runs, on="company_id", how="left")
    gazelle_either = gazelle_either[gazelle_either["gazelle_10pct"] | gazelle_either["gazelle_20pct"]]
    intersection = gazelle_either[gazelle_either["turnover_2030"] >= FIFTY_MILLION].copy()
    evidence_group_indexed = intersection.set_index("company_id")["forecast_evidence_group"]
    credibility_pass = compute_credibility_gate(full_trajectories, evidence_group_indexed)
    intersection["credibility_status"] = intersection["company_id"].map(
        lambda cid: "meets_credibility_threshold" if credibility_pass.get(cid, False) else "does_not_meet_credibility_threshold"
    )
    intersection_path = DATA_DIR / "forecast_gazelle_50m_intersection.csv"
    intersection.to_csv(intersection_path, index=False)

    # --- 4. Operational scaling (independent of turnover) ---
    scaling = compute_operational_scaling(real_panel, baseline)
    scaling_out = base.merge(scaling, on="company_id", how="left")
    scaling_out = scaling_out[scaling_out["operational_scaling_10pct"].fillna(False)]
    scaling_path = DATA_DIR / "forecast_operational_scaling.csv"
    scaling_out.to_csv(scaling_path, index=False)

    # --- Report ---
    print("=== 1. £10M-by-2030 crossings ===")
    print(f"Total: {len(crossings)}")
    print(crossings.groupby("mission").size().to_string())

    print("\n=== 2. Gazelle >=10% YoY, 3+ consecutive years ===")
    print(f"Total: {len(gazelle_10pct)}")
    print(gazelle_10pct.groupby("mission").size().to_string())

    print("\n=== 2. Gazelle >=20% YoY, 3+ consecutive years ===")
    print(f"Total: {len(gazelle_20pct)}")
    print(gazelle_20pct.groupby("mission").size().to_string())

    print("\n=== 3. Gazelle (either tier) AND >=£50M by 2030 ===")
    print(f"Total: {len(intersection)}")
    print(intersection.groupby("mission").size().to_string())
    print(intersection["credibility_status"].value_counts().to_string())

    print("\n=== 4. Operational scaling (employee_growth or asset_growth, >=10%, 3+ real years) ===")
    print(f"Total: {len(scaling_out)}")
    print(scaling_out.groupby("mission").size().to_string())

    print(f"\nWrote {crossings_path}")
    print(f"Wrote {gazelle_10_path}")
    print(f"Wrote {gazelle_20_path}")
    print(f"Wrote {intersection_path}")
    print(f"Wrote {scaling_path}")


if __name__ == "__main__":
    main()
