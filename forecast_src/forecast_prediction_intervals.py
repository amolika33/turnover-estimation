"""Residual-based confidence intervals for the recursive 2030 forecast,
stratified by evidence group (A/B/C/D).

METHOD (documented here + PROJECT_NOTES.md, per project convention):

1. Per-step residual spread. The deployed one-year-ahead mechanism is
   Persistence for the "stable" majority (forecast_selection.py's actual
   horizon=1 winner in all 3 real missions) — and Persistence's real-data
   residual has a clean closed form: predicting turnover_t+1 = turnover_t
   means the LOG-space residual log1p(actual) - log1p(predicted) is
   exactly log1p(turnover_t+1) - log1p(turnover_t), i.e. log_growth_1y
   itself. So Persistence's historical out-of-fold residual distribution
   is just the real, already-computed log_growth_1y values — no
   refitting needed, and no leakage risk to guard against (unlike an ML
   model's OOF predictions, Persistence has no parameters to fit in the
   first place).

   sigma_log_residual is the STANDARD DEVIATION of real log_growth_1y,
   computed separately per (mission, forecast_evidence_group) from
   forecast_ordered_panel.csv's real transitions only (never from
   recursion's own synthetic predictions — same "stable anchor" principle
   as forecast_recursive.py's mission-average growth decay target).

   SCOPE, STATED NOT SILENT: this uses Persistence's residual spread
   uniformly for every predicted row, even the "growing" steps actually
   produced by Ridge/CAGR. Those trend-continuation models likely carry
   their own, probably WIDER, true uncertainty — not modelled separately
   here. The reported band should be read as a lower bound specifically
   for growth_classification="growing" rows.

2. Fallback for thin groups. Group C (exactly 1 real year) and Group D
   (0 real years, estimated baseline) have NO real year-over-year
   transition at all — log_growth_1y is undefined for them structurally,
   not just sparse. MIN_SAMPLES_FOR_GROUP_SPREAD=5 also catches any
   (mission, group) cell with too few real transitions to trust a std
   estimate (checked empirically: Group B ranges 6-15 real transitions
   per mission, above this bar; Group A ranges 575-1564).

   Cells below the bar fall back to the WORST (highest-std) evidence tier
   in that mission that still clears the sample-size bar — NOT the
   mission's pooled std across all groups. Pooling was tried first and
   rejected: Group A's much larger sample (575-1564 vs Group B's 6-15)
   dominates a size-weighted pool, so a naive pooled fallback landed
   within a few percent of Group A's OWN spread — backwards for a company
   with LESS evidence than Group A, and it would make Group D look nearly
   as confident as the best-evidenced tier, defeating the entire point of
   stratifying by evidence group at all. Anchoring to the worst reliable
   tier (empirically Group B in every mission, since B's real transitions
   are consistently 2.5-3.5x more volatile than A's) keeps the ordering
   correctly monotonic: A (tightest) < B < C = D (widest, both anchored to
   B's spread, since neither has any real evidence of its own to differ
   by). `used_fallback_spread` records which cells this applied to.

3. Widening with horizon. Standard random-walk h-step-ahead forecast
   variance under an independent-errors assumption: if each recursive
   step's log-residual has variance sigma^2, the cumulative log-space
   variance after h steps is h * sigma^2, so the cumulative STANDARD
   DEVIATION is sigma * sqrt(h) — the same sqrt(horizon) growth used in
   ARIMA-style h-step-ahead intervals. `step` (1, 2, 3, ...) is
   forecast_recursive.py's own recursive-step counter, already in
   forecast_trajectories.csv.

   turnover_lower/upper = expm1(log1p(turnover) -+ Z * sigma_log_residual
   * sqrt(step)), Z=1.96 (~95% interval under a normal approximation —
   a stated, standard choice, not derived from this data specifically).
   Clipped at 0 on the lower side (turnover can't be negative).

Baseline/observed/estimated_baseline rows get no interval (NaN) — they're
either real filed data or the turnover-ESTIMATION pipeline's own point
value, not this pipeline's own uncertainty to characterise.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_src.forecast_data_prep import DATA_DIR
from forecast_src.forecast_feature_engineering import build_engineered_panel

REPO_ROOT = Path(__file__).resolve().parent.parent
ORDERED_PANEL_PATH = DATA_DIR / "forecast_ordered_panel.csv"
BASELINE_VALIDATED_PATH = DATA_DIR / "forecast_baseline_validated.csv"
EVIDENCE_GROUPS_PATH = DATA_DIR / "forecast_evidence_groups.csv"
TRAJECTORIES_PATH = DATA_DIR / "forecast_trajectories.csv"

Z_SCORE = 1.96  # ~95% interval, standard normal approximation — a stated choice
MIN_SAMPLES_FOR_GROUP_SPREAD = 5


def compute_residual_spread(real_panel: pd.DataFrame, baseline: pd.DataFrame, evidence_groups: pd.DataFrame) -> pd.DataFrame:
    engineered, _ = build_engineered_panel(real_panel, baseline)
    engineered = engineered.merge(evidence_groups, on="company_id", how="left")
    valid = engineered.dropna(subset=["log_growth_1y"])

    group_spread = valid.groupby(["mission", "forecast_evidence_group"])["log_growth_1y"].agg(["std", "count"]).reset_index()

    # Fallback anchor for groups with no/insufficient real transitions
    # (Group C has exactly 1 real year -> 0 transitions structurally;
    # Group D has 0 -> same). Deliberately the WORST (highest-std) tier
    # that still clears MIN_SAMPLES_FOR_GROUP_SPREAD, not the mission-wide
    # pooled std: pooling is dominated by Group A's much larger sample
    # (575-1564 vs Group B's 6-15 real transitions), so a naive pooled
    # fallback lands within a few percent of Group A's OWN spread —
    # exactly backwards for a company with LESS evidence than Group A,
    # and it would make Group D look nearly as confident as Group A,
    # defeating the entire point of stratifying by evidence at all.
    reliable = group_spread[group_spread["count"] >= MIN_SAMPLES_FOR_GROUP_SPREAD]
    worst_reliable = (
        reliable.loc[reliable.groupby("mission")["std"].idxmax()][["mission", "std"]]
        .rename(columns={"std": "fallback_std"})
    )

    all_combos = pd.MultiIndex.from_product(
        [engineered["mission"].unique(), ["A", "B", "C", "D"]], names=["mission", "forecast_evidence_group"]
    ).to_frame(index=False)
    result = all_combos.merge(group_spread, on=["mission", "forecast_evidence_group"], how="left")
    result = result.merge(worst_reliable, on="mission", how="left")
    result["count"] = result["count"].fillna(0)
    result["used_fallback_spread"] = result["count"] < MIN_SAMPLES_FOR_GROUP_SPREAD
    result["sigma_log_residual"] = np.where(result["used_fallback_spread"], result["fallback_std"], result["std"])

    return result[["mission", "forecast_evidence_group", "sigma_log_residual", "count", "used_fallback_spread"]]


def apply_prediction_intervals(trajectories: pd.DataFrame, residual_spread: pd.DataFrame) -> pd.DataFrame:
    # Idempotent: drop any columns this function (or a prior run of it)
    # already added, so re-running against an already-augmented
    # forecast_trajectories.csv doesn't collide on merge.
    stale_cols = ["sigma_log_residual", "used_fallback_spread", "turnover_lower", "turnover_upper", "count"]
    trajectories = trajectories.drop(columns=[c for c in stale_cols if c in trajectories.columns])

    merged = trajectories.merge(
        residual_spread.rename(columns={"forecast_evidence_group": "evidence_group"}),
        on=["mission", "evidence_group"],
        how="left",
    )
    half_width_log = Z_SCORE * merged["sigma_log_residual"] * np.sqrt(merged["step"])
    log_point = np.log1p(merged["turnover"])
    merged["turnover_lower"] = np.expm1(log_point - half_width_log).clip(lower=0)
    merged["turnover_upper"] = np.expm1(log_point + half_width_log)
    return merged.drop(columns=["count"])


def main() -> None:
    for path in (ORDERED_PANEL_PATH, BASELINE_VALIDATED_PATH, EVIDENCE_GROUPS_PATH, TRAJECTORIES_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}: run the earlier forecast_src build-order steps first.")

    real_panel = pd.read_csv(ORDERED_PANEL_PATH)
    baseline = pd.read_csv(BASELINE_VALIDATED_PATH)
    evidence_groups = pd.read_csv(EVIDENCE_GROUPS_PATH)[["company_id", "forecast_evidence_group"]]
    trajectories = pd.read_csv(TRAJECTORIES_PATH)

    residual_spread = compute_residual_spread(real_panel, baseline, evidence_groups)
    print("=== Residual spread (sigma_log_residual) per mission x evidence group ===")
    print(residual_spread.to_string(index=False))

    trajectories_with_intervals = apply_prediction_intervals(trajectories, residual_spread)
    trajectories_with_intervals.to_csv(TRAJECTORIES_PATH, index=False)
    print(f"\nWrote {TRAJECTORIES_PATH} (added turnover_lower/turnover_upper)")

    spread_path = DATA_DIR / "forecast_residual_spread.csv"
    residual_spread.to_csv(spread_path, index=False)
    print(f"Wrote {spread_path}")


if __name__ == "__main__":
    main()
