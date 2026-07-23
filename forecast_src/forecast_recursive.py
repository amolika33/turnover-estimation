"""Applies each company's routed one-year-ahead model recursively from its
own baseline year out to 2030 (build order step 7).

ROUTING RULE (stated assumption, not implicit in the code — see
PROJECT_NOTES.md's "2030 Forecasting Pipeline" section for the full investigation
that produced it): forecast_selection.py found Persistence (no-growth)
winning horizon=1 in all 3 missions, but a follow-up check
(forecast_evidence_group_diagnostic.py) split that result by actual growth
trajectory and found it was NOT uniform —

  - ACE: Persistence's win WAS a flat-majority artifact. Restricted to
    companies growing >=10% YoY, Ridge clearly wins (R2=0.982 vs
    Persistence not even in the top 5).
  - Beyond Earth / Resilient Earth: Persistence's win holds even
    restricted to growing companies — genuinely the better one-step
    predictor there, not an artifact.

But "best one-step predictor on average" and "fit for this project's
purpose" are different questions: Persistence is structurally INCAPABLE of
ever forecasting growth (turnover_t+1 = turnover_t, always), so applying it
recursively for 5-17 steps to 2030 would flatten every company's
trajectory regardless of mission — directly undermining the project's core
objective of identifying companies that cross £10M/£50M or sustain
gazelle-tier growth. So routing is GROWTH-TRAJECTORY-CONDITIONAL PER
COMPANY, not fixed per mission:

  GROWTH_THRESHOLD = log1p(0.10) (~0.0953): a company is "growing" if its
  log_growth_3y_mean (falling back to log_growth_1y, then "stable" if
  neither is available) exceeds this. Chosen to match the same 10% YoY cut
  the diagnostic used and the planned gazelle criteria will use — one
  reused, already-validated number, not a second arbitrary threshold.

  log_growth_3y_mean over log_growth_1y as the PRIMARY signal: recursive
  stepping means later years' "history" includes the model's OWN earlier
  predictions, not just real observations — a single noisy predicted year
  feeding a single-year growth signal risks the routing flipping
  unpredictably step to step. A 3-year rolling average is far more stable,
  and is thematically consistent with the project's own planned gazelle
  definition requiring 3 CONSECUTIVE years of sustained growth, not a
  single-year spike.

  "growing" -> Ridge (ACE, per the diagnostic) or Company Historical CAGR
  (Beyond Earth / Resilient Earth / Cross-Cutting). Ridge vs CAGR for the
  non-ACE missions is an explicit choice, not a default: re-examining the
  diagnostic's growing-company subset, CAGR beats Ridge in BOTH remaining
  missions (Beyond Earth: CAGR R2=0.912 vs Ridge not in top 5; Resilient
  Earth: CAGR R2=0.969 vs Ridge R2=0.962) — and CAGR is philosophically the
  right tool for "continue THIS company's own established trend" in a
  per-company recursive context, whereas Ridge's growth signal comes from
  patterns across the whole mission population. Cross-Cutting companies
  use CAGR only, never Ridge — there is no Cross-Cutting-specific fitted
  model (mission_segmentation.py never trains one), consistent with why
  they were segregated from mission-specific modelling in the first place.
  "stable" (including declining) -> Persistence, uniformly. The diagnostic
  only examined growing vs not-growing; there's no evidence a separate
  declining-company treatment is needed, so none is invented here.

DYNAMIC vs FIXED-AT-BASELINE CLASSIFICATION: re-evaluated at EVERY
recursive step, not fixed once at baseline. Fixing it at baseline would let
a company with one early growth burst get a trend-continuation model
(Ridge/CAGR) blindly applied for up to 17 consecutive steps — compounding
an anomaly that long is not credible (a company appearing to grow 30% in
one year, extrapolated at that rate for 7 years, would appear to reach
~6.6x its baseline turnover by 2030 without ever being allowed to
decelerate). Re-evaluating lets a company's routing shift to Persistence
the moment its own trailing growth genuinely cools off, using only
information available at or before that step (sec 7's rule, extended
naturally into the recursive phase). The oscillation risk this could
introduce is why log_growth_3y_mean (smoothed) is the primary signal
instead of the noisier log_growth_1y — the smoothing and the dynamic
re-evaluation are one combined design decision, not two independent ones.

NON-DETERMINISTIC vs DETERMINISTIC future predictors (per the original task
brief): company_age updates deterministically every step
(= year - founded_year, always computable). employees/total_assets/
export_revenue/company_size have no future values available at all (Source
2 has no columns beyond its real historical years) and are held at each
company's own last observed-or-predicted value going forward — carried
forward automatically here since each new synthetic row copies these
fields straight from the row used as that step's prediction origin.

MINIMUM-EVIDENCE GATE ON TREND-CONTINUATION ROUTING (added after the first
full run surfaced runaway compounding — stated assumption, same treatment
as GROWTH_THRESHOLD): the first full 2030 run produced 16 companies with a
>100x baseline multiple by 2030 (9 of them >1000x, one >7 million x). Every
one traced back to a company with only 1-2 real year-over-year transitions
showing one large early jump (e.g. TerraFarmer: £75,592 -> £721,789,
2022->2023, its only 2 real years) — genuine data, but weak evidence for a
SUSTAINABLE rate. Company Historical CAGR/Ridge under pure recursion has no
natural deceleration: the realized one-year growth between two
model-PREDICTED years exactly equals whatever rate was applied to produce
them, so once anchored to an extreme rate from thin evidence, a company's
own synthetic history "confirms" that same rate every subsequent step
forever — dynamic re-classification (above) can't catch this, because the
smoothed growth signal computed from that synthetic history keeps
reporting the same extreme value it was fed.

Fix: MIN_EVIDENCE_GROUP_FOR_TREND_CONTINUATION = "A" (>=3 REAL historical
turnover years, forecast_sample_construction.py's evidence_group — a
STATIC, pre-recursion classification that never changes during a
company's own recursive run, unlike growth_signal above). A company
classified "growing" by growth_signal but in evidence Group B/C/D falls
back to Persistence regardless of its measured rate — one or two real
transitions is real data, but not enough to trust as a sustainable
per-year rate to compound for up to 17 years. This does NOT fully resolve
the problem: SaxaVord Spaceport (Group A, 3 real years: 47,465 -> 508,000
-> 1,866,000) and Eutelsat OneWeb (Group A, 9 real years, volatile) both
have genuinely evidence-backed extreme historical rates and are NOT
filtered by this gate — see the re-run report for whether they still
produce non-credible 2030 values and what (if anything) further mitigation
would look like. The gate targets thin EVIDENCE specifically, not
extreme RATE MAGNITUDE — those are different failure modes and this fix
only addresses the first.

GROWTH-RATE DECAY (added after the gate turned out to be a partial fix —
6 companies remained >100x baseline, ALL evidence Group A: SaxaVord
Spaceport, Infleqtion, Map of Agriculture, Eutelsat OneWeb, Sierra Nevada
Corporation, Oxa — genuinely evidence-backed small-base, high-volatility
companies the gate isn't designed to catch). The remaining root cause: a
trend-continuation model's applied rate never decelerates under pure
recursion regardless of how much real evidence backed it, because the
realized growth between two model-PREDICTED years always exactly equals
whatever rate produced them — there's no mechanism pulling an extreme
company-specific rate back toward anything more moderate as the projection
gets further from real data.

Fix: every "growing"-routed prediction (CAGR's formula output or Ridge's
regression output — both go through the same path, apply_growth_decay,
so neither needs separate handling) is converted to an implied one-year
log-growth rate and blended toward that MISSION's real-data median
log_growth_1y (compute_mission_average_growth — median, not mean,
deliberately: this project has repeatedly shown small-base companies can
produce extreme one-off values, exactly what a mean would be distorted by
and exactly what decay is trying to pull companies away from), with a
blend weight that shrinks by recursive STEP (not calendar year, since what
matters is distance from real evidence, not the calendar):

  weight_company(step) = 0.5 ** (step / GROWTH_DECAY_HALF_LIFE_STEPS),
  HALF_LIFE_STEPS = 2.0 -> ~0.71 at step 1 (mostly the company's own
  rate), 0.5 at step 2, ~0.18 at step 5, ~0.09 at step 7 (mostly mission
  average) — matches "mostly-company-rate at step 1, mostly-mission-
  average by step 5+" from the brief. Exponential/half-life over linear
  decay-to-zero: never fully reaches 0, so a company's own evidence always
  retains SOME influence no matter how far out the projection runs, rather
  than assuming every company's trajectory becomes indistinguishable from
  the mission average given enough time (a stronger, less defensible
  claim than "regress toward the mission, don't ignore the company").

Persistence is untouched by decay — it has no rate to decay (it already
represents "no growth"), and decay is specifically about tempering a
trend-continuation model's rate, not a second growth mechanism of its own.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from forecast_src.forecast_bakeoff import (
    BENCHMARK_PREDICT_FNS,
    FEATURE_COLUMNS,
    cast_categoricals,
)
from forecast_src.forecast_data_prep import DATA_DIR
from forecast_src.forecast_feature_engineering import build_engineered_panel
from forecast_src.forecast_selection import fit_final_ml_model
from src.data_prep import COMPANY_ID_COL, prepare_source2
from src.mission_segmentation import MISSION_COL, REAL_MISSIONS, load_mapping, segment_missions

REPO_ROOT = Path(__file__).resolve().parents[1]
ORDERED_PANEL_PATH = DATA_DIR / "forecast_ordered_panel.csv"
BASELINE_VALIDATED_PATH = DATA_DIR / "forecast_baseline_validated.csv"

FORECAST_END_YEAR = 2030
GROWTH_THRESHOLD = np.log1p(0.10)  # ~0.0953, same 10% YoY cut used throughout this project

GROWTH_ROUTING_MODEL = {
    "ACE": "Ridge",
    "Beyond Earth": "Company Historical CAGR",
    "Resilient Earth": "Company Historical CAGR",
    "Cross-Cutting": "Company Historical CAGR",
}
STABLE_MODEL = "Persistence"

EVIDENCE_GROUPS_PATH = DATA_DIR / "forecast_evidence_groups.csv"
# Minimum REAL-history evidence a company must have (forecast_sample_
# construction.py's static, pre-recursion evidence_group) to be eligible
# for trend-continuation routing even when growth_signal says "growing" —
# see module docstring's "MINIMUM-EVIDENCE GATE" section for why this was
# added and what it does/doesn't fix.
MIN_EVIDENCE_GROUP_FOR_TREND_CONTINUATION = "A"
MISSIONS_NEEDING_FITTED_ROUTING_MODEL = ["ACE"]  # only Ridge (ACE) needs an actual fit; CAGR/Persistence are formulas

# GROWTH-RATE DECAY (added after the evidence gate turned out to be a
# partial fix — see module docstring's "GROWTH-RATE DECAY" section):
# half-life in RECURSIVE STEPS, not calendar years, since decay must track
# each company's own distance from real evidence, not the calendar. weight
# = 0.5 ** (step / HALF_LIFE): ~0.71 at step 1 (mostly the company's own
# rate), 0.5 at step 2, ~0.18 at step 5, ~0.09 at step 7 (mostly mission
# average) — matches the brief's "mostly-company at step 1, mostly-
# mission-average by step 5+" without a hard cutoff (never reaches exactly
# 0, unlike a linear schedule — some company-specific signal always
# remains, which is more defensible than assuming every company reverts
# fully to the mission mean given enough time).
GROWTH_DECAY_HALF_LIFE_STEPS = 2.0

STATIC_COLS = {"Founded": "founded_year", "SIC Code 1": "sic_code_1", "Value Stream": "value_stream"}


def classify_growth(row: pd.Series) -> str:
    signal = row.get("log_growth_3y_mean")
    if pd.isna(signal):
        signal = row.get("log_growth_1y")
    if pd.isna(signal):
        return "stable"
    return "growing" if signal > GROWTH_THRESHOLD else "stable"


def compute_mission_average_growth(real_panel: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    """MEDIAN (not mean) log_growth_1y per mission, computed ONCE from the
    REAL, pre-recursion historical panel only — a stable anchor to decay
    toward, unaffected by the recursion's own synthetic predictions. Median
    over mean is deliberate: this project has already shown small-base
    companies can produce extreme one-off growth values (Earth-i's 394x
    jump, the 6 remaining outlier companies here) that would badly distort
    a mean — the whole point of this anchor is to be a level-headed
    reference point uncontaminated by exactly the kind of extreme value
    decay is meant to pull companies away from."""
    engineered, _ = build_engineered_panel(real_panel, baseline)
    return engineered.groupby("mission")["log_growth_1y"].median().to_dict()


def apply_growth_decay(
    turnover_t: np.ndarray, raw_predicted_turnover: np.ndarray, step: int, mission_avg_growth_rate: float
) -> np.ndarray:
    """Converts a trend-continuation model's raw prediction (CAGR's formula
    output or Ridge's regression output — both go through this same path,
    so both get decayed identically rather than needing separate logic) to
    an implied one-year log-growth rate, blends it toward the mission's
    real-data median rate by `step`'s decay weight, then reconstructs the
    final turnover from the blended rate."""
    implied_rate = np.log1p(raw_predicted_turnover) - np.log1p(turnover_t)
    weight_company = 0.5 ** (step / GROWTH_DECAY_HALF_LIFE_STEPS)
    blended_rate = weight_company * implied_rate + (1 - weight_company) * mission_avg_growth_rate
    return np.expm1(np.log1p(turnover_t) + blended_rate)


def fit_growth_routing_models() -> dict:
    """Only ACE's "growing" branch (Ridge) needs an actual fitted model —
    CAGR and Persistence are formulas (BENCHMARK_PREDICT_FNS), applied
    directly at each step, never fitted or saved."""
    from forecast_src.forecast_bakeoff import get_mission_training_data

    models = {}
    for mission in MISSIONS_NEEDING_FITTED_ROUTING_MODEL:
        model_name = GROWTH_ROUTING_MODEL[mission]
        mission_df = get_mission_training_data(mission)
        fitted, params = fit_final_ml_model(mission_df, model_name)
        models[mission] = fitted
        model_path = DATA_DIR / f"forecast_growth_routing_model_{mission.lower().replace(' ', '_')}.joblib"
        joblib.dump(fitted, model_path)
        print(f"Fit growth-routing model for {mission}: {model_name}, wrote {model_path}")
    return models


def seed_missing_companies(panel: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Companies with turnover_source=="estimated" (baseline came from the
    turnover-ESTIMATION pipeline's own prediction, real-mission "predicted"
    or cross-cutting "approximate") have ZERO rows in the historical
    panel — that's exactly why they were predicted in the first place. Each
    needs one seed row at its own baseline_year/baseline_turnover before
    recursion can step forward from it. employees/total_assets/
    export_revenue/company_size are left null (no covariate data exists for
    these companies either) — the model pipeline's own imputer fills them
    in, same as predict.py's zero-covariate inference companies."""
    prepped, _ = prepare_source2()
    mapping = load_mapping()
    segmented, _ = segment_missions(prepped, mapping)
    static = segmented[[COMPANY_ID_COL] + list(STATIC_COLS)].rename(columns=STATIC_COLS)

    existing_keys = set(zip(panel["company_id"], panel["accounting_year"]))
    needs_seed = baseline[baseline["turnover_source"] == "estimated"].copy()
    needs_seed = needs_seed[
        ~needs_seed.apply(lambda r: (r["company_id"], r["baseline_year"]) in existing_keys, axis=1)
    ]

    seed_rows = needs_seed[["company_id", "mission", "baseline_year", "baseline_turnover"]].rename(
        columns={"baseline_year": "accounting_year", "baseline_turnover": "turnover"}
    )
    seed_rows = seed_rows.merge(static, on="company_id", how="left")
    for col in ["total_assets", "employees", "export_revenue", "company_size"]:
        seed_rows[col] = np.nan
    seed_rows["company_age"] = seed_rows["accounting_year"] - seed_rows["founded_year"]

    return pd.concat([panel, seed_rows], ignore_index=True)


def load_evidence_groups() -> pd.Series:
    df = pd.read_csv(EVIDENCE_GROUPS_PATH)
    return df.set_index("company_id")["forecast_evidence_group"]


def predict_step(
    engineered: pd.DataFrame,
    origin_keys: pd.DataFrame,
    routing_models: dict,
    evidence_groups: pd.Series,
    step: int,
    mission_avg_growth: dict,
) -> pd.DataFrame:
    """`origin_keys` is one row per active company: (company_id, accounting_year)
    of the row to predict FROM. Returns the new (company_id, next_year,
    turnover, model_used, growth_classification, ...carried-forward fields)
    rows — one per active company.

    `growth_signal` (raw, from the growth features) and `growth_classification`
    (the gated, ACTUALLY-USED-FOR-ROUTING decision) are kept as separate
    columns — a company can show growth_signal="growing" but still get
    routed to Persistence if its evidence_group is below the minimum (see
    module docstring's "MINIMUM-EVIDENCE GATE" section), and that
    distinction needs to stay visible, not collapsed into one label.

    The "growing" branch's raw model output (CAGR formula or Ridge
    regression) is passed through apply_growth_decay before being used —
    Persistence is untouched, since decay is specifically about a
    trend-continuation model's applied RATE, and Persistence has no rate
    to decay (it's already "no growth")."""
    origin = engineered.merge(origin_keys, on=["company_id", "accounting_year"], how="inner")
    origin["growth_signal"] = origin.apply(classify_growth, axis=1)
    origin["evidence_group"] = origin["company_id"].map(evidence_groups).fillna("D")
    origin["growth_classification"] = np.where(
        (origin["growth_signal"] == "growing") & (origin["evidence_group"] == MIN_EVIDENCE_GROUP_FOR_TREND_CONTINUATION),
        "growing",
        "stable",
    )

    origin_cast = cast_categoricals(origin)
    predictions = np.full(len(origin), np.nan)

    for mission, growth_model_name in GROWTH_ROUTING_MODEL.items():
        is_growing_this_mission = (origin["mission"] == mission) & (origin["growth_classification"] == "growing")
        if not is_growing_this_mission.any():
            continue
        subset = origin_cast[is_growing_this_mission]
        if mission in routing_models:
            raw_preds = routing_models[mission].predict(subset[FEATURE_COLUMNS])
        else:
            raw_preds = BENCHMARK_PREDICT_FNS[growth_model_name](pd.DataFrame(), subset)
        turnover_t = origin.loc[is_growing_this_mission, "turnover_t"].to_numpy()
        decayed_preds = apply_growth_decay(turnover_t, raw_preds, step, mission_avg_growth[mission])
        predictions[is_growing_this_mission.to_numpy()] = decayed_preds

    is_stable = origin["growth_classification"] == "stable"
    if is_stable.any():
        predictions[is_stable.to_numpy()] = BENCHMARK_PREDICT_FNS[STABLE_MODEL](pd.DataFrame(), origin[is_stable])

    origin["predicted_turnover"] = predictions
    origin["model_used"] = np.where(
        origin["growth_classification"] == "growing", origin["mission"].map(GROWTH_ROUTING_MODEL), STABLE_MODEL
    )

    is_invalid = ~np.isfinite(origin["predicted_turnover"]) | (origin["predicted_turnover"] < 0)
    if is_invalid.any():
        origin.loc[is_invalid, "predicted_turnover"] = np.nan

    new_rows = origin[
        ["company_id", "mission", "accounting_year", "founded_year", "sic_code_1", "value_stream",
         "total_assets", "employees", "export_revenue", "company_size",
         "predicted_turnover", "model_used", "growth_signal", "evidence_group", "growth_classification"]
    ].copy()
    new_rows["accounting_year"] = new_rows["accounting_year"] + 1
    new_rows["company_age"] = new_rows["accounting_year"] - new_rows["founded_year"]
    new_rows = new_rows.rename(columns={"predicted_turnover": "turnover"})
    return new_rows


def run_recursive_forecast(panel: pd.DataFrame, baseline: pd.DataFrame, routing_models: dict) -> pd.DataFrame:
    # Mission-average growth anchor computed ONCE, from the real (pre-seed,
    # pre-recursion) panel only — must not be recomputed each step, or it
    # would itself get contaminated by the recursion's own synthetic
    # predictions, defeating the point of a stable decay target.
    mission_avg_growth = compute_mission_average_growth(panel, baseline)
    print(f"Mission-average (median) log_growth_1y, decay anchor: {mission_avg_growth}")

    panel = seed_missing_companies(panel, baseline)
    evidence_groups = load_evidence_groups()

    current_year = baseline.set_index("company_id")["baseline_year"].to_dict()
    trajectory_rows = []

    step = 0
    while any(y < FORECAST_END_YEAR for y in current_year.values()):
        step += 1
        active_ids = [cid for cid, y in current_year.items() if y < FORECAST_END_YEAR]
        origin_keys = pd.DataFrame(
            {"company_id": active_ids, "accounting_year": [current_year[cid] for cid in active_ids]}
        )

        engineered, _ = build_engineered_panel(panel, baseline)
        new_rows = predict_step(engineered, origin_keys, routing_models, evidence_groups, step, mission_avg_growth)

        valid_mask = new_rows["turnover"].notna()
        n_invalid = int((~valid_mask).sum())
        if n_invalid:
            print(f"  step {step}: {n_invalid} invalid predictions dropped (company stops advancing from here)")

        new_rows_valid = new_rows[valid_mask].copy()
        new_rows_valid["step"] = step
        trajectory_rows.append(new_rows_valid.drop(columns=["founded_year"]))

        panel = pd.concat(
            [panel, new_rows_valid.drop(columns=["model_used", "growth_signal", "evidence_group", "growth_classification", "step"])],
            ignore_index=True,
        )

        for _, row in new_rows_valid.iterrows():
            current_year[row["company_id"]] = row["accounting_year"]
        for cid in active_ids:
            if cid not in {r for r in new_rows_valid["company_id"]}:
                current_year[cid] = FORECAST_END_YEAR  # stop advancing a company whose prediction failed

        print(f"  step {step}: predicted {len(new_rows_valid)} company-years, {sum(1 for y in current_year.values() if y < FORECAST_END_YEAR)} companies still active")

    return pd.concat(trajectory_rows, ignore_index=True) if trajectory_rows else pd.DataFrame()


def main() -> None:
    for path in (ORDERED_PANEL_PATH, BASELINE_VALIDATED_PATH, EVIDENCE_GROUPS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}: run the earlier forecast_src build-order steps first.")

    panel = pd.read_csv(ORDERED_PANEL_PATH)
    baseline = pd.read_csv(BASELINE_VALIDATED_PATH)

    print("Fitting growth-routing models...")
    routing_models = fit_growth_routing_models()

    print("\nRunning recursive forecast to 2030...")
    trajectories = run_recursive_forecast(panel, baseline, routing_models)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "forecast_trajectories.csv"
    trajectories.to_csv(out_path, index=False)

    print(f"\nTotal predicted company-years: {len(trajectories)}")
    print(f"Companies reaching 2030: {(trajectories[trajectories['accounting_year'] == FORECAST_END_YEAR]['company_id'].nunique())}")
    print("\nModel usage breakdown (all steps, all companies):")
    print(trajectories["model_used"].value_counts().to_string())

    gated = (trajectories["growth_signal"] == "growing") & (trajectories["growth_classification"] == "stable")
    print(f"\nCompany-year steps where the evidence gate downgraded growing->stable: {int(gated.sum())}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
