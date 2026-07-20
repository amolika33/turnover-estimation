"""Applies each company's routed one-year-ahead model recursively from its
own baseline year out to 2030 (build order step 7).

ROUTING RULE (stated assumption, not implicit in the code — see
CLAUDE.md's "2030 Forecasting Pipeline" section for the full investigation
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
MISSIONS_NEEDING_FITTED_ROUTING_MODEL = ["ACE"]  # only Ridge (ACE) needs an actual fit; CAGR/Persistence are formulas

STATIC_COLS = {"Founded": "founded_year", "SIC Code 1": "sic_code_1", "Value Stream": "value_stream"}


def classify_growth(row: pd.Series) -> str:
    signal = row.get("log_growth_3y_mean")
    if pd.isna(signal):
        signal = row.get("log_growth_1y")
    if pd.isna(signal):
        return "stable"
    return "growing" if signal > GROWTH_THRESHOLD else "stable"


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


def predict_step(engineered: pd.DataFrame, origin_keys: pd.DataFrame, routing_models: dict) -> pd.DataFrame:
    """`origin_keys` is one row per active company: (company_id, accounting_year)
    of the row to predict FROM. Returns the new (company_id, next_year,
    turnover, model_used, growth_classification, ...carried-forward fields)
    rows — one per active company."""
    origin = engineered.merge(origin_keys, on=["company_id", "accounting_year"], how="inner")
    origin["growth_classification"] = origin.apply(classify_growth, axis=1)

    origin_cast = cast_categoricals(origin)
    predictions = np.full(len(origin), np.nan)

    for mission, growth_model_name in GROWTH_ROUTING_MODEL.items():
        is_growing_this_mission = (origin["mission"] == mission) & (origin["growth_classification"] == "growing")
        if not is_growing_this_mission.any():
            continue
        subset = origin_cast[is_growing_this_mission]
        if mission in routing_models:
            preds = routing_models[mission].predict(subset[FEATURE_COLUMNS])
        else:
            preds = BENCHMARK_PREDICT_FNS[growth_model_name](pd.DataFrame(), subset)
        predictions[is_growing_this_mission.to_numpy()] = preds

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
         "predicted_turnover", "model_used", "growth_classification"]
    ].copy()
    new_rows["accounting_year"] = new_rows["accounting_year"] + 1
    new_rows["company_age"] = new_rows["accounting_year"] - new_rows["founded_year"]
    new_rows = new_rows.rename(columns={"predicted_turnover": "turnover"})
    return new_rows


def run_recursive_forecast(panel: pd.DataFrame, baseline: pd.DataFrame, routing_models: dict) -> pd.DataFrame:
    panel = seed_missing_companies(panel, baseline)

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
        new_rows = predict_step(engineered, origin_keys, routing_models)

        valid_mask = new_rows["turnover"].notna()
        n_invalid = int((~valid_mask).sum())
        if n_invalid:
            print(f"  step {step}: {n_invalid} invalid predictions dropped (company stops advancing from here)")

        new_rows_valid = new_rows[valid_mask].copy()
        new_rows_valid["step"] = step
        trajectory_rows.append(new_rows_valid.drop(columns=["founded_year"]))

        panel = pd.concat([panel, new_rows_valid.drop(columns=["model_used", "growth_classification", "step"])], ignore_index=True)

        for _, row in new_rows_valid.iterrows():
            current_year[row["company_id"]] = row["accounting_year"]
        for cid in active_ids:
            if cid not in {r for r in new_rows_valid["company_id"]}:
                current_year[cid] = FORECAST_END_YEAR  # stop advancing a company whose prediction failed

        print(f"  step {step}: predicted {len(new_rows_valid)} company-years, {sum(1 for y in current_year.values() if y < FORECAST_END_YEAR)} companies still active")

    return pd.concat(trajectory_rows, ignore_index=True) if trajectory_rows else pd.DataFrame()


def main() -> None:
    for path in (ORDERED_PANEL_PATH, BASELINE_VALIDATED_PATH):
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
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
