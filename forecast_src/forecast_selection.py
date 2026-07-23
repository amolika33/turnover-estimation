"""Select the strongest one-year-ahead forecasting model per (mission,
horizon) from forecast_bakeoff.py's saved results (methodology sec 7.11's
closing line — build order step 6). Mirrors src/model_selection.py's
selection convention exactly: robustness filter -> composite rank (6
metrics: MAE/RMSE/R2 mean+std) -> simplicity tie-break when comparable —
with one addition the bake-off's own headline finding demands.

The 3 benchmark "models" (Persistence, Mission-Average Growth, Company
Historical CAGR) require no fitting, no hyperparameters, and no
preprocessing pipeline at all — the simplest possible candidates, ranked
accordingly (SIMPLICITY_RANK=0, below even Linear Regression) — and they
won or tied in most mission/horizon combinations during the bake-off. Every
selection here is tagged `is_benchmark`: printed distinctly, written as its
own column in both output CSVs, never silently presented as if an ML model
had won. Downstream code (forecast_recursive.py) MUST branch on this flag —
a benchmark winner has no `.joblib` file to load, only a formula to apply
(see BENCHMARK_PREDICT_FNS in forecast_bakeoff.py).

Reads forecast_bakeoff_{mission}_summary.csv / _folds.csv rather than
re-running the bake-off — same reason as model_selection.py: wasted
compute (the rolling-origin bake-off takes real time per mission), and
run-to-run drift risk (guarantees this selection matches the one bake-off
run everyone is looking at).

Selection happens independently per (mission, horizon) — sec 7.11 asks for
per-horizon reporting, and a different model can legitimately win at
different horizons. But only the horizon="1" winner is ever actually
refit-on-full-data and deployed: forecast_recursive.py (build order step 7)
applies a single one-year-ahead model RECURSIVELY, one step at a time, out
to 2030 — there is no separate "horizon-4 model" to deploy. Horizons
2/3/4+ are diagnostic here (how well a horizon-1-trained model's errors
compound at increasing distance from its own training cutoff), not a
second axis for a model object to save. `is_primary_recursive_model` flags
the one row per mission that's actually refit and (for ML winners) saved.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from forecast_src.forecast_bakeoff import (
    BENCHMARK_MODELS,
    FEATURE_COLUMNS,
    INNER_CUTOFF_OFFSETS,
    MODELS,
    SCALE_SENSITIVE,
    TARGET_COL,
    TARGET_YEAR_COL,
    build_preprocessor,
    get_mission_training_data,
    make_inner_temporal_cv,
)
from forecast_src.forecast_data_prep import DATA_DIR
from src.mission_segmentation import REAL_MISSIONS

REPO_ROOT = Path(__file__).resolve().parents[1]

ACCURACY_METRICS = [("MAE_mean", True), ("RMSE_mean", True), ("R2_mean", False)]
CONSISTENCY_METRICS = [("MAE_std", True), ("RMSE_std", True), ("R2_std", True)]
RANK_METRICS = ACCURACY_METRICS + CONSISTENCY_METRICS
COMPARABLE_TOLERANCE = 1.0
R2_COMPARABLE_TOLERANCE = 0.05
BROKEN_FOLD_R2 = -2.0
BLOWUP_FOLD_MAE_MULTIPLE = 3.0
USABILITY_R2_THRESHOLD = 0.0

# Same style as src/model_selection.py's SIMPLICITY_RANK, extended with the
# 3 benchmarks at rank 0 — genuinely simpler than any fitted model (no
# fitting, no hyperparameters, no preprocessing pipeline).
SIMPLICITY_RANK = {
    "Persistence": 0,
    "Mission-Average Growth": 0,
    "Company Historical CAGR": 0,
    "Linear Regression": 1,
    "Ridge": 2,
    "Lasso": 2,
    "Elastic Net": 2,
    "SVR": 4,
    "Gradient Boosting": 5,
    "Histogram Gradient Boosting": 5,
    "Random Forest": 5,
    "Extra Trees": 5,
    "CatBoost": 5,
}

HORIZONS = ["1", "2", "3", "4+"]
PRIMARY_HORIZON = "1"


def compute_robustness(fold_detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in fold_detail.groupby("Model"):
        broken_fold = bool((g["R2"] < BROKEN_FOLD_R2).any())
        mae_median = g["MAE"].median()
        blowup_fold = bool(mae_median > 0 and (g["MAE"] > BLOWUP_FOLD_MAE_MULTIPLE * mae_median).any())
        rows.append(
            {
                "Model": model,
                "robustness_violation": broken_fold or blowup_fold,
                "had_broken_fold_r2": broken_fold,
                "had_mae_blowup_fold": blowup_fold,
                "worst_fold_r2": g["R2"].min(),
                "max_fold_mae": g["MAE"].max(),
            }
        )
    return pd.DataFrame(rows)


def rank_models(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    rank_cols = []
    for col, ascending in RANK_METRICS:
        rank_col = f"rank_{col}"
        df[rank_col] = df[col].rank(ascending=ascending, method="average")
        rank_cols.append(rank_col)
    df["composite_rank"] = df[rank_cols].mean(axis=1).round(2)
    return df


def pick_winner(ranked: pd.DataFrame) -> dict:
    survivors = ranked[~ranked["robustness_violation"]]
    fallback_used = survivors.empty
    if fallback_used:
        survivors = ranked

    top_row = survivors.sort_values("composite_rank").iloc[0]
    best_rank = top_row["composite_rank"]
    comparable = survivors[
        (survivors["composite_rank"] <= best_rank + COMPARABLE_TOLERANCE)
        & ((survivors["R2_mean"] - top_row["R2_mean"]).abs() < R2_COMPARABLE_TOLERANCE)
    ]
    comparable = comparable.sort_values(["simplicity_rank", "composite_rank"])
    winner_row = comparable.iloc[0]
    tie_break_used = len(comparable) > 1

    others = survivors[survivors["Model"] != winner_row["Model"]]
    runner_up_row = others.sort_values("composite_rank").iloc[0] if not others.empty else None

    beats_on, margin_r2, margin_mae_pct = [], None, None
    if runner_up_row is not None:
        for col, ascending in RANK_METRICS:
            w, r = winner_row[col], runner_up_row[col]
            if (w < r if ascending else w > r):
                beats_on.append(col)
        margin_r2 = float(winner_row["R2_mean"] - runner_up_row["R2_mean"])
        if runner_up_row["MAE_mean"] > 0:
            margin_mae_pct = float((runner_up_row["MAE_mean"] - winner_row["MAE_mean"]) / runner_up_row["MAE_mean"] * 100)

    return {
        "winner": winner_row["Model"],
        "is_benchmark": bool(winner_row["Model"] in BENCHMARK_MODELS),
        "ranked_table": ranked,
        "tie_break_used": tie_break_used,
        "fallback_used_no_robust_survivor": fallback_used,
        "runner_up": None if runner_up_row is None else runner_up_row["Model"],
        "runner_up_is_benchmark": None if runner_up_row is None else bool(runner_up_row["Model"] in BENCHMARK_MODELS),
        "beats_runner_up_on": beats_on,
        "margin_r2": margin_r2,
        "margin_mae_pct": margin_mae_pct,
        "winner_row": winner_row,
        "top_ranked_model": top_row["Model"],
        "winner_is_top_ranked": winner_row["Model"] == top_row["Model"],
    }


def select_model(summary: pd.DataFrame, fold_detail: pd.DataFrame) -> dict:
    robustness = compute_robustness(fold_detail)
    ranked = rank_models(summary).merge(robustness, on="Model")
    ranked["simplicity_rank"] = ranked["Model"].map(SIMPLICITY_RANK)
    ranked = ranked.sort_values("composite_rank").reset_index(drop=True)
    return pick_winner(ranked)


def load_bakeoff_results(mission: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    slug = mission.lower().replace(" ", "_")
    summary_path = DATA_DIR / f"forecast_bakeoff_{slug}_summary.csv"
    folds_path = DATA_DIR / f"forecast_bakeoff_{slug}_folds.csv"
    if not summary_path.exists() or not folds_path.exists():
        raise FileNotFoundError(
            f"Missing bake-off results for {mission}: run `python -m forecast_src.forecast_bakeoff` first."
        )
    return pd.read_csv(summary_path), pd.read_csv(folds_path)


def fit_final_ml_model(mission_df: pd.DataFrame, model_name: str) -> tuple:
    """Refits the horizon="1" ML winner on the mission's FULL training
    table (all target_years, not just one outer cutoff's train portion),
    with its own fresh nested-temporal-CV hyperparameter search — same
    "re-tune on all data rather than reuse one fold's params" approach as
    src/model_selection.py's fit_final_model."""
    estimator, param_grid = MODELS[model_name]
    reset_df = mission_df.reset_index(drop=True)
    X, y = reset_df[FEATURE_COLUMNS], reset_df[TARGET_COL]

    preprocessor = build_preprocessor(scale=model_name in SCALE_SENSITIVE)
    target_model = TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)
    pipe = Pipeline([("preprocess", preprocessor), ("model", target_model)])

    max_year = int(reset_df[TARGET_YEAR_COL].max())
    inner_cv = make_inner_temporal_cv(reset_df, max_year, INNER_CUTOFF_OFFSETS)
    if param_grid and inner_cv:
        search = GridSearchCV(pipe, param_grid=param_grid, cv=inner_cv, scoring="neg_mean_absolute_error", n_jobs=-1)
        search.fit(X, y)
        return search.best_estimator_, search.best_params_
    pipe.fit(X, y)
    return pipe, {}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ranking_rows, selected_rows = [], []

    for mission in REAL_MISSIONS:
        print(f"\n=== {mission} ===")
        summary, fold_detail = load_bakeoff_results(mission)

        mission_results = {}
        for horizon in HORIZONS:
            summary_h = summary[summary["horizon"] == horizon]
            folds_h = fold_detail[fold_detail["horizon"] == horizon]
            if summary_h.empty:
                continue
            result = select_model(summary_h, folds_h)
            mission_results[horizon] = result

            ranked_out = result["ranked_table"].copy()
            ranked_out.insert(0, "Mission", mission)
            ranked_out["selected"] = ranked_out["Model"] == result["winner"]
            ranked_out["is_benchmark"] = ranked_out["Model"].isin(BENCHMARK_MODELS)
            ranking_rows.append(ranked_out)

            tag = "[BENCHMARK]" if result["is_benchmark"] else "[ML]"
            print(f"\n--- horizon {horizon} ---")
            print(f"Winner: {tag} {result['winner']}")
            print(f"  Runner-up: {result['runner_up']} ({'benchmark' if result['runner_up_is_benchmark'] else 'ML'})")
            if result["margin_r2"] is not None:
                print(f"  Margin over runner-up: R2 {result['margin_r2']:+.4f}, MAE {result['margin_mae_pct']:+.1f}% better")
            print(f"  Tie-break (simplicity) invoked: {result['tie_break_used']}")
            if not result["winner_is_top_ranked"]:
                print(
                    f"  NOTE: tie-break overrode the raw top-ranked model — {result['top_ranked_model']} had a "
                    f"better composite rank but {result['winner']} was within tolerance and simpler. This is why "
                    "the margin above can read negative: it's comparing the CHOSEN (simpler) winner against the "
                    "next-best model, which in an override case is the one the tie-break passed over."
                )
            if result["fallback_used_no_robust_survivor"]:
                print("  WARNING: every model had a robustness violation; selected from full set anyway.")

            r2_mean = float(result["winner_row"]["R2_mean"])
            selected_rows.append(
                {
                    "mission": mission,
                    "horizon": horizon,
                    "is_primary_recursive_model": horizon == PRIMARY_HORIZON,
                    "selected_model": result["winner"],
                    "is_benchmark": result["is_benchmark"],
                    "runner_up": result["runner_up"],
                    "runner_up_is_benchmark": result["runner_up_is_benchmark"],
                    "margin_r2": result["margin_r2"],
                    "margin_mae_pct": result["margin_mae_pct"],
                    "r2_mean": r2_mean,
                    "usable": r2_mean > USABILITY_R2_THRESHOLD,
                    "tie_break_used": result["tie_break_used"],
                    "winner_is_top_ranked": result["winner_is_top_ranked"],
                    "top_ranked_model": result["top_ranked_model"],
                }
            )

        # Only the horizon="1" winner is refit/deployed — see module docstring.
        primary = mission_results.get(PRIMARY_HORIZON)
        slug = mission.lower().replace(" ", "_")
        if primary is None:
            print(f"\nNo horizon-1 result for {mission} — nothing to refit.")
        elif primary["is_benchmark"]:
            print(
                f"\nPrimary recursive model for {mission}: [BENCHMARK] {primary['winner']} — "
                "no .joblib saved, forecast_recursive.py applies the formula directly "
                "(see BENCHMARK_PREDICT_FNS in forecast_bakeoff.py)."
            )
        else:
            mission_df = get_mission_training_data(mission)
            final_model, final_params = fit_final_ml_model(mission_df, primary["winner"])
            model_path = DATA_DIR / f"forecast_final_model_{slug}.joblib"
            joblib.dump(final_model, model_path)
            for row in selected_rows:
                if row["mission"] == mission and row["is_primary_recursive_model"]:
                    row["best_params"] = json.dumps(final_params, default=str)
            print(f"\nPrimary recursive model for {mission}: [ML] {primary['winner']} — refit on full data, wrote {model_path}")

    ranking_all = pd.concat(ranking_rows, ignore_index=True)
    selected_all = pd.DataFrame(selected_rows)

    ranking_path = DATA_DIR / "forecast_selection_ranking.csv"
    selected_path = DATA_DIR / "forecast_selected_models.csv"
    ranking_all.to_csv(ranking_path, index=False)
    selected_all.to_csv(selected_path, index=False)

    print(f"\nWrote {ranking_path}")
    print(f"Wrote {selected_path}")

    print("\n=== Summary: winner per mission x horizon ===")
    display_cols = ["mission", "horizon", "selected_model", "is_benchmark", "runner_up", "margin_r2", "margin_mae_pct", "r2_mean"]
    print(selected_all[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
