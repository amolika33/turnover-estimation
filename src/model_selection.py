"""Pick one model per mission from the model_bakeoff.py results (methodology
doc sec 1.8.2), then refit the winner on the full labelled dataset for that
mission — never just the CV training folds — so it's ready for predict.py.

Selection is computed from the bake-off's fold-level results, not read off
a summary table by hand:

1. Robustness filter (hard exclusion): a model is dropped from contention if
   any single outer CV fold shows R2 < -2 (a clearly broken fold) or a fold
   MAE more than 3x its own across-fold median (an internal blow-up) — this
   catches instability even when the mean metrics look fine, per the
   "extreme values in any fold" requirement.
2. Composite rank (accuracy + consistency, combined): for each of
   MAE_mean, RMSE_mean, R2_mean, MAE_std, RMSE_std, R2_std, rank all
   surviving models 1..N (best=1) and average the six ranks. This gives
   accuracy and fold-to-fold consistency equal weight without hand-picking
   a blend, and keeps the "no single metric decides" rule from the
   checklist — now applied across 6 metrics instead of 3.
3. Interpretability tie-break: composite rank alone is too loose to mean
   "genuine tie" — it blends 6 ranked metrics, so two models can be within
   1.0 composite-rank points while one has decisively better R2 (e.g.
   Resilient Earth: Extra Trees R2=0.59 vs Elastic Net R2=-0.09 were only
   0.66 composite-rank points apart). A model only enters the "comparable"
   set, and is eligible for the simplicity tie-break, if BOTH its composite
   rank is within COMPARABLE_TOLERANCE of the top-ranked survivor AND its
   R2_mean is within R2_COMPARABLE_TOLERANCE (0.05) of the top-ranked
   survivor's R2_mean. Among that comparable set, pick the lowest
   SIMPLICITY_RANK (plain Linear Regression simplest; Ridge/Lasso/Elastic
   Net next; k-NN; SVR; tree ensembles least interpretable).

Reads the bake-off's saved model_bakeoff_{mission}_summary.csv /
_folds.csv (written by `python -m src.model_bakeoff`) rather than calling
run_bakeoff() again. Two reasons: (1) wasted compute — the repeated 5x5
grouped CV bake-off takes minutes per mission, and re-running it here to
get numbers already sitting on disk was pure waste; (2) run-to-run drift —
GroupKFold's shuffling is seeded, so a second run *should* reproduce the
same folds, but reading the one bake-off run everyone is looking at
guarantees model_selection's decision matches the reported bake-off table
byte-for-byte, rather than trusting that two separate invocations agree.

Writes selected_models.csv (mission, selected_model, r2_mean, usable,
exclusion_reason, best_params) — the single source of truth predict.py and
assemble.py read instead of a hardcoded mission-name dict, so ACE's
exclusion is a reviewable data fact (recorded here, re-evaluated every time
this module runs), not baked into another module's source code.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import has_fit_parameter

from src.model_bakeoff import (
    CATBOOST_FIT_KWARGS,
    CATEGORICAL_FEATURES,
    GROUP_COL,
    MODELS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    REAL_MISSIONS,
    SCALE_SENSITIVE,
    TARGET_COL,
    WEIGHT_COL,
    cast_categoricals,
    get_mission_features,
    get_preprocessor,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "processed"

ACCURACY_METRICS = [("MAE_mean", True), ("RMSE_mean", True), ("R2_mean", False)]
CONSISTENCY_METRICS = [("MAE_std", True), ("RMSE_std", True), ("R2_std", True)]
RANK_METRICS = ACCURACY_METRICS + CONSISTENCY_METRICS
COMPARABLE_TOLERANCE = 1.0
R2_COMPARABLE_TOLERANCE = 0.05
BROKEN_FOLD_R2 = -2.0
BLOWUP_FOLD_MAE_MULTIPLE = 3.0

# ASSUMPTION (documented here + PROJECT_NOTES.md "Model usability threshold" +
# the usable/exclusion_reason columns in selected_models.csv): a selected
# model is only usable for prediction if it beats predicting the mission's
# own mean turnover, i.e. R2_mean > 0. This is what took ACE out of
# predict.py (R2_mean=-1.03 under repeated CV) — expressed as a threshold
# instead of a hardcoded mission name, a future data refresh that lifts
# ACE's R2 above 0 flips its usability automatically, no code change.
USABILITY_R2_THRESHOLD = 0.0

SIMPLICITY_RANK = {
    "Linear Regression": 1,
    "Ridge": 2,
    "Lasso": 2,
    "Elastic Net": 2,
    "k-NN": 3,
    "SVR": 4,
    "Gradient Boosting": 5,
    "Random Forest": 5,
    "Extra Trees": 5,
    # One tier above the sklearn tree ensembles: not more complex as a model
    # class (still gradient-boosted trees), but it's an extra third-party
    # dependency (not part of scikit-learn) and its native categorical
    # handling (build_catboost_preprocessor) makes it a genuinely different,
    # less-standard preprocessing path than every other candidate — both
    # reasons to rank it least simple among the tree-based models on a tie.
    "CatBoost": 6,
}


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
    """Expects `ranked` to already carry composite_rank, robustness_violation,
    R2_mean, and simplicity_rank (from rank_models + compute_robustness)."""
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

    runner_up_row = None
    others = survivors[survivors["Model"] != winner_row["Model"]]
    if not others.empty:
        runner_up_row = others.sort_values("composite_rank").iloc[0]

    beats_on = []
    if runner_up_row is not None:
        for col, ascending in RANK_METRICS:
            w, r = winner_row[col], runner_up_row[col]
            better = w < r if ascending else w > r
            if better:
                beats_on.append(col)

    return {
        "winner": winner_row["Model"],
        "ranked_table": ranked,
        "tie_break_used": tie_break_used,
        "fallback_used_no_robust_survivor": fallback_used,
        "runner_up": None if runner_up_row is None else runner_up_row["Model"],
        "beats_runner_up_on": beats_on,
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


def fit_final_model(mission_df: pd.DataFrame, model_name: str, n_splits: int = 5):
    df = cast_categoricals(mission_df)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]
    groups = df[GROUP_COL]
    weights = df[WEIGHT_COL]

    estimator, param_grid = MODELS[model_name]
    preprocessor = get_preprocessor(model_name)
    target_model = TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)
    pipe = Pipeline([("preprocess", preprocessor), ("model", target_model)])

    cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)

    fit_kwargs = {"groups": groups}
    if has_fit_parameter(estimator, "sample_weight"):
        fit_kwargs["model__sample_weight"] = weights.to_numpy()
    if model_name == "CatBoost":
        fit_kwargs.update(CATBOOST_FIT_KWARGS)
    search.fit(X, y, **fit_kwargs)

    return search.best_estimator_, search.best_params_


def load_bakeoff_results(mission: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    slug = mission.lower().replace(" ", "_")
    summary_path = OUTPUT_DIR / f"model_bakeoff_{slug}_summary.csv"
    folds_path = OUTPUT_DIR / f"model_bakeoff_{slug}_folds.csv"
    if not summary_path.exists() or not folds_path.exists():
        raise FileNotFoundError(
            f"Missing bake-off results for {mission}: run `python -m src.model_bakeoff "
            f"--mission \"{mission}\"` first (expected {summary_path.name} and {folds_path.name})."
        )
    return pd.read_csv(summary_path), pd.read_csv(folds_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", action="append", choices=REAL_MISSIONS, default=None)
    args = parser.parse_args()
    missions = args.mission or REAL_MISSIONS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selection_rows = []
    selected_models_rows = []
    for mission in missions:
        print(f"\n=== {mission} ===")
        mission_df = get_mission_features(mission)
        summary, fold_detail = load_bakeoff_results(mission)

        result = select_model(summary, fold_detail)
        ranked = result["ranked_table"]
        print(
            ranked[
                [
                    "Model",
                    "MAE_mean",
                    "RMSE_mean",
                    "R2_mean",
                    "MAE_std",
                    "RMSE_std",
                    "R2_std",
                    "composite_rank",
                    "robustness_violation",
                    "simplicity_rank",
                ]
            ].to_string(index=False)
        )

        winner = result["winner"]
        print(f"\nSelected: {winner}")
        print(f"  Top-ranked by composite_rank alone: {result['top_ranked_model']}")
        if not result["winner_is_top_ranked"]:
            print(
                f"  NOTE: tie-break overrode the raw top-ranked model — {result['top_ranked_model']} "
                f"had a better composite rank but {winner} was within tolerance and simpler."
            )
        print(f"  Runner-up: {result['runner_up']}")
        print(f"  Beats runner-up on: {result['beats_runner_up_on']}")
        print(f"  Tie-break (simplicity) invoked: {result['tie_break_used']}")
        if result["fallback_used_no_robust_survivor"]:
            print("  WARNING: every model had a robustness violation; selected from full set anyway.")

        final_model, final_params = fit_final_model(mission_df, winner)
        print(f"  Refit on full labelled data ({len(mission_df)} rows): best_params={final_params}")

        slug = mission.lower().replace(" ", "_")
        model_path = OUTPUT_DIR / f"final_model_{slug}.joblib"
        joblib.dump(final_model, model_path)
        print(f"  Wrote {model_path}")

        r2_mean = float(result["winner_row"]["R2_mean"])
        usable = r2_mean > USABILITY_R2_THRESHOLD
        exclusion_reason = (
            ""
            if usable
            else (
                f"no reliable model available — best model ({winner}) has R2_mean={r2_mean:.2f}, "
                f"at or below the usability threshold ({USABILITY_R2_THRESHOLD}); insufficient "
                "labelled data / no genuine predictive value under repeated cross-validation"
            )
        )
        print(f"  Usable (R2_mean={r2_mean:.2f} > {USABILITY_R2_THRESHOLD}): {usable}")
        selected_models_rows.append(
            {
                "mission": mission,
                "selected_model": winner,
                "r2_mean": r2_mean,
                "usable": usable,
                "exclusion_reason": exclusion_reason,
                "best_params": json.dumps(final_params, default=str),
            }
        )

        ranked_out = ranked.copy()
        ranked_out.insert(0, "Mission", mission)
        ranked_out["selected"] = ranked_out["Model"] == winner
        selection_rows.append(ranked_out)

    # --mission can target a subset (e.g. re-running just Resilient Earth
    # after a fix); merge with any prior run's rows for the other missions
    # instead of overwriting them, since these two files are the persistent
    # record other modules (predict.py, assemble.py) depend on.
    summary_path = OUTPUT_DIR / "model_selection_summary.csv"
    new_summary = pd.concat(selection_rows, ignore_index=True)
    if summary_path.exists():
        prior_summary = pd.read_csv(summary_path)
        prior_summary = prior_summary[~prior_summary["Mission"].isin(missions)]
        new_summary = pd.concat([prior_summary, new_summary], ignore_index=True)
    new_summary.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")

    selected_path = OUTPUT_DIR / "selected_models.csv"
    new_selected = pd.DataFrame(selected_models_rows)
    if selected_path.exists():
        prior_selected = pd.read_csv(selected_path)
        prior_selected = prior_selected[~prior_selected["mission"].isin(missions)]
        new_selected = pd.concat([prior_selected, new_selected], ignore_index=True)
    new_selected.to_csv(selected_path, index=False)
    print(f"Wrote {selected_path}")


if __name__ == "__main__":
    main()
