"""Diagnostic run BEFORE forecast_recursive.py was built: forecast_selection.py
found Persistence (no-growth) winning horizon=1 in all 3 missions — applied
recursively to 2030, that flattens most companies' trajectories, which is a
real risk for a project whose core goal is identifying £10M/£50M-by-2030
GROWTH candidates. Two questions, both answered here:

1. Is Persistence's win driven by the flat/stable majority, with ML models
   actually winning on genuinely-growing companies? Checked two ways:
   (a) the literal ask — split by forecast_evidence_group (A/B); (b) a
   supplementary split by actual single-year growth trajectory
   (growing/flat/declining at a 10% YoY cut, the same threshold the
   project's planned gazelle criteria use), because (a) alone turns out not
   to discriminate much — see the finding below.
2. If Persistence still wins even among growing companies, is it fit for
   the project's actual purpose?

Not a pipeline stage (no build-order number) — a decision-support check
read once, kept here because the project's convention is to keep every
real analysis referenceable, not just quoted in chat.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from forecast_src.forecast_bakeoff import (
    BENCHMARK_MODELS,
    BENCHMARK_PREDICT_FNS,
    FEATURE_COLUMNS,
    INNER_CUTOFF_OFFSETS,
    MODELS,
    OUTER_CUTOFF_YEARS,
    SCALE_SENSITIVE,
    TARGET_COL,
    TARGET_YEAR_COL,
    build_preprocessor,
    compute_horizon_bucket,
    get_mission_training_data,
    make_inner_temporal_cv,
    make_outer_temporal_splits,
)
from forecast_src.forecast_selection import SIMPLICITY_RANK
from src.mission_segmentation import REAL_MISSIONS

# Lighter-weight proxy for "genuine growth trajectory" than the full
# 3-consecutive-year gazelle definition (that needs forecast_recursive's
# 2030 trajectories to exist first) — single-year growth direction at the
# same 10% YoY cut the gazelle criteria will eventually use.
GROWTH_CUT = np.log1p(0.10)  # ~0.0953


def growth_trajectory_bucket(log_growth_1y: pd.Series) -> np.ndarray:
    bucket = np.select(
        [log_growth_1y > GROWTH_CUT, log_growth_1y < -GROWTH_CUT, log_growth_1y.notna()],
        ["growing (>=10% YoY)", "declining (<=-10% YoY)", "flat (-10% to +10%)"],
        default="insufficient_history (no lag_1)",
    )
    return bucket


def collect_horizon1_predictions(mission: str) -> pd.DataFrame:
    """Refits every candidate at each outer cutoff (same as forecast_bakeoff.
    evaluate_ml_model/evaluate_benchmark_model) but keeps ROW-LEVEL
    predictions instead of aggregating straight to MAE/RMSE/R2 — needed to
    slice horizon=1 predictions by evidence_group / growth_trajectory
    afterward, which forecast_bakeoff's saved fold_detail doesn't retain."""
    mission_df = get_mission_training_data(mission)
    outer_splits = make_outer_temporal_splits(mission_df, OUTER_CUTOFF_YEARS)

    rows = []
    for cutoff, train_idx, test_idx in outer_splits:
        train_df = mission_df.iloc[train_idx].reset_index(drop=True)
        test_df = mission_df.iloc[test_idx].reset_index(drop=True)
        horizon_bucket = compute_horizon_bucket(test_df[TARGET_YEAR_COL], cutoff)
        is_horizon1 = horizon_bucket == "1"
        if not is_horizon1.any():
            continue

        for model_name, (estimator, param_grid) in MODELS.items():
            preprocessor = build_preprocessor(scale=model_name in SCALE_SENSITIVE)
            target_model = TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)
            pipe = Pipeline([("preprocess", preprocessor), ("model", target_model)])
            inner_cv = make_inner_temporal_cv(train_df, cutoff, INNER_CUTOFF_OFFSETS)
            X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COL]
            if param_grid and inner_cv:
                search = GridSearchCV(pipe, param_grid=param_grid, cv=inner_cv, scoring="neg_mean_absolute_error", n_jobs=-1)
                search.fit(X_train, y_train)
                fitted = search.best_estimator_
            else:
                fitted = pipe.fit(X_train, y_train)
            y_pred = fitted.predict(test_df[FEATURE_COLUMNS])

            sub = test_df.loc[is_horizon1, ["forecast_evidence_group", "log_growth_1y", TARGET_COL]].copy()
            sub["Model"] = model_name
            sub["cutoff_year"] = cutoff
            sub["y_pred"] = y_pred[is_horizon1]
            rows.append(sub)

        for model_name in BENCHMARK_MODELS:
            y_pred = BENCHMARK_PREDICT_FNS[model_name](train_df, test_df)
            sub = test_df.loc[is_horizon1, ["forecast_evidence_group", "log_growth_1y", TARGET_COL]].copy()
            sub["Model"] = model_name
            sub["cutoff_year"] = cutoff
            sub["y_pred"] = y_pred[is_horizon1]
            rows.append(sub)

    all_preds = pd.concat(rows, ignore_index=True)
    all_preds["growth_trajectory"] = growth_trajectory_bucket(all_preds["log_growth_1y"])
    return all_preds


def summarise_by_subgroup(preds: pd.DataFrame, subgroup_col: str) -> pd.DataFrame:
    rows = []
    for (model, subgroup), g in preds.groupby(["Model", subgroup_col]):
        y_true, y_pred = g[TARGET_COL].to_numpy(), g["y_pred"].to_numpy()
        rows.append(
            {
                "Model": model,
                subgroup_col: subgroup,
                "n_rows": len(g),
                "MAE": mean_absolute_error(y_true, y_pred),
                "RMSE": root_mean_squared_error(y_true, y_pred),
                "R2": r2_score(y_true, y_pred) if len(g) >= 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def rank_within_subgroup(summary: pd.DataFrame, subgroup_col: str) -> pd.DataFrame:
    """Lightweight version of forecast_selection.py's composite rank —
    single-fold-per-subgroup here (no std across folds available at this
    granularity), so rank directly on MAE/RMSE/R2 (3 metrics, not 6), same
    simplicity tie-break for consistency with the main selection logic."""
    out = []
    for subgroup, g in summary.groupby(subgroup_col):
        g = g.copy()
        g["rank_MAE"] = g["MAE"].rank(ascending=True)
        g["rank_RMSE"] = g["RMSE"].rank(ascending=True)
        g["rank_R2"] = g["R2"].rank(ascending=False)
        g["composite_rank"] = g[["rank_MAE", "rank_RMSE", "rank_R2"]].mean(axis=1)
        g["simplicity_rank"] = g["Model"].map(SIMPLICITY_RANK)
        g = g.sort_values(["composite_rank", "simplicity_rank"])
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    for mission in REAL_MISSIONS:
        print(f"\n{'=' * 60}\n{mission} — horizon=1 only\n{'=' * 60}")
        preds = collect_horizon1_predictions(mission)

        print(f"\nRow counts by evidence group (horizon=1 test rows, all cutoffs pooled):")
        print(preds.groupby("forecast_evidence_group")["Model"].count().to_string())
        print(f"\nRow counts by growth trajectory:")
        print(preds.groupby("growth_trajectory")["Model"].count().to_string())

        print("\n--- 1. Split by forecast_evidence_group (as literally requested) ---")
        by_group = summarise_by_subgroup(preds, "forecast_evidence_group")
        ranked_group = rank_within_subgroup(by_group, "forecast_evidence_group")
        for group, g in ranked_group.groupby("forecast_evidence_group"):
            print(f"\nEvidence group {group} (n={int(g['n_rows'].iloc[0]) if len(g) else 0} total-model-rows/model):")
            print(g[["Model", "n_rows", "MAE", "RMSE", "R2", "composite_rank"]].head(5).to_string(index=False))

        print("\n--- 2. Split by growth trajectory (supplementary — evidence group doesn't discriminate growth) ---")
        by_growth = summarise_by_subgroup(preds[preds["growth_trajectory"] != "insufficient_history (no lag_1)"], "growth_trajectory")
        ranked_growth = rank_within_subgroup(by_growth, "growth_trajectory")
        for bucket, g in ranked_growth.groupby("growth_trajectory"):
            print(f"\n{bucket} (n={int(g['n_rows'].iloc[0])} rows/model):")
            print(g[["Model", "n_rows", "MAE", "RMSE", "R2", "composite_rank"]].head(5).to_string(index=False))

        winner_growing = ranked_growth[ranked_growth["growth_trajectory"] == "growing (>=10% YoY)"].iloc[0]
        tag = "[BENCHMARK]" if winner_growing["Model"] in BENCHMARK_MODELS else "[ML]"
        print(f"\n>>> {mission}: winner on GROWING companies specifically: {tag} {winner_growing['Model']}")


if __name__ == "__main__":
    main()
