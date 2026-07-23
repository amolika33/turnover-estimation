"""POOLED bake-off variant — trains on ALL training-eligible space companies
across all 3 real missions TOGETHER, with `mission` added as a categorical
FEATURE rather than the dataset partition. Independent of, and does not
replace, the mission-specific bake-off in src/model_bakeoff.py.

WHY THIS EXISTS (a hedge/comparison, not a replacement): PROJECT_NOTES.md's "Three
independent mission models" rule is the project's primary, deliberately
chosen approach — pooling missions into one model is explicitly listed as
something NOT to do for the real pipeline. This module exists only to
answer one specific diagnostic question: given ACE's persistent weakness
(only ~80 labelled companies, R2_mean=0.15 even after linkedin_industry
removal — see PROJECT_NOTES.md), does training on the ~3-4x larger pooled dataset
(borrowing Beyond Earth/Resilient Earth's signal, with `mission` itself as a
feature so the model can still learn a mission-level adjustment) do better
than ACE's own mission-specific model? Both results are kept and reported
side by side; nothing here feeds predict.py/assemble.py.

`mission` is only meaningful as a feature once training is pooled — in the
mission-specific bake-off every row already belongs to one mission (it's
the partition variable, carries zero within-model information); here it can
carry real signal (does being an ACE company predict a different turnover
level than an otherwise-identical Beyond Earth company?).

SELECTION METHODOLOGY, kept identical to model_selection.py's mission-
specific procedure so the comparison is apples-to-apples (same robustness
filter -> composite rank -> simplicity tie-break, reusing
model_selection.py's functions directly on the pooled summary/fold_detail)
rather than picking whichever pooled candidate happens to look best on any
one mission's slice — that would be a different, softer standard than how
each mission's own model was chosen.

Deliberately near-duplicates src/model_bakeoff.py's preprocessing/CV/model
code rather than importing and parameterising it, to keep this effort
textually independent and auditable on its own (per the task's explicit
"do these as two separate, clearly labeled efforts" instruction) — same
precedent as forecast_bakeoff.py maintaining its own bake-off code instead
of sharing model_bakeoff.py's, given a genuinely different feature-column
configuration (mission as a feature) that doesn't exist in the
mission-specific pipeline.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import has_fit_parameter

from src.mission_segmentation import MISSION_COL, REAL_MISSIONS
from src.model_bakeoff import (
    GROUP_COL,
    LOG_NUMERIC_FEATURES,
    MODELS,
    PLAIN_NUMERIC_FEATURES,
    RANDOM_STATE,
    SCALE_SENSITIVE,
    TARGET_COL,
    WEIGHT_COL,
    _cast_value,
    check_negative_log_inputs,
    get_mission_features,
    make_repeated_group_kfold_splits,
)
from src.model_selection import select_model

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "processed"

# `mission` appended to the same 4 categorical columns the mission-specific
# bake-off uses — everything else about the feature set is identical, so
# any performance difference is attributable to pooling + this one added
# feature, not a different feature-engineering pass.
POOLED_CATEGORICAL_FEATURES = ["employee_count_source", "company_size", "sic_code_1", "value_stream", MISSION_COL]
NUMERIC_FEATURES = PLAIN_NUMERIC_FEATURES + LOG_NUMERIC_FEATURES
# NOT the mission-specific model_bakeoff.CATBOOST_FIT_KWARGS — that one
# lists only the 4 mission-specific categorical columns, missing `mission`
# itself (which would make CatBoost try to parse "ACE"/"Beyond Earth"/
# "Resilient Earth" as a numeric feature and crash). Own copy, built from
# POOLED_CATEGORICAL_FEATURES.
CATBOOST_FIT_KWARGS = {"model__cat_features": POOLED_CATEGORICAL_FEATURES}


def build_pooled_dataset() -> pd.DataFrame:
    """Concatenates all 3 real missions' labelled (training-eligible) rows —
    Cross-cutting companies are still never in this dataset, same exclusion
    the mission-specific bake-off already applies (get_mission_features only
    ever returns real-mission rows)."""
    return pd.concat([get_mission_features(m) for m in REAL_MISSIONS], ignore_index=True)


def cast_categoricals_pooled(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in POOLED_CATEGORICAL_FEATURES:
        df[col] = df[col].apply(_cast_value).astype(object)
    return df


def build_preprocessor(scale: bool) -> ColumnTransformer:
    plain_steps = [("imputer", SimpleImputer(strategy="median"))]
    log_steps = [
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale:
        plain_steps.append(("scaler", StandardScaler()))
        log_steps.append(("scaler", StandardScaler()))
    plain_numeric_pipeline = Pipeline(plain_steps)
    log_numeric_pipeline = Pipeline(log_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )
    return ColumnTransformer(
        [
            ("plain_numeric", plain_numeric_pipeline, PLAIN_NUMERIC_FEATURES),
            ("log_numeric", log_numeric_pipeline, LOG_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, POOLED_CATEGORICAL_FEATURES),
        ]
    )


def build_catboost_preprocessor() -> ColumnTransformer:
    plain_numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    log_numeric_pipeline = Pipeline(
        [
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="missing"))])
    preprocessor = ColumnTransformer(
        [
            ("plain_numeric", plain_numeric_pipeline, PLAIN_NUMERIC_FEATURES),
            ("log_numeric", log_numeric_pipeline, LOG_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, POOLED_CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def get_preprocessor(name: str) -> ColumnTransformer:
    if name == "CatBoost":
        return build_catboost_preprocessor()
    return build_preprocessor(scale=name in SCALE_SENSITIVE)


def evaluate_model(
    name: str,
    estimator,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    weights: pd.Series,
    missions: pd.Series,
    outer_splits: list[tuple[int, int, np.ndarray, np.ndarray]],
    inner_cv,
) -> tuple[list[dict], list[dict]]:
    """Returns (fold_rows, per_mission_fold_rows) — the second is the whole
    point of this module: the same fitted-on-pooled-data fold model, scored
    separately on just the ACE / Beyond Earth / Resilient Earth rows of each
    outer test fold, so "how does the pooled model do on ACE specifically"
    is a like-for-like slice of genuine held-out predictions, not a
    re-fit or a different CV scheme."""
    preprocessor = get_preprocessor(name)
    fold_rows = []
    per_mission_rows = []
    for repeat, fold_idx, train_idx, test_idx in outer_splits:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        w_train, w_test = weights.iloc[train_idx], weights.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        missions_test = missions.iloc[test_idx]

        target_model = TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)
        pipe = Pipeline([("preprocess", preprocessor), ("model", target_model)])
        search = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        fit_kwargs = {"groups": groups_train}
        if has_fit_parameter(estimator, "sample_weight"):
            fit_kwargs["model__sample_weight"] = w_train.to_numpy()
        if name == "CatBoost":
            fit_kwargs.update(CATBOOST_FIT_KWARGS)
        search.fit(X_train, y_train, **fit_kwargs)

        y_pred = search.best_estimator_.predict(X_test)
        fold_rows.append(
            {
                "Model": name,
                "repeat": repeat,
                "fold": fold_idx,
                "n_train_companies": groups_train.nunique(),
                "n_test_companies": groups.iloc[test_idx].nunique(),
                "n_test_rows": len(test_idx),
                "MAE": mean_absolute_error(y_test, y_pred, sample_weight=w_test),
                "RMSE": root_mean_squared_error(y_test, y_pred, sample_weight=w_test),
                "R2": r2_score(y_test, y_pred, sample_weight=w_test),
                "best_params": json.dumps(search.best_params_, default=str),
            }
        )

        for mission in REAL_MISSIONS:
            mission_mask = (missions_test == mission).to_numpy()
            if mission_mask.sum() < 2:
                continue  # r2_score needs >=2 points to be meaningful
            per_mission_rows.append(
                {
                    "Model": name,
                    "mission": mission,
                    "repeat": repeat,
                    "fold": fold_idx,
                    "n_test_rows": int(mission_mask.sum()),
                    "MAE": mean_absolute_error(y_test[mission_mask], y_pred[mission_mask], sample_weight=w_test[mission_mask]),
                    "RMSE": root_mean_squared_error(y_test[mission_mask], y_pred[mission_mask], sample_weight=w_test[mission_mask]),
                    "R2": r2_score(y_test[mission_mask], y_pred[mission_mask], sample_weight=w_test[mission_mask]),
                }
            )
    return fold_rows, per_mission_rows


def run_pooled_bakeoff(
    pooled_df: pd.DataFrame, n_outer_splits: int = 5, n_outer_repeats: int = 5, n_inner_splits: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = cast_categoricals_pooled(pooled_df)
    X = df[NUMERIC_FEATURES + POOLED_CATEGORICAL_FEATURES]
    y = df[TARGET_COL]
    groups = df[GROUP_COL]
    weights = df[WEIGHT_COL]
    missions = df[MISSION_COL]

    negative_log_inputs = check_negative_log_inputs(df, LOG_NUMERIC_FEATURES)

    outer_splits = make_repeated_group_kfold_splits(
        X, y, groups, n_splits=n_outer_splits, n_repeats=n_outer_repeats, random_state=RANDOM_STATE
    )
    inner_cv = GroupKFold(n_splits=n_inner_splits, shuffle=True, random_state=RANDOM_STATE)

    all_folds = []
    all_per_mission_folds = []
    for name, (estimator, param_grid) in MODELS.items():
        fold_rows, per_mission_rows = evaluate_model(
            name, estimator, param_grid, X, y, groups, weights, missions, outer_splits, inner_cv
        )
        all_folds.extend(fold_rows)
        all_per_mission_folds.extend(per_mission_rows)
    fold_detail = pd.DataFrame(all_folds)
    per_mission_fold_detail = pd.DataFrame(all_per_mission_folds)

    summary = fold_detail.groupby("Model")[["MAE", "RMSE", "R2"]].agg(["mean", "std"]).round(2)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values("MAE_mean")

    per_mission_summary = (
        per_mission_fold_detail.groupby(["Model", "mission"])[["MAE", "RMSE", "R2"]]
        .agg(["mean", "std"])
        .round(2)
    )
    per_mission_summary.columns = [f"{metric}_{stat}" for metric, stat in per_mission_summary.columns]
    per_mission_summary = per_mission_summary.reset_index()

    return summary, per_mission_summary, fold_detail, per_mission_fold_detail, negative_log_inputs


def select_pooled_winner(summary: pd.DataFrame, fold_detail: pd.DataFrame) -> dict:
    """Same robustness-filter -> composite-rank -> simplicity-tie-break
    procedure model_selection.py uses per mission, applied here to the
    OVERALL pooled metrics — one pooled winner, chosen the same way each
    mission's own winner was chosen, so the head-to-head comparison isn't
    stacking a looser selection standard against the mission-specific
    models' stricter one."""
    return select_model(summary, fold_detail)


def build_comparison_table(
    pooled_winner: str,
    per_mission_summary: pd.DataFrame,
    mission_specific_results: dict[str, dict],
) -> pd.DataFrame:
    """One row per mission: the pooled winner's performance sliced to that
    mission's held-out rows, next to that mission's own SELECTED model
    (model_selection.py's actual robustness-filter -> composite-rank ->
    simplicity-tie-break winner, not just whichever model happens to have
    the lowest MAE_mean in the summary table) — the direct "does pooling
    help ACE" comparison the task asked for, using the identical selection
    standard on both sides."""
    rows = []
    for mission in REAL_MISSIONS:
        pooled_row = per_mission_summary[
            (per_mission_summary["Model"] == pooled_winner) & (per_mission_summary["mission"] == mission)
        ]
        mission_result = mission_specific_results[mission]
        mission_winner_row = mission_result["winner_row"]
        rows.append(
            {
                "mission": mission,
                "pooled_model": pooled_winner,
                "pooled_R2_mean": float(pooled_row["R2_mean"].iloc[0]) if len(pooled_row) else np.nan,
                "pooled_MAE_mean": float(pooled_row["MAE_mean"].iloc[0]) if len(pooled_row) else np.nan,
                "mission_specific_model": mission_result["winner"],
                "mission_specific_R2_mean": float(mission_winner_row["R2_mean"]),
                "mission_specific_MAE_mean": float(mission_winner_row["MAE_mean"]),
                "pooled_beats_mission_specific_R2": (
                    float(pooled_row["R2_mean"].iloc[0]) > float(mission_winner_row["R2_mean"])
                    if len(pooled_row)
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pooled_df = build_pooled_dataset()
    n_companies = pooled_df[GROUP_COL].nunique()
    print(f"=== POOLED (all 3 real missions): {len(pooled_df)} panel rows, {n_companies} companies ===")
    print(pooled_df[MISSION_COL].value_counts().to_string())

    summary, per_mission_summary, fold_detail, per_mission_fold_detail, negative_log_inputs = run_pooled_bakeoff(
        pooled_df
    )
    print("\n--- Overall pooled bake-off summary ---")
    print(summary.to_string(index=False))
    print("\n--- Per-mission sliced performance (every candidate model) ---")
    print(per_mission_summary.sort_values(["mission", "MAE_mean"]).to_string(index=False))

    summary.to_csv(OUTPUT_DIR / "model_bakeoff_pooled_summary.csv", index=False)
    per_mission_summary.to_csv(OUTPUT_DIR / "model_bakeoff_pooled_per_mission_summary.csv", index=False)
    fold_detail.to_csv(OUTPUT_DIR / "model_bakeoff_pooled_folds.csv", index=False)
    per_mission_fold_detail.to_csv(OUTPUT_DIR / "model_bakeoff_pooled_per_mission_folds.csv", index=False)
    if len(negative_log_inputs):
        negative_log_inputs.to_csv(OUTPUT_DIR / "log1p_negative_values_pooled.csv", index=False)

    result = select_pooled_winner(summary, fold_detail)
    winner = result["winner"]
    print(f"\nSelected pooled winner (same selection procedure as model_selection.py): {winner}")
    print(f"  Top-ranked by composite_rank alone: {result['top_ranked_model']}")
    print(f"  Runner-up: {result['runner_up']}")

    mission_specific_results = {}
    for mission in REAL_MISSIONS:
        slug = mission.lower().replace(" ", "_")
        summary_path = OUTPUT_DIR / f"model_bakeoff_{slug}_summary.csv"
        folds_path = OUTPUT_DIR / f"model_bakeoff_{slug}_folds.csv"
        if not summary_path.exists() or not folds_path.exists():
            raise FileNotFoundError(
                f"Missing {summary_path} / {folds_path}: run `python -m src.model_bakeoff` "
                "(mission-specific bake-off) first so this comparison has something to compare against."
            )
        mission_summary = pd.read_csv(summary_path)
        mission_folds = pd.read_csv(folds_path)
        mission_specific_results[mission] = select_model(mission_summary, mission_folds)

    comparison = build_comparison_table(winner, per_mission_summary, mission_specific_results)
    print("\n=== Pooled winner vs. each mission's own best mission-specific model ===")
    print(comparison.to_string(index=False))
    comparison.to_csv(OUTPUT_DIR / "pooled_vs_mission_specific_comparison.csv", index=False)
    print(f"\nWrote {OUTPUT_DIR / 'pooled_vs_mission_specific_comparison.csv'}")


if __name__ == "__main__":
    main()
