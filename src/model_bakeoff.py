"""Cross-validated regression bake-off per mission. Follows the checklist in
CLAUDE.md (methodology doc secs 1.7-1.8): all 9 candidate models, grouped CV
by company ID, per-fold preprocessing pipelines, scaling only for
scale-sensitive models, hyperparameter search for every model, MAE/RMSE/R2
for every model. Hyperparameter tuning happens inside the training fold only
(nested CV: a repeated outer GroupKFold for performance estimation, an inner
GroupKFold inside GridSearchCV for tuning), never on the full dataset.

Outer CV is repeated grouped k-fold (5 folds x 5 repeats, 25 outer
train/test splits, a different random_state per repeat) per methodology
sec 1.8's requirement to repeat the split across multiple subsets rather
than rely on a single k-fold pass — a single pass is one arbitrary
partition of a ~80-200 company dataset, and results (especially the
robustness fallback on ACE) were sensitive to it.

Every model predicts log1p(total_turnover) via TransformedTargetRegressor,
fit inside the pipeline per fold (CLAUDE.md's leakage rule already lists
"target transforms" as something that belongs inside the training fold).
Turnover is heavily right-skewed — BT (~£20-24bn), BAE Systems (~£26bn),
Rolls Royce (~£19bn) and other primes sit 2-3 orders of magnitude above the
per-mission median (~£17-21M), and raw-scale regression was dominated by
those few companies regardless of algorithm. Predictions are inverse
log1p'd back to £ before MAE/RMSE/R2 are computed, so metrics stay in
original units.

The same primes also dominate the numeric features (total_employees,
balance_sheet_total_assets, total_export_revenue and the two derived
ratios) — same multi-order-of-magnitude skew as the target. Left
untransformed, unregularized/weakly-regularized linear models (Linear
Regression, Ridge, Elastic Net) occasionally produced a wild log-space
prediction that exploded through inverse log1p (results up to 1e173).
Lasso stayed sane because L1 zeroes out unstable coefficients; the others
didn't. Fix: log1p those 5 numeric features too (LOG_NUMERIC_FEATURES),
same justification and same per-fold fitting as the target transform.

k-NN is scale-sensitive (distance-based) but isn't named in either bucket in
the CLAUDE.md checklist (only Linear/Ridge/Lasso/ElasticNet/SVR are named
scale-sensitive; only RF/Extra Trees/GB are named as using raw scale). Scaled
it here as the standard-practice assumption — flagged for confirmation.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.utils.validation import has_fit_parameter

from src.data_prep import prepare_source2
from src.feature_engineering import build_features
from src.mission_segmentation import MISSION_COL, REAL_MISSIONS, load_mapping, segment_missions

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "processed"

RANDOM_STATE = 42
TARGET_COL = "total_turnover"
GROUP_COL = "CH No. (full)"
WEIGHT_COL = "sample_weight"

LOG_NUMERIC_FEATURES = [
    "total_employees",
    "balance_sheet_total_assets",
    "total_export_revenue",
    "assets_per_employee",
    "export_revenue_per_employee",
]
PLAIN_NUMERIC_FEATURES = ["year", "company_age_years"]
NUMERIC_FEATURES = PLAIN_NUMERIC_FEATURES + LOG_NUMERIC_FEATURES
CATEGORICAL_FEATURES = [
    "employee_count_source",
    "company_size",
    "sic_code_1",
    "linkedin_industry",
    "value_stream",
]

SCALE_SENSITIVE = {"Linear Regression", "Ridge", "Lasso", "Elastic Net", "SVR", "k-NN"}

MODELS = {
    "Linear Regression": (LinearRegression(), {"model__regressor__fit_intercept": [True, False]}),
    "Ridge": (Ridge(random_state=RANDOM_STATE), {"model__regressor__alpha": [0.1, 1.0, 10.0, 100.0]}),
    "Lasso": (
        Lasso(random_state=RANDOM_STATE, max_iter=20000),
        {"model__regressor__alpha": [0.01, 0.1, 1.0, 10.0]},
    ),
    "Elastic Net": (
        ElasticNet(random_state=RANDOM_STATE, max_iter=20000),
        {"model__regressor__alpha": [0.01, 0.1, 1.0], "model__regressor__l1_ratio": [0.2, 0.5, 0.8]},
    ),
    "Random Forest": (
        RandomForestRegressor(random_state=RANDOM_STATE),
        {
            "model__regressor__n_estimators": [200, 400],
            "model__regressor__max_depth": [None, 8],
            "model__regressor__min_samples_leaf": [1, 3],
        },
    ),
    "Extra Trees": (
        ExtraTreesRegressor(random_state=RANDOM_STATE),
        {
            "model__regressor__n_estimators": [200, 400],
            "model__regressor__max_depth": [None, 8],
            "model__regressor__min_samples_leaf": [1, 3],
        },
    ),
    "Gradient Boosting": (
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {
            "model__regressor__n_estimators": [100, 200],
            "model__regressor__learning_rate": [0.05, 0.1],
            "model__regressor__max_depth": [2, 3],
        },
    ),
    "SVR": (
        SVR(),
        {"model__regressor__C": [1, 10, 100], "model__regressor__epsilon": [0.1, 1.0], "model__regressor__kernel": ["rbf"]},
    ),
    "k-NN": (
        KNeighborsRegressor(),
        {"model__regressor__n_neighbors": [3, 5, 10, 15], "model__regressor__weights": ["uniform", "distance"]},
    ),
}


def _cast_value(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
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
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def make_repeated_group_kfold_splits(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int, n_repeats: int, random_state: int
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Methodology sec 1.8: repeat the train/validation split across multiple
    subsets rather than relying on a single k-fold pass, so the outer
    performance estimate isn't sensitive to one unlucky partition. Each
    repeat is a full, independent GroupKFold partition (different
    random_state per repeat) — companies are grouped correctly within every
    repeat, but which fold a company lands in varies repeat to repeat."""
    splits = []
    for repeat in range(n_repeats):
        cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state + repeat)
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
            splits.append((repeat, fold_idx, train_idx, test_idx))
    return splits


def evaluate_model(
    name: str,
    estimator,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    weights: pd.Series,
    outer_splits: list[tuple[int, int, np.ndarray, np.ndarray]],
    inner_cv,
) -> list[dict]:
    preprocessor = build_preprocessor(scale=name in SCALE_SENSITIVE)
    fold_rows = []
    for repeat, fold_idx, train_idx, test_idx in outer_splits:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        w_train, w_test = weights.iloc[train_idx], weights.iloc[test_idx]
        groups_train = groups.iloc[train_idx]

        target_model = TransformedTargetRegressor(
            regressor=estimator, func=np.log1p, inverse_func=np.expm1
        )
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
                "best_params": search.best_params_,
            }
        )
    return fold_rows


def run_bakeoff(
    mission_df: pd.DataFrame, n_outer_splits: int = 5, n_outer_repeats: int = 5, n_inner_splits: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = cast_categoricals(mission_df)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]
    groups = df[GROUP_COL]
    weights = df[WEIGHT_COL]

    outer_splits = make_repeated_group_kfold_splits(
        X, y, groups, n_splits=n_outer_splits, n_repeats=n_outer_repeats, random_state=RANDOM_STATE
    )
    inner_cv = GroupKFold(n_splits=n_inner_splits, shuffle=True, random_state=RANDOM_STATE)

    all_folds = []
    for name, (estimator, param_grid) in MODELS.items():
        all_folds.extend(
            evaluate_model(name, estimator, param_grid, X, y, groups, weights, outer_splits, inner_cv)
        )
    fold_detail = pd.DataFrame(all_folds)

    summary = (
        fold_detail.groupby("Model")[["MAE", "RMSE", "R2"]]
        .agg(["mean", "std"])
        .round(2)
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values("MAE_mean")

    return summary, fold_detail


def get_mission_features(mission: str) -> pd.DataFrame:
    prepped, _ = prepare_source2()
    mapping = load_mapping()
    segmented, _ = segment_missions(prepped, mapping)
    features, _ = build_features(segmented)
    return features[features[MISSION_COL] == mission].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", action="append", choices=REAL_MISSIONS, default=None)
    args = parser.parse_args()
    missions = args.mission or REAL_MISSIONS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for mission in missions:
        mission_df = get_mission_features(mission)
        n_companies = mission_df[GROUP_COL].nunique()
        print(f"\n=== {mission}: {len(mission_df)} panel rows, {n_companies} companies ===")

        summary, fold_detail = run_bakeoff(mission_df)
        print(summary.to_string(index=False))

        slug = mission.lower().replace(" ", "_")
        summary.to_csv(OUTPUT_DIR / f"model_bakeoff_{slug}_summary.csv", index=False)
        fold_detail.to_csv(OUTPUT_DIR / f"model_bakeoff_{slug}_folds.csv", index=False)
        print(f"Wrote data/processed/model_bakeoff_{slug}_summary.csv and _folds.csv")


if __name__ == "__main__":
    main()
