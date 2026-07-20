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

GROUP_COL is company_id (data_prep.make_company_id), not the raw CH number
column. GroupKFold errors/misbehaves on a null group label, and CH number
can be null (GeoData Institute has none of its own) or shared by genuinely
different companies (a handful of shared_ch_number_anomaly cases) — see
data_prep.py's make_company_id docstring for the full reasoning. This was
a live bug: GeoData Institute is in Resilient Earth's labelled panel, so
grouping by raw CH number would have handed GroupKFold a null group.
"""
import argparse
import json
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

from src.data_prep import COMPANY_ID_COL, prepare_source2
from src.feature_engineering import build_features
from src.mission_segmentation import MISSION_COL, REAL_MISSIONS, load_mapping, segment_missions

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "processed"

RANDOM_STATE = 42
TARGET_COL = "total_turnover"
GROUP_COL = COMPANY_ID_COL
WEIGHT_COL = "sample_weight"

# NOT YET WIRED UP — no adjacent data exists to test against (see
# ADJACENT_DATA_REQUIREMENTS.md and CLAUDE.md "Current status / build
# order" step 3). Once adjacent rows are merged in, this should scale
# their sample_weight down relative to space companies' inverse-frequency
# weight (sample_construction.build_long_panel), so the model trusts
# adjacent-company labels less. The value itself is a placeholder — to be
# tuned empirically once adjacent data exists, by comparing candidate
# weights (e.g. 0.2/0.5/1.0) against held-out SPACE-COMPANY-ONLY
# validation performance, not pooled space+adjacent performance (see the
# outer-CV note in CLAUDE.md's "Documented assumptions and thresholds").
ADJACENT_SAMPLE_WEIGHT = 0.5

LOG_NUMERIC_FEATURES = [
    "total_employees",
    "balance_sheet_total_assets",
    "total_export_revenue",
    "assets_per_employee",
    "export_revenue_per_employee",
    # Source 3 (grants/accelerator/funding enrichment): counts and
    # monetary totals, same skewed-feature treatment as the rest of this
    # list — see feature_engineering.py's Source 3 section.
    "grants_count",
    "grants_total_amount",
    "fundraising_count",
    "fundraising_total_amount",
]
PLAIN_NUMERIC_FEATURES = [
    "year",
    "company_age_years",
    # Source 3: recency (years-since, same pattern as company_age_years)
    # and small bounded counts/binary indicators — not log-transformed,
    # range is too small (accelerator_count caps at 5; signals are 0/1) to
    # benefit from it.
    "grant_recency_years",
    "fundraising_recency_years",
    "accelerator_count",
    "signal_equity_fundraising",
    "signal_debt_fundraising",
    "signal_mbo_mbi",
    "signal_acquired",
    "signal_made_acquisition",
    "signal_ipo",
    "signal_rd_grant",
    "signal_patent",
    "has_attended_accelerator",
    "is_academic_spinout",
]
NUMERIC_FEATURES = PLAIN_NUMERIC_FEATURES + LOG_NUMERIC_FEATURES
CATEGORICAL_FEATURES = [
    "employee_count_source",
    "company_size",
    "sic_code_1",
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


def check_negative_log_inputs(df: pd.DataFrame, columns: list[str], id_col: str = GROUP_COL) -> pd.DataFrame:
    """log1p requires x > -1, and every LOG_NUMERIC_FEATURES column is a
    financial quantity or ratio that should be non-negative by definition
    (employee counts, assets, export revenue). A negative value here means
    bad upstream data, not something to silently pass through log1p (which
    would still run, just on a nonsensical input) or clip away. Logged so a
    future data source (e.g. adjacent companies) that does produce
    negatives gets caught, not silently absorbed."""
    rows = []
    for col in columns:
        mask = df[col] < 0
        if not mask.any():
            continue
        subset = pd.DataFrame({id_col: df.loc[mask, id_col], "column": col, "value": df.loc[mask, col]})
        rows.append(subset)
    if not rows:
        return pd.DataFrame(columns=[id_col, "column", "value"])
    return pd.concat(rows, ignore_index=True)


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
                "best_params": json.dumps(search.best_params_, default=str),
            }
        )
    return fold_rows


def run_bakeoff(
    mission_df: pd.DataFrame, n_outer_splits: int = 5, n_outer_repeats: int = 5, n_inner_splits: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = cast_categoricals(mission_df)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]
    groups = df[GROUP_COL]
    weights = df[WEIGHT_COL]

    negative_log_inputs = check_negative_log_inputs(df, LOG_NUMERIC_FEATURES)

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

    return summary, fold_detail, negative_log_inputs


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

        summary, fold_detail, negative_log_inputs = run_bakeoff(mission_df)
        print(summary.to_string(index=False))

        slug = mission.lower().replace(" ", "_")
        summary.to_csv(OUTPUT_DIR / f"model_bakeoff_{slug}_summary.csv", index=False)
        fold_detail.to_csv(OUTPUT_DIR / f"model_bakeoff_{slug}_folds.csv", index=False)
        print(f"Wrote data/processed/model_bakeoff_{slug}_summary.csv and _folds.csv")

        if len(negative_log_inputs):
            log_path = OUTPUT_DIR / f"log1p_negative_values_{slug}.csv"
            negative_log_inputs.to_csv(log_path, index=False)
            print(f"WARNING: {len(negative_log_inputs)} negative values in log-transformed features, wrote {log_path}")


if __name__ == "__main__":
    main()
