"""13-candidate one-year-ahead turnover forecasting bake-off
(FORECASTING_METHODOLOGY.md sec 7.11), one independent run per real mission
(ACE / Beyond Earth / Resilient Earth — filtered via `primary_model_eligible`,
same convention as src/model_bakeoff.py never training a Cross-cutting
model; Cross-cutting companies are forecasted later via the benchmark-only
company-level growth models, which this module also builds and needs no
fitted regression at all).

VALIDATION SCHEME — genuinely different from src/model_bakeoff.py's
GroupKFold, and here's why: this panel is chronological (company-year), not
just company-grouped. Random/grouped k-fold would let a model train on 2023
data and get "tested" on a 2015 row — meaningless for a model that will
only ever be applied FORWARD in time (forecast_recursive.py projects
2026-2030 using a model trained on 2013-2025, never the reverse). Rolling-
origin (expanding-window) validation instead:

  - OUTER (performance reporting, OUTER_CUTOFF_YEARS): for each cutoff
    year C, train on every row with target_year <= C, test on every row
    with target_year > C. Test rows are bucketed by
    horizon = target_year - C into {"1","2","3","4+"} — sec 7.11's
    "performance reported separately per forecast horizon" — so a fixed
    one-year-ahead model's accuracy is measured at increasing distance from
    its own training cutoff, exactly the situation forecast_recursive.py
    will face applying a static model forward to 2030 without retraining.
  - INNER (hyperparameter tuning, INNER_CUTOFF_OFFSETS): the same
    expanding-window logic, nested strictly inside each outer cutoff's
    training portion (never touches that cutoff's test rows) — passed to
    GridSearchCV as a custom temporally-ordered `cv` iterable instead of
    KFold/GroupKFold.

NUMERIC STABILITY — the BT/BAE Systems lesson from src/model_bakeoff.py,
checked empirically before this module was built (see PROJECT_NOTES.md's "2030
Forecasting Pipeline" section): raw turnover-scale features measure skew
8-9.5 against real training data, the same shape that destabilised the
estimation pipeline's linear models. TransformedTargetRegressor(log1p/
expm1) on the target + log1p on LOG_NUMERIC_FEATURES below, same pattern.
Already-log-space growth features (log_growth_1y and friends, plus
employee_growth/asset_growth after their log-difference reformulation) are
NOT re-transformed — see forecast_feature_engineering.py's docstring for
why a second log1p pass would even be mechanically invalid for some of them.

NOT YET DONE, documented rather than silently skipped: this bake-off
ignores the `sample_weight` column carried through from the estimation
pipeline (1 / company's row count in ITS labelled panel) — that weighting
was designed to equalise companies regardless of turnover-history length
for a "predict overall level" task; whether it's the right scheme for a
"predict one-year-ahead transitions" task is an open question, not
resolved here. All training rows are weighted equally for now.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from forecast_src.forecast_data_prep import DATA_DIR
from src.mission_segmentation import REAL_MISSIONS

try:
    from catboost import CatBoostRegressor

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_FEATURES_PATH = DATA_DIR / "forecast_training_features.csv"

RANDOM_STATE = 42
TARGET_COL = "target_turnover_next_year"
TARGET_YEAR_COL = "target_year"
MISSION_COL = "mission"

# Expanding-window outer cutoffs: chosen so the earliest cutoff still leaves
# a workable training set for the smallest mission (ACE), and the latest
# leaves horizon-1..4 test data (target_year maxes out at 2025).
OUTER_CUTOFF_YEARS = [2018, 2019, 2020, 2021]
# Inner (nested) cutoffs, expressed as an offset back from whichever outer
# cutoff is active — e.g. outer cutoff 2021 -> inner cutoffs 2018, 2019.
INNER_CUTOFF_OFFSETS = [3, 2]

# Raw turnover/asset/employee-scale features — skew 8-9.5 confirmed against
# real data, same shape as the estimation pipeline's BT/BAE Systems problem.
LOG_NUMERIC_FEATURES = [
    "turnover_t",
    "turnover_lag_1",
    "turnover_lag_2",
    "turnover_lag_3",
    "rolling_turnover_mean_2",
    "rolling_turnover_mean_3",
    "rolling_turnover_median_3",
    "historical_turnover_max",
    "historical_turnover_min",
    "total_assets",
    "assets_per_employee",
    "employees",
    "export_revenue",
]
# Already bounded/near-symmetric quantities (log-differences, counts, small
# integers, booleans) — no further transform, see module docstring.
PLAIN_NUMERIC_FEATURES = [
    "company_age",
    "n_prior_turnover_values",
    "history_span_years",
    "years_since_previous_turnover",
    "consecutive_history_length_to_date",
    "log_growth_1y",
    "log_growth_2y_mean",
    "log_growth_3y_mean",
    "growth_volatility",
    "growth_acceleration",
    "positive_growth_count",
    "negative_growth_count",
    "employee_growth",
    "asset_growth",
    "export_share",
    "missing_feature_count",
    "missing_feature_proportion",
    "has_growth_history",
    "has_three_year_history",
    "has_employee_data",
    "has_financial_statement_data",
]
CATEGORICAL_FEATURES = ["value_stream", "company_size", "sic_code_1", "estimation_reliability"]

NUMERIC_FEATURES = LOG_NUMERIC_FEATURES + PLAIN_NUMERIC_FEATURES
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

SCALE_SENSITIVE = {"Linear Regression", "Ridge", "Lasso", "Elastic Net", "SVR"}

MODELS = {
    "Linear Regression": (LinearRegression(), {}),
    "Ridge": (Ridge(random_state=RANDOM_STATE), {"model__regressor__alpha": [0.1, 1.0, 10.0]}),
    "Lasso": (Lasso(random_state=RANDOM_STATE, max_iter=5000), {"model__regressor__alpha": [0.01, 0.1, 1.0]}),
    "Elastic Net": (
        ElasticNet(random_state=RANDOM_STATE, max_iter=5000),
        {"model__regressor__alpha": [0.01, 0.1, 1.0], "model__regressor__l1_ratio": [0.2, 0.5, 0.8]},
    ),
    "Random Forest": (
        RandomForestRegressor(random_state=RANDOM_STATE),
        {"model__regressor__n_estimators": [200], "model__regressor__max_depth": [None, 8]},
    ),
    "Extra Trees": (
        ExtraTreesRegressor(random_state=RANDOM_STATE),
        {"model__regressor__n_estimators": [200], "model__regressor__max_depth": [None, 8]},
    ),
    "Gradient Boosting": (
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {
            "model__regressor__n_estimators": [100, 200],
            "model__regressor__max_depth": [2, 3],
            "model__regressor__learning_rate": [0.05, 0.1],
        },
    ),
    "Histogram Gradient Boosting": (
        HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        {"model__regressor__max_depth": [None, 6], "model__regressor__learning_rate": [0.05, 0.1]},
    ),
    "SVR": (SVR(), {"model__regressor__C": [1.0, 10.0], "model__regressor__epsilon": [0.1, 0.5]}),
}
if CATBOOST_AVAILABLE:
    MODELS["CatBoost"] = (
        CatBoostRegressor(random_state=RANDOM_STATE, verbose=False),
        {"model__regressor__depth": [4, 6], "model__regressor__learning_rate": [0.05, 0.1]},
    )

BENCHMARK_MODELS = ["Persistence", "Mission-Average Growth", "Company Historical CAGR"]


def check_negative_log_inputs(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Same guard as src/model_bakeoff.py's check_negative_log_inputs: a
    negative value in a column that's about to be log1p'd means bad
    upstream data, not something to silently pass through."""
    rows = []
    for col in cols:
        negative = df[df[col] < 0]
        for _, row in negative.iterrows():
            rows.append({"company_id": row["company_id"], "column": col, "value": row[col]})
    return pd.DataFrame(rows)


def build_preprocessor(scale: bool) -> ColumnTransformer:
    scale_step = [("scale", StandardScaler())] if scale else []
    log_numeric = Pipeline(
        [
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="median")),
        ]
        + scale_step
    )
    plain_numeric = Pipeline([("impute", SimpleImputer(strategy="median"))] + scale_step)
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )
    return ColumnTransformer(
        [
            ("log_numeric", log_numeric, LOG_NUMERIC_FEATURES),
            ("plain_numeric", plain_numeric, PLAIN_NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def _cast_value(v):
    """Same helper as src/model_bakeoff.py's _cast_value: pandas' nullable
    "string" dtype (pd.NA) breaks sklearn's SimpleImputer (_object_dtype_isnan
    can't bool-evaluate pd.NA) — plain object dtype with real np.nan avoids
    it. Also strips the trailing ".0" a float-typed code like sic_code_1
    would otherwise carry into its string form."""
    if pd.isna(v):
        return np.nan
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].apply(_cast_value).astype(object)
    for col in ["has_growth_history", "has_three_year_history", "has_employee_data", "has_financial_statement_data"]:
        df[col] = df[col].astype(int)
    return df


def make_outer_temporal_splits(df: pd.DataFrame, cutoff_years: list[int]) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """One split per outer cutoff year: train = target_year <= cutoff,
    test = target_year > cutoff. Skips a cutoff entirely if it produces an
    empty train or test set (can happen for the smaller missions)."""
    splits = []
    years = df[TARGET_YEAR_COL].to_numpy()
    for cutoff in cutoff_years:
        train_idx = np.where(years <= cutoff)[0]
        test_idx = np.where(years > cutoff)[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append((cutoff, train_idx, test_idx))
    return splits


def make_inner_temporal_cv(train_df: pd.DataFrame, outer_cutoff: int, offsets: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Nested expanding-window splits STRICTLY inside an outer cutoff's own
    training portion — never sees that cutoff's outer test rows. `train_df`
    must already be positionally indexed 0..n-1 (reset_index(drop=True))
    so the returned arrays line up with what GridSearchCV will index into."""
    years = train_df[TARGET_YEAR_COL].to_numpy()
    positions = np.arange(len(train_df))
    folds = []
    for offset in offsets:
        inner_cutoff = outer_cutoff - offset
        inner_train_mask = years <= inner_cutoff
        inner_test_mask = (years > inner_cutoff) & (years <= outer_cutoff)
        if inner_train_mask.sum() == 0 or inner_test_mask.sum() == 0:
            continue
        folds.append((positions[inner_train_mask], positions[inner_test_mask]))
    return folds


def compute_horizon_bucket(target_year: pd.Series, cutoff: int) -> np.ndarray:
    horizon = target_year.to_numpy() - cutoff
    return np.where(horizon >= 4, "4+", horizon.astype(str))


def _score_rows(model_name, cutoff, y_test, y_pred, horizon_bucket, best_params) -> list[dict]:
    records = []
    for bucket in sorted(set(horizon_bucket), key=lambda b: (b == "4+", b)):
        mask = horizon_bucket == bucket
        n = int(mask.sum())
        if n == 0:
            continue
        records.append(
            {
                "Model": model_name,
                "cutoff_year": cutoff,
                "horizon": bucket,
                "n_test_rows": n,
                "MAE": mean_absolute_error(y_test[mask], y_pred[mask]),
                "RMSE": root_mean_squared_error(y_test[mask], y_pred[mask]),
                "R2": r2_score(y_test[mask], y_pred[mask]) if n >= 2 else np.nan,
                "best_params": json.dumps(best_params, default=str),
            }
        )
    return records


def evaluate_ml_model(model_name, estimator, param_grid, mission_df, outer_splits) -> list[dict]:
    records = []
    preprocessor = build_preprocessor(scale=model_name in SCALE_SENSITIVE)
    for cutoff, train_idx, test_idx in outer_splits:
        train_df = mission_df.iloc[train_idx].reset_index(drop=True)
        test_df = mission_df.iloc[test_idx].reset_index(drop=True)

        X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COL]
        X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COL].to_numpy()

        target_model = TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)
        pipe = Pipeline([("preprocess", preprocessor), ("model", target_model)])

        inner_cv = make_inner_temporal_cv(train_df, cutoff, INNER_CUTOFF_OFFSETS)
        if param_grid and inner_cv:
            search = GridSearchCV(pipe, param_grid=param_grid, cv=inner_cv, scoring="neg_mean_absolute_error", n_jobs=-1)
            search.fit(X_train, y_train)
            best_estimator, best_params = search.best_estimator_, search.best_params_
        else:
            best_estimator, best_params = pipe.fit(X_train, y_train), {}

        y_pred = best_estimator.predict(X_test)
        horizon_bucket = compute_horizon_bucket(test_df[TARGET_YEAR_COL], cutoff)
        records += _score_rows(model_name, cutoff, y_test, y_pred, horizon_bucket, best_params)
    return records


def predict_persistence(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """No-growth benchmark: next year's turnover = this year's."""
    return test_df["turnover_t"].to_numpy()


def predict_mission_average_growth(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Applies the TRAIN portion's mean one-year log growth to each test
    row's own current turnover — the mean is fit only on train_df, no
    leakage from the test rows it's being applied to."""
    mean_log_growth = train_df["log_growth_1y"].mean()
    if pd.isna(mean_log_growth):
        mean_log_growth = 0.0
    return np.expm1(np.log1p(test_df["turnover_t"]) + mean_log_growth)


def predict_company_cagr(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Each test row's OWN trailing growth-rate estimate (log_growth_3y_mean,
    falling back to shorter windows, then 0 for a company with no growth
    history at all) — information available at that row's own forecast
    origin t, so train_df isn't used here; there's nothing to "fit" for a
    per-company trend."""
    growth = (
        test_df["log_growth_3y_mean"]
        .fillna(test_df["log_growth_2y_mean"])
        .fillna(test_df["log_growth_1y"])
        .fillna(0.0)
    )
    return np.expm1(np.log1p(test_df["turnover_t"]) + growth)


BENCHMARK_PREDICT_FNS = {
    "Persistence": predict_persistence,
    "Mission-Average Growth": predict_mission_average_growth,
    "Company Historical CAGR": predict_company_cagr,
}


def evaluate_benchmark_model(model_name, predict_fn, mission_df, outer_splits) -> list[dict]:
    records = []
    for cutoff, train_idx, test_idx in outer_splits:
        train_df = mission_df.iloc[train_idx].reset_index(drop=True)
        test_df = mission_df.iloc[test_idx].reset_index(drop=True)
        y_test = test_df[TARGET_COL].to_numpy()
        y_pred = predict_fn(train_df, test_df)
        horizon_bucket = compute_horizon_bucket(test_df[TARGET_YEAR_COL], cutoff)
        records += _score_rows(model_name, cutoff, y_test, y_pred, horizon_bucket, {})
    return records


def get_mission_training_data(mission: str) -> pd.DataFrame:
    df = pd.read_csv(TRAINING_FEATURES_PATH)
    df = df[(df[MISSION_COL] == mission) & df["primary_model_eligible"]].reset_index(drop=True)
    return cast_categoricals(df)


def summarise(fold_detail: pd.DataFrame) -> pd.DataFrame:
    grouped = fold_detail.groupby(["Model", "horizon"])
    summary = grouped.agg(
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
        n_folds=("MAE", "count"),
        n_test_rows_total=("n_test_rows", "sum"),
    ).reset_index()
    return summary


def run_bakeoff(mission: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mission_df = get_mission_training_data(mission)
    negative_log_inputs = check_negative_log_inputs(mission_df, LOG_NUMERIC_FEATURES)

    outer_splits = make_outer_temporal_splits(mission_df, OUTER_CUTOFF_YEARS)

    all_records = []
    for model_name, (estimator, param_grid) in MODELS.items():
        all_records += evaluate_ml_model(model_name, estimator, param_grid, mission_df, outer_splits)
    for model_name in BENCHMARK_MODELS:
        all_records += evaluate_benchmark_model(model_name, BENCHMARK_PREDICT_FNS[model_name], mission_df, outer_splits)

    fold_detail = pd.DataFrame(all_records)
    summary = summarise(fold_detail)
    return summary, fold_detail, negative_log_inputs


def main() -> None:
    if not TRAINING_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TRAINING_FEATURES_PATH}: run `python -m forecast_src.forecast_feature_engineering` first."
        )
    print(f"Candidate models: {list(MODELS) + BENCHMARK_MODELS} ({len(MODELS) + len(BENCHMARK_MODELS)} total)")
    if not CATBOOST_AVAILABLE:
        print("NOTE: catboost not installed — CatBoost skipped ('where available' per sec 7.11).")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for mission in REAL_MISSIONS:
        print(f"\n=== {mission} ===")
        summary, fold_detail, negative_log_inputs = run_bakeoff(mission)

        print(f"Training rows: {len(get_mission_training_data(mission))}")
        for horizon in sorted(summary["horizon"].unique(), key=lambda b: (b == "4+", b)):
            print(f"\n--- horizon {horizon} ---")
            sub = summary[summary["horizon"] == horizon].sort_values("MAE_mean")
            print(sub[["Model", "MAE_mean", "RMSE_mean", "R2_mean", "R2_std", "n_folds", "n_test_rows_total"]].to_string(index=False))

        slug = mission.lower().replace(" ", "_")
        summary_path = DATA_DIR / f"forecast_bakeoff_{slug}_summary.csv"
        folds_path = DATA_DIR / f"forecast_bakeoff_{slug}_folds.csv"
        summary.to_csv(summary_path, index=False)
        fold_detail.to_csv(folds_path, index=False)
        print(f"\nWrote {summary_path}")
        print(f"Wrote {folds_path}")

        if len(negative_log_inputs):
            log_path = DATA_DIR / f"forecast_log1p_negative_values_{slug}.csv"
            negative_log_inputs.to_csv(log_path, index=False)
            print(f"WARNING: {len(negative_log_inputs)} negative values in log-transformed features, wrote {log_path}")


if __name__ == "__main__":
    main()
