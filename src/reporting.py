"""Proof-of-concept output artifacts: feature weights, performance summary,
out-of-fold actual-vs-predicted, and residuals, per mission. Structured
CSVs only — no visualisation tooling assumed.

Uses the CURRENTLY saved final_model_*.joblib artifacts (produced by
model_selection.py from the latest model_bakeoff.py run) — rerun the
bake-off + model_selection first if the feature set or data has changed.
feature_weights_{mission}.csv has two possible shapes depending on the
mission's selected model, distinguished by a `metric_type` column (and, for
the importance shape, a leading '#' comment line in the file itself):
linear models (Lasso/Ridge/Elastic Net/Linear Regression) get log-space
`coefficient` + `direction` (sign is meaningful); tree ensembles with no
coef_ (Gradient Boosting/Random Forest/Extra Trees) fall back to
`feature_importances_` as an `importance` column (magnitude only, no
sign/direction — not comparable to a coefficient). A selected model with
neither is skipped with a printed reason, but still gets its
performance_summary/actual_vs_predicted/residuals artifacts.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.utils.validation import has_fit_parameter

from src.data_prep import COMPANY_ID_COL, NAME_COL
from src.mission_segmentation import MISSION_COL, REAL_MISSIONS
from src.model_bakeoff import (
    CATEGORICAL_FEATURES,
    GROUP_COL,
    MODELS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    SCALE_SENSITIVE,
    TARGET_COL,
    WEIGHT_COL,
    build_preprocessor,
    cast_categoricals,
    get_mission_features,
    make_repeated_group_kfold_splits,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "processed"

INNER_N_SPLITS = 3
OUTER_N_SPLITS = 5
OUTER_N_REPEATS = 5


def extract_feature_weights(mission: str) -> tuple[pd.DataFrame, str | None]:
    """Returns (weights_df, header_note). header_note is None for the
    coefficient path (linear models) and a plain-text explanatory line for
    the importance path (tree ensembles), meant to be written as the first
    line of the output CSV — see main().

    Coefficient path: coefficients are in log-turnover space (target was
    log1p-transformed via TransformedTargetRegressor) and, for these
    scale-sensitive models, standardised-feature space too (numeric features
    pass through StandardScaler) — multiplicative effects on turnover, not
    additive raw ones, and not directly comparable to a coefficient on an
    unscaled feature. Sign/direction is still directly meaningful.

    Importance path (models with no coef_, e.g. Gradient Boosting/Random
    Forest/Extra Trees): sklearn's `feature_importances_` is magnitude-only
    (non-negative, sums to 1 across features) — it says how much a feature
    reduced impurity/loss across the ensemble's splits, not whether higher
    values of that feature push turnover up or down. This is NOT the same
    quantity as `coefficient` in the linear-model output, so it's written to
    its own `importance` column (no `direction` column, since one doesn't
    exist for this metric) with a `metric_type` flag and a header comment,
    rather than silently reusing the coefficient schema."""
    slug = mission.lower().replace(" ", "_")
    model = joblib.load(OUTPUT_DIR / f"final_model_{slug}.joblib")
    preprocessor = model.named_steps["preprocess"]
    ttr = model.named_steps["model"]
    regressor = ttr.regressor_
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(regressor, "coef_"):
        df = pd.DataFrame(
            {
                "Mission": mission,
                "feature": feature_names,
                "coefficient": regressor.coef_,
            }
        )
        df["direction"] = np.select(
            [df["coefficient"] > 0, df["coefficient"] < 0], ["positive", "negative"], default="zero"
        )
        df["metric_type"] = "coefficient"
        df["abs_coefficient"] = df["coefficient"].abs()
        df = df.sort_values("abs_coefficient", ascending=False).drop(columns="abs_coefficient").reset_index(drop=True)
        return df, None

    if hasattr(regressor, "feature_importances_"):
        df = pd.DataFrame(
            {
                "Mission": mission,
                "feature": feature_names,
                "importance": regressor.feature_importances_,
            }
        )
        df["metric_type"] = "importance"
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        header_note = (
            f"# {mission}: {type(regressor).__name__}.feature_importances_ — magnitude only "
            "(impurity/loss reduction, sums to 1 across features), NOT a coefficient. No sign/"
            "direction is available for this metric_type, unlike the coefficient-based "
            "feature_weights files for missions with a linear selected model."
        )
        return df, header_note

    raise ValueError(
        f"{mission}: selected model ({type(regressor).__name__}) supports neither `coef_` nor "
        "`feature_importances_` — extract_feature_weights has no extraction path for it."
    )


def performance_summary() -> pd.DataFrame:
    selected = pd.read_csv(OUTPUT_DIR / "selected_models.csv")
    rows = []
    for _, row in selected.iterrows():
        mission = row["mission"]
        slug = mission.lower().replace(" ", "_")
        bakeoff_summary = pd.read_csv(OUTPUT_DIR / f"model_bakeoff_{slug}_summary.csv")
        model_row = bakeoff_summary[bakeoff_summary["Model"] == row["selected_model"]].iloc[0]
        mission_df = get_mission_features(mission)
        rows.append(
            {
                "Mission": mission,
                "selected_model": row["selected_model"],
                "MAE_mean": model_row["MAE_mean"],
                "MAE_std": model_row["MAE_std"],
                "RMSE_mean": model_row["RMSE_mean"],
                "RMSE_std": model_row["RMSE_std"],
                "R2_mean": model_row["R2_mean"],
                "R2_std": model_row["R2_std"],
                "n_companies": mission_df[GROUP_COL].nunique(),
                "n_panel_rows": len(mission_df),
                "usable": row["usable"],
                "exclusion_reason": row["exclusion_reason"],
            }
        )
    return pd.DataFrame(rows)


def generate_oof_predictions(mission: str) -> pd.DataFrame:
    """Repeated grouped CV (same 5-fold x 5-repeat scheme as model_bakeoff.py)
    for the mission's selected model only, capturing each row's prediction
    whenever it landed in a test fold. A row is held out exactly once per
    repeat (5 times total); its OOF prediction is the mean of those 5 —
    genuine held-out performance, never a training-fit (in-sample) value."""
    selected = pd.read_csv(OUTPUT_DIR / "selected_models.csv")
    model_name = selected.loc[selected["mission"] == mission, "selected_model"].iloc[0]
    estimator, param_grid = MODELS[model_name]

    mission_df = get_mission_features(mission)
    df = cast_categoricals(mission_df)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]
    groups = df[GROUP_COL]
    weights = df[WEIGHT_COL]

    outer_splits = make_repeated_group_kfold_splits(
        X, y, groups, n_splits=OUTER_N_SPLITS, n_repeats=OUTER_N_REPEATS, random_state=RANDOM_STATE
    )
    inner_cv = GroupKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    preprocessor = build_preprocessor(scale=model_name in SCALE_SENSITIVE)

    preds_accum: dict = {}
    for _repeat, _fold_idx, train_idx, test_idx in outer_splits:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        w_train = weights.iloc[train_idx]
        groups_train = groups.iloc[train_idx]

        target_model = TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)
        pipe = Pipeline([("preprocess", preprocessor), ("model", target_model)])
        search = GridSearchCV(pipe, param_grid=param_grid, cv=inner_cv, scoring="neg_mean_absolute_error", n_jobs=-1)
        fit_kwargs = {"groups": groups_train}
        if has_fit_parameter(estimator, "sample_weight"):
            fit_kwargs["model__sample_weight"] = w_train.to_numpy()
        search.fit(X_train, y_train, **fit_kwargs)

        y_pred = search.best_estimator_.predict(X_test)
        for idx, pred in zip(df.index[test_idx], y_pred):
            preds_accum.setdefault(idx, []).append(pred)

    oof_pred = pd.Series({idx: float(np.mean(v)) for idx, v in preds_accum.items()}, name="predicted_total_turnover_oof")
    oof_n_folds = pd.Series({idx: len(v) for idx, v in preds_accum.items()}, name="n_test_folds")

    result = df[[GROUP_COL, NAME_COL, MISSION_COL, "year"]].copy()
    result["actual_total_turnover"] = y
    result = result.join(oof_pred).join(oof_n_folds)
    return result


def build_residuals(actual_vs_predicted: pd.DataFrame) -> pd.DataFrame:
    df = actual_vs_predicted.copy()
    df["residual"] = df["actual_total_turnover"] - df["predicted_total_turnover_oof"]
    df["abs_residual"] = df["residual"].abs()
    df["pct_error"] = np.where(
        df["actual_total_turnover"] != 0, df["residual"] / df["actual_total_turnover"] * 100, np.nan
    )
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Model performance summary ===")
    perf = performance_summary()
    print(perf.to_string(index=False))
    perf.to_csv(OUTPUT_DIR / "model_performance_summary.csv", index=False)
    print(f"Wrote {OUTPUT_DIR / 'model_performance_summary.csv'}\n")

    for mission in REAL_MISSIONS:
        slug = mission.lower().replace(" ", "_")
        print(f"=== {mission}: feature weights ===")
        try:
            weights, header_note = extract_feature_weights(mission)
        except ValueError as exc:
            # A selected model with neither coef_ nor feature_importances_
            # (not currently reachable by any of the 9 bake-off candidates,
            # but kept as a safety net) — skip it here rather than aborting
            # the OOF/residuals generation below for every mission.
            print(f"SKIPPED: {exc}\n")
            continue

        out_path = OUTPUT_DIR / f"feature_weights_{slug}.csv"
        if header_note:
            # Importance-based file: a leading '#' comment line makes the
            # metric_type switch (coefficient vs. importance) visible even
            # to someone opening the CSV directly, not just reading the
            # metric_type column programmatically.
            with open(out_path, "w") as f:
                f.write(header_note + "\n")
            weights.to_csv(out_path, mode="a", index=False)
        else:
            weights.to_csv(out_path, index=False)
        print(weights.head(10).to_string(index=False))
        print(f"Wrote {out_path} ({len(weights)} features)\n")

    for mission in REAL_MISSIONS:
        slug = mission.lower().replace(" ", "_")
        print(f"=== {mission}: out-of-fold actual vs predicted ===")
        avp = generate_oof_predictions(mission)
        avp_path = OUTPUT_DIR / f"actual_vs_predicted_{slug}.csv"
        avp.to_csv(avp_path, index=False)
        print(f"Wrote {avp_path} ({len(avp)} rows)")

        residuals = build_residuals(avp)
        resid_path = OUTPUT_DIR / f"residuals_{slug}.csv"
        residuals.to_csv(resid_path, index=False)
        oof_mae = mean_absolute_error(residuals["actual_total_turnover"], residuals["predicted_total_turnover_oof"])
        oof_r2 = r2_score(residuals["actual_total_turnover"], residuals["predicted_total_turnover_oof"])
        print(f"OOF MAE (single run, informal check against bake-off mean): {oof_mae:,.0f}, R2: {oof_r2:.2f}")
        print(f"Wrote {resid_path}\n")


if __name__ == "__main__":
    main()
