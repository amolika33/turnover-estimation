"""Apply the selected mission model to each mission's inference population
(companies with zero observed turnover) and attach a reliability indicator.
Observed turnover is never touched here — this only ever produces
`turnover_source="predicted"` rows for companies with nothing observed.

Which missions get a numeric prediction is read from selected_models.csv
(written by model_selection.py: mission, selected_model, r2_mean, usable,
exclusion_reason), not hardcoded here — see model_selection.py's
USABILITY_R2_THRESHOLD docstring for why. A mission with usable=False (ACE,
currently: its selected model has R2<0 under repeated cross-validation —
worse than predicting the mission mean, no genuine predictive value) gets
its exclusion_reason passed through as a status instead of a fabricated-
looking number."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import IDENTITY_COLS, MERGE_KEY, STATIC_COLS, _melt_year_indexed
from src.mission_segmentation import MISSION_COL, REAL_MISSIONS, load_mapping, segment_missions
from src.model_bakeoff import (
    CATEGORICAL_FEATURES,
    LOG_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
    cast_categoricals,
    check_negative_log_inputs,
    get_mission_features,
)
from src.data_prep import COMPANY_ID_COL, prepare_source2
from src.sample_construction import YEARS, construct_samples

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "processed"

COVARIATE_FIELDS = {
    "Total Employees (CH {year})": "total_employees_ch",
    "Total Employees (Est. {year})": "total_employees_est",
    "Balance Sheet Total Assets ({year})": "balance_sheet_total_assets",
    "Total Export Revenue {year}": "total_export_revenue",
    "Size {year}": "company_size",
}

NUMERIC_OOD_TOLERANCE = 1.5


def build_covariate_snapshot(inference_df: pd.DataFrame) -> pd.DataFrame:
    """One row per inference company: the most recent year with any
    covariate populated (turnover isn't required here — these companies
    have none, by definition of being in the inference population).
    Companies with no covariate data in any year fall back to the most
    recent year with everything missing (imputed downstream by the model's
    own pipeline), flagged via is_fallback_year for the reliability check."""
    value_cols = list(COVARIATE_FIELDS.values())
    if inference_df.empty:
        meta_cols = [c for c in IDENTITY_COLS if c not in MERGE_KEY] + [MISSION_COL, "sample_weight"]
        return pd.DataFrame(columns=MERGE_KEY + ["year"] + value_cols + ["is_fallback_year"] + meta_cols)

    long = None
    for prefix_fmt, out_col in COVARIATE_FIELDS.items():
        piece = _melt_year_indexed(inference_df, prefix_fmt, out_col)
        long = piece if long is None else long.merge(piece, on=MERGE_KEY + ["year"], how="outer")

    has_data = long[value_cols].notna().any(axis=1)
    covered = long[has_data].sort_values("year", ascending=False)
    best = covered.drop_duplicates(subset=MERGE_KEY, keep="first").copy()
    best["is_fallback_year"] = False

    all_companies = inference_df[MERGE_KEY].drop_duplicates()
    missing = all_companies.merge(best[MERGE_KEY], on=MERGE_KEY, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"][MERGE_KEY]
    if not missing.empty:
        fallback = missing.copy()
        fallback["year"] = max(YEARS)
        for c in value_cols:
            fallback[c] = np.nan
        fallback["is_fallback_year"] = True
        best = pd.concat([best, fallback], ignore_index=True)

    # IDENTITY_COLS (name/URL/CH number), not just MERGE_KEY (company_id
    # alone): _melt_year_indexed's id_vars is MERGE_KEY, so the human-
    # readable identity columns were dropped by the melt above and need
    # re-attaching here for a readable predictions_all.csv.
    meta_cols = [c for c in IDENTITY_COLS if c not in MERGE_KEY] + [MISSION_COL, "sample_weight"]
    best = best.merge(
        inference_df[MERGE_KEY + meta_cols].drop_duplicates(subset=MERGE_KEY), on=MERGE_KEY, how="left"
    )
    return best


def add_prediction_features(snapshot: pd.DataFrame, segmented_df: pd.DataFrame) -> pd.DataFrame:
    df = snapshot.merge(
        segmented_df[MERGE_KEY + list(STATIC_COLS)].rename(columns=STATIC_COLS),
        on=MERGE_KEY,
        how="left",
    )

    df["company_age_years"] = df["year"] - df["founded_year"]
    df.loc[df["company_age_years"] < 0, "company_age_years"] = np.nan

    df["total_employees"] = df["total_employees_ch"].where(
        df["total_employees_ch"].notna(), df["total_employees_est"]
    )
    df["employee_count_source"] = pd.NA
    df.loc[df["total_employees_ch"].notna(), "employee_count_source"] = "filed"
    df.loc[
        df["total_employees_ch"].isna() & df["total_employees_est"].notna(), "employee_count_source"
    ] = "estimated"

    with np.errstate(divide="ignore", invalid="ignore"):
        df["assets_per_employee"] = df["balance_sheet_total_assets"] / df["total_employees"]
        df["export_revenue_per_employee"] = df["total_export_revenue"] / df["total_employees"]
    df["assets_per_employee"] = df["assets_per_employee"].replace([np.inf, -np.inf], np.nan)
    df["export_revenue_per_employee"] = df["export_revenue_per_employee"].replace([np.inf, -np.inf], np.nan)

    return df


def compute_reliability(pred_df: pd.DataFrame, training_df: pd.DataFrame) -> pd.DataFrame:
    """Flags predictions as low-reliability when: the company had no real
    covariate data in any year (pure imputation), a numeric feature exceeds
    1.5x the max seen in the mission's own labelled training data (simple
    magnitude OOD check), or a categorical value was never seen in training
    (the model's OneHotEncoder would silently zero it out otherwise)."""
    df = pred_df.copy()

    numeric_max = {
        col: training_df[col].dropna().max() if training_df[col].notna().any() else np.nan
        for col in ["total_employees", "balance_sheet_total_assets"]
    }
    seen_categories = {
        col: set(training_df[col].dropna()) for col in ["sic_code_1", "linkedin_industry", "value_stream", "company_size"]
    }

    reasons = []
    for _, row in df.iterrows():
        row_reasons = []
        if bool(row.get("is_fallback_year", False)):
            row_reasons.append("no_covariate_data_any_year")
        for col, max_val in numeric_max.items():
            v = row.get(col)
            if pd.notna(v) and pd.notna(max_val) and v > NUMERIC_OOD_TOLERANCE * max_val:
                row_reasons.append(f"{col}_exceeds_training_range")
        for col, seen in seen_categories.items():
            v = row.get(col)
            if pd.notna(v) and v not in seen:
                row_reasons.append(f"unseen_{col}_category")
        reasons.append("; ".join(row_reasons))

    df["reliability_reason"] = reasons
    df["reliability"] = np.where(df["reliability_reason"] == "", "standard", "low")
    return df


def validate_predictions(features: pd.DataFrame) -> pd.DataFrame:
    """A turnover prediction that's non-finite (inf/-inf/nan — possible if a
    row's features are entirely out-of-distribution) or negative (turnover
    can't be negative) is a modelling failure for that row, not a value to
    export silently. Flagged via prediction_valid + prediction_invalid_reason
    and the value is nulled rather than written out looking legitimate."""
    df = features.copy()
    pred = df["predicted_total_turnover"].to_numpy(dtype=float)
    is_finite = np.isfinite(pred)
    is_negative = is_finite & (pred < 0)

    df["prediction_valid"] = is_finite & ~is_negative
    df["prediction_invalid_reason"] = np.select(
        [~is_finite, is_negative], ["non_finite_prediction", "negative_prediction"], default=""
    )
    df.loc[~df["prediction_valid"], "predicted_total_turnover"] = np.nan
    return df


def load_selected_models() -> pd.DataFrame:
    path = OUTPUT_DIR / "selected_models.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}: run `python -m src.model_selection` first.")
    return pd.read_csv(path)


def predict_mission(mission: str, segmented: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, inference_all, _ = construct_samples(segmented)
    inference_df = inference_all[inference_all[MISSION_COL] == mission].copy()

    snapshot = build_covariate_snapshot(inference_df)
    features = add_prediction_features(snapshot, segmented)
    features = cast_categoricals(features)

    training_df = cast_categoricals(get_mission_features(mission))
    features = compute_reliability(features, training_df)

    # Same guard as model_bakeoff.run_bakeoff before training: log1p on a
    # negative employees/assets/export-revenue value means bad upstream
    # data, not something to silently pass through the loaded model's
    # pipeline (which applies log1p internally).
    negative_log_inputs = check_negative_log_inputs(features, LOG_NUMERIC_FEATURES, id_col=COMPANY_ID_COL)

    slug = mission.lower().replace(" ", "_")
    model = joblib.load(OUTPUT_DIR / f"final_model_{slug}.joblib")

    X = features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    features["predicted_total_turnover"] = model.predict(X) if len(X) else []
    features["turnover_source"] = "predicted"
    features["prediction_year"] = features["year"]
    features = validate_predictions(features)

    return features, negative_log_inputs


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prepped, _ = prepare_source2()
    mapping = load_mapping()
    segmented, _ = segment_missions(prepped, mapping)
    selected_models = load_selected_models()

    all_predictions = []
    for mission in REAL_MISSIONS:
        row = selected_models[selected_models["mission"] == mission]
        if row.empty or not bool(row.iloc[0]["usable"]):
            reason = row.iloc[0]["exclusion_reason"] if not row.empty else "mission not found in selected_models.csv"
            print(f"\n=== {mission}: SKIPPED — {reason} ===")
            continue

        print(f"\n=== {mission} ===")
        preds, negative_log_inputs = predict_mission(mission, segmented)
        print(f"Predicted turnover for {len(preds)} inference companies")
        if len(preds):
            print("Reliability breakdown:", preds["reliability"].value_counts().to_dict())
            n_invalid = int((~preds["prediction_valid"]).sum())
            if n_invalid:
                print(f"WARNING: {n_invalid} invalid predictions nulled (see prediction_invalid_reason)")

        slug = mission.lower().replace(" ", "_")
        out_path = OUTPUT_DIR / f"predictions_{slug}.csv"
        preds.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")
        all_predictions.append(preds)

        if len(negative_log_inputs):
            log_path = OUTPUT_DIR / f"log1p_negative_values_predict_{slug}.csv"
            negative_log_inputs.to_csv(log_path, index=False)
            print(f"WARNING: {len(negative_log_inputs)} negative values in log-transformed features, wrote {log_path}")

    if all_predictions:
        combined = pd.concat(all_predictions, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / "predictions_all.csv", index=False)
        print(f"\nWrote {OUTPUT_DIR / 'predictions_all.csv'} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
