"""Apply the selected mission model to each mission's inference population
(companies with zero observed turnover) and attach a reliability indicator.
Observed turnover is never touched here — this only ever produces
`turnover_source="predicted"` rows for companies with nothing observed.

ACE is deliberately skipped: its selected model (Elastic Net) has R2<0 under
repeated cross-validation — worse than predicting the mission mean — so it
has no genuine predictive value. Running it anyway would put
fabricated-looking numbers into the final dataset with false confidence.
ACE's inference companies get a "no reliable model available" status
instead of a numeric prediction; assemble.py should pass that status
through rather than a turnover figure."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import MERGE_KEY, STATIC_COLS, _melt_year_indexed
from src.mission_segmentation import MISSION_COL, REAL_MISSIONS, load_mapping, segment_missions
from src.model_bakeoff import CATEGORICAL_FEATURES, NUMERIC_FEATURES, cast_categoricals, get_mission_features
from src.data_prep import prepare_source2
from src.sample_construction import YEARS, construct_samples

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "processed"

NO_MODEL_MISSIONS = {
    "ACE": "no reliable model available — insufficient labelled data (best model R2 < 0 under repeated cross-validation)",
}
PREDICTABLE_MISSIONS = [m for m in REAL_MISSIONS if m not in NO_MODEL_MISSIONS]

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
        return pd.DataFrame(columns=MERGE_KEY + ["year"] + value_cols + ["is_fallback_year"])

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

    meta_cols = [MISSION_COL, "sample_weight"]
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


def predict_mission(mission: str, segmented: pd.DataFrame) -> pd.DataFrame:
    _, inference_all, _ = construct_samples(segmented)
    inference_df = inference_all[inference_all[MISSION_COL] == mission].copy()

    snapshot = build_covariate_snapshot(inference_df)
    features = add_prediction_features(snapshot, segmented)
    features = cast_categoricals(features)

    training_df = cast_categoricals(get_mission_features(mission))
    features = compute_reliability(features, training_df)

    slug = mission.lower().replace(" ", "_")
    model = joblib.load(OUTPUT_DIR / f"final_model_{slug}.joblib")

    X = features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    features["predicted_total_turnover"] = model.predict(X) if len(X) else []
    features["turnover_source"] = "predicted"
    features["prediction_year"] = features["year"]

    return features


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prepped, _ = prepare_source2()
    mapping = load_mapping()
    segmented, _ = segment_missions(prepped, mapping)

    all_predictions = []
    for mission in REAL_MISSIONS:
        if mission in NO_MODEL_MISSIONS:
            print(f"\n=== {mission}: SKIPPED — {NO_MODEL_MISSIONS[mission]} ===")
            continue

        print(f"\n=== {mission} ===")
        preds = predict_mission(mission, segmented)
        print(f"Predicted turnover for {len(preds)} inference companies")
        if len(preds):
            print("Reliability breakdown:", preds["reliability"].value_counts().to_dict())

        slug = mission.lower().replace(" ", "_")
        out_path = OUTPUT_DIR / f"predictions_{slug}.csv"
        preds.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")
        all_predictions.append(preds)

    if all_predictions:
        combined = pd.concat(all_predictions, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / "predictions_all.csv", index=False)
        print(f"\nWrote {OUTPUT_DIR / 'predictions_all.csv'} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
