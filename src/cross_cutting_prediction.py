"""SUPERSEDED, no longer called by run_full_pipeline.py or anything else —
kept only as a record of the original approach. Once Cross-cutting had
enough of its own labelled companies to support real dedicated models
(the sub-segmentation investigation, PROJECT_NOTES.md "Extended validation
round" section 6), src/predict.py's predict_cross_cutting replaced this
module's approach entirely: scoring cross-cutting companies with their own
validated models (Consultancy/Other's dedicated model + a whole-population
blended fallback) instead of borrowing a real mission's model via
best-guess similarity. Both write the same predictions_cross_cutting.csv
path — do not run both in the same pipeline invocation.

Prediction-time-only best-guess mission scoring for cross-cutting
companies with no observed turnover (PROJECT_NOTES.md "Planned: cross-cutting
predictions" — this is that build). Cross-cutting companies NEVER enter
training data for any mission model, no exceptions — this module only
affects what happens at inference time, exactly as scoped.

For each cross-cutting company with zero observed turnover in any year:
  1. Assign a best-guess mission (ACE / Beyond Earth / Resilient Earth) via
     similarity to the three real missions' own companies, on two signals:
       - SIC Code 1 (categorical exact-match frequency)
       - LinkedIn Specialties (Keywords) — free-text buzzwords, comma-
         separated (the "buzzword" half of "buzzword/SIC-code similarity";
         previously logged in feature_engineering.py's DROPPED_COLUMNS as
         "natural fit for the planned buzzword-similarity logic... not this
         pass" — this is that use).
     Each signal is scored as (matching-company-count in mission m) /
     (mission m's company count), so a large mission isn't favoured just for
     being large. Both signals are then normalised to sum to 1 across the
     3 missions and added together (equal weight, no tuned blend — same
     "don't hand-pick a blend" spirit as model_selection.py's composite
     rank). A company with no signal on either axis (no SIC match anywhere,
     no keyword overlap anywhere) falls back to the mission with the most
     companies — logged distinctly (assignment_method="fallback_plurality_
     mission_no_signal") so it's visibly the weakest-evidence case.
  2. Score the company through THAT mission's existing final_model_*.joblib,
     using the exact same feature-building path as predict.py's real
     inference population (build_covariate_snapshot / add_prediction_features
     / cast_categoricals / compute_reliability, imported directly — not
     reimplemented, to avoid the kind of drift that broke predict.py's
     Source3 merge earlier in this project).
  3. Reliability is unconditionally overwritten to "approximate" — distinct
     from every other reliability value in the pipeline (observed/standard/
     low) precisely because the *mission itself* is inferred here, not
     given. The OOD-check reasons compute_reliability() would normally
     report are preserved, appended after the assignment info, not lost.

Cross-cutting companies WITH observed turnover history never reach this
module at all — assemble.py's "observed" branch (evaluated before the
predicted branch) already covers them with real data, untouched.
"""
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_prep import COMPANY_ID_COL, prepare_source2
from src.mission_segmentation import MISSION_COL, REAL_MISSIONS, load_mapping, segment_missions
from src.sample_construction import split_labelled_inference
from src.model_bakeoff import (
    CATEGORICAL_FEATURES,
    LOG_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
    cast_categoricals,
    check_negative_log_inputs,
    get_mission_features,
)
from src.predict import (
    add_prediction_features,
    build_covariate_snapshot,
    compute_reliability,
    validate_predictions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "processed"

CROSS_CUTTING_VALUE = "Cross-cutting"
SIC_COL = "SIC Code 1"
KEYWORDS_COL = "LinkedIn Specialties (Keywords)"


def split_keywords(raw) -> set:
    if pd.isna(raw):
        return set()
    return {kw.strip().lower() for kw in str(raw).split(",") if kw.strip()}


def build_mission_profiles(real_df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Per real mission: SIC Code 1 frequency Counter, keyword frequency
    Counter (counts companies with that keyword, not raw occurrences), and
    company count (the denominator both signals normalise by)."""
    sic_profiles, keyword_profiles, sizes = {}, {}, {}
    for mission in REAL_MISSIONS:
        sub = real_df[real_df[MISSION_COL] == mission]
        sizes[mission] = len(sub)
        sic_profiles[mission] = Counter(sub[SIC_COL].dropna())
        kw_counter = Counter()
        for raw in sub[KEYWORDS_COL]:
            kw_counter.update(split_keywords(raw))
        keyword_profiles[mission] = kw_counter
    return sic_profiles, keyword_profiles, sizes


def _normalize(scores: dict) -> dict:
    total = sum(scores.values())
    if total <= 0:
        return {k: 0.0 for k in scores}
    return {k: v / total for k, v in scores.items()}


def score_missions(
    sic_value, keywords: set, sic_profiles: dict, keyword_profiles: dict, sizes: dict
) -> tuple[str, str, dict]:
    sic_scores = {
        m: (sic_profiles[m].get(sic_value, 0) / sizes[m] if sizes[m] and pd.notna(sic_value) else 0.0)
        for m in REAL_MISSIONS
    }
    kw_scores = {
        m: (sum(keyword_profiles[m].get(kw, 0) for kw in keywords) / sizes[m] if sizes[m] else 0.0)
        for m in REAL_MISSIONS
    }

    has_sic_signal = sum(sic_scores.values()) > 0
    has_kw_signal = sum(kw_scores.values()) > 0

    if not has_sic_signal and not has_kw_signal:
        assigned = max(sizes, key=sizes.get)
        return assigned, "fallback_plurality_mission_no_signal", {m: 0.0 for m in REAL_MISSIONS}

    combined = {m: _normalize(sic_scores)[m] + _normalize(kw_scores)[m] for m in REAL_MISSIONS}
    assigned = max(combined, key=combined.get)
    if has_sic_signal and has_kw_signal:
        method = "sic_and_keyword_similarity"
    elif has_sic_signal:
        method = "sic_similarity_only_no_keyword_match"
    else:
        method = "keyword_similarity_only_no_sic_match"
    return assigned, method, combined


def assign_best_guess_missions(cross_cutting_inference_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    sic_profiles, keyword_profiles, sizes = build_mission_profiles(real_df)
    rows = []
    for _, row in cross_cutting_inference_df.iterrows():
        assigned, method, scores = score_missions(
            row.get(SIC_COL), split_keywords(row.get(KEYWORDS_COL)), sic_profiles, keyword_profiles, sizes
        )
        rows.append(
            {
                COMPANY_ID_COL: row[COMPANY_ID_COL],
                "assigned_mission": assigned,
                "assignment_method": method,
                **{f"assignment_score_{m.lower().replace(' ', '_')}": scores[m] for m in REAL_MISSIONS},
            }
        )
    return pd.DataFrame(rows)


def predict_cross_cutting(segmented_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    real_df = segmented_df[segmented_df[MISSION_COL].isin(REAL_MISSIONS)]

    cc_df = segmented_df[
        (segmented_df[MISSION_COL] == CROSS_CUTTING_VALUE) & ~segmented_df["is_true_duplicate"]
    ].copy()
    _labelled_cc, inference_cc = split_labelled_inference(cc_df)

    if inference_cc.empty:
        empty = pd.DataFrame(columns=[COMPANY_ID_COL, "prediction_year", "predicted_total_turnover", "reliability", "reliability_reason"])
        return empty, pd.DataFrame(), pd.DataFrame()

    assignments = assign_best_guess_missions(inference_cc, real_df)

    snapshot = build_covariate_snapshot(inference_cc)
    features = add_prediction_features(snapshot, segmented_df)
    features = cast_categoricals(features)
    features = features.merge(assignments, on=COMPANY_ID_COL, how="left")

    negative_log_inputs = check_negative_log_inputs(features, LOG_NUMERIC_FEATURES, id_col=COMPANY_ID_COL)

    scored = []
    for mission in REAL_MISSIONS:
        subset = features[features["assigned_mission"] == mission].copy()
        if subset.empty:
            continue
        training_df = cast_categoricals(get_mission_features(mission))
        subset = compute_reliability(subset, training_df)

        slug = mission.lower().replace(" ", "_")
        model = joblib.load(OUTPUT_DIR / f"final_model_{slug}.joblib")
        X = subset[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        subset["predicted_total_turnover"] = model.predict(X) if len(X) else []
        subset["prediction_year"] = subset["year"]
        subset["scoring_mission"] = mission

        # The OOD reasons compute_reliability already worked out are kept,
        # not discarded — just prefixed with the mission-assignment context,
        # then reliability itself is force-set to "approximate" regardless
        # of what the OOD check said (even a "no OOD flags" cross-cutting
        # row is still only as good as the guessed mission it was scored
        # against).
        ood_reason = subset["reliability_reason"].replace("", pd.NA)
        subset["reliability_reason"] = (
            "cross_cutting_approximate_mission: assigned "
            + mission
            + " via "
            + subset["assignment_method"].astype(str)
            + ood_reason.map(lambda r: f"; {r}" if pd.notna(r) else "")
        )
        subset["reliability"] = "approximate"
        subset = validate_predictions(subset)
        scored.append(subset)

    result = pd.concat(scored, ignore_index=True) if scored else features.iloc[0:0].copy()
    return result, assignments, negative_log_inputs


def main() -> None:
    prepped, _ = prepare_source2()
    mapping = load_mapping()
    segmented, _ = segment_missions(prepped, mapping)

    predictions, assignments, negative_log_inputs = predict_cross_cutting(segmented)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "predictions_cross_cutting.csv"
    predictions.to_csv(out_path, index=False)

    print(f"Cross-cutting companies scored: {len(predictions)}")
    if len(assignments):
        print("\nAssignment method breakdown:")
        print(assignments["assignment_method"].value_counts().to_string())
        print("\nAssigned mission breakdown:")
        print(assignments["assigned_mission"].value_counts().to_string())
    if len(predictions):
        n_invalid = int((~predictions["prediction_valid"]).sum())
        if n_invalid:
            print(f"\nWARNING: {n_invalid} invalid predictions nulled (see prediction_invalid_reason)")

    if len(negative_log_inputs):
        log_path = OUTPUT_DIR / "log1p_negative_values_cross_cutting.csv"
        negative_log_inputs.to_csv(log_path, index=False)
        print(f"WARNING: {len(negative_log_inputs)} negative values in log-transformed features, wrote {log_path}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
