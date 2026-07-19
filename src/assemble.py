"""Recombine observed + predicted turnover into one company-level completed
dataset (methodology sec 1.10). One row per company, in priority order:

1. Observed turnover (any mission, any eligibility, including cross-cutting)
   — the company's most recent year with a non-null Total Turnover value.
   Never overwritten by a prediction. Mission status doesn't invalidate
   ground-truth data that's independent of mission assignment.
2. Predicted turnover — Beyond Earth / Resilient Earth inference companies
   scored by predict.py.
3. ACE inference companies with no observed turnover — ACE's selected
   model has R2<0 (worse than the mission mean) under repeated CV, so no
   numeric prediction is produced; turnover_source flags this explicitly
   rather than passing off a number with false confidence.
4. Cross-cutting companies with no observed turnover — retained for
   reference per the methodology; the buzzword-based best-guess mission
   assignment + scoring extension isn't built yet (see CLAUDE.md "Planned:
   cross-cutting predictions").
5. Data-quality exclusions (e.g. the Sky UK data-entry-error row, and any
   future true-duplicate company records) — retained for reference, not
   silently dropped.
"""
from pathlib import Path

import pandas as pd

from src.data_prep import CH_COL, NAME_COL, URL_COL, prepare_source2
from src.mission_segmentation import MISSION_COL, load_mapping, segment_missions
from src.predict import NO_MODEL_MISSIONS
from src.sample_construction import YEARS, turnover_col

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "processed"

MERGE_KEY = [NAME_COL, URL_COL, CH_COL]

CROSS_CUTTING_REASON = (
    "Cross-cutting companies retained for reference; buzzword-based best-guess "
    "mission assignment + scoring not yet built (see CLAUDE.md 'Planned: "
    "cross-cutting predictions')."
)


def latest_observed_turnover(segmented_df: pd.DataFrame) -> pd.DataFrame:
    cols = {turnover_col(y): y for y in YEARS}
    long = segmented_df[MERGE_KEY + list(cols)].melt(
        id_vars=MERGE_KEY, var_name="_col", value_name="turnover_value"
    )
    long["year"] = long["_col"].map(cols)
    long = long.drop(columns="_col")
    observed = long[long["turnover_value"].notna()].sort_values("year", ascending=False)
    observed = observed.drop_duplicates(subset=MERGE_KEY, keep="first")
    return observed[MERGE_KEY + ["year", "turnover_value"]]


def assemble(segmented_df: pd.DataFrame, predictions_df: pd.DataFrame) -> pd.DataFrame:
    base = segmented_df[MERGE_KEY + [MISSION_COL, "training_eligible"]].copy()

    observed = latest_observed_turnover(segmented_df)
    base = base.merge(observed, on=MERGE_KEY, how="left")
    base["turnover_source"] = pd.NA
    base["reliability"] = pd.NA
    base["reliability_reason"] = ""

    is_observed = base["turnover_value"].notna()
    base.loc[is_observed, "turnover_source"] = "observed"
    base.loc[is_observed, "reliability"] = "observed"

    pred = predictions_df[
        MERGE_KEY + ["prediction_year", "predicted_total_turnover", "reliability", "reliability_reason"]
    ].rename(columns={"prediction_year": "pred_year", "predicted_total_turnover": "pred_value"})
    base = base.merge(pred, on=MERGE_KEY, how="left", suffixes=("", "_pred"))

    is_predicted = base["turnover_source"].isna() & base["pred_value"].notna()
    base.loc[is_predicted, "turnover_source"] = "predicted"
    base.loc[is_predicted, "year"] = base.loc[is_predicted, "pred_year"]
    base.loc[is_predicted, "turnover_value"] = base.loc[is_predicted, "pred_value"]
    base.loc[is_predicted, "reliability"] = base.loc[is_predicted, "reliability_pred"]
    base.loc[is_predicted, "reliability_reason"] = base.loc[is_predicted, "reliability_reason_pred"]
    base = base.drop(columns=["pred_year", "pred_value", "reliability_pred", "reliability_reason_pred"])

    is_ace_no_model = base["turnover_source"].isna() & (base[MISSION_COL] == "ACE") & base["training_eligible"]
    base.loc[is_ace_no_model, "turnover_source"] = "no_model_insufficient_data"
    base.loc[is_ace_no_model, "reliability"] = "n/a"
    base.loc[is_ace_no_model, "reliability_reason"] = NO_MODEL_MISSIONS["ACE"]

    is_cross_cutting = base["turnover_source"].isna() & (base[MISSION_COL] == "Cross-cutting")
    base.loc[is_cross_cutting, "turnover_source"] = "cross_cutting_unmodelled"
    base.loc[is_cross_cutting, "reliability"] = "n/a"
    base.loc[is_cross_cutting, "reliability_reason"] = CROSS_CUTTING_REASON

    is_excluded_dup = base["turnover_source"].isna() & base[MISSION_COL].notna() & ~base["training_eligible"]
    base.loc[is_excluded_dup, "turnover_source"] = "excluded_duplicate_company_record"
    base.loc[is_excluded_dup, "reliability"] = "n/a"
    base.loc[is_excluded_dup, "reliability_reason"] = "True duplicate (same CH number and same normalised name), excluded pending manual review — see data_prep.py."

    is_data_error = base["turnover_source"].isna() & base[MISSION_COL].isna()
    base.loc[is_data_error, "turnover_source"] = "excluded_data_entry_error"
    base.loc[is_data_error, "reliability"] = "n/a"
    base.loc[is_data_error, "reliability_reason"] = "Value Stream was a data-entry error (e.g. Sky UK's own name pasted into the field) — excluded from mission mapping entirely, see DATA_SCHEMA.md."

    base = base.drop(columns=["training_eligible"])
    return base


def main() -> None:
    prepped, _ = prepare_source2()
    mapping = load_mapping()
    segmented, _ = segment_missions(prepped, mapping)

    predictions_path = OUTPUT_DIR / "predictions_all.csv"
    predictions_df = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame(
        columns=MERGE_KEY + ["prediction_year", "predicted_total_turnover", "reliability", "reliability_reason"]
    )

    final = assemble(segmented, predictions_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "final_completed_dataset.csv"
    final.to_csv(out_path, index=False)

    print("Row counts by turnover_source:")
    print(final["turnover_source"].value_counts().to_string())

    n_companies = len(final)
    n_unique = final[MERGE_KEY].drop_duplicates().shape[0]
    print(f"\nTotal rows: {n_companies}")
    print(f"Unique companies (by name+URL+CH number): {n_unique}")
    print(f"No company appears more than once: {n_companies == n_unique}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
