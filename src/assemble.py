"""Recombine observed + predicted turnover into one company-level completed
dataset (methodology sec 1.10). One row per company, in priority order:

1. Observed turnover (any mission, any eligibility, including cross-cutting)
   — the company's most recent year with a non-null Total Turnover value.
   Never overwritten by a prediction. Mission status doesn't invalidate
   ground-truth data that's independent of mission assignment — EXCEPT for
   mission_segmentation.py's KNOWN_MISCATEGORIZED_COMPANIES (distinct from
   the Sky UK data-entry-error case): those are excluded from this branch
   too, since the turnover value's own relevance is in question, not just
   the mission tag (see KNOWN_MISCATEGORIZED_COMPANIES' per-company reason).
2. Predicted turnover — inference companies in missions selected_models.csv
   marks usable=True, scored by predict.py.
3. Inference companies in a mission marked usable=False (e.g. ACE: its
   selected model has R2<0 under repeated CV, worse than the mission mean)
   — turnover_source="no_model_insufficient_data" flags this explicitly
   rather than passing off a number with false confidence. Read from
   selected_models.csv, not hardcoded — see model_selection.py.
4. Cross-cutting companies with no observed turnover — scored via
   src/cross_cutting_prediction.py's buzzword/SIC-code best-guess mission
   assignment (see that module's docstring and CLAUDE.md "Planned:
   cross-cutting predictions"), reliability="approximate" (distinct from
   observed/standard/low — the mission itself is inferred, not given).
   Companies for which even that produced no valid prediction fall through
   to turnover_source="cross_cutting_unmodelled" below.
5. Data-quality exclusions (Sky UK's data-entry-error row,
   KNOWN_MISCATEGORIZED_COMPANIES entries, and any future true-duplicate
   company records) — retained for reference, not silently dropped; the
   real per-row reason from mission_segmentation.py's exclusion_reason
   column is carried straight through, not a second hardcoded copy of it.

Also flags (not yet acts on) stale observed turnover — see
STALE_THRESHOLD_YEARS below — and enforces one-row-per-company by
company_id, exporting any real duplicates for manual review instead of
silently dropping or keeping both (see enforce_one_company_per_row).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_prep import CH_COL, COMPANY_ID_COL, NAME_COL, URL_COL, prepare_source2
from src.mission_segmentation import KNOWN_MISCATEGORIZED_COMPANIES, MISSION_COL, load_mapping, segment_missions
from src.sample_construction import YEARS, turnover_col

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "processed"

# Join/dedup key: company_id (data_prep.make_company_id), not the old
# name+URL+CH composite — see feature_engineering.py's MERGE_KEY comment
# for why (null URLs silently broke that merge for 2 companies).
JOIN_KEY = [COMPANY_ID_COL]
IDENTITY_COLS = [COMPANY_ID_COL, NAME_COL, URL_COL, CH_COL]

CROSS_CUTTING_REASON = (
    "Cross-cutting company retained for reference; the buzzword/SIC-code best-guess "
    "mission assignment (src/cross_cutting_prediction.py) produced no valid prediction "
    "for this company (e.g. no covariate data to score at all) — see "
    "predictions_cross_cutting.csv / prediction_invalid_reason for detail."
)

# ASSUMPTION (documented here + CLAUDE.md "Stale observed turnover" +
# turnover_age_years/turnover_is_stale columns below): "stale" means a
# company's most recent observed turnover year is more than 3 years older
# than the most recent year ANY company in the dataset filed. UK companies
# must file accounts annually, so >3 consecutive years with nothing filed
# suggests the company has genuinely stopped reporting turnover, not just
# an administrative lag. This is a proposed threshold, not validated
# against how many companies it actually flags — see CLAUDE.md for the
# still-open decision on what to do about stale values (currently: nothing,
# they stay turnover_source="observed").
STALE_THRESHOLD_YEARS = 3


def enforce_one_company_per_row(df: pd.DataFrame, stage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """"One row per company" is enforced, not just reported. Any company_id
    appearing more than once is pulled into a separate export for manual
    review, rather than silently dropped (loses information) or silently
    kept as multiple rows (breaks the guarantee this file exists to give)."""
    dup_mask = df[COMPANY_ID_COL].duplicated(keep=False)
    if not dup_mask.any():
        return df, df.iloc[0:0].copy()
    duplicates = df[dup_mask].copy()
    duplicates["duplicate_detected_at_stage"] = stage
    deduped = df[~dup_mask].copy()
    return deduped, duplicates


def latest_observed_turnover(segmented_df: pd.DataFrame) -> pd.DataFrame:
    cols = {turnover_col(y): y for y in YEARS}
    long = segmented_df[IDENTITY_COLS + list(cols)].melt(
        id_vars=IDENTITY_COLS, var_name="_col", value_name="turnover_value"
    )
    long["year"] = long["_col"].map(cols)
    long = long.drop(columns="_col")
    observed = long[long["turnover_value"].notna()].sort_values("year", ascending=False)
    observed = observed.drop_duplicates(subset=JOIN_KEY, keep="first")
    return observed[JOIN_KEY + ["year", "turnover_value"]]


def load_unusable_missions() -> dict:
    path = OUTPUT_DIR / "selected_models.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}: run `python -m src.model_selection` first.")
    selected = pd.read_csv(path)
    unusable = selected[~selected["usable"]]
    return dict(zip(unusable["mission"], unusable["exclusion_reason"]))


def assemble(
    segmented_df: pd.DataFrame, predictions_df: pd.DataFrame, unusable_missions: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = segmented_df[IDENTITY_COLS + [MISSION_COL, "training_eligible", "exclusion_reason"]].copy()
    base, dup_source = enforce_one_company_per_row(base, "segmented_df_source")

    is_known_miscategorized = base[COMPANY_ID_COL].isin(KNOWN_MISCATEGORIZED_COMPANIES)

    observed = latest_observed_turnover(segmented_df)
    base = base.merge(observed, on=JOIN_KEY, how="left")
    base["turnover_source"] = pd.NA
    base["reliability"] = pd.NA
    base["reliability_reason"] = ""

    # KNOWN_MISCATEGORIZED_COMPANIES rows never win the observed branch below
    # even when a real Total Turnover figure exists — see the module
    # docstring's point 1 for why this differs from Sky UK's treatment.
    base.loc[is_known_miscategorized, "turnover_value"] = np.nan
    base.loc[is_known_miscategorized, "year"] = np.nan

    is_observed = ~is_known_miscategorized & base["turnover_value"].notna()
    base.loc[is_observed, "turnover_source"] = "observed"
    base.loc[is_observed, "reliability"] = "observed"

    pred = predictions_df[
        JOIN_KEY + ["prediction_year", "predicted_total_turnover", "reliability", "reliability_reason"]
    ].rename(columns={"prediction_year": "pred_year", "predicted_total_turnover": "pred_value"})
    base = base.merge(pred, on=JOIN_KEY, how="left", suffixes=("", "_pred"))

    is_predicted = base["turnover_source"].isna() & base["pred_value"].notna()
    base.loc[is_predicted, "turnover_source"] = "predicted"
    base.loc[is_predicted, "year"] = base.loc[is_predicted, "pred_year"]
    base.loc[is_predicted, "turnover_value"] = base.loc[is_predicted, "pred_value"]
    base.loc[is_predicted, "reliability"] = base.loc[is_predicted, "reliability_pred"]
    base.loc[is_predicted, "reliability_reason"] = base.loc[is_predicted, "reliability_reason_pred"]
    base = base.drop(columns=["pred_year", "pred_value", "reliability_pred", "reliability_reason_pred"])

    is_no_model = (
        base["turnover_source"].isna() & base[MISSION_COL].isin(unusable_missions) & base["training_eligible"]
    )
    base.loc[is_no_model, "turnover_source"] = "no_model_insufficient_data"
    base.loc[is_no_model, "reliability"] = "n/a"
    base.loc[is_no_model, "reliability_reason"] = base.loc[is_no_model, MISSION_COL].map(unusable_missions)

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
    # The real per-row reason from mission_segmentation.py's exclusion_reason
    # (Sky UK's Value-Stream typo, a KNOWN_MISCATEGORIZED_COMPANIES entry, or
    # both) — not a second hardcoded copy that only ever described Sky UK.
    base.loc[is_data_error, "reliability_reason"] = base.loc[is_data_error, "exclusion_reason"]

    most_recent_filed_year = base.loc[base["turnover_source"] == "observed", "year"].max()
    has_year = base["year"].notna()
    base["turnover_age_years"] = pd.NA
    base.loc[has_year, "turnover_age_years"] = most_recent_filed_year - base.loc[has_year, "year"]
    base["turnover_is_stale"] = False
    is_obs = base["turnover_source"] == "observed"
    base.loc[is_obs, "turnover_is_stale"] = base.loc[is_obs, "turnover_age_years"] > STALE_THRESHOLD_YEARS

    base = base.drop(columns=["training_eligible", "exclusion_reason"])
    base, dup_final = enforce_one_company_per_row(base, "post_assembly")
    duplicates = pd.concat([dup_source, dup_final], ignore_index=True)
    return base, duplicates


def main() -> None:
    prepped, _ = prepare_source2()
    mapping = load_mapping()
    segmented, _ = segment_missions(prepped, mapping)
    unusable_missions = load_unusable_missions()

    prediction_cols = JOIN_KEY + ["prediction_year", "predicted_total_turnover", "reliability", "reliability_reason"]
    predictions_path = OUTPUT_DIR / "predictions_all.csv"
    predictions_df = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame(columns=prediction_cols)

    # Cross-cutting best-guess predictions (src/cross_cutting_prediction.py)
    # are a separate population from the 3 real missions' inference_all —
    # same predictions_all.csv schema, concatenated in here rather than
    # merged into predict.py's own output, since they come from a distinct
    # process (mission itself is guessed, not given) that assemble() doesn't
    # need to know about beyond reading the same 5 columns.
    cross_cutting_path = OUTPUT_DIR / "predictions_cross_cutting.csv"
    if cross_cutting_path.exists():
        cross_cutting_df = pd.read_csv(cross_cutting_path)
        if len(cross_cutting_df):
            predictions_df = pd.concat(
                [predictions_df, cross_cutting_df[prediction_cols]], ignore_index=True
            )

    final, duplicates = assemble(segmented, predictions_df, unusable_missions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "final_completed_dataset.csv"
    final.to_csv(out_path, index=False)

    print("Row counts by turnover_source:")
    print(final["turnover_source"].value_counts().to_string())

    n_stale = int(final["turnover_is_stale"].sum())
    print(f"\nStale observed turnover (>{ STALE_THRESHOLD_YEARS } years old): {n_stale} companies (flagged, not reclassified)")

    n_companies = len(final)
    n_unique = final[JOIN_KEY].drop_duplicates().shape[0]
    print(f"\nTotal rows: {n_companies}")
    print(f"Unique companies (by company_id): {n_unique}")
    print(f"No company appears more than once: {n_companies == n_unique}")

    if len(duplicates):
        dup_path = OUTPUT_DIR / "assemble_duplicate_company_id.csv"
        duplicates.to_csv(dup_path, index=False)
        print(f"WARNING: {len(duplicates)} rows removed as duplicate company_id, wrote {dup_path}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
