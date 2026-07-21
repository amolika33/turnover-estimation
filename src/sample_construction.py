"""Within each mission dataset, split training-eligible companies into a
labelled (observed turnover) population and an inference (missing turnover)
population, and reshape the labelled population into long/panel format."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_prep import CH_COL, COMPANY_ID_COL, NAME_COL, URL_COL, prepare_source2
from src.mission_segmentation import MISSION_COL, REAL_MISSIONS, load_mapping, segment_missions

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "processed"

YEARS = list(range(2013, 2026))

# ASSUMPTION (documented here + PROJECT_NOTES.md "Panel row weighting" +
# sample_weight column itself): the methodology's unit of analysis is the
# company, not the company-year. A company with 13 years of turnover history
# and one with 1 year should count equally in training, not 13:1 — so
# sample_weight = 1 / that company's row count in the labelled panel.
# Consciously accepted tradeoff: this down-weights well-documented companies
# relative to naive equal-row weighting.
POPULATION_TYPE_SPACE = "space"
# Adjacent-company rows (src/adjacent_data_prep.py) — training-only, never
# part of the inference population, always tagged so model_bakeoff.py can
# tell them apart (space-only outer CV test folds, ADJACENT_SAMPLE_WEIGHT).
POPULATION_TYPE_ADJACENT = "adjacent"


def turnover_col(year: int) -> str:
    return f"Total Turnover (CH {year})"


def year_value_cols(year: int) -> dict:
    return {
        turnover_col(year): "total_turnover",
        f"Total Employees (CH {year})": "total_employees_ch",
        f"Total Employees (Est. {year})": "total_employees_est",
        f"Balance Sheet Total Assets ({year})": "balance_sheet_total_assets",
        f"Total Export Revenue {year}": "total_export_revenue",
    }


ID_COLS = [COMPANY_ID_COL, NAME_COL, URL_COL, CH_COL, MISSION_COL, "sample_weight"]


def split_labelled_inference(mission_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    has_turnover = mission_df[[turnover_col(y) for y in YEARS]].notna().any(axis=1)
    labelled = mission_df[has_turnover].copy()
    inference = mission_df[~has_turnover].copy()
    return labelled, inference


def check_turnover(df: pd.DataFrame, col: str = "total_turnover") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Null-and-log guard on the estimation pipeline's own training target —
    same shape as forecast_data_prep.check_turnover (not imported directly:
    src/ and forecast_src/ are kept independently auditable, same precedent
    as forecast_bakeoff.py's own CV logic instead of sharing model_bakeoff.py's).

    Closes ASSUMPTIONS_REGISTER.md #20: until this existed, the estimation
    pipeline had no validation on Source 2's raw OBSERVED turnover value
    before it became a training target — only the forecasting pipeline
    (forecast_data_prep.check_turnover) and predict.validate_predictions
    (which checks the model's PREDICTED output, not the observed input) had
    an equivalent guard. Verified before this was added: zero negative/
    non-finite total_turnover values exist in the current labelled panel —
    a no-op today, a guard against a future data refresh (including the
    adjacent-company merge), not a fix for an active problem.

    Non-numeric/negative/non-finite turnover is converted to missing and
    logged, never silently corrected. Unlike forecast_data_prep's version
    (which keeps the row — its historical panel has uses for a company-year
    beyond just its turnover value), a row here that loses its only reason
    for being labelled (a valid turnover) is dropped entirely by the
    `notna()` filter build_long_panel already applies per year — this
    function only needs to null the bad value first so that filter catches
    it, rather than re-implementing a drop of its own."""
    numeric = pd.to_numeric(df[col], errors="coerce")
    was_non_numeric = df[col].notna() & numeric.isna()
    is_negative = numeric.notna() & (numeric < 0)
    is_non_finite = numeric.notna() & ~np.isfinite(numeric)

    reasons = pd.Series("", index=df.index, dtype="object")
    reasons[was_non_numeric] = "non_numeric"
    reasons[is_negative] = "negative"
    reasons[is_non_finite] = "non_finite"
    invalid_mask = reasons != ""

    cleaned = df.copy()
    cleaned[col] = numeric
    cleaned.loc[invalid_mask, col] = np.nan

    log = df.loc[invalid_mask, ID_COLS + ["year", col]].copy() if invalid_mask.any() else df.iloc[0:0][ID_COLS + ["year", col]].copy()
    log["exclusion_reason"] = "turnover_" + reasons[invalid_mask]
    return cleaned, log


def build_long_panel(labelled_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    year_frames = []
    for year in YEARS:
        cols = year_value_cols(year)
        present_cols = {src: dst for src, dst in cols.items() if src in labelled_df.columns}
        year_df = labelled_df[ID_COLS + list(present_cols)].rename(columns=present_cols)
        year_df = year_df[year_df["total_turnover"].notna()].copy()
        if year_df.empty:
            continue
        year_df.insert(len(ID_COLS), "year", year)
        year_frames.append(year_df)
    if not year_frames:
        empty = pd.DataFrame(columns=ID_COLS + ["year", "total_turnover"])
        return empty, empty.assign(exclusion_reason=pd.Series(dtype="object"))
    panel = pd.concat(year_frames, ignore_index=True)

    panel, turnover_quality_log = check_turnover(panel)
    panel = panel[panel["total_turnover"].notna()].copy()

    panel["total_employees"] = panel["total_employees_ch"].where(
        panel["total_employees_ch"].notna(), panel["total_employees_est"]
    )
    panel["employee_count_source"] = pd.NA
    panel.loc[panel["total_employees_ch"].notna(), "employee_count_source"] = "filed"
    panel.loc[
        panel["total_employees_ch"].isna() & panel["total_employees_est"].notna(),
        "employee_count_source",
    ] = "estimated"

    # Overrides the uniform sample_weight=1.0 inherited from
    # mission_segmentation.py: see POPULATION_TYPE_SPACE comment above for
    # why panel rows are weighted by inverse company row-count.
    panel["sample_weight"] = 1.0 / panel.groupby(COMPANY_ID_COL)[COMPANY_ID_COL].transform("count")

    # Stub for the adjacent-company merge (PROJECT_NOTES.md "Current status / build
    # order" step 3): every row here is a space company today, but tagging
    # it now means adjacent rows can slot in as "adjacent" tomorrow without
    # a rename or a migration of already-written data.
    panel["population_type"] = POPULATION_TYPE_SPACE

    return panel, turnover_quality_log


def construct_samples(
    segmented_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labelled_panels = []
    inference_frames = []
    summary_rows = []
    turnover_quality_logs = []

    for mission in REAL_MISSIONS:
        mission_df = segmented_df[
            (segmented_df[MISSION_COL] == mission) & segmented_df["training_eligible"]
        ]
        labelled, inference = split_labelled_inference(mission_df)
        panel, turnover_quality_log = build_long_panel(labelled)

        labelled_panels.append(panel)
        inference_frames.append(inference)
        turnover_quality_logs.append(turnover_quality_log)
        summary_rows.append(
            {
                "Mission": mission,
                "training_eligible_companies": len(mission_df),
                "labelled_companies": len(labelled),
                "inference_companies": len(inference),
                "labelled_panel_rows": len(panel),
            }
        )

    panel_all = pd.concat(labelled_panels, ignore_index=True) if labelled_panels else pd.DataFrame()
    inference_all = pd.concat(inference_frames, ignore_index=True) if inference_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    turnover_quality_log_all = (
        pd.concat(turnover_quality_logs, ignore_index=True) if turnover_quality_logs else pd.DataFrame()
    )
    return panel_all, inference_all, summary, turnover_quality_log_all


def main() -> None:
    prepped, _ = prepare_source2()
    mapping = load_mapping()
    segmented, _ = segment_missions(prepped, mapping)

    panel_all, inference_all, summary, turnover_quality_log = construct_samples(segmented)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_all.to_csv(OUTPUT_DIR / "labelled_panel.csv", index=False)
    inference_all.to_csv(OUTPUT_DIR / "inference_companies.csv", index=False)
    turnover_quality_log.to_csv(OUTPUT_DIR / "turnover_quality_log.csv", index=False)

    print(summary.to_string(index=False))
    if len(turnover_quality_log):
        print(
            f"\nWARNING: {len(turnover_quality_log)} negative/non-finite/non-numeric observed "
            f"turnover value(s) nulled and excluded — see {OUTPUT_DIR / 'turnover_quality_log.csv'}"
        )
    else:
        print("\nNo negative/non-finite/non-numeric observed turnover values found.")
    print(f"\nWrote {OUTPUT_DIR / 'labelled_panel.csv'} ({len(panel_all)} rows)")
    print(f"Wrote {OUTPUT_DIR / 'inference_companies.csv'} ({len(inference_all)} rows)")


if __name__ == "__main__":
    main()
