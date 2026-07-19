"""Within each mission dataset, split training-eligible companies into a
labelled (observed turnover) population and an inference (missing turnover)
population, and reshape the labelled population into long/panel format."""
from pathlib import Path

import pandas as pd

from src.data_prep import CH_COL, COMPANY_ID_COL, NAME_COL, URL_COL, prepare_source2
from src.mission_segmentation import MISSION_COL, REAL_MISSIONS, load_mapping, segment_missions

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "processed"

YEARS = list(range(2013, 2026))

# ASSUMPTION (documented here + CLAUDE.md "Panel row weighting" +
# sample_weight column itself): the methodology's unit of analysis is the
# company, not the company-year. A company with 13 years of turnover history
# and one with 1 year should count equally in training, not 13:1 — so
# sample_weight = 1 / that company's row count in the labelled panel.
# Consciously accepted tradeoff: this down-weights well-documented companies
# relative to naive equal-row weighting.
POPULATION_TYPE_SPACE = "space"


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


def build_long_panel(labelled_df: pd.DataFrame) -> pd.DataFrame:
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
        return pd.DataFrame(columns=ID_COLS + ["year", "total_turnover"])
    panel = pd.concat(year_frames, ignore_index=True)

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

    # Stub for the adjacent-company merge (CLAUDE.md "Current status / build
    # order" step 3): every row here is a space company today, but tagging
    # it now means adjacent rows can slot in as "adjacent" tomorrow without
    # a rename or a migration of already-written data.
    panel["population_type"] = POPULATION_TYPE_SPACE

    return panel


def construct_samples(segmented_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labelled_panels = []
    inference_frames = []
    summary_rows = []

    for mission in REAL_MISSIONS:
        mission_df = segmented_df[
            (segmented_df[MISSION_COL] == mission) & segmented_df["training_eligible"]
        ]
        labelled, inference = split_labelled_inference(mission_df)
        panel = build_long_panel(labelled)

        labelled_panels.append(panel)
        inference_frames.append(inference)
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
    return panel_all, inference_all, summary


def main() -> None:
    prepped, _ = prepare_source2()
    mapping = load_mapping()
    segmented, _ = segment_missions(prepped, mapping)

    panel_all, inference_all, summary = construct_samples(segmented)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_all.to_csv(OUTPUT_DIR / "labelled_panel.csv", index=False)
    inference_all.to_csv(OUTPUT_DIR / "inference_companies.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nWrote {OUTPUT_DIR / 'labelled_panel.csv'} ({len(panel_all)} rows)")
    print(f"Wrote {OUTPUT_DIR / 'inference_companies.csv'} ({len(inference_all)} rows)")


if __name__ == "__main__":
    main()
