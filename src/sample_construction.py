"""Within each mission dataset, split training-eligible companies into a
labelled (observed turnover) population and an inference (missing turnover)
population, and reshape the labelled population into long/panel format."""
from pathlib import Path

import pandas as pd

from src.data_prep import CH_COL, NAME_COL, URL_COL, prepare_source2
from src.mission_segmentation import MISSION_COL, REAL_MISSIONS, load_mapping, segment_missions

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "processed"

YEARS = list(range(2013, 2026))


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


ID_COLS = [NAME_COL, URL_COL, CH_COL, MISSION_COL, "sample_weight"]


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
    return pd.concat(year_frames, ignore_index=True)


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
