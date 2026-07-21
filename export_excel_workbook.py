"""Consolidates the key turnover-estimation + forecasting outputs into one
presentable .xlsx workbook — a deliverable for people who won't open CSVs
directly. Reads already-computed files, doesn't recompute anything.

Sheets:
  1. Current Turnover Baseline   <- data/processed/final_completed_dataset.csv (assemble.py)
  2. 2030 Full Trajectories      <- forecast_full_trajectories.csv
  3. £10M Crossings              <- forecast_10m_crossings.csv
  4. Gazelle 10%                 <- forecast_gazelle_10pct.csv
  5. Gazelle 20%                 <- forecast_gazelle_20pct.csv
  6. £50M Intersection           <- forecast_gazelle_50m_intersection.csv
  7. Operational Scaling         <- forecast_operational_scaling.csv

Column headers are renamed to presentable labels (not raw code variable
names) via COLUMN_LABELS below — one shared mapping applied per sheet, so
the same underlying column (e.g. `forecast_evidence_group`) reads
identically everywhere it appears.
"""
from pathlib import Path

import pandas as pd
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_PATH = REPO_ROOT / "data" / "output" / "turnover_forecast_workbook.xlsx"

COLUMN_LABELS = {
    "company_id": "Company ID",
    "company_name": "Company Name",
    "Organisation Name": "Company Name",
    "Beauhurst URL": "Beauhurst URL",
    "CH No. (full)": "Companies House Number",
    "Mission": "Mission",
    "mission": "Mission",
    "year": "Turnover Year",
    "turnover_value": "Turnover (£)",
    "turnover_source": "Turnover Source",
    "reliability": "Reliability",
    "reliability_reason": "Reliability Notes",
    "turnover_age_years": "Years Since Last Filing",
    "turnover_is_stale": "Stale Filing Flag",
    "accounting_year": "Year",
    "turnover": "Turnover (£)",
    "data_type": "Data Type",
    "model_used": "Forecast Model Used",
    "growth_classification": "Growth Classification",
    "baseline_year": "Baseline Year",
    "baseline_turnover": "Baseline Turnover (£)",
    "turnover_2030": "Forecast Turnover 2030 (£)",
    "forecast_evidence_group": "Evidence Group",
    "baseline_turnover_source": "Baseline Source",
    "growth_decay_applied": "Growth Model Ever Applied",
    "n_real_years": "Real Turnover Years",
    "evidence_gate_triggered": "Evidence Gate Triggered",
    "longest_run_10pct": "Longest Growth Streak >=10% (yrs)",
    "longest_run_20pct": "Longest Growth Streak >=20% (yrs)",
    "gazelle_10pct": "Gazelle (>=10%)",
    "gazelle_20pct": "Gazelle (>=20%)",
    "credibility_status": "Credibility Status",
    "n_real_years_employee_growth": "Real Employee-Growth Years",
    "employee_growth_10pct": "Employee Growth >=10% (3+ yrs)",
    "employee_growth_20pct": "Employee Growth >=20% (3+ yrs)",
    "n_real_years_asset_growth": "Real Asset-Growth Years",
    "asset_growth_10pct": "Asset Growth >=10% (3+ yrs)",
    "asset_growth_20pct": "Asset Growth >=20% (3+ yrs)",
    "operational_scaling_10pct": "Operational Scaling Flag (>=10%)",
    "operational_scaling_20pct": "Operational Scaling Flag (>=20%)",
}

SHEETS = [
    ("Current Turnover Baseline", DATA_DIR / "final_completed_dataset.csv"),
    ("2030 Full Trajectories", DATA_DIR / "forecast_full_trajectories.csv"),
    ("£10M Crossings", DATA_DIR / "forecast_10m_crossings.csv"),
    ("Gazelle 10pct", DATA_DIR / "forecast_gazelle_10pct.csv"),
    ("Gazelle 20pct", DATA_DIR / "forecast_gazelle_20pct.csv"),
    ("£50M Intersection", DATA_DIR / "forecast_gazelle_50m_intersection.csv"),
    ("Operational Scaling", DATA_DIR / "forecast_operational_scaling.csv"),
]

HEADER_FONT = Font(bold=True)


def load_and_label(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns=COLUMN_LABELS)


def format_sheet(worksheet) -> None:
    """Freeze the header row and bold it; size columns to fit their
    (now presentable) header text at minimum, capped so one long column
    doesn't blow out the whole sheet's readability."""
    worksheet.freeze_panes = "A2"
    for cell in worksheet[1]:
        cell.font = HEADER_FONT

    for col_cells in worksheet.columns:
        col_letter = get_column_letter(col_cells[0].column)
        max_len = max((len(str(c.value)) for c in col_cells if c.value is not None), default=10)
        worksheet.column_dimensions[col_letter].width = min(max(max_len + 2, 12), 40)


def main() -> None:
    missing = [path for _, path in SHEETS if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input file(s), run the full pipeline first: {missing}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        for sheet_name, path in SHEETS:
            df = load_and_label(path)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            format_sheet(writer.sheets[sheet_name])
            print(f"Wrote sheet '{sheet_name}' ({len(df)} rows, {len(df.columns)} columns)")

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
