"""Runs every stage of both pipelines — turnover estimation, then 2030
forecasting, then the presentation-layer exports — in the one order that
actually works (each stage reads files the previous one wrote), so the
whole project can be rebuilt end to end with one command instead of
running each of the ~20 stage scripts by hand.

Usage:
    python run_full_pipeline.py

Each stage is run as its own subprocess (`python -m <module>`, or the
plain script for the two top-level ones), streaming that stage's normal
stdout/stderr live rather than hiding it — this is exactly what running
each script by hand would print, just automated and in order. Stops
immediately at the first stage that fails (non-zero exit code): later
stages read files earlier ones wrote, so continuing past a failure would
just fail again on a missing/stale file with a more confusing error.

Not included, deliberately: `src/model_bakeoff_pooled.py` (a hedge/
comparison finding, not part of the deployed pipeline — see PROJECT_NOTES.md's
"Model improvement investigation" §5) and `dashboard.py` (an interactive
Streamlit app, not a batch stage — run it separately once the pipeline
below has produced the CSVs it reads; see README.md).
"""
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# (label, argv) — argv is passed to subprocess.run as-is. Module stages use
# `python -m package.module` so relative imports inside src/ and
# forecast_src/ resolve the same way they would if run by hand; the two
# top-level scripts (no package) are run directly by path.
STAGES: list[tuple[str, list[str]]] = [
    # --- Turnover estimation pipeline (src/) ---
    ("Estimation: data_prep", [sys.executable, "-m", "src.data_prep"]),
    ("Estimation: mission_segmentation", [sys.executable, "-m", "src.mission_segmentation"]),
    ("Estimation: sample_construction", [sys.executable, "-m", "src.sample_construction"]),
    ("Estimation: feature_engineering", [sys.executable, "-m", "src.feature_engineering"]),
    ("Estimation: model_bakeoff", [sys.executable, "-m", "src.model_bakeoff"]),
    ("Estimation: model_selection", [sys.executable, "-m", "src.model_selection"]),
    # cross_cutting_prediction (the old buzzword/SIC-code best-guess mission
    # assignment) is superseded — src.predict's own main() now scores
    # Cross-cutting directly with its 2 locked sub-models (Consultancy/Other
    # + whole-population blended fallback, see model_selection.py's
    # CROSS_CUTTING_ROUTING) and writes predictions_cross_cutting.csv
    # itself. Running the old stage after this one would silently
    # overwrite that file with the retired approach.
    ("Estimation: predict", [sys.executable, "-m", "src.predict"]),
    ("Estimation: assemble", [sys.executable, "-m", "src.assemble"]),
    ("Estimation: reporting", [sys.executable, "-m", "src.reporting"]),
    # --- 2030 forecasting pipeline (forecast_src/) ---
    ("Forecasting: forecast_data_prep", [sys.executable, "-m", "forecast_src.forecast_data_prep"]),
    ("Forecasting: forecast_panel_construction", [sys.executable, "-m", "forecast_src.forecast_panel_construction"]),
    ("Forecasting: forecast_sample_construction", [sys.executable, "-m", "forecast_src.forecast_sample_construction"]),
    ("Forecasting: forecast_feature_engineering", [sys.executable, "-m", "forecast_src.forecast_feature_engineering"]),
    ("Forecasting: forecast_bakeoff", [sys.executable, "-m", "forecast_src.forecast_bakeoff"]),
    ("Forecasting: forecast_selection", [sys.executable, "-m", "forecast_src.forecast_selection"]),
    ("Forecasting: forecast_recursive", [sys.executable, "-m", "forecast_src.forecast_recursive"]),
    # forecast_prediction_intervals must run BEFORE forecast_assemble: it adds
    # turnover_lower/turnover_upper to forecast_trajectories.csv, and
    # forecast_assemble carries those columns through into
    # forecast_full_trajectories.csv if (and only if) they're already there.
    ("Forecasting: forecast_prediction_intervals", [sys.executable, "-m", "forecast_src.forecast_prediction_intervals"]),
    ("Forecasting: forecast_assemble", [sys.executable, "-m", "forecast_src.forecast_assemble"]),
    ("Forecasting: forecast_reporting", [sys.executable, "-m", "forecast_src.forecast_reporting"]),
    # --- Reporting / exports ---
    ("Exports: export_excel_workbook", [sys.executable, str(REPO_ROOT / "export_excel_workbook.py")]),
]


def main() -> None:
    total = len(STAGES)
    pipeline_start = time.monotonic()

    for i, (label, argv) in enumerate(STAGES, start=1):
        print(f"\n{'=' * 70}\n[{i}/{total}] {label}\n{'=' * 70}", flush=True)
        stage_start = time.monotonic()
        result = subprocess.run(argv, cwd=REPO_ROOT)
        elapsed = time.monotonic() - stage_start

        if result.returncode != 0:
            print(
                f"\nFAILED at stage [{i}/{total}] '{label}' "
                f"(exit code {result.returncode}, after {elapsed:.1f}s). "
                "Stopping — later stages depend on this one's output.",
                file=sys.stderr,
            )
            sys.exit(result.returncode)

        print(f"\n[{i}/{total}] {label} — done in {elapsed:.1f}s", flush=True)

    total_elapsed = time.monotonic() - pipeline_start
    print(f"\n{'=' * 70}\nAll {total} stages completed in {total_elapsed / 60:.1f} minutes.")
    print("Final outputs: data/output/turnover_forecast_workbook.xlsx")
    print("Interactive view: streamlit run dashboard.py")


if __name__ == "__main__":
    main()
