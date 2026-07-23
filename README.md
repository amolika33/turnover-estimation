# Turnover Estimation Framework

Two linked pipelines for UK space-sector companies (ACE / Beyond Earth /
Resilient Earth / Cross-cutting missions):

1. **Turnover estimation** (`src/`) — supervised ML to fill in missing
   current turnover for companies that don't report it.
2. **2030 forecasting** (`forecast_src/`) — projects each company's
   turnover trajectory forward from the completed baseline above.

See `PROJECT_NOTES.md` for full methodology context, `PROJECT_SUMMARY.docx` for a
plain-language overview with no code, and `FORECASTING_METHODOLOGY.md` /
`DATA_SCHEMA.md` for the underlying specs.

## Structure

- `src/` — turnover estimation pipeline modules (one per stage)
- `forecast_src/` — 2030 forecasting pipeline modules (one per stage), built
  on the estimation pipeline's completed output
- `data/raw/` — input datasets (not committed — see `.gitignore`)
- `data/processed/`, `data/output/` — intermediate and final outputs
- `export_excel_workbook.py` — consolidates the completed baseline + all
  forecasting reports into one `.xlsx` workbook
  (`data/output/turnover_forecast_workbook.xlsx`)
- `dashboard.py` — interactive Streamlit company explorer

## Getting started

```bash
pip install -r requirements.txt
```

### Run everything with one command

```bash
python run_full_pipeline.py
```

Runs every stage of both pipelines in the one order that works (each stage
reads files an earlier one wrote) — turnover estimation, then 2030
forecasting, then the Excel export — printing progress and timing per
stage, and stopping immediately if a stage fails. **Timing note**: the
original estimate here was 30-40 minutes, dominated by the two
cross-validated model bake-offs (`model_bakeoff` and `forecast_bakeoff`).
Since ACE and Beyond Earth were switched to their adjacent-augmented
models, `model_selection` now also refits Extra Trees on tens of
thousands of adjacent-augmented rows for those two missions (each refit
alone can take upwards of 10-15 minutes), and `reporting`'s
out-of-fold re-derivation for Resilient Earth adds further time — a
genuinely fresh, from-scratch run (starting at `data_prep`, including
both bake-offs) has not been re-timed end-to-end since that switch;
budget meaningfully more than the original 30-40 minutes.

Then explore the result interactively:

```bash
streamlit run dashboard.py
```

### Running stages individually

Estimation build order: `data_prep` → `mission_segmentation` →
`sample_construction` → `feature_engineering` → `model_bakeoff` →
`model_selection` → `predict` → `assemble` → `reporting` (each run as
`python -m src.<module>`). `cross_cutting_prediction` is retired
(superseded by `predict.py`'s own Cross-cutting handling — see that
module's docstring) and is not part of the build order; do not run it.
`model_selection` refits ACE/Beyond Earth's locked adjacent-augmented
models and Cross-cutting's 2 sub-models (Consultancy/Other + blended
fallback) in addition to the standard per-mission ranking — pass
`--only-cross-cutting` to refit just Cross-cutting's 2 sub-models without
touching the 3 real missions' deployed models.

Forecasting build order (after the estimation pipeline has produced
`data/processed/final_completed_dataset.csv`): `forecast_data_prep` →
`forecast_panel_construction` → `forecast_sample_construction` →
`forecast_feature_engineering` → `forecast_bakeoff` → `forecast_selection`
→ `forecast_recursive` → `forecast_prediction_intervals` →
`forecast_assemble` → `forecast_reporting` (each run as
`python -m forecast_src.<module>`), then `python export_excel_workbook.py`.

Not part of either build order — separate, standalone analyses, not
required for the deployed pipeline:
`python -m src.model_bakeoff_pooled` (a hedge/comparison model, see
PROJECT_NOTES.md's "Model improvement investigation" §5).
