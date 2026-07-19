# Turnover Estimation Framework

Supervised ML pipeline to estimate missing turnover values for UK space-sector
companies (ACE / Beyond Earth / Resilient Earth missions). See `CLAUDE.md` for
full methodology context.

## Structure

- `src/` — pipeline modules (one per stage)
- `data/raw/` — input datasets (not committed — see `.gitignore`)
- `data/processed/`, `data/output/` — intermediate and final outputs
- `tests/` — pytest tests, one toy-example test per module
- `notebooks/` — exploratory analysis only, not production logic

## Getting started

```bash
pip install -r requirements.txt
```

Build order: `data_prep` → `mission_segmentation` → `sample_construction` →
`feature_engineering` → `model_bakeoff` → `model_selection` → `predict` → `assemble`.
