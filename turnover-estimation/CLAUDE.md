# Turnover Estimation Framework — Project Context

This file gives Claude Code persistent context on this project. Read it before making changes.

## Objective

Estimate missing annual turnover for UK space-sector companies using supervised
machine learning, to complete a baseline dataset used later for a separate 2030
forecasting framework. This is **current-turnover estimation**, not future forecasting.

Formally: learn `y_hat_i = f(x_i)` from companies with observed turnover, then apply
`f` to companies with missing turnover.

## Key structural rules (do not violate these)

- **Three independent mission models**: Autonomous and Connected Earth (ACE),
  Beyond Earth, Resilient Earth. Each mission gets its own dataset, its own
  regression bake-off, and its own selected model. Do not pool missions into one model.
- **Cross-cutting companies** are excluded from modelling entirely (retained for
  reference only) — they can't be assigned to a single mission without extra assumptions.
- **Adjacent-company datasets** (one per mission) exist only to enlarge the labelled
  training set. They are never part of the inference/prediction population.
  Model *selection* must be validated on held-out **space companies only**, not
  pooled space+adjacent performance.
- **No leakage, ever**: all preprocessing (imputation, encoding, scaling, target
  transforms), feature selection, and hyperparameter tuning must be fit inside the
  training fold only (sklearn `Pipeline` + cross-validation), never on the full
  dataset before splitting.
- **Observed turnover values are never overwritten by predictions.** Predictions
  are only generated for companies with missing turnover.
- Every predicted value carries a `turnover_source` (observed/predicted) and a
  reliability/confidence indicator.

## Pipeline stages (implement as separate modules — see `src/`)

1. `data_prep.py` — schema validation, duplicate resolution (Companies House
   number preferred, name/domain fallback), variable standardisation, missingness
   assessment, eligibility filtering.
2. `mission_segmentation.py` — split space-company dataset into ACE / Beyond Earth /
   Resilient Earth / cross-cutting (excluded); merge each with its adjacent dataset.
3. `sample_construction.py` — within each mission dataset, split into labelled
   (observed turnover) vs inference (missing turnover) populations.
4. `feature_engineering.py` — company characteristics, financial indicators,
   commercial activity, categorical features, composite indicators. Must only use
   information available at prediction time (no target leakage).
5. `model_bakeoff.py` — candidate models: Linear, Ridge, Lasso, Elastic Net,
   Random Forest, Extra Trees, Gradient Boosting, SVR, k-NN. Cross-validated
   hyperparameter search per model, per mission. Metrics: MAE, RMSE, R².
6. `model_selection.py` — pick best model per mission (accuracy + stability +
   prefer simpler model on ties); refit on full labelled data for that mission.
7. `predict.py` — apply selected mission models to each mission's inference
   population; attach reliability/out-of-distribution indicators.
8. `assemble.py` — recombine observed + predicted turnover across all three
   missions (+ cross-cutting passthrough) into the final completed dataset.

## Tech stack

- Python, pandas, scikit-learn (Pipeline, ColumnTransformer, cross_val_score / GridSearchCV or RandomizedSearchCV)
- Fixed random seeds everywhere for reproducibility
- pytest for tests in `tests/`

## Data locations

- Raw input datasets: `data/raw/` (main space dataset + 3 adjacent datasets — not yet added)
- Processed/intermediate outputs: `data/processed/`
- Final completed dataset: `data/output/`

## Current status

Scaffold only — no data or implementation yet. Build and test module by module,
starting with `data_prep.py`, validating each stage against a small toy sample
before moving to the next.
