# Turnover Estimation Framework — Project Context

This file provides persistent context on this project's methodology,
decisions, and status. Read it before making changes.

## Objective

Estimate missing annual turnover for UK space-sector companies using supervised
machine learning, to complete a baseline dataset used later for a separate 2030
forecasting framework. This is **current-turnover estimation**, not future forecasting.

Formally: learn `y_hat_i = f(x_i)` from companies with observed turnover, then apply
`f` to companies with missing turnover.

## Target and data shape (see DATA_SCHEMA.md for full detail)

- Target = **Total Turnover** (not Space Turnover — that field is unused).
- Data is used in **long/panel format**: one row per (company, year), not one
  row per company. This multiplies effective sample size.
- Because multiple rows belong to the same company, **all cross-validation
  must group by company ID** (`GroupKFold`/`GroupShuffleSplit`), never plain
  random k-fold — otherwise the same company leaks across train/validation.
  The group key is `company_id` (`data_prep.make_company_id`), never the raw
  CH number column — see "Documented assumptions and thresholds" below.
- Mission assignment: space companies are segmented via
  `data/mission_mapping.csv` (Value Stream -> Mission). Adjacent companies
  arrive pre-segmented (already exported as 3 separate mission-specific
  Beauhurst collections) — no run-time mapping needed for them.
- Adjacent-company data will likely follow Source 1's raw Beauhurst schema
  (repeated Financial Statement blocks), not Source 2's curated master-sheet
  schema — design shared feature engineering around Source 1's structure.

## Key structural rules (do not violate these)

- **Three independent mission models**: Autonomous and Connected Earth (ACE),
  Beyond Earth, Resilient Earth. Each mission gets its own dataset, its own
  regression bake-off, and its own selected model. Do not pool missions into one model.
- **Cross-cutting companies** are excluded from modelling (training) entirely
  (retained for reference only) — they can't be assigned to a single mission
  without extra assumptions. At inference time only, a best-guess mission
  assignment now scores their turnover — see "Cross-cutting company
  predictions" below.
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

1. `data_prep.py` — schema validation, duplicate resolution, variable
   standardisation, missingness assessment, eligibility filtering.
   **Duplicate resolution rule**: two rows are the same company only if they
   share a CH number *and* a normalised name. A shared CH number with
   genuinely different names is a data-quality anomaly, not a duplicate
   identity — log it, but never merge those rows' financials or mission
   assignments. Company name is the deciding factor, not CH number alone
   (some CH numbers are legitimately reused across distinct legal entities,
   e.g. a parent charity's number appearing on multiple affiliated orgs).
   Expect more of these once the ~23k adjacent dataset is added.
2. `mission_segmentation.py` — split space-company dataset into ACE / Beyond Earth /
   Resilient Earth / cross-cutting (excluded); merge each with its adjacent dataset.
3. `sample_construction.py` — within each mission dataset, split into labelled
   (observed turnover) vs inference (missing turnover) populations.
4. `feature_engineering.py` — company characteristics, financial indicators,
   commercial activity, categorical features, composite indicators. Must only use
   information available at prediction time (no target leakage).
   **Commercial activity is now built** from Source 3 (grants/accelerator/
   funding enrichment) — see "Data sources" below for the feature list.
   Composite indicators are still not built.
5. `model_bakeoff.py` — candidate models: Linear, Ridge, Lasso, Elastic Net,
   Random Forest, Extra Trees, Gradient Boosting, SVR, k-NN. Cross-validated
   hyperparameter search per model, per mission. Metrics: MAE, RMSE, R².
6. `model_selection.py` — pick best model per mission (accuracy + stability +
   prefer simpler model on ties); refit on full labelled data for that mission.
7. `predict.py` — apply selected mission models to each mission's inference
   population; attach reliability/out-of-distribution indicators.
   **Prediction year is per-company, not "today" or a fixed year** — see
   "Prediction year assignment" in "Documented assumptions and thresholds"
   below.
8. `assemble.py` — recombine observed + predicted turnover across all three
   missions (+ cross-cutting passthrough) into the final completed dataset.

## model_bakeoff.py checklist (methodology doc §1.7-1.8 — unambiguous, do not deviate)

**Candidate models — all of these, not a subset:**
- Linear Regression
- Ridge, Lasso, Elastic Net
- Random Forest, Extra Trees
- Gradient Boosting
- Support Vector Regression (SVR)
- k-Nearest Neighbours (k-NN)

**Requirements:**
- All preprocessing (imputation, encoding, scaling) is fit inside the pipeline,
  per CV fold, on training data only — never before the split.
- Scaling applied only to scale-sensitive models (Linear/Ridge/Lasso/Elastic
  Net/SVR); tree-based models (Random Forest/Extra Trees/Gradient Boosting)
  use raw feature scale — do not scale features for them.
- Hyperparameter tuning via cross-validation for every model — no fixed
  defaults for any candidate.
- Cross-validation must be **grouped by company ID** (not plain k-fold),
  since the panel data has multiple rows per company.
- Validation performance is reported primarily on held-out **space companies
  specifically**, not pooled space+adjacent (relevant once adjacent data is
  merged in later).
- Report all three metrics for every model: **MAE, RMSE, R²** — no single
  metric decides the winner.
- **Tie-breaking rule**: when models show comparable performance, prefer the
  simpler one.
- **One model selected independently per mission** — three separate winners
  are allowed, not one model forced across all three.

**10th candidate added post-launch: CatBoost** (`model_bakeoff.py`), specifically
for its native categorical handling — see "Model improvement investigation"
near the end of this file for the full rationale, the sklearn-`clone()`
compatibility bug it required working around, and per-mission results.

## Tech stack

- Python, pandas, scikit-learn (Pipeline, ColumnTransformer, cross_val_score / GridSearchCV or RandomizedSearchCV)
- Fixed random seeds everywhere for reproducibility
- pytest for tests in `tests/`

## Data locations

- Raw input datasets: `data/raw/` (main space dataset + 3 adjacent datasets — not yet added)
- Processed/intermediate outputs: `data/processed/`
- Final completed dataset: `data/output/`

## Data sources

See `DATA_SCHEMA.md` for the structure of the source Excel files (Beauhurst
raw export + mission-tagged master sheet + grants/accelerator/funding
enrichment) and open questions that need resolving before feature
engineering can be finalised.

- **Source 1**: `space_companies_beauhurst_financials.xlsx` — raw
  Beauhurst export, repeated Financial Statement blocks.
- **Source 2**: `Published Space Capabilities Catalogue_Cleaned.xlsx` —
  curated master/mission-tagged sheet, year-panel format. Primary source
  `feature_engineering.py` is built around.
- **Source 3**: `space_companies_beauhurst_grants_accelerators.xlsx` —
  grants/accelerator/funding enrichment for Source 1's company universe
  (same 1,372 rows), joined into the pipeline by Beauhurst URL. Added
  `feature_engineering.py` features: 8 boolean commercial/growth signals
  (`signal_equity_fundraising`, `signal_debt_fundraising`, `signal_mbo_mbi`,
  `signal_acquired`, `signal_made_acquisition`, `signal_ipo`,
  `signal_rd_grant`, `signal_patent` — `Growth signals - Accelerator` and
  `Innovation signals - Academic spinout` were dropped as ~100% redundant
  with the derived features below), `has_attended_accelerator` /
  `accelerator_count` / `is_academic_spinout` (derived from event-slot
  columns, not the raw names/dates themselves), and `grants_count` /
  `grants_total_amount` / `grant_recency_years` + the fundraising equivalents
  (recency computed
  relative to each panel row's year, nulled rather than negative when the
  event postdates that row — see `feature_engineering.py`'s `add_features`
  comment). 3 boolean signals (the two "scaleup" flags and "High growth
  list") were excluded as an unresolved turnover-derivation risk — see
  DATA_SCHEMA.md "Source 3" section.
- **Source 1 financial ratios** (post-launch addition, same Source 1 file):
  Financial Statement 1's ~20 financial ratios per company, checked for
  turnover-derivation leakage before including any (formula-reconstructed
  each ratio from its own components against real data — 6 reconstruct
  exactly from a Turnover-based formula and are excluded; 9 never do and
  are kept: `fs1_current_ratio`, `fs1_liquidity_acid_test`,
  `fs1_gearing_pct`, `fs1_equity_pct`, `fs1_current_debt_ratio`,
  `fs1_total_debt_ratio`, `fs1_roce_pct`, `fs1_rota_pct`, `fs1_ronae_pct`).
  Merged on (company_id, year), not company_id alone — Statement 1 is one
  snapshot tied to its own filing date, so treating it as a company-constant
  (like `founded_year`) would leak a recent balance sheet into historical
  panel rows. Full detail in "Model improvement investigation" below and
  DATA_SCHEMA.md's "Source 1 financial ratios" section.

## Current status / build order (adjusted)

No adjacent-company data available yet (~23k planned, not ready). Build order
is deliberately re-sequenced to prove out the ML core first, using only the
~400 space companies with known turnover:

1. `data_prep.py` + `feature_engineering.py` — scoped to the space-company
   dataset only (skip adjacent-company harmonisation for now)
2. `model_bakeoff.py` — cross-validated comparison on the ~400 labelled
   companies. **Build `sample_weight` support into the model-fitting code
   from the start**, even though the adjacent-data use case (lower weight
   for adjacent rows) is unused until that data arrives — this avoids
   re-architecting later. `sample_weight` is *not* uniformly 1.0: see
   "Panel row weighting" in "Documented assumptions and thresholds" below —
   it's currently inverse-frequency by company (1 / that company's row
   count in the labelled panel), a decision made once the panel was built.
   With only ~400 rows, use careful CV (e.g. repeated k-fold) and watch fold
   variance.
3. Once adjacent-company data (~23k rows) is available: extend
   `mission_segmentation.py` to merge it in, and set adjacent rows to a
   lower `sample_weight` than space companies so the model trusts them
   less (`ADJACENT_SAMPLE_WEIGHT` placeholder already in `model_bakeoff.py`,
   unused — see "Documented assumptions and thresholds"). See
   `ADJACENT_DATA_REQUIREMENTS.md` for the format the incoming files need
   to be in.
   **Mission assignment is simpler than earlier assumed**: adjacent
   companies arrive as three separate mission-specific exports (one per
   mission), pre-tagged or filename-tagged — *not* requiring the
   buzzword/SIC-based inference originally planned for them (that
   inference logic is scoped to cross-cutting companies only, at
   prediction time — see "Cross-cutting company predictions" below).
   **Not optional, must change when this happens**: `model_bakeoff.py`'s
   outer `GroupKFold` (`make_repeated_group_kfold_splits`) currently
   groups every row in the mission dataset by `company_id` with no
   awareness of `population_type`. Per the methodology's explicit
   requirement (already stated above under "Adjacent-company datasets"),
   once adjacent rows are merged into training, the **outer** test folds
   used for final performance reporting must be restricted to space
   companies only — adjacent rows can appear in outer *training* folds
   (that's their entire purpose) but never in the outer *test* fold. The
   inner CV (hyperparameter tuning) can stay pooled. This needs an actual
   code change to the outer-split construction, not just a data filter
   applied afterward.

Validate each stage against a small toy sample before moving to the next.

## Cross-cutting company predictions (`src/cross_cutting_prediction.py`)

The written methodology only said cross-cutting companies (Consultancy /
Other, Explore New Markets Value Streams) are "retained for reference." This
extension (originally deferred, now built) generates real turnover values
for the 197 with zero observed turnover history — the other 108 cross-cutting
companies have real observed turnover and are untouched, handled entirely by
`assemble.py`'s existing observed-turnover branch.

**Why this exists at all**: `data/mission_mapping.csv` only maps Value
Streams that belong to one of the three real missions. `Consultancy /
Other` and `Explore New Markets` are their own separate Value Streams —
they don't map to ACE, Beyond Earth, or Resilient Earth by design, not by
omission. So a cross-cutting company has no unambiguous mission to apply
any mission-specific model to in the first place; the SIC/keyword
best-guess process below exists solely to give these companies a
plausible mission to borrow a model from, since none is given.

- **At prediction time only** (never for training): each cross-cutting
  company with no observed turnover is assigned a best-guess mission (ACE /
  Beyond Earth / Resilient Earth) via similarity to the three real missions'
  own companies, on two signals — SIC Code 1 (categorical exact-match) and
  `LinkedIn Specialties (Keywords)` (free-text buzzwords, comma-separated;
  this is the concrete use of the buzzword-similarity approach that
  `feature_engineering.py`'s `DROPPED_COLUMNS` flagged this column for
  instead of using it as an ML feature). Each signal scores a mission as
  (companies in that mission matching this value) / (that mission's company
  count) — normalising by mission size so Beyond Earth (459 companies) isn't
  favoured just for being the largest — then both signals are normalised to
  sum to 1 across missions and added (equal weight, no tuned blend). A
  company with zero signal on both axes (SIC code and every keyword unmatched
  anywhere in the 3 real missions) falls back to the largest real mission by
  company count, tagged `assignment_method="fallback_plurality_mission_no_
  signal"` so it's visibly the weakest-evidence case rather than indistinguishable
  from a real match. Current run: 197 scored, 150 `sic_and_keyword_similarity`,
  19 keyword-only, 18 SIC-only, 10 fallback. Assigned-mission distribution:
  Resilient Earth 89, Beyond Earth 80, ACE 28.
- The company is then scored with whichever mission model that best-guess
  assignment points to — same `build_covariate_snapshot`/
  `add_prediction_features`/`compute_reliability` feature path as a real
  inference-population company (imported directly from `predict.py`, not
  reimplemented).
- `reliability` is unconditionally set to **`"approximate"`** — distinct
  from every other value in the pipeline (observed/standard/low) — since the
  mission itself is inferred here, not given, regardless of what the
  underlying OOD check found. `reliability_reason` keeps both the assignment
  method and the OOD reason (if any), not just one or the other.
  **`"approximate"` should be read as ranking BELOW `"standard"` and even
  below `"low"`**, not just as a same-tier alternative label: a real
  mission-matched company's `"low"` reflects one source of uncertainty (an
  out-of-distribution covariate check), while a cross-cutting company's
  `"approximate"` stacks that same model-quality uncertainty on top of a
  second, independent one — whether the best-guess mission assignment
  itself is even correct.
- Cross-cutting companies still never enter training data for any mission
  model, no exceptions — this only affects what happens at inference time.
- Writes `predictions_cross_cutting.csv` (same schema as `predictions_all.csv`)
  — `assemble.py`'s `main()` concatenates it onto `predictions_all.csv` before
  calling `assemble()`, so the function itself needed no logic change; a
  cross-cutting company only still lands in `turnover_source=
  "cross_cutting_unmodelled"` if even best-guess scoring produced no valid
  prediction for it (e.g. zero covariate data anywhere to build a snapshot from).

## Data quality exclusions from training (see DATA_SCHEMA.md for detail)

- The `Value Stream == "Sky UK"` row is a data-entry error (own company name
  pasted into the field) — excluded from mission mapping entirely, not folded
  into Cross-cutting.
- **Volante Global** (`ch_10993763_volanteglobal`) is excluded from mission
  mapping entirely — `mission_segmentation.py`'s `KNOWN_MISCATEGORIZED_
  COMPANIES` dict. Not a genuine space-sector company: SIC Code 1 = 70100,
  LinkedIn Industry/Specialties = "Insurance"/"Insurance, Underwriting"
  (Source 2), independently corroborated by Source 1's own description
  ("Volante provides reinsurance and insurance products to clients across
  various industries") and matching SIC code — same legal entity confirmed
  across both sources (identical CH number and Beauhurst URL), but an
  insurance/reinsurance holding group with no evident space-sector
  business, regardless of which of its two reported turnover figures
  (Source 1: £19,474,210 FY2024; Source 2: -£3,757,048) would apply. Same
  treatment as Sky UK: excluded entirely, not folded into Cross-cutting.
- 6 Companies House numbers in Source 2 were originally shared by multiple
  company rows (11 rows). 2 were confirmed data-entry errors and corrected
  in `data_prep.py` (GeoData Institute's bogus CH number nulled; ISVR
  Consulting's corrected to `14701170`). The remaining 4 groups (9 rows) are
  genuinely different companies that happen to share a reused CH number
  (e.g. a parent charity's number) — logged as `shared_ch_number_anomaly`
  but **not** excluded from training, since company name (not CH number
  alone) decides identity. Only a true duplicate (same CH number *and* same
  normalised name) is excluded pending manual review; there are currently
  none. See DATA_SCHEMA.md for the full list.

## Documented assumptions and thresholds (data_prep.py -> assemble.py)

Every entry here is also a code comment at the point of decision, and
(where it affects output data) a column/flag in the relevant CSV — this
list is the third leg, not the only place it's written down.

- **`company_id`** (`data_prep.make_company_id`): stable, guaranteed-non-null
  per-company identifier — `ch_<CH number>_<normalised name>` where a CH
  number exists, else `fallback_<sha1(name+URL)[:12]>`. Used as the
  `GroupKFold` group key everywhere and the merge/dedup key in
  `feature_engineering.py`, `predict.py`, `assemble.py`. Why: `GroupKFold`
  errors/misbehaves on a null group, and the raw CH number column can be
  null (GeoData Institute has none of its own — this was the live trigger,
  since it's in Resilient Earth's labelled panel) or shared by genuinely
  different companies (the `shared_ch_number_anomaly` cases) — the
  normalised name is folded into the CH-prefixed branch specifically to
  stop those anomaly companies colliding onto one `company_id`. Column
  `company_id` is on every processed CSV from `data_prep.py` onward.
  Side effect caught while wiring this up: the old name+URL+CH composite
  merge key silently failed for any company with a null Beauhurst URL
  (`pandas.merge` doesn't treat `NaN == NaN` as a match) — UK Hydrographic
  Office's 13 labelled panel rows had `founded_year`/`sic_code_1`/
  `linkedin_industry`/`value_stream` silently blank because of this before
  the `company_id` switch.
- **Panel row weighting** (`sample_construction.build_long_panel`):
  `sample_weight = 1 / company's row count` in the labelled panel. The
  methodology's unit of analysis is the company, not the company-year, so a
  company with 13 years of history and one with 1 year should count equally
  in training rather than 13:1. Consciously accepted tradeoff: this
  down-weights well-documented companies relative to naive equal-row
  weighting. Column `sample_weight` in `labelled_panel.csv` /
  `labelled_features.csv`.
- **`population_type`** (`sample_construction.build_long_panel`): every
  panel row is stamped `"space"` now. Cheap stub so adjacent-company rows
  can slot in as `"adjacent"` later (per the build-order step above)
  without a rename or migrating already-written data.
- **`log1p` negative-value guard** (`model_bakeoff.check_negative_log_inputs`):
  every `LOG_NUMERIC_FEATURES` column (employees, assets, export revenue,
  and their derived ratios) should be non-negative by definition — a
  negative value means bad upstream data, not something to silently log1p
  anyway or clip. Checked before the pipeline runs; any hits are written to
  `log1p_negative_values_{mission}.csv` rather than passed through. None
  found in the current Source 2 data — this is future-proofing for
  adjacent/refreshed data.
- **Prediction year assignment** (`predict.build_covariate_snapshot`) — a
  direct answer, since it determines what "the predicted turnover" actually
  represents: it is **neither** today's date **nor** a single fixed year
  shared across all companies. For each inference company, it's the most
  recent year (within `sample_construction.YEARS`, 2013-2025) that has
  *any* covariate populated (Total Employees CH/Est, Balance Sheet Total
  Assets, Total Export Revenue, or Size) — i.e. that company's own latest
  real financial snapshot, whatever year that happens to be. A company
  whose last real data point was 2022 gets `year=2022` and its prediction
  represents an estimate of 2022 turnover, not current/2026 turnover, even
  though it sits alongside a company predicted for `year=2025`. Companies
  with **zero** covariate data in any year (no real snapshot to anchor to
  at all) fall back to `year=max(YEARS)=2025` — never 2026, since 2026's
  columns are ~100% empty in Source 2 (confirmed against the current export) — with
  every numeric feature left null for the pipeline's imputer to fill in.
  Both facts are in the output data, not just here: every predicted row
  carries `prediction_year` (the year used) and `is_fallback_year` (True =
  pure imputation, no real snapshot). Current distribution (Beyond Earth +
  Resilient Earth combined, 437 predictions): year 2022 (3), 2023 (51),
  2024 (257), 2025 (126); 27 of 437 (6.2%) are `is_fallback_year=True`.
  `company_age_years` and the grant/fundraising recency features are
  computed relative to this same per-company `year`, so they inherit the
  same "as of that company's own latest snapshot" framing, not "as of
  today."
- **Model usability threshold** (`model_selection.USABILITY_R2_THRESHOLD =
  0.0`): a selected model is only usable for prediction if it beats
  predicting the mission's own mean turnover (`R2_mean > 0`). This is what
  currently excludes ACE (`R2_mean=-1.03` under repeated CV) — expressed as
  a threshold against `selected_models.csv`'s `usable`/`exclusion_reason`
  columns, not a hardcoded mission name, so a future data refresh that
  lifts ACE's R² above 0 flips its usability automatically. `predict.py`
  and `assemble.py` read this file; neither contains mission-specific logic.
- **Prediction validation** (`predict.validate_predictions`): a predicted
  turnover that's non-finite or negative is a modelling failure for that
  row, not a value to export. Flagged via `prediction_valid` /
  `prediction_invalid_reason` and nulled rather than written out looking
  legitimate.
- **Observed-turnover validation** (`sample_construction.check_turnover`) —
  **closed, previously a gap**: mirrors the forecasting pipeline's
  `forecast_data_prep.check_turnover` exactly (same null-and-log shape,
  never silently corrected) — non-numeric/negative/non-finite values in
  Source 2's raw `total_turnover` are converted to missing and logged to
  `turnover_quality_log.csv` before the row can become a training target,
  closing the gap `predict.validate_predictions` above didn't cover (that
  one validates the model's PREDICTED output, not the observed input).
  Runs inside `build_long_panel`, so every caller (`sample_construction.
  main()`, `feature_engineering.build_features()` — logged combined with
  the negative-company-age check into `feature_engineering_quality_
  log.csv`) gets the check automatically. Verified a no-op against the
  current labelled panel (0 rows flagged, identical row counts
  before/after) — a guard against a future data refresh (including the
  adjacent-company merge), not a fix for an active problem today.
- **Stale observed turnover** (`assemble.STALE_THRESHOLD_YEARS = 3`) —
  **flagged, not acted on yet**: `turnover_age_years` = (most recent year
  any company in the dataset filed) − (this company's observed year);
  `turnover_is_stale` = that gap > 3 years, computed only for
  `turnover_source="observed"` rows. UK companies must file annually, so
  >3 consecutive years with nothing filed suggests a company has genuinely
  stopped reporting turnover, not just an administrative lag — but this is
  a proposed threshold, not validated against how many companies it
  actually flags. Stale values are **not** reclassified — they stay
  `turnover_source="observed"` — pending a decision once we can see the
  real impact.
- **One-row-per-company enforcement** (`assemble.enforce_one_company_per_row`):
  applied at the source (right after selecting from `segmented_df`) and
  again after the full assembly cascade. Any `company_id` appearing more
  than once is pulled out and written to
  `assemble_duplicate_company_id.csv`, not silently dropped (loses
  information) or silently kept as multiple rows (breaks the
  one-row-per-company guarantee `assemble.py` exists to provide).
- **`model_selection.py` reads saved bake-off results** rather than calling
  `model_bakeoff.run_bakeoff()` again: avoids wasted compute (the repeated
  5x5 grouped CV bake-off takes minutes per mission) and run-to-run drift
  (guarantees the selection decision matches the one bake-off run everyone
  is looking at, rather than trusting two separate invocations agree).
  Requires `python -m src.model_bakeoff` to have been run first; errors
  clearly if the CSVs are missing.
- **`best_params` serialised as JSON** (`json.dumps(..., default=str)`) in
  `model_bakeoff.py`'s fold detail and `model_selection.py`'s
  `selected_models.csv`, instead of relying on pandas' implicit
  `str()`-on-write — keeps the column machine-readable
  (`json.loads`-able) rather than a Python-repr string that's awkward to
  parse back.
- **`OneHotEncoder(min_frequency=5)`** (`model_bakeoff.build_preprocessor`):
  any categorical value seen fewer than 5 times in a training fold is
  bucketed into a shared "infrequent" column rather than getting its own
  one-hot column — the actual fix for `sic_code_1`'s high cardinality (see
  this file's `model_bakeoff.py` docstring reference above and the module's
  own docstring for the 1e83+ linear-model blow-up this was built to stop).
  `5` is a standard, conservative default, not empirically tuned against
  alternatives for this dataset specifically — flagged as a threshold worth
  revisiting once adjacent-company data changes the categorical cardinality
  profile.
- **`linkedin_industry` dropped from the feature set entirely**
  (`feature_engineering.py`'s `DROPPED_COLUMNS`, removed from `STATIC_COLS`
  and `FEATURE_COLUMNS`; also removed from `model_bakeoff.py`'s
  `CATEGORICAL_FEATURES` and `predict.py`'s `compute_reliability`
  out-of-distribution category check): it's a raw, externally-scraped
  LinkedIn classification, not part of the project's own deliberate
  mission/industry taxonomy — `value_stream` and `sic_code_1` already cover
  company categorisation more reliably (curated for this project
  specifically) and overlap heavily with what `linkedin_industry` captures.
  It was also one of the two high-cardinality columns (158 categories) that
  caused the original linear-model numerical instability (Linear
  Regression/Ridge/Elastic Net coefficients blowing up to 1e83+, see
  `model_bakeoff.py`'s module docstring) before `min_frequency` bucketing
  was added — removing it outright is more robust than continuing to rely
  on that bucketing to contain it. Full bake-off, model selection,
  predictions, and reporting artifacts re-run against the reduced feature
  set; see `model_performance_summary.csv` for the resulting per-mission
  metrics.

## 2030 Forecasting Pipeline (`forecast_src/`) — build order

Separate pipeline stage, built on the completed dataset above — see
`FORECASTING_METHODOLOGY.md` for the full verbatim spec. Kept in its own
`forecast_src/` package, never mixed into `src/`.

**What "2030" actually means, stated explicitly (known simplification, not
a silent gap)**: `accounting_year` (and `FORECAST_END_YEAR = 2030`) is a
bare integer throughout this pipeline — `forecast_recursive.py` advances it
`+1` per recursive step with no month/day attached anywhere, and the
original methodology spec (`FORECASTING_METHODOLOGY.md`) never defines one
either. Source 2's own column is literally `Total Turnover (CH year)` —
whatever calendar-year label Companies House/Beauhurst assigned to that
filing. So "a company reaches 2030" currently means **that company's own
fiscal-year-labelled-2030 turnover figure** — not a value pinned to any
specific calendar date, and specifically NOT verified to align with the UK
government fiscal year start (April 2030) or any other fixed date. Since
company year-ends vary (a March year-end company's "2030" figure covers
Apr 2029-Mar 2030 and ends before April 2030; a December year-end
company's covers all of calendar 2030 and straddles it), two companies
both labelled "2030" are not necessarily describing the same real-world
period. Pinning precisely to April 2030 would require carrying each
company's actual fiscal year-end month (present in Source 1's `Date of
accounts`, not currently propagated into the year-indexed panel or the
forecast pipeline at all) and a real modelling decision about interpolating
within a fiscal year — a separate, larger design question, not implemented
here.

1. ✅ `forecast_data_prep.py` — validates the completed baseline + historical
   company-year panel (methodology sec 2-3); also owns two exclusion
   policies layered on top of the written spec, both documented in their
   own module: `mission_segmentation.py`'s `KNOWN_MISCATEGORIZED_COMPANIES`
   (sector-mismatch exclusions, e.g. Volante Global — an insurance/
   reinsurance group, not a space company, despite sitting in the space-
   capabilities catalogue) and `forecast_data_prep.py`'s company-wide
   invalid-turnover-history exclusion (any negative/non-finite/non-numeric
   turnover anywhere in a company's history removes that company from the
   forecasting pipeline entirely, not just the offending row — e.g. Price
   Forbes, excluded despite 10 otherwise-clean years).
2. ✅ `forecast_panel_construction.py` — ordered per-company time series +
   history-quality properties (sec 4).
3. ✅ `forecast_sample_construction.py` — one-year-ahead training rows via
   shift-based target construction (sec 5) + evidence groups A-D (sec 6).
4. ✅ `forecast_feature_engineering.py` — lag/log/growth/rolling/history-
   quality/company/derived-financial/data-quality features (sec 7.1-7.8).
5. ✅ `forecast_bakeoff.py` — 12-candidate model comparison (13th, CatBoost,
   skipped — not installed), rolling-origin temporal validation (not
   grouped k-fold — panel data here is chronological, not just
   company-grouped), per-horizon reporting.
   **Numeric-stability carryover from the estimation pipeline's BT/BAE
   Systems lesson, checked empirically before this module is built**: raw
   turnover-scale features (`target_turnover_next_year`,
   `turnover_lag_1/2/3`, `rolling_turnover_mean_2/3`,
   `rolling_turnover_median_3`, `historical_turnover_max/min`,
   `total_assets`, `assets_per_employee`) all show skew 9-17 — the same
   shape as the original BT/BAE-driven instability — and need the same
   fix: `TransformedTargetRegressor(log1p/expm1)` on the target, log1p on
   this module's `LOG_NUMERIC_FEATURES`-equivalent list. `log_growth_1y`
   itself needs no further transform (skew ≈ -0.02, already effectively
   symmetric — it's a log-difference, so a 394x raw multiplicative jump
   becomes a bounded 5.98 additive value; a second log1p pass would also be
   mechanically invalid, since log1p requires x > -1 and log_growth_1y goes
   down to -6.06). `log_growth_3y_mean`/`growth_volatility` are more mildly
   skewed (2.10/4.43) — left untransformed for now, flagged as a secondary
   watch item if bake-off fold variance looks unstable for them.
   **`employee_growth`/`asset_growth` — identified and resolved**: sec
   7.7's original simple, non-log growth-ratio formula measured far more
   skewed than raw turnover (28.25/55.79, `asset_growth` max = 2,553 —
   255,300% in one year) — a real BT/BAE-shaped risk of their own, not
   covered by the log_growth_1y fix since it was a different construction.
   `forecast_feature_engineering.py`'s `add_derived_financial_features` was
   revised to a log-difference (`log1p(X_t) - log1p(X_lag1)`), matching
   log_growth_1y's proven-symmetric construction exactly. Re-measuring
   after the change surfaced a SECOND, independent problem the reformulation
   itself exposed: log-differences are unbounded below (unlike the old
   simple ratio, floored at -1.0 no matter how bad the prior value), so 3
   rows with an implausible upstream `employees` estimate (GMV 2013:
   39,187,246; Added Value Solutions 2016/2017: 630,431 twice — all
   `employee_count_source="estimated"`, never "filed"; the largest
   genuinely filed value anywhere in the dataset is BAE Systems' ~97,000)
   were blowing up `employee_growth` to -13.6 in log-difference space.
   Fixed with a new `forecast_data_prep.py` check,
   `check_plausible_employees` (`PLAUSIBLE_EMPLOYEES_MAX = 500,000`,
   comfortably above BAE Systems' real max) — nulls the 3 implausible
   values (never drops the row), same null-and-log shape as `check_turnover`.
   Final skew after both fixes: `employee_growth` 28.25 → **-2.80**,
   `asset_growth` 55.79 → **3.07** — both now in the same moderate range as
   `log_growth_3y_mean`/`growth_volatility` (2-4), no further transform
   planned. (One remaining outlier double-checked and confirmed genuine,
   not a data error: Cobham's filed employee count really did drop from
   10,185 to 51 in 2020, consistent with its real Advent International
   breakup/restructuring that year.)
   **Headline finding, not a bug**: the 3 simple benchmarks (Persistence,
   Mission-Average Growth, Company Historical CAGR) consistently matched or
   beat every ML candidate across all 3 missions and most horizons —
   one-year-ahead turnover is highly autocorrelated for established
   companies, so "no growth" and "simple trend" are hard baselines to beat.
   SVR and Histogram Gradient Boosting consistently underperformed
   everywhere. 0 negative-log-input warnings on the full run.
6. ✅ `forecast_selection.py` — selects the strongest model per (mission,
   horizon) from the bake-off results, mirroring `model_selection.py`'s
   robustness-filter → composite-rank → simplicity-tie-break convention,
   with `SIMPLICITY_RANK` extended so the 3 benchmarks rank simplest of
   all (below Linear Regression — no fitting, no hyperparameters, no
   preprocessing). Every selection is tagged `is_benchmark`, printed and
   written distinctly (`forecast_selected_models.csv`,
   `forecast_selection_ranking.csv`) — never silently presented as if an
   ML model had won. **Result: all 3 missions selected a benchmark
   (Persistence) as the horizon="1" model — the one actually refit and
   deployed by `forecast_recursive.py`** (horizons 2/3/4+ are diagnostic
   only, since a one-year-ahead model is applied recursively, not
   separately re-selected per horizon) — no ML model beat Persistence at
   horizon 1 in any mission. No `.joblib` files exist for any mission as a
   result; `forecast_recursive.py` must apply the Persistence formula
   directly (`BENCHMARK_PREDICT_FNS` in `forecast_bakeoff.py`), not load a
   fitted model.
7. ✅ `forecast_recursive.py` — built and run to 2030 for all 1,222 baseline
   companies (0 NaN/negative/non-finite values in the output).
   **Not a per-mission fixed model after all**: forecast_selection.py's
   Persistence-everywhere result was checked with
   `forecast_evidence_group_diagnostic.py` before this module was built,
   splitting horizon=1 performance by evidence group (uninformative — Group
   B has only 1-2 rows per model per mission) and by actual growth
   trajectory (informative, and NOT uniform): ACE's Persistence win was a
   flat-majority artifact (Ridge clearly wins among growing companies,
   R2=0.982 vs Persistence not in the top 5); Beyond Earth/Resilient
   Earth's Persistence win holds even restricted to growing companies. But
   "best average one-step predictor" and "fit for a growth-identification
   objective" are different questions — Persistence is structurally
   incapable of ever forecasting growth, so applying it recursively for up
   to 17 steps would flatten every company's trajectory regardless of
   mission, undermining the £10M/£50M/gazelle objective. Routing is
   **growth-trajectory-conditional per company**, re-evaluated at every
   recursive step (not fixed at baseline — a company's trend can genuinely
   change over a multi-year projection, and fixing it would let one early
   growth burst compound blindly for up to 17 years):
   - `GROWTH_THRESHOLD = log1p(0.10)` (~0.0953): reuses the same 10% YoY
     cut the diagnostic and the planned gazelle criteria both use, rather
     than inventing a second number.
   - Primary signal is `log_growth_3y_mean`, not `log_growth_1y` — smoothed
     against a single noisy predicted year (later recursive steps' history
     includes the model's own earlier predictions), and thematically
     consistent with the gazelle definition's own 3-consecutive-year
     framing. Falls back to `log_growth_1y`, then "stable" if neither
     exists (e.g. Group D companies with zero real history).
   - "growing" → Ridge (ACE only, per the diagnostic) or Company Historical
     CAGR (Beyond Earth / Resilient Earth / Cross-Cutting — CAGR beats
     Ridge in both remaining missions' growing-company subset, and is
     philosophically the right tool for "continue this company's own
     trend" vs Ridge's cross-company signal). Cross-Cutting never uses
     Ridge — no Cross-Cutting-specific fitted model exists.
   - "stable" (including declining) → Persistence, uniformly; the
     diagnostic only examined growing vs not-growing, so no separate
     declining-company treatment is invented.
   Verified against a company sample before committing to the full run:
   Air Liquide (Beyond Earth) has log_growth_1y=-0.37 but
   log_growth_3y_mean=+0.10 → correctly "growing" (reads through one bad
   year to an established uptrend); AXA XL (Cross-Cutting) has
   log_growth_1y=+0.65 but log_growth_3y_mean=-0.76 → correctly "stable"
   (doesn't chase a one-year spike). Baseline-year classification counts:
   ACE 33 growing/162 stable, Beyond Earth 88/371, Resilient Earth 36/229,
   Cross-Cutting 42/261.

   **Runaway compounding found on the first full run, partially fixed —
   MIN_EVIDENCE_GROUP_FOR_TREND_CONTINUATION = "A"**: the first full run
   produced 16 companies >100x their baseline turnover by 2030 (9 of them
   >1000x, max ~22 million x). Every one traced back to a company with
   only 1-2 real year-over-year transitions and one large early jump (e.g.
   TerraFarmer: £75,592→£721,789, its only 2 real years, evidence Group B).
   Root cause: Company Historical CAGR/Ridge under pure recursion has no
   deceleration mechanism — the realized one-year growth between two
   model-PREDICTED years exactly equals whatever rate was applied to
   produce them, so once anchored to an extreme rate, a company's own
   synthetic history keeps "confirming" that same rate forever (dynamic
   re-classification doesn't catch this, since the smoothed signal
   computed from that synthetic history just keeps reporting the same
   extreme value). Fix: a company classified "growing" by growth_signal
   but in evidence Group B/C/D (< 3 real historical years,
   forecast_sample_construction.py's static, pre-recursion
   forecast_evidence_group) falls back to Persistence regardless of its
   measured rate — real data, but not enough evidence for a sustainable
   rate worth compounding for up to 17 years.
   **This is a partial fix, not a complete one — stated explicitly, not
   silently assumed**: re-running after the gate, all originally-traced
   Group-B/C companies (TerraFarmer, Agrimetrics, Space Skills Alliance)
   dropped out as intended (Resilient Earth's growing-company max fell
   5.22 trillion → 1.43 billion). But 6 companies remain >100x (4 still
   >1000x, max ~230,151x) — SaxaVord Spaceport, Infleqtion, Map of
   Agriculture, Eutelsat OneWeb, Sierra Nevada Corporation, Oxa — and every
   one of these is evidence Group A (3-9 real years). The gate targets
   thin EVIDENCE specifically; it doesn't address extreme RATE MAGNITUDE
   from genuinely evidence-backed small-base, high-volatility companies
   (the same population that produced Earth-i's 394x growth outlier and
   the GMV/Added Value Solutions employee data issue earlier in this
   project). A growth-rate cap or a decay/mean-reversion mechanism would
   be the next candidate fix, not yet built pending a decision on it.
   Air Liquide/AXA XL re-verified unaffected by the gate (both Group A,
   never needed it — their own growth signal self-corrects without
   intervention). Full run: 1,222/1,222 companies reached 2030, 75/1,222
   (6.1%) flipped growing↔stable classification at least once, 52
   company-year steps were downgraded by the evidence gate. Model usage
   across all steps: Persistence 6,785, Company Historical CAGR 900,
   Ridge 100.

   **Growth-rate decay — the fix for the 6 remaining Group-A outliers**:
   every "growing"-routed prediction (CAGR formula or Ridge regression,
   both through the same `apply_growth_decay` path) is converted to an
   implied one-year log-growth rate and blended toward that mission's
   real-data MEDIAN log_growth_1y (`compute_mission_average_growth` — median
   over mean, deliberately, for the same small-base-distortion reason
   documented throughout this section), with blend weight
   `0.5 ** (step / GROWTH_DECAY_HALF_LIFE_STEPS)`, `HALF_LIFE_STEPS = 2.0`
   (~71% company-rate at step 1, 50% at step 2, ~18% at step 5, ~9% at
   step 7 — "mostly company early, mostly mission average by step 5+", per
   the brief; exponential rather than linear decay-to-zero so a company's
   own evidence never gets fully discounted no matter how far out the
   projection runs). Persistence is untouched — it has no rate to decay.
   **Result: dramatic improvement.** Max 2030-vs-baseline multiple fell
   from 230,151x to **22.7x**; >100x companies: 6 → **0**; >1000x: 4 → 0;
   >10x: 29 → 1 (SaxaVord Spaceport itself, now a plausible 22.7x with
   visibly decelerating year-over-year growth: 2.35x→1.55x→1.29x→1.14x→
   1.08x→1.06x). All 6 originally-traced outliers re-checked individually
   and now land at credible multiples (SaxaVord 22.7x, Eutelsat OneWeb
   8.8x, Infleqtion 5.2x, Oxa 4.4x, Sierra Nevada Corp 4.4x, Map of
   Agriculture 4.7x) with visibly decelerating trajectories, not runaway
   compounding. Air Liquide/AXA XL re-verified unaffected in shape (still
   one growth step then reverting to stable), values shifted only
   slightly from the decay blend. Side effect, expected not a bug:
   decayed predictions feed back into later steps' own growth signals, so
   classification flips rose from 75 to 180/1,222 (14.7%) and model usage
   shifted toward more Persistence (7,098 vs 6,785) — decay naturally
   cools a company's measured rate faster, so more companies settle into
   "stable" sooner than under undamped compounding.
8. ✅ `forecast_assemble.py` — combines the real historical panel +
   forecast_recursive.py's predictions into two outputs:
   `forecast_full_trajectories.csv` (long format, one row per company-year,
   `data_type` = observed/estimated_baseline/predicted — needed by
   forecast_reporting.py's planned gazelle check, which must walk every
   year-over-year transition including real pre-baseline history, not just
   baseline-vs-2030) and `forecast_2030_summary.csv` (one row per company —
   the "final company-level 2030 trajectories" the build order names,
   growth_multiple_2030, annualized_growth_rate_to_2030,
   n_years_growing/n_years_stable across the projected horizon,
   forecast_evidence_group carried through). Validated: 12,629 trajectory
   rows (4,095 observed + 749 estimated_baseline + 7,785 predicted, exactly
   matching each source file's row count) across 1,222 companies, 100%
   turnover_2030 coverage, 0 duplicate company_id. Growth multiple median
   1.0x (flat majority) with the known outliers landing where expected
   (SaxaVord Spaceport 22.7x, ~56%/year annualized). 203/1,222 companies
   were classified "growing" at least once during their projection.
9. `forecast_reporting.py` — **added to the plan, not yet built**.
   Business-facing outputs once 2030 trajectories exist from step 8:
   - **£10M-by-2030 companies**: filtered from the main forecast — companies
     currently under £10M turnover whose predicted 2030 turnover crosses
     £10M (i.e. the crossing must happen *from below*; already-above-£10M
     companies don't count here) — this is the project's core stated
     objective.
   - **Gazelle/high-growth companies**: two tiers, not one fixed threshold —
     companies sustaining ≥10% YoY growth, and companies sustaining ≥20%
     YoY growth, each independently reported so the tiers can be compared.
     **Decided**: both tiers require **3 consecutive years** of sustained
     growth at that threshold — the OECD's standard "high-growth
     enterprise" definition (same underlying concept as the Beauhurst
     10%/20% scaleup flags excluded from `feature_engineering.py` for
     leakage reasons — see `DROPPED_COLUMNS` — but computed independently
     here from this project's own forecast/growth features, not borrowed
     from Beauhurst's version, so the leakage concern doesn't apply to this
     output). A stated assumption, not a hardcoded silent default — will
     live as a named constant (e.g. `GAZELLE_CONSECUTIVE_YEARS = 3`) in
     `forecast_reporting.py` once built, with this same reasoning repeated
     as a code comment there.
   - **Intersection output**: companies meeting the high-growth criteria
     (either tier) AND predicted to reach ≥£50M by 2030.
   - **Reliability/evidence-group carry-through, required for all three**:
     none of these outputs are presented as equally trustworthy regardless
     of evidence quality — a gazelle candidate built on Group D (estimated
     baseline, zero observed turnover history) must visibly carry weaker
     confidence than one built on Group A (3+ observed years), via the same
     `forecast_evidence_group` / reliability columns the main forecast
     already carries, not a separate ad hoc confidence score.

   ✅ **Built.** `forecast_10m_crossings.csv`, `forecast_gazelle_10pct.csv`,
   `forecast_gazelle_20pct.csv`, `forecast_gazelle_50m_intersection.csv`
   all carry `forecast_evidence_group`, `baseline_turnover_source`,
   `n_real_years`, `evidence_gate_triggered` (was this company's raw
   growth signal ever downgraded by the Group-A evidence gate?), and
   `growth_decay_applied` (was Ridge/CAGR ever actually used for it, i.e.
   did decay apply at all, vs. a trajectory decided entirely by
   Persistence). Gazelle tiers walk `forecast_full_trajectories.csv`'s
   FULL year-by-year sequence (real history + predicted years together),
   not baseline-vs-2030 — a company already gazelle-qualifying in its real
   history counts, matching the brief's "not just the endpoint."

   **Two additions beyond the original plan, both stated assumptions:**

   - **Operational scaling indicator** (`forecast_operational_scaling.csv`,
     independent of the turnover gazelle tiers): flags sustained
     `employee_growth` or `asset_growth` (sec 7.7's log-difference
     features, same construction as `log_growth_1y` after their earlier
     reformulation) at the same 10%/20%-for-3+-years logic as the turnover
     gazelle tiers — reusing the identical thresholds/consecutive-year rule
     rather than inventing a third growth-rate convention. Requires 3+ REAL
     years of the SPECIFIC covariate (employees for employee_growth,
     total_assets for asset_growth) — not `forecast_evidence_group`, which
     measures turnover-history depth specifically and is a different axis:
     a company can have thin turnover evidence but rich filed-employee
     history, or vice versa, and this flag should track its own evidence.
     Catches companies scaling operationally (headcount, balance sheet)
     even in years without a usable turnover figure — the FEATURE this
     draws on was already built (sec 7.7), this is a new REPORTING use of
     it, not new feature engineering.
   - **Credibility gate on the £50M intersection specifically**
     (`credibility_status` column, `forecast_gazelle_50m_intersection.csv`):
     a company is only flagged as a genuine £50M candidate if it has
     `forecast_evidence_group=="A"` (3+ real turnover years) AND at least
     £1M turnover in one of its 5 most recent real reporting years.
     Rationale: Group A alone doesn't rule out "genuinely real, but a tiny
     early-stage company nowhere near £50M-scale operations" — exactly the
     profile of 3 of the 6 growth-decay outlier companies (SaxaVord
     Spaceport, Infleqtion, Map of Agriculture — all Group A on real data
     that's still only in the tens/hundreds of thousands of pounds).
     Companies failing the gate are NOT excluded from the output —
     excluding them would look like they were never considered, when the
     point is the opposite: they were considered and explicitly found not
     to meet the credibility bar. **Current run: 0/77 companies in the
     £50M intersection fail this gate** — not because the gate is
     vacuous, but because growth-rate decay (already fixed, an earlier
     step) already pulled the thin-base outliers' 2030 values
     below £50M before this gate ever runs (SaxaVord £42.4M, Infleqtion
     £376K, Map of Agriculture £935K — none reach the £50M bar at all,
     gazelle-qualifying or not). The gate is a safeguard against a future
     data refresh or parameter change reintroducing that failure mode, not
     evidence that it's unnecessary now.
10. ✅ `forecast_prediction_intervals.py` — presentation-layer additions
    beyond the original build order, added as a follow-on requirement once
    the 9-step pipeline was complete.
    - **`export_excel_workbook.py`**: consolidates the completed baseline
      (`assemble.py`) + all 6 `forecast_reporting.py` outputs into one
      `.xlsx` (`data/output/turnover_forecast_workbook.xlsx`), presentable
      column headers via a shared `COLUMN_LABELS` map, frozen + bolded
      header row, content-sized columns.
    - **Residual-based prediction intervals, stratified by evidence
      group**: added to `forecast_trajectories.csv`
      (`turnover_lower`/`turnover_upper`), carried through
      `forecast_full_trajectories.csv` by `forecast_assemble.py`. Method:
      - *Per-step spread*: the deployed horizon=1 mechanism is Persistence
        for the "stable" majority (forecast_selection.py's actual winner
        in all 3 real missions), and Persistence's log-space residual has
        a clean closed form — predicting turnover_t+1=turnover_t means
        log1p(actual)-log1p(predicted) is exactly log1p(turnover_t+1)-
        log1p(turnover_t), i.e. `log_growth_1y` itself. So Persistence's
        historical out-of-fold residual distribution IS the real,
        already-computed `log_growth_1y` values — no refitting, no
        leakage risk (Persistence has no parameters to fit). `sigma_log_
        residual` = std of real `log_growth_1y`, computed separately per
        (mission, `forecast_evidence_group`) from real transitions only,
        never from recursion's own synthetic predictions.
      - *Stated scope, not silent*: this uses Persistence's residual
        spread uniformly for every predicted row, even "growing" steps
        actually produced by Ridge/CAGR — those likely carry their own,
        probably wider, true uncertainty, not modelled separately here.
        Read the band as a lower bound specifically for
        growth_classification="growing" rows.
      - *Fallback for thin groups* (`MIN_SAMPLES_FOR_GROUP_SPREAD = 5`):
        Group C (exactly 1 real year) and Group D (0 real years) have NO
        real transition at all — structurally undefined, not just sparse.
        **First attempt rejected on inspection**: falling back to the
        mission's pooled std (all groups combined) landed Group D within
        a few percent of Group A's own spread, since Group A's much
        larger sample (575-1564 vs Group B's 6-15 real transitions)
        dominates a size-weighted pool — backwards for a group with LESS
        evidence, and it would make Group D look nearly as confident as
        the best-evidenced tier. **Fixed**: fall back to the WORST
        (highest-std) tier that still clears the sample-size bar
        (empirically Group B in every mission, 2.5-3.5x more volatile
        than Group A's real transitions) — keeps the ordering correctly
        monotonic, A tightest, B/C/D anchored to B's wider spread.
      - *Widening with horizon*: standard random-walk h-step-ahead
        variance under an independent-errors assumption — cumulative
        log-space variance after `step` recursive steps is
        `step * sigma_log_residual^2`, so cumulative log-space std is
        `sigma_log_residual * sqrt(step)` (same sqrt(horizon) growth used
        in ARIMA-style h-step intervals). `turnover_lower/upper =
        expm1(log1p(turnover) -+ 1.96 * sigma_log_residual * sqrt(step))`,
        Z=1.96 (~95%, a stated standard choice) — clipped at 0 below.
        Baseline/observed/estimated_baseline rows get no interval (NaN):
        they're real filed data or the OTHER pipeline's point value, not
        this pipeline's own uncertainty to characterise.
      - **Verified stratification works**: comparing Air Liquide (Group A)
        against DEOS Consultancy (Group D) at matching step counts — step
        1: upper/lower width ratio 5.7x (Group A) vs 348x (Group D, 61x
        wider); step 6-7 (2030): 99x vs ~1.68 million x. Confirms Group A
        bands are visibly, dramatically tighter than Group D's, as
        intended.
    - **`dashboard.py`**: Streamlit company explorer (`streamlit run
      dashboard.py`, no setup beyond `pip install -r requirements.txt` —
      `streamlit`/`plotly`/`openpyxl` added there). Search by name or CH
      number; Plotly chart with solid observed/baseline turnover, dashed
      forecast to 2030, and the shaded confidence band above; a summary
      panel (mission, evidence group, baseline source, real-years count,
      2030 forecast/multiple/annualized rate, and all 5
      `forecast_reporting.py` flags including £50M credibility status);
      a data-provenance breakdown and full year-by-year table. Reads only
      already-computed CSVs, no recomputation. Verified: server starts
      cleanly (HTTP 200, no exceptions), and the underlying data logic
      (company lookup, trajectory split, summary fields, flag membership)
      tested directly against Air Liquide with correct results.

## Filing-period annualization (post-launch fix — non-standard accounting periods)

`Financial Statement N - Number of weeks in the accounting year` (Source 1)
was never used anywhere in either pipeline before this — a real gap,
checked against real data rather than assumed: **4.0% of Source 1
statement-years in the space-company universe have a non-52-week
period** (a company's first "stub" filing, or a year-end change), ranging
from a 4-week stub (13x if naively annualised) to an 82-week extended
period (0.63x). Left uncorrected, this silently distorts any year-over-
year growth comparison spanning one of these years — `log_growth_1y`,
CAGR, and the gazelle consecutive-growth-streak logic all walk exactly
these transitions, purely from the reporting-period mismatch, not real
business change.

**Fix**: `feature_engineering.build_source1_annualization_factors` extends
the existing Source1<->Source2 merge (previously Statement 1 / most-recent-
filing only, for the financial ratios above) across **all 10 Financial
Statement blocks** — growth calculations span a company's entire observed
history, not just its latest filing. `forecast_data_prep.annualize_turnover`
applies `turnover x 52/weeks` (missing weeks -> factor 1.0, i.e. no
correction — no evidence of a non-standard period is not evidence it
wasn't standard) once, at the earliest point turnover enters the
forecasting pipeline: the historical panel (`build_historical_panel_source`)
and the completed baseline (only `turnover_source="observed"` rows — a
`"predicted"` baseline is the estimation model's own output, not tied to
any real filing period, so there's nothing to annualize). Fixing it this
early means every downstream growth formula inherits the correction
automatically, with no per-formula patch needed. Every correction is logged
(`forecast_panel_annualization_log.csv`, `forecast_baseline_annualization_log.csv`)
rather than applied silently.

**Scope, deliberately NOT touching the estimation pipeline**: this only
corrects the FORECASTING pipeline's own historical panel/baseline copies —
`src/model_bakeoff.py`'s target (`total_turnover`, "Total Turnover (CH
year)" per DATA_SCHEMA.md) and the mission-specific R² numbers reported
elsewhere in this file are computed from a separate build
(`labelled_features.csv`) that this fix never touches. Confirmed by a full
pipeline re-run: `model_performance_summary.csv` (ACE 0.14 / Beyond Earth
0.63 / Resilient Earth 0.65) is byte-identical before and after.

**Real impact, measured via a full before/after pipeline re-run**:
- Forecasting bake-off R² (12-candidate rolling-origin CV, all 3 missions):
  small, single-digit-percentage-point movements in both directions (e.g.
  ACE Elastic Net 0.9959 -> 0.9967; Resilient Earth Elastic Net 0.9398 ->
  0.9295) — real but modest, consistent with ~2-4% of rows being corrected.
- The actual DEPLOYED recursive mechanism (Persistence at horizon 1, used
  to build every company's 2030 trajectory) is unchanged in all 3
  missions — only some horizon-2/3/4+ benchmark comparisons (reporting-only,
  never used to generate the trajectory itself) shifted.
- Gazelle 10%: same total count (167) but real membership churn — 3
  companies added, 3 removed (net zero, not "no change").
- Gazelle 20%: 54 -> 56 (4 added, 2 removed).
- £50M intersection: unchanged (same 77 companies).
- £10M-by-2030 crossings: 9 -> 11 (2 clean additions: TransitionZero,
  Trimble Maps — both cross once their corrected growth trajectory pushes
  them over).
- **TerraFarmer** (flagged earlier as a runaway-compounding outlier, £75,592
  -> £721,789 in its only 2 real years): confirmed **partly, not wholly, a
  filing-period artifact** — its 2023 filing covered 74 weeks (0.703x
  factor), so the corrected baseline is £507,203, not £721,789 (30% lower).
  The underlying jump from £75,592 is still substantial even corrected —
  genuine strong growth remains, but the raw reported magnitude was
  inflated by the non-standard filing period.
- **GMV**: its own baseline year (2023) wasn't itself irregular, but an
  earlier year in its history (2020, 61 weeks, 0.852x factor) was — once
  corrected, GMV's overall historical growth trajectory shifts modestly,
  moving its 2030 forecast from £22.8M (2.42x baseline) to £24.9M (2.64x),
  a ~9% increase from a historical correction 3 years before its baseline,
  not a dramatic change.

## Model improvement investigation (post-launch, independent of adjacent-company data)

Four related pieces of work, requested together but kept independently
attributable per instruction (separate commits, separate findings below) —
none of this replaces the mission-specific models (`src/model_bakeoff.py`,
`src/model_selection.py`), which remain the pipeline's primary approach.

### 1. CatBoost added to the mission-specific bake-off

`model_bakeoff.py`'s candidate list is now 10 (9 original + CatBoost),
specifically for its native categorical handling — relevant given
`sic_code_1`'s high cardinality already forced `min_frequency` one-hot
bucketing to stabilise the linear models (see the module docstring's
1e83+ blow-up history). CatBoost gets its OWN preprocessor
(`build_catboost_preprocessor`) that skips one-hot entirely and passes
categorical columns through as imputed strings via CatBoost's
`cat_features` — reusing the standard one-hot pipeline would have reduced
it to "just another tree ensemble" and defeated the reason it was added.

**Bug hit and fixed**: passing `cat_features` via the `CatBoostRegressor`
constructor breaks sklearn's `clone()` (`RuntimeError: Cannot clone
object ... constructor either does not set or modifies parameter
cat_features` — a known CatBoost/sklearn `get_params` round-trip gap).
`clone()` runs on every fold inside `GridSearchCV`/`TransformedTargetRegressor`,
so this isn't avoidable by working around it once. Fixed by passing
`cat_features` as a FIT-time parameter instead (`CATBOOST_FIT_KWARGS`,
threaded through the same `"model__<param>"` mechanism already used for
`sample_weight`) — the constructor stays clone-safe.

`get_preprocessor(name)` centralises the preprocessor choice so
`model_selection.py`'s final refit and `reporting.py`'s OOF-residual
regeneration pick the right path when CatBoost wins a mission, not just
the bake-off itself — both were updated. `catboost>=1.2` added to
`requirements.txt`; `catboost_info/` (its auto-generated per-run training
log directory) added to `.gitignore`.

### 2. Source 1 financial ratios

See "Data sources" above for the feature list and DATA_SCHEMA.md's "Source
1 financial ratios" section for the full coverage/leakage-check tables.
Headline: 9 balance-sheet ratios kept (confirmed never turnover-derived),
6 excluded (confirmed or strongly suspected turnover-derived by formula
reconstruction against real data). Year-anchored to Statement 1's own
filing date — coverage in the merged panel is real but modest (~33% of
the 367 labelled companies, ~4% of panel rows), a stated tradeoff to avoid
leaking a recent balance sheet into historical rows.

### 3. Mission-specific bake-off re-run: does CatBoost + the new ratios move R²?

Re-ran `python -m src.model_bakeoff` (all 3 missions, 5x5 repeated grouped
CV) with both additions in place, then `model_selection.py` to reselect.
Comparison against the prior committed summary CSVs (9 models, Source
2/3 features only):

| Mission | Before (model, R2_mean) | After (model, R2_mean) | Selected model changed? |
|---|---|---|---|
| ACE | Lasso, 0.15 | Lasso, 0.14 | No — CatBoost R2_mean=-1.69 for ACE, far worse than Lasso |
| Beyond Earth | Lasso, 0.62 | Lasso, 0.63 | No — CatBoost R2_mean=0.55, worse than Lasso |
| Resilient Earth | Gradient Boosting, 0.61 | **CatBoost, 0.65** | **Yes** — CatBoost beat Gradient Boosting (0.62 after the ratios) |

**Conclusion**: the ratio features moved R² by ~0.01 for ACE/Beyond Earth
(noise-level — none of the 9 `fs1_*` features cracked ACE's or Resilient
Earth's top-10 feature weights; `fs1_roce_pct` appears, minor, in Beyond
Earth's) and by a real but modest ~0.01 for Resilient Earth's own best
non-CatBoost model. **CatBoost does NOT help ACE** — it performs
dramatically worse than Lasso there (-1.69 vs 0.14) despite the
native-categorical advantage; the diagnostic below explains why (ACE's
weakness isn't categorical-handling-shaped at all). CatBoost DOES help
**Resilient Earth**, becoming its new selected model (R2_mean 0.61 -> 0.65)
— the low-cardinality-categorical advantage plus tree-based handling of
that mission's financial ratios evidently helps there specifically.
`selected_models.csv` / `final_model_resilient_earth.joblib` updated
accordingly; ACE remains `usable=True` (R2_mean=0.14 still clears the
`USABILITY_R2_THRESHOLD=0.0` bar) but essentially unchanged.

### 4. ACE worst-predicted-companies diagnostic — refining "small sample size"

Regenerated `residuals_ace.csv` (post re-run) and traced the largest
out-of-fold residuals. The naive "sum of absolute residual per row" view
is misleading (dominated by BT's 12 high-scale rows) — the real diagnostic
had to match `model_bakeoff.py`'s actual weighting (`sample_weight = 1 /
company's row count`, i.e. every company gets equal total vote regardless
of how many years it has).

**Under that weighting, a tiny handful of companies dominate ACE's
weighted sum-of-squared-error**: BT alone = 92.9%; BT + Babcock + Costain
Group + Intelsat + Avanti together = 99.97% (of 80 companies total). BT,
Babcock and Costain are large, DIVERSIFIED conglomerates (telecoms,
defence/engineering, infrastructure) — confirmed via the raw Source 2 file
that Beauhurst itself has **0% `Space %` documented for all three across
every year** — their Total Turnover reflects their whole business, not a
space-specific segment, and their scale (BT: 83,400 employees, ~1,242x
ACE's median company) sits far outside the range ACE's other ~75 companies
establish. Intelsat/Avanti are a second, distinct pattern: capital-intensive
satellite operators (SIC 61300) with atypical revenue-per-employee ratios
(Intelsat: ~£12M/employee vs ACE's median ~£203K, ~60x).

**This concentration is NOT unique to ACE** — Beyond Earth shows an almost
identical pattern (BAE Systems + Rolls Royce = 79.6% of its weighted SSE,
top 5 = 99.1%; BAE Systems is even underestimated by almost exactly the
same ~50-60% every year that BT is) — so "a few giant companies dominate
the metric" is just what a heavy-tailed turnover target looks like in
every mission, not an ACE-specific flaw. A single-fit, all-companies-at-once
OOF check even shows ACE's R2≈0.69, comparable to Beyond Earth's ≈0.70 —
the underlying model fit is not obviously worse for ACE than for Beyond
Earth.

**What actually differs, and the real mechanism connecting this to "small
sample size"**: ACE has only 80 companies vs Beyond Earth's 202 (2.5x
fewer), so under 5-fold `GroupKFold` each ACE outer test fold holds only
~16 companies vs Beyond Earth's ~40. Checked directly against the 25
repeated-CV folds: the worst observed fold (R2=-9.45) had **Avanti** —
not BT — as its only large/volatile test company; when BT is in a
fold's TRAINING set (not its test set), that fold's mean R2 collapses to
~0.01 (std 2.26) vs ~0.66 (std 0.17) when BT is in the test set instead —
consistent with a single extreme, high-leverage training point distorting
the fitted coefficients enough to hurt generalisation to whichever OTHER
16 companies land in that fold's test set. With ~2.5x fewer companies per
fold than Beyond Earth, ACE has far less room to buffer against this.

**Refined conclusion, replacing the bare "small sample size" framing**:
ACE's low, volatile R2_mean/R2_std isn't primarily "not enough data to
learn the relationship" (the aggregate fit is fine) — it's "too few
companies per CV fold to dilute the leverage of a small number of
structurally atypical companies" (huge diversified conglomerates with
undocumented space-revenue share, plus capital-intensive satellite
operators with volatile financials). This also explains CatBoost's poor
showing for ACE specifically: the problem isn't categorical cardinality at
all, so CatBoost's actual advantage is irrelevant there, and tree-based
splitting generalises even worse than regularised linear regression with
this few training companies and this much per-company leverage.

### 5. Pooled model bake-off — a hedge, not a replacement (`src/model_bakeoff_pooled.py`)

Trained on all 367 training-eligible space companies across all 3 real
missions together (3,133 panel rows: 665 ACE + 1,781 Beyond Earth + 687
Resilient Earth — the real, verified count, not the ~919 estimate floated
before this was built), `mission` added as a categorical feature. Winner
selected via the identical robustness-filter -> composite-rank ->
simplicity-tie-break procedure as the mission-specific pipeline
(`model_selection.select_model`, reused directly), then that one winner's
held-out predictions sliced per mission for a same-fold, no-refit,
apples-to-apples comparison against each mission's own selected model:

| Mission | Pooled winner (R2_mean) | Mission-specific model (R2_mean) | Pooled beats mission-specific? |
|---|---|---|---|
| ACE | Random Forest, **0.36** | Lasso, 0.14 | **Yes** |
| Beyond Earth | Random Forest, 0.62 | Lasso, 0.63 | No (essentially tied) |
| Resilient Earth | Random Forest, 0.64 | CatBoost, 0.65 | No (essentially tied) |

**The pooled model beats ACE's mission-specific model, by a real margin
(0.36 vs 0.14) — not just noise.** This is exactly consistent with the ACE
diagnostic above: ACE's core weakness was identified as too few companies
(80) per CV fold to dilute the leverage of a handful of structurally
atypical companies (BT, Babcock, Costain, Intelsat, Avanti). Pooling gives
the model ~4.7x more companies (367) to train on, and Random Forest's
bagging/ensemble structure is specifically robust to any single
high-leverage training point in a way a single Lasso fit on just ~64 ACE
training companies per fold isn't — directly addressing the diagnosed
mechanism, not a generic "more data helps" effect (which is also why
pooling does NOT help Beyond Earth/Resilient Earth: they already had
enough companies per fold that a few extra didn't move the needle, and
pooling in fact costs Beyond Earth's Lasso its edge, since the pooled
data's added cross-mission heterogeneity works against a single global
linear fit — a real, if very minor, downside of pooling for the
already-well-served missions).

**Notable side effect**: pooling destabilises the linear models far more
than any single mission's bake-off ever did — Lasso/Elastic Net/Ridge/
Linear Regression all produced R2 as extreme as -1e96 in the pooled
setting (vs. merely "occasionally exploding to 1e83" in isolated
mission bake-offs). The combination of `mission`'s new one-hot columns,
the already-marginal Source 1 debt-ratio outliers (documented above), and
the pooled data's much larger scale range (spanning BT/BAE Systems down to
Frontier Agriculture-scale companies all at once) evidently compounds the
project's known linear-model instability. Only tree-based/ensemble models
(Random Forest, Gradient Boosting, Extra Trees, CatBoost) and SVR/k-NN
stayed sane pooled — consistent with, not contradicting, this file's
existing "Lasso survives via L1 zeroing, other linear models don't"
finding, just at a larger scale.

**Status**: this is a hedge/fallback finding for ACE specifically, kept
alongside — not instead of — the mission-specific models. ACE's deployed
model in `predict.py`/`assemble.py` remains whatever `selected_models.csv`
says (currently Lasso, R2_mean=0.14, `usable=True`); switching ACE to the
pooled Random Forest would be a deliberate, separate decision (it would
also mean ACE's inference-time predictions depend on Beyond Earth/
Resilient Earth's own labelled data, a real architectural change PROJECT_NOTES.md's
"Three independent mission models" rule was written to avoid) — not made
here. Outputs: `model_bakeoff_pooled_summary.csv`,
`model_bakeoff_pooled_per_mission_summary.csv`,
`pooled_vs_mission_specific_comparison.csv`.

## Adjacent-company groundwork (`src/adjacent_data_prep.py`) — initial data prep, not yet integrated

The 3 planned adjacent-company files arrived (`SatApps ACE/Beyond
Earth/Resilient Earth training data.xlsx`, 2,505/3,659/5,963 rows —
smaller than the originally-planned ~23k, but real data, not a stub).
Validated against `ADJACENT_DATA_REQUIREMENTS.md` first (see that doc):
confirmed single-export Source1/3-style schema, 100%-populated CH ID and
Beauhurst URL, and good coverage of the required financial/grants/
accelerator fields. This section covers the 6 follow-up decisions made
once real files were in hand, implemented in `src/adjacent_data_prep.py`.
**This module produces two standalone CSVs
(`adjacent_static_features.csv`, `adjacent_turnover_panel.csv`) for review
— it does NOT wire into `sample_construction.py`/`feature_engineering.py`/
`model_bakeoff.py` yet.** That integration (setting `population_type` to
`"adjacent"`, tuning `ADJACENT_SAMPLE_WEIGHT`, restricting outer CV folds
to space companies per the build-order note above) remains a separate,
later decision.

1. **`company_age_years`**: searched all 437 raw columns for anything
   incorporation/registration-date-equivalent (not just an exact
   `Founded` match) — genuinely absent. Left null for every adjacent
   company; the pipeline's existing median imputation handles it like any
   other missing numeric feature. **Flagged, not built**: a Companies
   House API lookup (one call per company, 12,127 companies total) could
   recover real incorporation dates — worth pursuing as its own scoped
   decision (rate limits, API key, caching) if adjacent-company age turns
   out to matter once these rows are actually used in training.
2. **`multi_mission_overlap`**: 322/2,505 ACE, 263/3,659 Beyond Earth, and
   323/5,963 Resilient Earth companies also appear in at least one other
   mission's file (26 companies appear in all 3 — verified via
   `company_id`, matching an independent URL-based cross-check).
   Companies are kept in **every** mission file they appear in (not
   deduplicated down to one), with a `multi_mission_overlap` column
   listing the other mission(s) — e.g. Chemring and Ford Motor Company
   appear in all 3 mission files and are flagged as such in each. This
   makes the overlap visible to any downstream analysis rather than
   letting each appearance look like an independent company.
3. **`sic_code_1`**: the adjacent files carry `SIC Codes (2007) - Code` as
   a comma-separated multi-code string (e.g. `"58110, 58142, 58190,
   58290"`), unlike Source 2's single `sic_code_1` value. Parsed as the
   first code only, matching the "primary SIC code" convention
   `sic_code_1` represents elsewhere (`parse_sic_code_1`). 97-98% parse
   successfully across all 3 files.
4. **`company_size`**: derived from `Financial Statement 1 - Number of
   employees` via the standard Micro (<10) / Small (10-49) / Medium
   (50-249) / Large (250+) UK/EU SME employee-count thresholds
   (`build_company_size`). Checked against real data before using it, not
   assumed: this rule reproduces Beauhurst's own `Size {year}` bucket for
   space companies **96.1%** of the time (2,876/2,994 `labelled_features.csv`
   rows with both fields populated) — the small remainder is explained by
   `company_size` and `total_employees` coming from different snapshot
   years for the same company (a known temporal-mismatch pattern already
   documented elsewhere in this file), not a different underlying rule.
5. **`total_export_revenue`**: no substitute field exists anywhere in the
   437 raw columns — left null, same imputation as any other missing
   feature.
6. **Turnover-by-year panel**: reconstructed from the 10 Financial
   Statement blocks (`Date of accounts` -> year, `Turnover` -> value),
   the same statement-to-year anchoring convention
   `build_source1_annualization_factors` already uses for space companies
   (`build_turnover_panel`). Returned **un-annualized** — raw turnover
   alongside its own `weeks` value — matching this project's existing
   precedent of leaving the estimation pipeline's own target
   un-annualized (see "Filing-period annualization" above for why that
   was a deliberate choice). Worth noting: **4.9% of the reconstructed
   panel's (company, year) rows have a non-52-week accounting period**
   (5,278/106,805) — essentially the same order of magnitude as the 4.0%
   found in the original space-company data, so the same annualization
   question will need answering here too, if/when this panel is actually
   merged into training or forecasting.
