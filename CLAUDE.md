# Turnover Estimation Framework — Project Context

This file gives Claude Code persistent context on this project. Read it before making changes.

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

- **Source 1**: `beauhurst_company_export_20260709-115139.csv.xlsx` — raw
  Beauhurst export, repeated Financial Statement blocks.
- **Source 2**: `Published Space Capabilities Catalogue_Cleaned.xlsx` —
  curated master/mission-tagged sheet, year-panel format. Primary source
  `feature_engineering.py` is built around.
- **Source 3**: `beauhurst_company_export_20260720-092535.csv.xlsx` —
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
  columns are ~100% empty in Source 2 (confirmed this session) — with
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
7. 🔶 `forecast_recursive.py` — built, growth-routing verified against a
   company sample, full 2030 run not yet executed (pending confirmation).
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
8. `forecast_assemble.py` — combine into final company-level 2030
   trajectories.
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
