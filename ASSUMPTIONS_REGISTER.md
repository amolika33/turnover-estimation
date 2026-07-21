# Assumptions Register

A single, complete index of every methodological assumption, threshold, and
design decision made across this project — turnover estimation
(`src/`) and 2030 forecasting (`forecast_src/`). Each entry cites where it's
actually documented (`PROJECT_NOTES.md`, `DATA_SCHEMA.md`,
`FORECASTING_METHODOLOGY.md`, or a module docstring) so nothing here is a
duplicate source of truth — this is an index, not a rewrite.

**Status legend**: ✅ already documented (citation given) · 🆕 was missing,
added this pass (both here and to the cited doc) · ⚠️ a real gap in the
CODE, not just the docs — documented honestly as a gap, not papered over.

## Estimation pipeline

1. **Target = Total Turnover, not Space Turnover.** ✅
   `DATA_SCHEMA.md` "Resolved decisions" #1; `PROJECT_NOTES.md` "Target and
   data shape".
2. **Join key = Beauhurst URL (+ name fallback), not Companies House
   number** — Source 1 has no CH number field. ✅
   `DATA_SCHEMA.md` "Resolved decisions" #0 and "Merging Source 1 and
   Source 2".
3. **Duplicate resolution: CH number AND name must both match to merge;** a
   shared CH number alone is a logged `shared_ch_number_anomaly`, entities
   kept separate. ✅
   `DATA_SCHEMA.md` "Mission mapping (confirmed)"; `PROJECT_NOTES.md`
   "Pipeline stages" #1 and "Data quality exclusions from training".
4. **Long/panel format** (one row per company-year), not one row per
   company. ✅
   `DATA_SCHEMA.md` "Resolved decisions" #2; `PROJECT_NOTES.md` "Target and
   data shape".
5. **Grouped CV by `company_id`** to prevent cross-year leakage within a
   company. ✅
   `DATA_SCHEMA.md` "Resolved decisions" #2; `PROJECT_NOTES.md` "Target and
   data shape" and "model_bakeoff.py checklist"; `src/model_bakeoff.py`
   module docstring.
6. **Space companies segmented via `mission_mapping.csv`; adjacent
   companies arrive pre-segmented** (3 separate Beauhurst collections), no
   mapping applied to them. ✅
   `DATA_SCHEMA.md` "Resolved decisions" #3; `PROJECT_NOTES.md` "Target and
   data shape" and "Current status / build order" step 3.
7. **Sky UK excluded** (data-entry error — company name pasted into Value
   Stream field). ✅
   `DATA_SCHEMA.md` "Mission mapping (confirmed)"; `PROJECT_NOTES.md` "Data
   quality exclusions from training".
8. **Volante Global excluded** (miscategorized — insurance/reinsurance
   holding company, not space sector). ✅🆕
   Already documented in `src/mission_segmentation.py`'s
   `KNOWN_MISCATEGORIZED_COMPANIES` dict (full reasoning) and mentioned in
   passing in `PROJECT_NOTES.md`'s forecasting build-order — but **not**
   previously cross-referenced in the "Data quality exclusions from
   training" section where a reader would look for it alongside Sky UK.
   Added there this pass.
9. **Internal-only Source 2 columns excluded from modelling** (`HPB:
   Normalised Score`, `SAC Tagging`, `S&H (2016)`, all `Space
   Employees`/`Space Turnover`/`Space Export Revenue`/`Space %` columns,
   `CH Check`, `Validated (CH & Beauhurst)`, `Comments`, `ST Type`). ✅
   `DATA_SCHEMA.md` "Excluded columns (internal-only, exclude from
   modelling)".
10. **Cross-cutting companies excluded from training entirely; included in
    final output via best-guess mission assignment** (SIC Code 1 +
    LinkedIn keyword similarity) for prediction only, flagged
    `reliability="approximate"`. ✅
    `PROJECT_NOTES.md` "Cross-cutting company predictions" (full section,
    including the "why" and the `"approximate"`-ranks-below-`"low"`
    clarification added this pass).
11. **`log1p` target transform, plus `log1p` on skewed numeric features**
    (employees, assets, export revenue, derived ratios). ✅
    `src/model_bakeoff.py` module docstring (extensive — the BT/BAE
    Systems instability history); `PROJECT_NOTES.md` "model_bakeoff.py
    checklist".
12. **`OneHotEncoder(min_frequency=5)`** to cap rare categorical levels
    (SIC code, etc.). ✅🆕
    The general "min_frequency bucketing" fix was already described in
    `PROJECT_NOTES.md`, but the specific value `5` was never justified
    anywhere (bare code, no comment). Added this pass: a code comment at
    `src/model_bakeoff.py`'s `build_preprocessor`, and an entry in
    `PROJECT_NOTES.md`'s "Documented assumptions and thresholds".
13. **`linkedin_industry` dropped entirely** (redundant with SIC/Value
    Stream, high-cardinality instability risk, hard to explain to
    stakeholders). ✅
    `PROJECT_NOTES.md` "Documented assumptions and thresholds";
    `src/feature_engineering.py`'s `DROPPED_COLUMNS`.
14. **`signal_accelerator`/`signal_academic_spinout` dropped** as redundant
    with derived `has_attended_accelerator`/`is_academic_spinout`. ✅
    `DATA_SCHEMA.md` "Source 3" section; `PROJECT_NOTES.md` "Data sources".
15. **Beauhurst's 10%/20% scaleup and High Growth List booleans excluded**
    (leakage risk — Beauhurst's own methodology is partly turnover-based).
    ✅
    `DATA_SCHEMA.md` "Leakage check performed before including any signal
    column"; `src/feature_engineering.py`'s `DROPPED_COLUMNS`.
16. **Grant/fundraising recency features nulled (not negative)** for panel
    rows predating the event, to prevent leakage. ✅
    `PROJECT_NOTES.md` "Data sources"; `src/feature_engineering.py`'s
    `merge_source3_features` comment.
17. **Panel rows weighted equally per company** (inverse-frequency by row
    count), not per company-year, per the methodology's stated unit of
    analysis. ✅
    `PROJECT_NOTES.md` "Documented assumptions and thresholds" ("Panel row
    weighting"); `src/sample_construction.py`.
18. **Stale observed turnover flagged** (`turnover_age_years`,
    `turnover_is_stale`) but **not** reclassified as needing prediction —
    still treated as "observed". ✅
    `PROJECT_NOTES.md` "Documented assumptions and thresholds" ("Stale
    observed turnover").
19. **`company_id` scheme**: CH-number-prefixed where available, name+URL
    fallback otherwise, guaranteed non-null. ✅
    `PROJECT_NOTES.md` "Documented assumptions and thresholds"
    (`company_id`); `src/data_prep.py`'s `make_company_id` docstring.
20. **Any negative/invalid turnover value is nulled and logged, never
    silently corrected.** ✅ (previously ⚠️ — code gap now closed)
    True for the FORECASTING pipeline (`forecast_data_prep.check_turnover`)
    and for PREDICTED turnover in the estimation pipeline
    (`predict.validate_predictions`) from the start. The estimation
    pipeline's gap for Source 2's raw, OBSERVED turnover value — flagged
    here in the previous pass, not merely undocumented but genuinely
    missing from the code — has since been closed:
    `sample_construction.check_turnover` (same null-and-log shape as
    `forecast_data_prep.check_turnover`, not imported directly — src/ and
    forecast_src/ stay independently auditable) now runs inside
    `build_long_panel`, so every path that builds the labelled panel
    (`sample_construction.main()`, `feature_engineering.build_features()`,
    and transitively every `model_bakeoff.get_mission_features()` caller)
    gets it automatically. Logged to `turnover_quality_log.csv`
    standalone, or combined into `feature_engineering_quality_log.csv`
    alongside the negative-company-age check when run via
    `feature_engineering.py`. Verified a no-op against the current
    labelled panel: 0 rows flagged, identical row counts before/after
    (665/1781/687 for ACE/Beyond Earth/Resilient Earth, unchanged) — this
    is now an ACTIVE check, not a documented absence.
    `PROJECT_NOTES.md` "Documented assumptions and thresholds"
    ("Observed-turnover validation"); `src/sample_construction.py`'s
    `check_turnover` docstring.
21. **Model-selection tie-break**: prefer the simpler model only when
    composite rank is within 1.0 AND R²_mean is within 0.05 of the
    top-ranked model. ✅
    `src/model_selection.py` module docstring (`COMPARABLE_TOLERANCE = 1.0`,
    `R2_COMPARABLE_TOLERANCE = 0.05`, both named and justified).
22. **Robustness-violation definition**: any CV fold with R²<-2 or fold
    MAE>3x that model's own median fold MAE. ✅
    `src/model_selection.py` module docstring (`BROKEN_FOLD_R2`,
    `BLOWUP_FOLD_MAE_MULTIPLE`, both named and justified).
23. **ACE usability threshold**: R²_mean must exceed 0.0 to be marked
    usable. ✅
    `PROJECT_NOTES.md` "Documented assumptions and thresholds" ("Model
    usability threshold"); `src/model_selection.py`'s
    `USABILITY_R2_THRESHOLD`.
24. **`best_params` serialised as JSON** in output CSVs, not relying on
    pandas' implicit string conversion. ✅
    `PROJECT_NOTES.md` "Documented assumptions and thresholds".

## Forecasting pipeline

25. **Recursive one-year-ahead forecasting is primary; direct multi-horizon
    forecasting is a secondary benchmark only.** ✅
    `FORECASTING_METHODOLOGY.md` §5 ("The principal supervised-learning
    task is one-year-ahead forecasting"); `PROJECT_NOTES.md` "2030
    Forecasting Pipeline" step 6.
26. **Rolling-origin/temporal validation** (not random or grouped k-fold)
    for the forecast bake-off, to respect chronological order. ✅
    `forecast_src/forecast_bakeoff.py` module docstring (full "VALIDATION
    SCHEME" section); `PROJECT_NOTES.md` "2030 Forecasting Pipeline" step 5.
27. **Evidence groups A/B/C/D** defined by real historical turnover-year
    count (A=3+, B=2, C=1, D=0). ✅
    `FORECASTING_METHODOLOGY.md` §6; `PROJECT_NOTES.md` "2030 Forecasting
    Pipeline" step 3.
28. **Training rows require strictly consecutive year pairs**
    (`target_year == accounting_year + 1`); non-consecutive gaps excluded.
    ✅
    `FORECASTING_METHODOLOGY.md` §5; `forecast_src/forecast_sample_
    construction.py`.
29. **Any company with negative/invalid turnover anywhere in its history is
    excluded ENTIRELY from forecasting** (not corrected, not partially
    salvaged) — applied to Price Forbes and Volante Global. ✅
    `PROJECT_NOTES.md` "2030 Forecasting Pipeline" step 1;
    `forecast_src/forecast_data_prep.py`'s
    `exclude_companies_with_invalid_turnover` docstring.
30. **`employee_growth`/`asset_growth` reformulated as log-differences**
    (not simple ratios) to fix extreme skew (28-56 -> ~3). ✅
    `PROJECT_NOTES.md` "2030 Forecasting Pipeline" step 5 (full
    before/after skew numbers).
31. **Three implausible employee-count values nulled** (GMV 2013, Added
    Value Solutions 2016/2017) as a data-quality issue, not a formula bug.
    ✅
    `PROJECT_NOTES.md` "2030 Forecasting Pipeline" step 5;
    `forecast_src/forecast_data_prep.py`'s `check_plausible_employees`
    docstring (`PLAUSIBLE_EMPLOYEES_MAX = 500,000`).
32. **Growth-trajectory-conditional model routing**: "growing" companies
    (via `log_growth_3y_mean`, falling back to `log_growth_1y`, threshold
    `log1p(0.10)`≈0.0953) get CAGR/Ridge; "stable" get Persistence. ✅
    `PROJECT_NOTES.md` "2030 Forecasting Pipeline" step 7 (`GROWTH_THRESHOLD`).
33. **Routing is re-evaluated dynamically at every recursive step**, not
    fixed at the baseline year. ✅
    `PROJECT_NOTES.md` "2030 Forecasting Pipeline" step 7 ("re-evaluated at
    every recursive step (not fixed at baseline...)").
34. **Evidence gate**: "growing" classification only eligible for CAGR/
    Ridge trend-continuation if the company has Evidence Group A (3+ real
    years); thinner evidence falls back to Persistence regardless of
    measured growth rate. ✅
    `PROJECT_NOTES.md` "2030 Forecasting Pipeline" step 7
    (`MIN_EVIDENCE_GROUP_FOR_TREND_CONTINUATION = "A"`, full root-cause
    trace: TerraFarmer et al.).
35. **Growth-rate decay**: `weight_company(step) = 0.5^(step/2.0)` — a
    2-step half-life — blending the company's own rate toward the
    mission's MEDIAN `log_growth_1y` as the horizon increases. ✅
    `PROJECT_NOTES.md` "2030 Forecasting Pipeline" step 7 ("Growth-rate
    decay" — full before/after: max multiple 230,151x -> 22.7x).
36. **Gazelle/high-growth thresholds**: BOTH ≥10% and ≥20% sustained YoY
    growth tiers, each requiring 3 consecutive years (OECD high-growth-
    enterprise definition). ✅
    `PROJECT_NOTES.md` "2030 Forecasting Pipeline" step 9
    (`GAZELLE_CONSECUTIVE_YEARS = 3`).
37. **£50M-by-2030 credibility gate**: requires Evidence Group A AND ≥£1M
    turnover in at least one of the last 5 real reporting years; failing
    companies stay visible but marked, not hidden. ✅
    `PROJECT_NOTES.md` "2030 Forecasting Pipeline" step 9 ("Credibility
    gate on the £50M intersection specifically").
38. **Operational scaling flag**: employee/asset-growth based, independent
    of turnover, requires 3+ years evidence, same threshold style as the
    turnover gazelle tiers. ✅
    `PROJECT_NOTES.md` "2030 Forecasting Pipeline" step 9 ("Operational
    scaling indicator").
39. **Prediction intervals**: residual-based, stratified by evidence group,
    using Persistence's log-space residual (`log_growth_1y` itself) per
    (mission, evidence group); widens via `sqrt(step)`; thin-evidence
    groups (C/D) fall back to the WORST reliable tier's spread (Group B),
    not a size-weighted pooled average. ✅
    `PROJECT_NOTES.md` "2030 Forecasting Pipeline" step 10 (full method,
    including the rejected pooled-average first attempt);
    `forecast_src/forecast_prediction_intervals.py` module docstring.
40. **"2030" forecast target = each company's own fiscal-year-labelled year
    as a bare integer** — NOT precisely pinned to any specific calendar
    date (e.g. UK government fiscal year start, April 2030). ✅
    `PROJECT_NOTES.md` "2030 Forecasting Pipeline" section intro ("What
    '2030' actually means, stated explicitly (known simplification, not a
    silent gap)") — added in the session that investigated this question;
    confirmed present and complete.

## Model-improvement additions

41. **CatBoost added to the estimation bake-off with native categorical
    handling** (separate preprocessing path, not one-hot encoded). ✅
    `PROJECT_NOTES.md` "Model improvement investigation" §1 (full rationale
    + the sklearn-`clone()` compatibility bug and its fix).
42. **Source 1 financial ratios**: 6 excluded as turnover-derived leakage
    (e.g. Debtor days = Trade debtors / Turnover × 365); 9 kept as genuine
    balance-sheet-only ratios, anchored to their exact filing year. ✅
    `PROJECT_NOTES.md` "Model improvement investigation" §2 and
    `DATA_SCHEMA.md` "Source 1 financial ratios" (full per-ratio
    reconstruction formulas).
43. **Pooled/mission-agnostic model built and documented as a hedge/
    fallback** (not deployed) — helps ACE specifically (R²=0.36 vs 0.14),
    does not help Beyond Earth/Resilient Earth. ✅
    `PROJECT_NOTES.md` "Model improvement investigation" §5 (full
    comparison table and mechanism explanation).

## Resolved this session (status reflects the RESOLVED state)

44. **Filing-period variation**: CONFIRMED a real, material issue (57% of
    affected training rows distorted >30%, touching 26%/39%/21% of
    gazelle-10%/gazelle-20%/£50M-intersection companies). FIXED via
    annualization (`turnover × 52/actual_weeks`) across all 10 Source-1
    statement blocks. ✅
    `PROJECT_NOTES.md` "Filing-period annualization" section — states the
    fix, the scope decision (forecasting pipeline only, estimation
    untouched — confirmed via a full pipeline re-run), and the complete
    before/after impact (bake-off R², gazelle counts, £50M intersection,
    £10M crossings, TerraFarmer/GMV specifically). This is the RESOLVED
    write-up, not the original open question — confirmed current.
45. **Cross-cutting mission-assignment process**: CONFIRMED already
    documented; clarification on the "why" (Value Streams that don't map
    to any real mission by design) and the confidence-ranking
    (`"approximate"` ranks below `"standard"`/`"low"`) added. ✅
    `PROJECT_NOTES.md` "Cross-cutting company predictions" section —
    confirmed both additions are present (the "Why this exists at all"
    paragraph and the ranking clarification after the `reliability`
    bullet).

## Adjacent-company groundwork (this session — `src/adjacent_data_prep.py`)

46. **`company_age_years` for adjacent companies**: no incorporation/
    registration-date-equivalent field exists anywhere in the 437 raw
    columns (checked exhaustively, not assumed) — left null, standard
    imputation applies. A Companies House API lookup could recover this
    for real but is flagged as a separate, larger decision, not built. ✅
    `PROJECT_NOTES.md` "Adjacent-company groundwork" #1;
    `ADJACENT_DATA_REQUIREMENTS.md` "Not covered here".
47. **`multi_mission_overlap` flag**: companies appearing in more than one
    mission's adjacent file are kept in every mission they appear in (not
    deduplicated), with a column naming the other mission(s) — 322/263/323
    companies flagged in ACE/Beyond Earth/Resilient Earth respectively (26
    appear in all 3). ✅ `PROJECT_NOTES.md` "Adjacent-company groundwork" #2.
48. **`sic_code_1` parsing for adjacent companies**: the raw
    `SIC Codes (2007) - Code` field is a comma-separated multi-code
    string, unlike Source 2's single value — first code taken as the
    primary SIC code. ✅ `PROJECT_NOTES.md` "Adjacent-company groundwork" #3.
49. **`company_size` bucketing for adjacent companies**: derived from
    `Financial Statement 1 - Number of employees` via the standard
    Micro(<10)/Small(10-49)/Medium(50-249)/Large(250+) UK/EU SME
    employee-count thresholds — verified this reproduces Beauhurst's own
    `Size {year}` bucket for space companies 96.1% of the time before
    relying on it for adjacent companies (which have no `Size {year}` at
    all). ✅ `PROJECT_NOTES.md` "Adjacent-company groundwork" #4.
50. **`total_export_revenue` for adjacent companies**: no substitute field
    exists in the raw export — left null, standard imputation applies. ✅
    `PROJECT_NOTES.md` "Adjacent-company groundwork" #5.
51. **Adjacent turnover-by-year panel, annualized before merge**:
    reconstructed from the 10 Financial Statement blocks (same
    statement-to-year anchoring as `build_source1_annualization_factors`),
    then corrected for non-standard accounting periods (`turnover x
    52/actual_weeks`) via forecast_data_prep's existing `annualize_turnover`
    — reused directly, applied before any merge into training, not
    deferred. 4.9% of (company, year) rows had a non-52-week period;
    3,719 of those had a turnover value to correct, 48.1% of which were
    distorted by more than 30% — comparable severity to the original
    space-company finding (57%). ✅ `PROJECT_NOTES.md` "Adjacent-company
    groundwork" #6; `ADJACENT_DATA_REQUIREMENTS.md` "Update" note.

## Summary

- **41 of 45** were already documented somewhere before this pass (33
  fully complete, 2 needing only a cross-reference/threshold-value
  addition to reach full documentation — see #8, #12).
- **2 real documentation gaps found and closed this pass**: #8 (Volante
  Global's exclusion wasn't cross-referenced where a reader would look for
  it) and #12 (`min_frequency=5`'s specific value was never justified).
- **1 genuine CODE gap, found, documented honestly, then closed**: #20 —
  the estimation pipeline had no negative-turnover check for observed (as
  opposed to predicted) values. Not an active risk (zero negative values
  in current data), but a real absence, not just an undocumented
  presence — `sample_construction.check_turnover` now closes it, verified
  as a no-op against current data.
- **#44 and #45** (this session's own resolved investigations) were
  verified to already reflect their RESOLVED state in `PROJECT_NOTES.md`,
  not the original open question.
- **#46-51** (a later session): 6 new decisions made once the real
  adjacent-company files arrived, all documented in `PROJECT_NOTES.md`'s
  "Adjacent-company groundwork" section at the time they were made.
