# Data Schema Notes

Two source spreadsheets feed the space-company dataset. This doc exists so
this structure doesn't need to be rediscovered from scratch on every review.

## Source 1: Beauhurst raw export ("financials" sheet)

- One row per company.
- Company info: name, Beauhurst URL, stage of evolution, employee count,
  descriptions, industries/sectors, SIC codes, tracking reasons, funding
  history, grants, accelerator attendance, academic spinout info, etc.
- Up to **10 repeated "Financial Statement N" blocks** (N = 1..10), each a
  full snapshot: turnover, profit, balance sheet (assets/liabilities/equity),
  cash flow, staff costs, and ~20 financial ratios (gearing, current ratio,
  ROCE, etc.)
- Statements are numbered, **not year-labelled** — use each statement's own
  `Financial Statement N - Date of accounts` field to anchor it to a
  calendar/accounting year. Statement 1 is presumably the most recent (needs
  confirming against real data — don't assume).

## Source 2: Master/derived sheet ("mission-tagged" sheet)

- One row per company, with **year-indexed columns from 2013–2026**:
  Size, Total Employees (CH and Estimated), Space Employees, Total Turnover,
  Space Turnover, Space %, Balance Sheet Total Assets, Total/Space Export
  Revenue, Filing Date — all repeated per year as separate columns
  (i.e. wide panel format, not long/tidy).
- Carries `Value Stream` — maps to one of the three missions (ACE / Beyond
  Earth / Resilient Earth). This is the mission-assignment field for space
  companies.
- Also has: LinkedIn industry/keywords, Companies House identifiers, SIC
  codes 1-4, growth-rate metrics (employee/turnover growth, sector CAGR),
  and an `HPB: Normalised Score`.
- Some columns visibly derive from Source 1 (same underlying Beauhurst data,
  reshaped) — expect overlap/duplication, not independent information.

## Source 3: Grants/accelerator/funding enrichment (`space_companies_beauhurst_grants_accelerators.xlsx`)

- One row per company, **same 1,372-company universe as Source 1** (same row
  count, confirmed) — this is an *enrichment* file, not a new population. It
  adds candidate features for companies already in Source 1; it does not add
  new training rows.
- No Companies House number field (same gap as Source 1) — joins to the rest
  of the pipeline by normalised **Beauhurst URL** only.
- 13 boolean signal columns (`Growth signals - ...` / `Innovation signals -
  ...`) — not 14 as originally described when this file was introduced; the
  actual file has 13.
- Up to 5 `Accelerator Attendances N` slots (name + entry/exit dates) and up
  to 2 `Academic Spinout Events N` slots (institution + date).
- Grants and Fundraisings summaries: count, total amount, latest/earliest
  dates. **Data quality note**: the amount columns (`Grants - Total amount
  ...`, `Grants - Amount received ... latest grant`, and the Fundraisings
  equivalents) mix real numbers with a literal `"(no value)"` string
  sentinel, forcing pandas to infer object dtype — `feature_engineering.py`'s
  `_clean_currency` coerces this to numeric, treating the sentinel as
  missing (not 0, which would misrepresent an unknown amount as a confirmed
  zero).
- IPO market capitalisation — present but too sparse (19/1,372 non-null) to
  be useful this pass.

**Join match rate against the space-company dataset** (segmented_df, 1,225
companies): 1,155/1,225 (94.3%) matched by normalised Beauhurst URL — in
line with Source 1's own ~94% match rate against Source 2 (same underlying
gap: Source 1/3's collection isn't a perfect superset of Source 2's). 70
unmatched companies, including the 2 companies with no Beauhurst URL at all
(UK Hydrographic Office, ONS Data Science Campus).

**Leakage check performed before including any signal column**: the two
"scaleup" booleans (`Growth signals - 10%/20% scaleup`) and `Growth signals
- High growth list` were checked against Source 2's own turnover-growth
columns (`Turnover Growth Rate (OECD)`, `Latest 3 Years: Growth Rate (OECD:
20%/10%)` — both already confirmed turnover-derived and excluded from
features). Only 14-17%/7-9% agreement — these are *not* a re-export of
Source 2's derived columns — but Beauhurst's own published "scaleup"/"high
growth" methodology is itself typically based on the OECD high-growth-
enterprise definition (>=10%/20% p.a. average growth in employees **or
turnover** over 3 years), so turnover-independence couldn't be confirmed.
Excluded per user decision, consistent with the project's absolute
no-turnover-derivation rule. See `feature_engineering.py`'s
`DROPPED_COLUMNS` for the full reasoning, and the 10 booleans that were
confirmed safe (fundraising/M&A/accelerator/patent-type events — none
turnover-related). Of those 10, 2 (`Growth signals - Accelerator`,
`Innovation signals - Academic spinout`) were subsequently dropped too —
not for leakage, but because they're ~100% redundant with the derived
`has_attended_accelerator`/`is_academic_spinout` features (1 disagreement
out of 1,372 rows / 0 disagreements respectively) — leaving **8** direct
boolean signals in the final feature set.

## Resolved decisions

0. **Join key confirmed: Beauhurst URL (+ company name fallback), not Companies House number.**
   Source 1's raw export has no Companies House number field at all — only
   `Beauhurst URL`. Company name is normalised (lowercased, legal suffixes
   stripped, punctuation stripped) as a fallback for the ~3% of Source 2
   companies that don't match on URL (including 2 rows with a null
   Beauhurst URL). Verified against real data: exact URL match covers
   93.8% of Source 2, normalising trailing-slash/case gets to 94.8%, and
   name fallback recovers most of the remainder — leaving ~32 companies
   (2.6%) genuinely absent from Source 1.
1. **Target variable**: `Total Turnover (CH year)` — NOT `Space Turnover`.
   `Space Turnover` is not used anywhere in this project.
2. **Row granularity: LONG/PANEL format, not one-row-per-company.**
   Reshape Excel 2's wide year columns (Total Turnover 2013, 2014, ... 2026,
   and the matching Total Employees / Balance Sheet Total Assets / Export
   Revenue columns for the same years) into one row per (company, year) with
   `year` as an explicit feature. This multiplies the effective training set
   size (~400 companies x however many years each has data = potentially
   several thousand rows) rather than collapsing to one row per company.
   - **Critical**: rows from the same company across different years are
     NOT independent. Cross-validation MUST group by company (e.g. sklearn
     `GroupKFold` / `GroupShuffleSplit` on company ID), never plain random
     k-fold, or the model will leak company identity across folds.
3. **Mission assignment: space companies need internal segmentation via a
   mapping table; adjacent companies arrive pre-segmented.** The mapping
   table (`data/mission_mapping.csv` — Mission, Value Stream, Master Tag) is
   used to split the comprehensive space-company list into ACE / Beyond
   Earth / Resilient Earth. Adjacent companies do NOT need this mapping
   applied programmatically — the user has built three separate Beauhurst
   collections, one per mission, so adjacent-company exports will already
   arrive as three mission-specific files. The mapping table mainly documents
   *why* those collections were built that way.
4. **Schema implication**: adjacent-company exports will likely come out in
   the **raw Beauhurst format** (Source 1's style — repeated "Financial
   Statement N" blocks), not the curated master-sheet format (Source 2 —
   year-panel, growth rates, HPB score), since Source 2's extra fields were
   purpose-built/curated for the space companies specifically. This means
   Source 1's structure, not Source 2's, is the realistic **common schema**
   to design `feature_engineering.py` around if the model is meant to
   eventually take both space and adjacent rows through the same pipeline.
   Source 2 should be treated as space-company-only enrichment for now.

## Excluded columns (internal-only, exclude from modelling)

Excel 2 (master sheet) contains internal metrics that are NOT to be used as
predictors or targets — they're internal working fields, not genuine company
characteristics:

- `HPB: Normalised Score`
- `SAC Tagging`, `No. of Tags Attributed`
- `S&H (2016)` (meaning unclear — exclude regardless)
- All `Space Employees (...)`, `Space Turnover (...)`, `Space Export Revenue (...)`,
  `Space %` columns — target is Total Turnover, not Space Turnover, and these
  space-specific breakdowns are not used at all.
- Any other clearly-internal QA/process columns (e.g. `CH Check`,
  `Validated (CH & Beauhurst)`, `Comments`, `ST Type`) — administrative, not
  predictive signal.

When in doubt about whether a column is an internal working field vs a
genuine company characteristic, ask before including it in the feature set.

## Merging Source 1 and Source 2 (space companies only)

Source 1 (raw Beauhurst financials) and Source 2 (master/mission sheet)
need to be joined at the company level — they are not guaranteed to contain
the same companies (some will be missing from one side or the other).

- Join key: **Beauhurst URL** (normalised for trailing slash/case), with
  **normalised company name** as fallback. Companies House number is NOT
  usable as the primary key — Source 1 doesn't have a CH number field.
- This merge is a **space-companies-only** step. Adjacent companies have no
  Source-2-equivalent file at all — they only ever have Source-1-style raw
  financial data. `data_prep.py` should keep the merge logic clearly scoped
  to the space-company path, not assumed to apply universally.
- Expect and handle partial mismatches gracefully (company present in one
  source but not the other) rather than silently dropping rows — flag for
  review per the eligibility criteria in PROJECT_NOTES.md.

1. **Statement-to-year anchoring** (Source 1) — **confirmed**: Financial
   Statement 1 is always the most recent filing. Verified across all 1,372
   companies by checking `Date of accounts` is non-increasing from
   Statement 1 through Statement 10; zero violations found.

## Mission mapping (confirmed)

- `data/mission_mapping.csv` maps Source 2's `Value Stream` to the three
  missions (ACE / Beyond Earth / Resilient Earth) plus a `Cross-cutting`
  bucket (`Consultancy / Other`, `Explore New Markets`).
- The single row with `Value Stream == "Sky UK"` (company: Sky UK itself)
  is a data-entry error — the company's own name was pasted into the
  Value Stream field. It is excluded from mission mapping entirely (not
  treated as a category, not folded into Cross-cutting).
- 6 Companies House numbers in Source 2 were originally found shared by more
  than one company row (11 rows total). Two are confirmed data-entry errors,
  corrected in `data_prep.py` (`KNOWN_CORRECTIONS`), not in the raw file:
  - **GeoData Institute** had no genuine CH number of its own (University of
    Southampton entity) — was carrying `RC000668` (Univ. of Southampton's
    charity number). Nulled; falls back to URL/name matching.
  - **ISVR Consulting**'s correct CH number is `14701170` — was also
    incorrectly carrying `RC000668`.
- A separate, unrelated data-entry error (not a shared-CH-number case) was
  found and corrected the same way (`KNOWN_CORRECTIONS`): **Open Cosmos**'s
  `Total Turnover (CH 2023)` was `£6,542,660,000` in the raw file — exactly
  1000x the real filed turnover for year ended 31 December 2023
  (`£6,542,660`, confirmed directly against the filed accounts). Traced and
  confirmed isolated, not systemic, before correcting — see PROJECT_NOTES.md
  "Beyond Earth worst-predicted diagnostic follow-ups" for the full
  investigation (Source 1 had no matching entry for this company/year at
  all, so the error predates this project's pipeline; two independent
  outlier scans across the full labelled panel found nothing else this
  extreme).
  - **General rule for the remaining 4 groups** (9 rows, e.g. CH `08750033`
    = "Seradata Ltd" vs. "Slingshot Aerospace"; CH `RC000817` = "RAL Space"
    vs. "Science and Technology Facilities Council" vs. "Centre for
    Environmental Data Analysis"): a shared CH number does **not** by itself
    make two rows the same company. `data_prep.py` only treats rows as a true
    duplicate (excluded from training pending manual review) if the CH number
    **and** the normalised company name both match. A shared CH number with
    genuinely different names is logged as a `shared_ch_number_anomaly` and
    the rows are kept as separate entities — never merged, not excluded from
    training on this basis alone. Currently 0 true duplicates, 9 anomalies.

## Planned future additions

- ~~Grants data (number/amount/dates) — partially present in Source 1
  already; to be expanded.~~ **Done** — see "Source 3" above and
  `feature_engineering.py` (`grants_count`, `grants_total_amount`,
  `grant_recency_years`, and the fundraising equivalents).
- ~~Accelerator attendance — partially present in Source 1; to be
  expanded.~~ **Done** — see "Source 3" above (`has_attended_accelerator`,
  `accelerator_count`). Still relevant for adjacent companies, where this +
  financials may be the *only* data available (Source 3-style enrichment
  hasn't been confirmed to exist for the adjacent-company universe yet).
- ~~Source 1's ~20 Financial Statement 1 financial ratios (gearing %,
  current ratio, ROCE, ROTA, debtor/creditor days, etc.) — not yet
  incorporated.~~ **Done** — see "Source 1 financial ratios" below.

## Source 1 financial ratios (Financial Statement 1, added this pass)

Source 1 has ~20 financial ratios per Financial Statement block. Only
Statement 1 is used (confirmed most recent filing, see "Statement-to-year
anchoring" above). Two checks were performed against real data before
including any of them, not assumed:

**1. Coverage** (out of 1,372 Source 1 rows): two tiers —
- 90.8-94.8% populated: Current ratio, Liquidity acid test, Gearing (%),
  Equity (%), Current debt ratio, Total debt ratio.
- 30.8-30.9% populated: Return on capital employed (%), Return on total
  assets employed (%), Return on net assets employed (%). Still well above
  the "not just 1-2%" bar, but a visibly sparser tier — the same 9
  columns the excluded (turnover-derived) Pretax profit margin/Debtor
  days/Creditor days/Exports turnover ratio/Stock turnover ratio also come
  from, i.e. companies that filed non-abbreviated (full P&L) accounts.

**2. Turnover-derivation leakage check** — reconstructed each candidate
ratio from its own component columns (e.g. Pretax profit / Turnover) and
compared to the actual column value:
- **6 excluded** (exact or near-exact reconstruction from a formula that
  divides by `Financial Statement 1 - Turnover`, i.e. Total Turnover in
  disguise — forbidden regardless of how indirect): Pretax profit margin
  (%), Debtor days, Creditor days, Exports turnover ratio (%), Sales
  networking capital (all exact), Stock turnover ratio (%) (excluded on
  definitional grounds — standard accounting term always includes Sales/
  Turnover, exact formula just wasn't pinned since this dataset lacks an
  averaged-stock figure).
- **9 kept**, confirmed balance-sheet-only (never reconstructs from
  Turnover; raw/log correlation with Turnover also far weaker than the
  confirmed-leaky ratios, consistent with — not the primary proof for —
  the formula evidence): Current ratio, Liquidity acid test, Gearing (%),
  Equity (%), Current debt ratio, Total debt ratio, Return on capital
  employed (%), Return on total assets employed (%), Return on net assets
  employed (%). See `feature_engineering.py`'s `SOURCE1_SAFE_RATIO_COLUMNS`
  / `DROPPED_COLUMNS` for the full per-ratio reconstruction formulas.

**3. Year-anchoring, not a company-constant** — unlike Source 3's grant/
funding signals (attached to every panel row of a company), these 9 ratios
are a single snapshot tied to Statement 1's own accounting date. Attaching
them to every year of a company's panel (like a static fact) would leak a
recent balance-sheet snapshot into historical rows — forbidden by this
project's own "no information from outside what's available at prediction
time" rule. `feature_engineering.py`'s `merge_source1_ratio_features` joins
on (company_id, year) instead, leaving every other year null. Real cost of
doing this correctly, checked against the actual merged panel: only ~33%
of the 367 labelled companies (~4% of panel rows) end up with a non-null
value for any of these 9 ratios — SimpleImputer(median) in
`model_bakeoff.py`'s preprocessor handles the rest, same as any other
partially-populated numeric feature.

**Side effect observed, not fixed here**: 2 of the 9 ratios (Current debt
ratio, Total debt ratio) have extreme outliers (min -659, max 320,272 —
near-zero-equity denominators) that reproduce the project's known
BT/BAE-Systems-style linear-model instability (see `model_bakeoff.py`'s
module docstring) when the new features were smoke-tested: Linear
Regression/Ridge/Elastic Net coefficients occasionally blow up
(R2 as extreme as -1e127 in one quick test run). Left as-is rather than
clipped/winsorised — `model_selection.py`'s existing robustness filter
(R2 < -2 or a 3x fold MAE blow-up excludes a model from winning) already
exists specifically to catch and exclude this failure mode, and Lasso/
tree ensembles/CatBoost are unaffected (L1/tree splits are robust to
unbounded single-feature outliers) — consistent with, not a regression
from, the project's existing design.
