# Data Schema Notes

Two source spreadsheets feed the space-company dataset. This doc exists so
Claude Code doesn't have to rediscover this structure from scratch each session.

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

## Resolved decisions

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

- Preferred join key: **Companies House number** (or Beauhurst URL as a
  fallback) — matches the entity-matching approach already specified in the
  methodology doc (see CLAUDE.md eligibility rules).
- This merge is a **space-companies-only** step. Adjacent companies have no
  Source-2-equivalent file at all — they only ever have Source-1-style raw
  financial data. `data_prep.py` should keep the merge logic clearly scoped
  to the space-company path, not assumed to apply universally.
- Expect and handle partial mismatches gracefully (company present in one
  source but not the other) rather than silently dropping rows — flag for
  review per the eligibility criteria in CLAUDE.md.

1. **Statement-to-year anchoring** (Source 1): confirm whether "Financial
   Statement 1" is always the most recent filing, or varies by company —
   needed if/when Source 1's richer financial-statement detail is joined
   onto the long-format panel from Source 2 by year.

## Planned future additions

- Grants data (number/amount/dates) — partially present in Source 1 already
  (`Grants - ...` columns); to be expanded.
- Accelerator attendance — partially present in Source 1
  (`Accelerator Attendances - ...`); to be expanded, especially for adjacent
  companies where this + financials may be the *only* data available.
