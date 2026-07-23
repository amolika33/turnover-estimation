# Adjacent-Company Data Requirements

This documents the format the incoming ~23k adjacent-company files need to
be in for smooth integration into the existing pipeline. The merge itself
isn't built yet (no data to test against) — this is what to ask for /
check against when the files arrive. See PROJECT_NOTES.md "Current status / build
order" step 3 and "Documented assumptions and thresholds" for the related
pipeline-side prep.

## 1. Schema: Source 1's raw Beauhurst format, not Source 2's

Adjacent-company exports should arrive in **Source 1's raw schema**
(repeated `Financial Statement N` blocks, N = 1..10, each anchored by its
own `Financial Statement N - Date of accounts` field — see
`DATA_SCHEMA.md`), **not** Source 2's curated year-panel master-sheet
schema. This is the schema `feature_engineering.py` was designed around
for exactly this reason (see its module docstring / DATA_SCHEMA.md
resolved decision 4): Source 2's extra fields (growth rates, HPB score,
year-indexed panel columns) were purpose-built for the ~1,225 space
companies specifically and won't exist for a general adjacent-company
export.

Confirmed from Source 1 exploration: Statement 1 is reliably
the most recent filing (verified across all 1,372 space companies, zero
violations) — safe to rely on for adjacent data too, but worth spot-checking
once real files arrive rather than assuming it holds universally.

**Key finding, changes what to ask for**: Source 1 and Source 3 (the
grants/accelerator/funding enrichment file) turned out to be **the same
underlying Beauhurst export schema, pulled ~11 days apart** — not two
structurally different data types. Verified directly: both files carry
identical-looking `Grants -`/`Fundraisings -`/`Accelerator Attendances -`/
`Academic Spinout Events -`/`Growth signals -`/`Innovation signals -`
columns, and cross-checking the two exports for the same companies shows
99.2% agreement on grants count (fundraisings count agreement is lower,
40.4%, because fundraising events get added on Beauhurst's platform between
export dates — the later export is simply more current, not a different
schema). **Practical implication: the adjacent-company pull needs only ONE
Beauhurst export per company** — one that contains both the Financial
Statement blocks and the grants/accelerator/signal columns together, as
both Source 1 and Source 3 individually already do — not two separate
files. Ask for a single, reasonably current export; don't request a
second "enrichment" file unless the first one turns out to be missing
these columns.

## 2. Companies House number, where available

Populate a Companies House number field where the adjacent company has
one. This feeds directly into `company_id` (`data_prep.make_company_id`):
a real CH number produces a `ch_<number>_<normalised name>` id; without
one, `company_id` falls back to a `fallback_<hash(name+URL)>` id. Both are
valid and the pipeline handles either automatically — this is a quality
preference, not a hard blocker.

Caveat worth flagging now: **Source 1's actual raw schema, as exported
today, has no Companies House number column at all** (verified directly
against the export — only `Beauhurst URL`). If the adjacent exports are a literal
match to Source 1's schema, they may have the same gap. If whoever
generates the adjacent export can add a CH number column that Source 1
itself doesn't have, that's a real improvement (fewer fallback-hash ids,
easier manual cross-referencing) — but don't block on it; confirm whether
it's feasible before assuming it'll be there.

## 3. Mission tagging: per-file, not inferred

Each adjacent file should be **pre-tagged or filename-tagged by mission**
(ACE / Beyond Earth / Resilient Earth) — three separate exports, one per
mission, matching how the Beauhurst collections were built. Adjacent
companies do **not** need the buzzword-based mission-inference logic
planned for cross-cutting companies (PROJECT_NOTES.md "Planned: cross-cutting
predictions") — that logic exists specifically because cross-cutting
companies and (potentially) future unsegmented data lack a given mission.
Adjacent companies arrive already knowing which mission they belong to;
inferring it would just reintroduce uncertainty that a filename or a
column can remove for free.

Practical suggestion: either a `Mission` column already populated with one
of the three exact mission names used elsewhere in the pipeline (`ACE`,
`Beyond Earth`, `Resilient Earth` — see `mission_segmentation.REAL_MISSIONS`),
or a clear filename convention (e.g. `adjacent_ace.xlsx`,
`adjacent_beyond_earth.xlsx`, `adjacent_resilient_earth.xlsx`) the merge
code can key off deterministically. Either works; consistency across all
three files matters more than which convention is picked.

## 4. Prioritized column shortlist (from `feature_source_mapping.csv`)

Built by tracing every feature that actually appears in any mission's
top-15 feature weights/importances — across both the estimation bake-off
and the forecasting bake-off — back to its raw column (see
`build_feature_source_mapping.py`, run to regenerate). The point: the
adjacent-company pull does not need to replicate Source 1's full
~1,100-column raw export (10 repeated statement blocks × ~110 base
fields). It needs these ~20 base fields, confirmed to matter, ranked by
how many of the 6 (mission, bake-off-system) combinations actually use a
feature derived from each:

**Source 2-equivalent fields** (or their nearest Source 1 analogue, if the
adjacent export follows Source 1's schema per section 1 above — see
DATA_SCHEMA.md's mapping between the two):
- `Balance Sheet Total Assets` — used by 8 different top-15 features
  across both bake-offs (the single highest-value field in the dataset).
- `Total Employees` (CH/Estimated) — used by 6 top-15 features.
- `Total Turnover` — the target itself; also feeds every lag/rolling/
  growth feature in the forecasting bake-off (see below — CRITICAL,
  jointly with the weeks field).
- `Founded` — feeds `company_age_years`/`company_age`, used by 4.
- `Size` (Company Size, year-indexed) — feeds `company_size`, used by 4.
- `SIC Code 1` — used by 3 (plus 2 more specific SIC categories in the
  forecasting bake-off).
- `Total Export Revenue` — used by 2.
- `Value Stream` — used by 1 (also the mission-assignment field itself —
  required regardless of feature ranking).

**Source 1/3 combined Beauhurst-export fields** (see the schema-overlap
finding above — one export covers both groups):
- **`Financial Statement N - Number of weeks in the accounting year` —
  CRITICAL.** Not itself a top-15 feature, but required to correctly use
  `Turnover` at all: this project's own historical data had 4.0% of
  statement-years with a non-52-week accounting period (a 4-week stub up
  to an 82-week extended filing), which silently distorted every
  year-over-year growth calculation before being fixed (see
  `PROJECT_NOTES.md`'s "Filing-period annualization" section). Without
  this field, adjacent-company growth features/gazelle flags will carry
  the same unfixed distortion this project's own data had.
- `Financial Statement 1 - Return on capital employed (%)` — the one
  Source 1 ratio that reached a top-15 list (Beyond Earth, estimation).
- `Grants - Number of grants received by the company`,
  `Grants - Total amount received by the company through grants (GBP)`.
- `Fundraisings - Number of fundraisings completed by the company`.
- `Growth signals - Debt fundraising`, `Growth signals - Equity
  fundraising` (2 of the 8 non-leakage growth/innovation signals — these 2
  specifically reached a top-15 list; the other 6 confirmed-safe signals
  are still worth including for completeness but ranked lower).
- `Accelerator Attendances 1-5 - Accelerator Name` (presence/count only).
- `Academic Spinout Events 1/2 - Academic Institution Name` (presence
  only).

**Confirmed safe to skip** (per `feature_source_mapping.csv`'s "dropped"/
"never_considered" rows — 44 of Source 1's ~110 base fields, 25 of Source
2's ~44, 7 of Source 3's columns): the OECD/Beauhurst growth-rate columns
(turnover-derived leakage), internal QA/administrative fields (`HPB:
Normalised Score`, `CH Check`, `SAC Tagging`, `Comments`, `ST Type`,
`Validated (CH & Beauhurst)`), all `Space *` breakdown columns (target is
Total Turnover, never Space Turnover), `LinkedIn Industry` (redundant,
high-cardinality instability risk), the raw grant/fundraising/accelerator
NAME and DATE columns (only their presence/count is used, never the raw
text), gender pay gap fields, valuations, banking provider, charges &
mortgages, and tracking-reason columns. Whoever sources the adjacent data
should not feel obliged to replicate any of these — see
`feature_source_mapping.csv` for the complete row-by-row list and reasons.

## Update: the files arrived, initial prep done (`src/adjacent_data_prep.py`)

The 3 planned files arrived (`SatApps ACE/Beyond Earth/Resilient Earth
training data.xlsx`, 2,505/3,659/5,963 rows) and were validated against
this document's checklist — schema, mission-tagging, and field coverage
all confirmed as expected (single Source1/3-style export per mission,
100%-populated CH ID and Beauhurst URL). `src/adjacent_data_prep.py` now
builds `company_id`, parses `sic_code_1`, derives `company_size`, flags
`multi_mission_overlap`, and reconstructs an un-annualized turnover-by-
year panel from the Financial Statement blocks — see PROJECT_NOTES.md's
"Adjacent-company groundwork" section for the full write-up of each
decision. Two things confirmed missing that this checklist didn't
originally call out: no incorporation/registration-date field of any kind
(so `company_age_years` is null for every adjacent company — a Companies
House API lookup could recover it, flagged as a separate future decision,
not built), and no `Total Export Revenue` substitute (also left null).

**This is still groundwork, not the merge**: `adjacent_static_features.csv`
and `adjacent_turnover_panel.csv` are standalone outputs for review: no
adjacent row has entered `sample_construction.py`/`feature_engineering.py`/
`model_bakeoff.py` yet.

## Not covered here (deferred to the actual merge implementation)

- Exact column-name mapping from Source 1's raw fields to the feature set
  `feature_engineering.py` currently builds from Source 2.
- How `sample_weight` gets set for adjacent rows (a `ADJACENT_SAMPLE_WEIGHT`
  constant exists in `model_bakeoff.py`, unused, to be tuned empirically
  once real data exists — see its docstring).
- Company-identity reconciliation between a space company and the *same*
  company appearing in an adjacent file (unlikely but not impossible given
  ~23k rows) — not addressed; flag if it turns out to matter once real
  data is in hand.
- A Companies House API lookup to recover real incorporation dates for
  `company_age_years` (12,127 companies) — flagged as worth pursuing, not
  scoped or built.

**Update**: the filing-period annualization fix (`turnover x
52/actual_weeks`) has now been applied to the reconstructed adjacent
turnover panel — see PROJECT_NOTES.md "Adjacent-company groundwork" #6.
Unlike the space-company estimation target, this panel had no existing
in-production number to preserve, so it was corrected immediately rather
than deferred. 3,719 of 106,805 (company, year) rows were corrected
(48.1% of those by more than 30%) — no longer an open item.
