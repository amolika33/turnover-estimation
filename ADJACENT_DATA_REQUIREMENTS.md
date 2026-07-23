# Adjacent-Company Data Requirements

**Historical note**: sections 1-4 below were written before any adjacent
data existed, as a spec for what to ask for. All 6 adjacent files have
since arrived (3 original + 3 later targeted-sourcing batches) and the
merge is fully built (`src/adjacent_data_prep.py`, wired into
`src/model_bakeoff.py`/`src/model_selection.py`/`src/predict.py`) — kept
here as-written for the historical record of what was originally
specified, with "Update" sections and a "Schema variations" section added
below documenting what was actually found once real files existed. See
PROJECT_NOTES.md's "Adjacent-company groundwork", "Adjacent-company
integration: bake-off results", and "Extended validation round" sections
for the full build/validation/decision narrative.

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

## Not covered here (deferred to the actual merge implementation) — STATUS UPDATE

All 4 items below were genuinely open when this section was first written.
All 4 are now resolved:

- ~~Exact column-name mapping from Source 1's raw fields to the feature set
  `feature_engineering.py` currently builds from Source 2.~~ **Done** —
  `src/adjacent_data_prep.py`'s `build_mission_training_features` (and the
  `build_adjacent_ratio_features`/`build_adjacent_source3_features` helpers
  it calls) is the full mapping, reusing `feature_engineering.py`'s own
  Source 1/3 builders unchanged.
- ~~How `sample_weight` gets set for adjacent rows...~~ **Done, tuned, and
  actively used** — `ADJACENT_SAMPLE_WEIGHT` is no longer an unused
  placeholder. Empirically tuned per mission (ACE=0.2, Beyond Earth=1.0,
  Resilient Earth=0.2) via a 3-model proxy sweep — see PROJECT_NOTES.md
  "Adjacent-company integration: bake-off results".
- ~~Company-identity reconciliation between a space company and the same
  company appearing in an adjacent file...~~ **Done, and it DID turn out
  to matter**: `exclude_space_company_overlap` in `adjacent_data_prep.py`
  checks every adjacent row's CH number against the full space-company
  population before it enters training — found 49 (ACE), 13 (Beyond
  Earth), 27 (Resilient Earth), 6 (Cross-cutting) rows that were actually
  already-labelled real space companies (including some in the ORIGINAL 3
  files too, not just the later targeted-sourcing batches — a gap that
  had gone unchecked until this was built). These are dropped from the
  adjacent pool entirely rather than double-counted under a weaker tag.
  See PROJECT_NOTES.md "Extended validation round" and this doc's own
  "Schema variations across the 6 adjacent files" section below.
- A Companies House API lookup to recover real incorporation dates for
  `company_age_years` (12,127 companies) — **still not built**, but worth
  re-flagging: the 3 targeted-sourcing batches (added later, see below)
  natively carry an `Incorporation date (Companies House)` field that the
  original 3 files never had — this could resolve `company_age_years` for
  those specific companies without any CH-API lookup at all. Not
  implemented (out of scope for the targeted-sourcing integration that
  found it) — flagged as a real, comparatively cheap follow-up opportunity
  for whoever picks this up next.

## Schema variations across the 6 adjacent files (found during the targeted-sourcing integration)

3 additional files arrived after the original 3 (`SatApps Extra ACE
training data.csv` — 566 rows deep-tech batch, `SatApps Extra Resilient
Earth training data.csv` — 624 rows hardware/photonics batch, `SatApps
Cross Cutting training data.csv` — 607 rows, plus a separate `SatApps
Extra ACE training data - Grants&Fundraising.csv` side-table). Validating
these against this document's original checklist surfaced real schema
inconsistencies **not just against the original 3 files, but among the 3
new files themselves** — the assumption that all adjacent exports share
one common schema turned out to be wrong once a second sourcing round
actually happened:

- **SIC code column**: the original 3 files and the new ACE batch use one
  comma-separated `SIC Codes (2007) - Code` column (`sic_style: "single"`
  in `adjacent_data_prep.py`'s `ADJACENT_SOURCES`). The new Resilient
  Earth and Cross-cutting batches instead use 4 numbered slots
  (`SIC Codes (2007) 1-4 - Code`, `sic_style: "numbered"`) — normalised to
  the single-column convention at load time (`_load_raw_file`) so
  `parse_sic_code_1` doesn't need to know which style it received.
- **Accelerator slots**: the original 3 files and the new ACE/Resilient
  Earth batches have 7 slots; the new Cross-cutting batch has only 4.
  `feature_engineering.py`'s `build_source3_features` now uses whichever
  slot columns a frame actually has (`accelerator_cols = [c for c in
  SOURCE3_ACCELERATOR_NAME_COLS if c in df.columns]`) instead of assuming
  a fixed count.
- **Grants/fundraising fields**: present in the new Resilient Earth and
  Cross-cutting batches, but **absent** from the new ACE batch — resolved
  by joining a separate side-table (`SatApps Extra ACE training data -
  Grants&Fundraising.csv`, 566 rows, joined on Beauhurst URL, verified
  566/566 exact match with zero name mismatches) rather than leaving ACE's
  new companies with null grants data.
- **CSV vs Excel**: the original 3 files are `.xlsx` (dates auto-parsed by
  `pandas.read_excel`); all 3 new batches are `.csv`, which needed
  explicit `pd.to_datetime` parsing for the Financial Statement date
  columns and the grants/fundraising latest-date columns before any
  `.dt` access downstream — handled in `_load_raw_file`'s `_CSV_DATE_COLS`
  list.
- **Within-mission duplicates**: a company appearing in both the original
  file and its mission's new batch (verified: 33 for ACE, 4 for Resilient
  Earth) is resolved by keeping the row from whichever file is listed
  FIRST in `ADJACENT_SOURCES` — the original, already weight-tuned file
  takes precedence.

See `src/adjacent_data_prep.py`'s `ADJACENT_SOURCES` registry for the
authoritative per-file configuration, and PROJECT_NOTES.md's "Extended
validation round" section for the full validation/integration narrative
and the Stage C bake-off results these batches produced (all 3
inconclusive/negative — see the "diminishing-returns finding").

**Update**: the filing-period annualization fix (`turnover x
52/actual_weeks`) has now been applied to the reconstructed adjacent
turnover panel — see PROJECT_NOTES.md "Adjacent-company groundwork" #6.
Unlike the space-company estimation target, this panel had no existing
in-production number to preserve, so it was corrected immediately rather
than deferred. 3,719 of 106,805 (company, year) rows were corrected
(48.1% of those by more than 30%) — no longer an open item.
