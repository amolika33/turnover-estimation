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
