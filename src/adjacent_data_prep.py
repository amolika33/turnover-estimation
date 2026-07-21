"""Prepares the 3 per-mission adjacent-company Beauhurst exports ("SatApps
<Mission> training data.xlsx") into a feature-engineering-ready shape:
stable company_id, mission tagging with multi-mission overlap made
explicit, and the handful of Source-2-equivalent fields these files don't
carry natively (SIC code, company size, company age, export revenue).

build_mission_training_features assembles the full NUMERIC_FEATURES +
CATEGORICAL_FEATURES + TARGET_COL set model_bakeoff.py needs, per mission,
ready to concatenate onto that mission's space-company labelled_features
rows (see model_bakeoff.py's get_mission_features_with_adjacent for the
actual merge — this module builds the rows, that function combines them).
Running main() writes adjacent_training_features.csv (the merge-ready
frame) alongside the earlier review artifacts (adjacent_static_features.csv,
adjacent_turnover_panel.csv) for inspection.

Six decisions implemented this pass (see PROJECT_NOTES.md "Adjacent-company
groundwork" for the full write-up):

1. company_age_years: checked all 437 raw columns for anything
   incorporation/registration-date-equivalent — none exists (verified by
   exhaustive name search, not assumed). Left null; the pipeline's existing
   median imputation handles it like any other missing feature. A
   Companies House API lookup could recover this for real, but that's a
   separate, much larger decision (12,127 companies) — flagged, not built.
2. multi_mission_overlap: a company appearing in more than one mission
   file is kept in every mission's output (not deduplicated into one), with
   a multi_mission_overlap column naming the other mission(s) it also
   appears in — so downstream analysis can see the overlap rather than
   silently treating each appearance as independent.
3. sic_code_1: parsed as the first code in the comma-separated
   "SIC Codes (2007) - Code" string.
4. company_size: derived from `Financial Statement 1 - Number of
   employees` via SIZE_BUCKET_EDGES below (see its docstring for the
   empirical validation against space-company data).
5. total_export_revenue: no substitute field exists anywhere in the raw
   export — left null, standard imputation applies.
6. Turnover-by-year panel: reconstructed from the 10 Financial Statement
   blocks (`Date of accounts` -> year, `Turnover` -> value), the same
   anchoring convention build_source1_annualization_factors already uses
   for space companies, then corrected for non-standard accounting
   periods (`turnover x 52/actual_weeks`) via forecast_data_prep's
   existing `annualize_turnover` — reused directly, not reimplemented —
   before this panel is ever merged into training. Unlike the space-
   company estimation target (deliberately left un-annualized, see
   PROJECT_NOTES.md's "Filing-period annualization"), this panel hasn't
   entered training yet, so there's no existing reported number to
   preserve — corrected up front instead.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_prep import CH_COL, COMPANY_ID_COL, NAME_COL, URL_COL, make_company_id
from src.feature_engineering import FEATURE_COLUMNS, build_source1_ratio_features, build_source3_features
from src.mission_segmentation import MISSION_COL, REAL_MISSIONS
from src.sample_construction import ID_COLS, POPULATION_TYPE_ADJACENT, check_turnover
from forecast_src.forecast_data_prep import annualize_turnover

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "processed"

ADJACENT_PATHS = {
    "ACE": REPO_ROOT / "data" / "raw" / "SatApps ACE training data.xlsx",
    "Beyond Earth": REPO_ROOT / "data" / "raw" / "SatApps Beyond Earth training data.xlsx",
    "Resilient Earth": REPO_ROOT / "data" / "raw" / "SatApps Resilient Earth training data.xlsx",
}

RAW_CH_COL = "Companies House ID"
RAW_NAME_COL = "Company name"
RAW_SIC_COL = "SIC Codes (2007) - Code"
RAW_EMPLOYEES_COL = "Financial Statement 1 - Number of employees"

N_STATEMENTS = 10

# Micro/Small/Medium/Large by employee-count only — the standard UK/EU SME
# convention (Companies Act 2006 / EU Recommendation 2003/361/EC,
# employee-count leg). Verified this reproduces Beauhurst's own Size
# {year} bucket for space companies 96.1% of the time (2,876/2,994 rows of
# labelled_features.csv with both fields populated) — the ~4% mismatch is
# explained by company_size and total_employees coming from different
# snapshot years for the same company (a known temporal-mismatch pattern
# elsewhere in this pipeline), not a different underlying rule. Used here
# because adjacent companies have no Size {year} column at all — this is
# the closest available substitute, not an assumption-free fact.
SIZE_BUCKET_BINS = [-np.inf, 10, 50, 250, np.inf]
SIZE_BUCKET_LABELS = ["Micro", "Small", "Medium", "Large"]


def load_adjacent_source(mission: str) -> pd.DataFrame:
    return pd.read_excel(ADJACENT_PATHS[mission], sheet_name=0)


def assign_company_id(df: pd.DataFrame) -> pd.DataFrame:
    """Reuses data_prep.make_company_id (identical CH-prefixed / fallback-
    hash logic as space companies) by renaming this export's columns to
    the canonical names it expects. Companies House ID is 100% populated
    in all 3 adjacent files (verified), so every row here gets a real
    ch_<number>_<name> id, not a fallback hash."""
    renamed = df.rename(columns={RAW_CH_COL: CH_COL, RAW_NAME_COL: NAME_COL})
    # One row (ACE, CH 8264515) has its company name read back as the
    # Python bool True rather than a string — Excel/openpyxl auto-typing a
    # literal name like "True" as a checkbox value. Cast defensively so
    # make_company_id's normalize_name (which calls .lower() on this
    # column) doesn't choke on it.
    renamed[NAME_COL] = renamed[NAME_COL].astype(str)
    df = df.copy()
    df[COMPANY_ID_COL] = make_company_id(renamed)
    return df


def compute_multi_mission_overlap(frames: dict) -> pd.DataFrame:
    """One row per (company_id, mission) appearance across all 3 files,
    with multi_mission_overlap listing the OTHER mission(s) that company_id
    also appears under (empty string if none). Deliberately not deduped
    down to one row per company — see module docstring, decision 2."""
    appearances = pd.concat(
        [pd.DataFrame({COMPANY_ID_COL: df[COMPANY_ID_COL], "mission": mission}) for mission, df in frames.items()],
        ignore_index=True,
    ).drop_duplicates()

    missions_by_company = appearances.groupby(COMPANY_ID_COL)["mission"].apply(list)
    appearances = appearances.merge(
        missions_by_company.rename("_all_missions"), on=COMPANY_ID_COL, how="left"
    )
    appearances["multi_mission_overlap"] = appearances.apply(
        lambda row: ", ".join(sorted(m for m in row["_all_missions"] if m != row["mission"])), axis=1
    )
    return appearances.drop(columns="_all_missions")


def parse_sic_code_1(series: pd.Series) -> pd.Series:
    """"SIC Codes (2007) - Code" is a comma-separated multi-code string
    (e.g. "58110, 58142, 58190, 58290"), unlike Source 2's single
    sic_code_1 value per company. Takes the first code only, matching the
    "primary SIC code" convention sic_code_1 represents elsewhere."""
    first = series.astype("string").str.split(",").str[0].str.strip()
    return pd.to_numeric(first, errors="coerce")


def build_company_size(employees: pd.Series) -> pd.Series:
    sizes = pd.cut(employees, bins=SIZE_BUCKET_BINS, labels=SIZE_BUCKET_LABELS, right=False)
    return sizes.astype("object")


def build_static_features(df: pd.DataFrame, mission: str, overlap_lookup: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            COMPANY_ID_COL: df[COMPANY_ID_COL],
            NAME_COL: df[RAW_NAME_COL],
            URL_COL: df[URL_COL],
            CH_COL: df[RAW_CH_COL],
            "mission": mission,
            "sic_code_1": parse_sic_code_1(df[RAW_SIC_COL]),
            "company_size": build_company_size(df[RAW_EMPLOYEES_COL]),
            # Decisions 1 and 5: no substitute field exists for either —
            # null rather than fabricated, same as any other missing
            # numeric feature the imputer already handles.
            "company_age_years": np.nan,
            "total_export_revenue": np.nan,
        }
    )
    mission_overlap = overlap_lookup[overlap_lookup["mission"] == mission][[COMPANY_ID_COL, "multi_mission_overlap"]]
    out = out.merge(mission_overlap, on=COMPANY_ID_COL, how="left")
    return out


def build_turnover_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstructs a (company_id, year, turnover, weeks, total_employees,
    balance_sheet_total_assets) long panel from the 10 Financial Statement
    blocks — the adjacent-file equivalent of Source 2's Total Turnover
    {year}/Total Employees {year}/Balance Sheet Total Assets {year}
    columns, which don't exist here. `Financial Statement N - Number of
    employees` / `- Total assets` are used as direct substitutes — checked
    against real data before relying on them (not assumed): for the 2,377
    space-company (company, year) rows where both a Source 1 FS value and
    Source 2's own year-indexed value exist, Total assets matches
    Balance Sheet Total Assets with 0.999 correlation (91% within 1%) and
    Number of employees matches total_employees with 0.982 correlation
    (96% within 5%) — both a good enough substitute to use directly, not
    a like-for-like identical field.

    Turnover here is still raw (un-annualized) — annualize_adjacent_panel
    applies the correction afterward, once across the combined multi-
    mission panel (see its docstring for why it can't be done per-mission
    here). Employees/assets are NOT annualized (a per-statement snapshot,
    not a flow figure like turnover — annualizing a stub period doesn't
    apply to a point-in-time balance sheet figure)."""
    frames = []
    for i in range(1, N_STATEMENTS + 1):
        date_col = f"Financial Statement {i} - Date of accounts"
        turnover_col = f"Financial Statement {i} - Turnover"
        weeks_col = f"Financial Statement {i} - Number of weeks in the accounting year"
        employees_col = f"Financial Statement {i} - Number of employees"
        assets_col = f"Financial Statement {i} - Total assets"
        needed = {date_col, turnover_col, weeks_col, employees_col, assets_col}
        if not needed.issubset(df.columns):
            continue
        sub = df[[COMPANY_ID_COL, date_col, turnover_col, weeks_col, employees_col, assets_col]].dropna(
            subset=[date_col]
        ).copy()
        sub = sub.rename(
            columns={
                date_col: "date_of_accounts",
                turnover_col: "turnover",
                weeks_col: "weeks",
                employees_col: "total_employees",
                assets_col: "balance_sheet_total_assets",
            }
        )
        sub["year"] = sub["date_of_accounts"].dt.year
        frames.append(
            sub[[COMPANY_ID_COL, "year", "turnover", "weeks", "total_employees", "balance_sheet_total_assets"]]
        )

    columns = [COMPANY_ID_COL, "year", "turnover", "weeks", "total_employees", "balance_sheet_total_assets"]
    if not frames:
        return pd.DataFrame(columns=columns)

    allrows = pd.concat(frames, ignore_index=True).dropna(subset=["year"])
    # Statement 1 is confirmed most recent (DATA_SCHEMA.md) — iterating
    # 1..10 in order and keeping the first (company_id, year) match means a
    # more recent statement wins if two statements ever resolve to the
    # same calendar year, same precedent as build_source1_ratio_features.
    return allrows.drop_duplicates(subset=[COMPANY_ID_COL, "year"])


def build_adjacent_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reuses feature_engineering.build_source1_ratio_features unchanged —
    adjacent files carry the identical Financial Statement 1 ratio columns
    (same schema, verified in the original file-validation pass), so the
    same leakage-checked ratio selection and Statement-1 anchoring apply
    as-is. Only the identity step differs: adjacent files already carry a
    directly-assigned company_id (from their own 100%-populated CH ID), so
    no Beauhurst-URL lookup is needed the way space companies require."""
    ratios = build_source1_ratio_features(df)
    ratios[COMPANY_ID_COL] = df.loc[ratios.index, COMPANY_ID_COL]
    return ratios.drop(columns="_url_norm")


def build_adjacent_source3_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reuses feature_engineering.build_source3_features unchanged — same
    reasoning as build_adjacent_ratio_features. Company-constant (not
    year-anchored): recency relative to each panel row's year is added
    separately by add_recency below, same split as merge_source3_features."""
    signals = build_source3_features(df)
    signals[COMPANY_ID_COL] = df.loc[signals.index, COMPANY_ID_COL]
    return signals.drop(columns="_url_norm")


def add_recency(panel: pd.DataFrame) -> pd.DataFrame:
    """Same recency-relative-to-row-year logic as
    feature_engineering.merge_source3_features's tail: a grant/fundraising
    dated after a given panel row's year hasn't happened yet from that
    row's point of view, so it's nulled rather than left as a
    leakage-implying negative recency."""
    panel = panel.copy()
    panel["grant_recency_years"] = panel["year"] - panel["_grants_latest_date"].dt.year
    panel.loc[panel["grant_recency_years"] < 0, "grant_recency_years"] = np.nan
    panel["fundraising_recency_years"] = panel["year"] - panel["_fundraising_latest_date"].dt.year
    panel.loc[panel["fundraising_recency_years"] < 0, "fundraising_recency_years"] = np.nan
    return panel.drop(columns=["_grants_latest_date", "_fundraising_latest_date"])


def annualize_adjacent_panel(panel_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applies the existing annualize_turnover (forecast_data_prep.py) to
    the combined, multi-mission adjacent turnover panel — reused directly,
    not reimplemented, same `turnover x 52/weeks` correction already
    proven on space-company data.

    Must run on the COMBINED panel (all 3 missions concatenated), not
    per-mission: a multi-mission-overlap company (decision 2) contributes
    one turnover-panel row per mission it appears in, so the same
    (company_id, year) pair can appear more than once in panel_all. The
    `factors` lookup annualize_turnover expects must be deduplicated to
    exactly one row per (company_id, year) — built here directly from
    this panel's own `weeks` column (no separate Source 1 join needed,
    unlike the space-company case, since turnover and weeks already come
    from the same Financial Statement row) — otherwise its merge would
    fan out across mission copies of the same company-year.

    Returns (annualized_panel, log) — log records only the genuinely
    corrected rows (raw_turnover, weeks, annualization_factor,
    annualized_turnover), same shape annualize_turnover always returns."""
    factors = (
        panel_all[[COMPANY_ID_COL, "year", "weeks"]]
        .drop_duplicates(subset=[COMPANY_ID_COL, "year"])
        .assign(annualization_factor=lambda d: 52.0 / d["weeks"])
    )
    df_for_annualize = panel_all.drop(columns=["weeks"])
    adjusted, log = annualize_turnover(df_for_annualize, "turnover", "year", factors, id_col=COMPANY_ID_COL)
    return adjusted, log


def build_mission_training_features(
    mission: str, raw_df: pd.DataFrame, annualized_panel: pd.DataFrame, overlap_lookup: pd.DataFrame
) -> pd.DataFrame:
    """Assembles one row per (company_id, year) adjacent-company training
    example for `mission`, with every column model_bakeoff.py's
    NUMERIC_FEATURES + CATEGORICAL_FEATURES + TARGET_COL + GROUP_COL +
    WEIGHT_COL + MISSION_COL + population_type expects — built to be
    concatenated directly onto that mission's space-company
    labelled_features rows (mismatched columns, e.g. total_employees_ch/
    founded_year which adjacent data has no equivalent of, are left for
    pandas.concat to fill with NaN; nothing downstream reads those on the
    adjacent side).

    `annualized_panel` must already be the 3-mission-combined, annualized
    turnover panel (see annualize_adjacent_panel) filtered to this mission
    — annualization cannot be done per-mission in isolation because of the
    multi-mission-overlap duplication (decision 2).

    sample_weight here is the UNSCALED base weight (1 / that company's row
    count within this mission, same inverse-frequency principle as
    sample_construction.build_long_panel for space companies) —
    ADJACENT_SAMPLE_WEIGHT is applied later, at merge/bake-off time, so
    tuning it doesn't require rebuilding these Excel-derived features."""
    panel = annualized_panel.rename(columns={"turnover": "total_turnover"}).drop(columns=["mission"])

    ratios = build_adjacent_ratio_features(raw_df)
    panel = panel.merge(ratios, on=[COMPANY_ID_COL, "year"], how="left")

    signals = build_adjacent_source3_features(raw_df)
    panel = panel.merge(signals, on=COMPANY_ID_COL, how="left")
    panel = add_recency(panel)

    static = build_static_features(raw_df, mission, overlap_lookup)
    static_cols = [
        COMPANY_ID_COL,
        NAME_COL,
        URL_COL,
        CH_COL,
        "sic_code_1",
        "company_age_years",
        "total_export_revenue",
        "multi_mission_overlap",
    ]
    panel = panel.merge(static[static_cols], on=COMPANY_ID_COL, how="left")

    # company_size is recomputed per (company_id, year) from this row's own
    # total_employees value, NOT taken from build_static_features' single
    # FS1 snapshot — matches how space companies get a year-varying
    # company_size from Size {year}, rather than one static value repeated
    # across every year.
    panel["company_size"] = build_company_size(panel["total_employees"])
    # No substitute exists for Value Stream anywhere in the adjacent
    # export (checked in the original file-validation pass) — left null,
    # same treatment as company_age_years/total_export_revenue.
    panel["value_stream"] = np.nan
    panel["employee_count_source"] = pd.NA
    panel.loc[panel["total_employees"].notna(), "employee_count_source"] = "filed"

    with np.errstate(divide="ignore", invalid="ignore"):
        panel["assets_per_employee"] = panel["balance_sheet_total_assets"] / panel["total_employees"]
        panel["export_revenue_per_employee"] = panel["total_export_revenue"] / panel["total_employees"]
    panel["assets_per_employee"] = panel["assets_per_employee"].replace([np.inf, -np.inf], np.nan)
    panel["export_revenue_per_employee"] = panel["export_revenue_per_employee"].replace([np.inf, -np.inf], np.nan)

    panel[MISSION_COL] = mission
    panel["population_type"] = POPULATION_TYPE_ADJACENT
    panel["sample_weight"] = 1.0 / panel.groupby(COMPANY_ID_COL)[COMPANY_ID_COL].transform("count")

    panel, turnover_quality_log = check_turnover(panel, col="total_turnover")
    panel = panel[panel["total_turnover"].notna()].copy()

    return panel, turnover_quality_log


def build_all_adjacent_training_features() -> pd.DataFrame:
    """Runs the full pipeline (load raw files, assign company_id, build
    the turnover panel, annualize it once across all 3 missions, then
    build each mission's full training-feature set) and returns ONE
    concatenated dataframe across all 3 missions.

    sample_weight is left as the UNSCALED base weight (1 / company row
    count) — model_bakeoff.get_mission_features_with_adjacent applies
    ADJACENT_SAMPLE_WEIGHT at merge time, so this Excel-derived build
    (the expensive part) only needs to run once regardless of how many
    candidate weights get tried."""
    raw = {mission: assign_company_id(load_adjacent_source(mission)) for mission in REAL_MISSIONS}
    overlap_lookup = compute_multi_mission_overlap(raw)

    panel_frames = [build_turnover_panel(raw[mission]).assign(mission=mission) for mission in REAL_MISSIONS]
    panel_all = pd.concat(panel_frames, ignore_index=True)
    panel_all, _ = annualize_adjacent_panel(panel_all)

    feature_frames = []
    for mission in REAL_MISSIONS:
        panel_m = panel_all[panel_all["mission"] == mission].copy()
        features_m, _ = build_mission_training_features(mission, raw[mission], panel_m, overlap_lookup)
        feature_frames.append(features_m)
    return pd.concat(feature_frames, ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = {mission: assign_company_id(load_adjacent_source(mission)) for mission in REAL_MISSIONS}
    overlap_lookup = compute_multi_mission_overlap(raw)

    static_frames = []
    panel_frames = []
    for mission in REAL_MISSIONS:
        df = raw[mission]
        static = build_static_features(df, mission, overlap_lookup)
        panel = build_turnover_panel(df).assign(mission=mission)
        static_frames.append(static)
        panel_frames.append(panel)

        n_overlap = (static["multi_mission_overlap"] != "").sum()
        print(f"\n=== {mission} ===")
        print(f"Rows: {len(static)}")
        print(f"Companies with a company_age_years value: {static['company_age_years'].notna().sum()} (expected 0 — see decision 1)")
        print(f"sic_code_1 parsed: {static['sic_code_1'].notna().sum()}/{len(static)}")
        print("company_size distribution:", static["company_size"].value_counts(dropna=False).to_dict())
        print(f"Also appears in another mission file: {n_overlap}")
        print(f"Turnover-by-year panel rows: {len(panel)} across {panel[COMPANY_ID_COL].nunique()} companies")

    static_all = pd.concat(static_frames, ignore_index=True)
    panel_all = pd.concat(panel_frames, ignore_index=True)

    n_raw = len(panel_all)
    n_non52 = (panel_all["weeks"] != 52).sum()
    panel_all, annualization_log = annualize_adjacent_panel(panel_all)

    print("\n=== Filing-period annualization (adjacent panel) ===")
    print(f"Total (company, year) rows: {n_raw}")
    print(f"Non-52-week rows: {n_non52} ({n_non52 / n_raw:.1%})")
    print(f"Rows actually corrected (factor != 1.0 and turnover present): {len(annualization_log)}")
    if len(annualization_log):
        pct_distortion = (annualization_log["annualization_factor"] - 1).abs()
        n_over_30pct = (pct_distortion > 0.30).sum()
        print(f"Rows with >30% distortion: {n_over_30pct} ({n_over_30pct / len(annualization_log):.1%} of corrected rows)")
        print(f"Distortion factor range: {annualization_log['annualization_factor'].min():.2f}x - {annualization_log['annualization_factor'].max():.2f}x")
        print(f"Distinct companies affected: {annualization_log[COMPANY_ID_COL].nunique()}")
        by_mission = annualization_log.merge(
            panel_all[[COMPANY_ID_COL, "year", "mission"]], on=[COMPANY_ID_COL, "year"], how="left"
        )
        print("Corrected rows by mission:", by_mission["mission"].value_counts().to_dict())

    static_path = OUTPUT_DIR / "adjacent_static_features.csv"
    panel_path = OUTPUT_DIR / "adjacent_turnover_panel.csv"
    log_path = OUTPUT_DIR / "adjacent_turnover_annualization_log.csv"
    static_all.to_csv(static_path, index=False)
    panel_all.to_csv(panel_path, index=False)
    annualization_log.to_csv(log_path, index=False)
    print(f"\nWrote {static_path} ({len(static_all)} rows, {static_all[COMPANY_ID_COL].nunique()} distinct companies)")
    print(f"Wrote {panel_path} ({len(panel_all)} rows, turnover now annualized)")
    print(f"Wrote {log_path} ({len(annualization_log)} corrected rows)")

    # Stage 1: full per-mission training-feature parity (see
    # build_mission_training_features) — the actual rows that will merge
    # into each mission's training population (task: real integration).
    print("\n=== Full training-feature build (per mission) ===")
    feature_frames = []
    quality_logs = []
    for mission in REAL_MISSIONS:
        panel_m = panel_all[panel_all["mission"] == mission].copy()
        features_m, quality_log_m = build_mission_training_features(mission, raw[mission], panel_m, overlap_lookup)
        feature_frames.append(features_m)
        quality_logs.append(quality_log_m)

        missing_expected = [c for c in FEATURE_COLUMNS if c not in features_m.columns]
        print(f"\n{mission}: {len(features_m)} training rows, {features_m[COMPANY_ID_COL].nunique()} companies")
        print(f"  Missing expected feature columns: {missing_expected or 'none'}")
        print(f"  Turnover values nulled/excluded by check_turnover: {len(quality_log_m)}")
        for c in FEATURE_COLUMNS:
            if c in features_m.columns:
                print(f"  {c:<28} non-null: {features_m[c].notna().sum()}/{len(features_m)}")

    features_all = pd.concat(feature_frames, ignore_index=True)
    quality_log_all = pd.concat(quality_logs, ignore_index=True) if quality_logs else pd.DataFrame()

    features_path = OUTPUT_DIR / "adjacent_training_features.csv"
    quality_log_path = OUTPUT_DIR / "adjacent_turnover_quality_log.csv"
    features_all.to_csv(features_path, index=False)
    quality_log_all.to_csv(quality_log_path, index=False)
    print(f"\nWrote {features_path} ({len(features_all)} rows)")
    print(f"Wrote {quality_log_path} ({len(quality_log_all)} rows)")


if __name__ == "__main__":
    main()
