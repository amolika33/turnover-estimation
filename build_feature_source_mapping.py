"""Maps every engineered feature that actually appears in any mission's
top-15 feature weights/importances — across BOTH bake-off systems
(src/model_bakeoff.py's estimation bake-off, forecast_src/forecast_bakeoff.py's
one-year-ahead forecasting bake-off) — back to its raw source column and
source file (Source 1 Beauhurst raw export, Source 2 SCC master sheet,
Source 3 grants/accelerator/funding enrichment).

Purpose: the ~23k adjacent-company data pull doesn't need to replicate
Source 1's full ~1,100-column raw export (per-year/per-statement
repetition of ~110 base fields) if most of those fields have never once
been used, dropped for leakage, or turned out unimportant. This report
ranks raw columns by how many of the 6 (mission, bake-off-system)
combinations actually use a feature derived from them, so the adjacent
data pull can prioritize what's proven to matter.

WHY THE RAW-COLUMN MAPPING IS HARDCODED, not derived generically: which
raw column feeds which engineered feature is domain knowledge baked into
feature_engineering.py / forecast_feature_engineering.py's construction
logic (STATIC_COLS, SOURCE3_SAFE_BOOLEAN_SIGNALS, SOURCE1_SAFE_RATIO_COLUMNS,
the year-indexed melts, the lag/rolling/growth derivations) — there's no
generic way to recover "assets_per_employee = balance_sheet_total_assets /
total_employees, both Source 2" from introspecting a fitted model's
feature name alone. RAW_COLUMN_MAP below is that mapping, verified against
the real column lists in data/raw/ and against feature_engineering.py's own
DROPPED_COLUMNS dict, not guessed.

Top-15 lists ARE computed live (not hardcoded) — estimation from the
already-saved feature_weights_{mission}.csv, forecasting by refitting each
mission's best non-benchmark ML candidate (forecast_bakeoff's own Persistence
benchmark wins in every mission, so it has no feature weights of its own —
see PROJECT_NOTES.md's "2030 Forecasting Pipeline" section; the next-best
ML candidate is what's introspected here, since it's the most informative
stand-in for "which covariates carry real one-year-ahead signal").
"""
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_src.forecast_bakeoff import BENCHMARK_MODELS, get_mission_training_data
from forecast_src.forecast_selection import fit_final_ml_model
from src.feature_engineering import DROPPED_COLUMNS
from src.mission_segmentation import REAL_MISSIONS

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "data" / "processed"
TOP_N = 15
NONZERO_EPS = 1e-9

# ---------------------------------------------------------------------------
# Raw column -> (source_file, engineered feature name(s), notes) mapping.
# Verified against feature_engineering.py, forecast_feature_engineering.py,
# and the real column lists in data/raw/, during this investigation.
# ---------------------------------------------------------------------------

# Engineered feature (as it appears, stripped of ColumnTransformer prefixes
# and one-hot suffixes) -> (source_file, raw_column_or_base_field, notes).
# A feature with a `_derived` note is a RATIO/DIFFERENCE of other raw
# columns, not a raw column read directly.
FEATURE_TO_RAW = {
    # --- Estimation-side (src/feature_engineering.py) ---
    "balance_sheet_total_assets": ("Source 2", "Balance Sheet Total Assets {year}", ""),
    "total_employees": ("Source 2", "Total Employees (CH {year}) / Total Employees (Est. {year})", "employee_count_source flags which"),
    "company_age_years": ("Source 2", "Founded", "derived: year - Founded"),
    "signal_debt_fundraising": ("Source 3", "Growth signals - Debt fundraising", ""),
    "is_academic_spinout": ("Source 3", "Academic Spinout Events 1/2 - Academic Institution Name", "derived: any non-null"),
    "fundraising_count": ("Source 3", "Fundraisings - Number of fundraisings completed by the company", ""),
    "assets_per_employee": ("Source 2", "Balance Sheet Total Assets {year} / Total Employees {year}", "derived ratio, both Source 2"),
    "accelerator_count": ("Source 3", "Accelerator Attendances 1-5 - Accelerator Name", "derived: count of non-null"),
    "fs1_roce_pct": ("Source 1", "Financial Statement 1 - Return on capital employed (%)", ""),
    "company_size": ("Source 2", "Size {year}", ""),
    "value_stream": ("Source 2", "Value Stream", "also the mission-assignment field"),
    "sic_code_1": ("Source 2", "SIC Code 1", ""),
    "signal_equity_fundraising": ("Source 3", "Growth signals - Equity fundraising", ""),
    "total_export_revenue": ("Source 2", "Total Export Revenue {year}", ""),
    "grants_count": ("Source 3", "Grants - Number of grants received by the company", ""),
    "export_revenue_per_employee": ("Source 2", "Total Export Revenue {year} / Total Employees {year}", "derived ratio, both Source 2"),
    "grants_total_amount": ("Source 3", "Grants - Total amount received by the company through grants (GBP)", ""),
    "year": ("Source 2", "(panel year index)", "not a raw column — the year-indexed column structure itself"),
    # --- Forecasting-side (forecast_src/forecast_feature_engineering.py) ---
    # All turnover_* / rolling_turnover_* / historical_turnover_* features
    # derive from the SAME underlying value: Source 2's Total Turnover
    # {year}, corrected for non-standard filing periods using Source 1's
    # Number of weeks in the accounting year (see PROJECT_NOTES.md's
    # "Filing-period annualization" section) — a genuine TWO-source
    # dependency, not just Source 2 alone.
    "turnover_t": ("Source 2 + Source 1", "Total Turnover {year} (annualized via Source 1 weeks field)", "current year's turnover"),
    "turnover_lag_1": ("Source 2 + Source 1", "Total Turnover {year} (annualized)", "1-year lag"),
    "turnover_lag_2": ("Source 2 + Source 1", "Total Turnover {year} (annualized)", "2-year lag"),
    "turnover_lag_3": ("Source 2 + Source 1", "Total Turnover {year} (annualized)", "3-year lag"),
    "rolling_turnover_mean_2": ("Source 2 + Source 1", "Total Turnover {year} (annualized)", "derived: 2-year rolling mean"),
    "rolling_turnover_mean_3": ("Source 2 + Source 1", "Total Turnover {year} (annualized)", "derived: 3-year rolling mean"),
    "rolling_turnover_median_3": ("Source 2 + Source 1", "Total Turnover {year} (annualized)", "derived: 3-year rolling median"),
    "historical_turnover_max": ("Source 2 + Source 1", "Total Turnover {year} (annualized)", "derived: expanding max"),
    "historical_turnover_min": ("Source 2 + Source 1", "Total Turnover {year} (annualized)", "derived: expanding min"),
    "log_growth_1y": ("Source 2 + Source 1", "Total Turnover {year} (annualized)", "derived: log1p difference"),
    "growth_volatility": ("Source 2 + Source 1", "Total Turnover {year} (annualized)", "derived: rolling std of log growth"),
    "growth_acceleration": ("Source 2 + Source 1", "Total Turnover {year} (annualized)", "derived: change in growth rate"),
    "employees": ("Source 2", "Total Employees (CH {year}) / Total Employees (Est. {year})", ""),
    "employee_growth": ("Source 2", "Total Employees {year}", "derived: log1p difference"),
    "has_employee_data": ("Source 2", "Total Employees {year}", "derived: missingness flag"),
    "total_assets": ("Source 2", "Balance Sheet Total Assets {year}", ""),
    "export_revenue": ("Source 2", "Total Export Revenue {year}", ""),
    "asset_growth": ("Source 2", "Balance Sheet Total Assets {year}", "derived: log1p difference"),
    "company_age": ("Source 2", "Founded", "derived: year - Founded"),
    "history_span_years": ("Source 2", "Total Turnover {year} (panel presence)", "derived: latest - first turnover year"),
}

# Raw columns/base fields considered and explicitly excluded, with reasons —
# these are the "safe to skip" candidates. Pulled directly from
# feature_engineering.py's DROPPED_COLUMNS dict (verbatim reasons) plus the
# Source 2 "internal/administrative" columns DATA_SCHEMA.md already
# excludes outright (never even considered as modelling candidates).
SOURCE1_DROPPED = {
    "Financial Statement N - Pretax profit margin (%)": DROPPED_COLUMNS["Financial Statement 1 - Pretax profit margin (%)"],
    "Financial Statement N - Debtor days": DROPPED_COLUMNS["Financial Statement 1 - Debtor days"],
    "Financial Statement N - Creditor days": DROPPED_COLUMNS["Financial Statement 1 - Creditor days"],
    "Financial Statement N - Exports turnover ratio (%)": DROPPED_COLUMNS["Financial Statement 1 - Exports turnover ratio (%)"],
    "Financial Statement N - Sales networking capital": DROPPED_COLUMNS["Financial Statement 1 - Sales networking capital"],
    "Financial Statement N - Stock turnover ratio (%)": DROPPED_COLUMNS["Financial Statement 1 - Stock turnover ratio (%)"],
}
# Source 1's ~80 non-Financial-Statement "company info" columns (gender pay
# gap, valuations, banking provider, charges & mortgages, tracking reasons,
# etc.) were never brought in as features at all — no DROPPED_COLUMNS entry
# exists because they were never candidates in the first place, not because
# they were considered and rejected. Distinct status from the leakage/
# redundancy exclusions above.
SOURCE1_NEVER_CONSIDERED_NOTE = (
    "Never evaluated as a feature candidate (not a rejected candidate) — "
    "~72 of Source 1's 80 non-Financial-Statement columns (gender pay gap, "
    "valuations, banking provider, charges & mortgages, tracking reasons, "
    "high-growth-list cohort details, exit/acquisition prices, etc.). "
    "Grants/Fundraisings/Accelerator/Academic-Spinout/Growth-Innovation-"
    "signal columns ARE used, but sourced from Source 3's later export, not "
    "Source 1's — see the Source 1 vs Source 3 overlap note below."
)

SOURCE2_DROPPED = {
    "Average Average": DROPPED_COLUMNS["Average Average"],
    "Average Employee Growth": DROPPED_COLUMNS["Average Employee Growth"],
    "Average Turnover Growth": DROPPED_COLUMNS["Average Turnover Growth"],
    "Employee Growth Rate (OECD)": DROPPED_COLUMNS["Employee Growth Rate (OECD)"],
    "Filing Date (year)": DROPPED_COLUMNS["Filing Date (year)"],
    "LinkedIn Industry": DROPPED_COLUMNS["LinkedIn Industry"],
    "LinkedIn Specialties (Keywords)": DROPPED_COLUMNS["LinkedIn Specialties (Keywords)"] + " (NOTE: still used, but only for cross_cutting_prediction.py's buzzword mission-assignment scoring, never as an ML feature)",
    "Latest 3 Years: Growth Rate (OECD-Esq: 10%)": DROPPED_COLUMNS["Latest 3 Years: Growth Rate (OECD-Esq: 10%)"],
    "Latest 3 Years: Growth Rate (OECD: 20%)": DROPPED_COLUMNS["Latest 3 Years: Growth Rate (OECD: 20%)"],
    "Company Size / Size (Power BI) / Size (LinkedIn)": DROPPED_COLUMNS["Company Size / Size (Power BI) / Size (LinkedIn)"],
    "SIC Code 2-4": DROPPED_COLUMNS["SIC Code 2-4"],
    "Sector CAGR": DROPPED_COLUMNS["Sector CAGR"],
    "Turnover Growth Rate (OECD)": DROPPED_COLUMNS["Turnover Growth Rate (OECD)"],
}
SOURCE2_NEVER_CONSIDERED = {
    "CH Check": "internal QA field (DATA_SCHEMA.md 'Excluded columns' — administrative, not predictive signal)",
    "Comments": "internal QA field, same as above",
    "HPB: Normalised Score": "internal working field, explicitly excluded (DATA_SCHEMA.md)",
    "No. of Tags Attributed": "internal QA field",
    "S&H": "meaning unclear, excluded regardless (DATA_SCHEMA.md)",
    "SAC Tagging": "internal QA field",
    "ST Type": "internal QA field",
    "Space Employees (CH)": "target is Total Turnover, not Space Turnover — these breakdowns are never used (DATA_SCHEMA.md)",
    "Space Export Revenue": "same as above",
    "Space Turnover": "same as above — explicitly NOT the target",
    "Space %": "same as above",
    "Validated (CH & Beauhurst)": "internal QA field",
}

# Source 3's non-signal columns never used: IPO market cap (too sparse,
# 19/1,372), and the raw accelerator/spinout NAME + date columns (used only
# internally to derive has_attended_accelerator/is_academic_spinout, never
# exposed as their own features).
SOURCE3_DROPPED = {
    "IPO market capitalisations (converted to GBP)": DROPPED_COLUMNS["IPO market capitalisations (converted to GBP)"],
    "Accelerator Attendances N - Accelerator Name / entry-exit dates": DROPPED_COLUMNS["Accelerator Attendances N - Accelerator Name / entry-exit dates"],
    "Academic Spinout Events N - Academic Institution Name / date": DROPPED_COLUMNS["Academic Spinout Events N - Academic Institution Name / date"],
    "Growth signals - Accelerator": DROPPED_COLUMNS["Growth signals - Accelerator"],
    "Innovation signals - Academic spinout": DROPPED_COLUMNS["Innovation signals - Academic spinout"],
    "Growth signals - 10%/20% scaleup": DROPPED_COLUMNS["Growth signals - 10% scaleup / 20% scaleup"],
    "Growth signals - High growth list": DROPPED_COLUMNS["Growth signals - High growth list"],
}

STRUCTURAL_COLUMNS = {
    ("Source 2", "Total Turnover (CH year)"): "THE TARGET VARIABLE — not a feature, required regardless of feature selection.",
    ("Source 2", "Beauhurst URL"): "join key (Source1<->Source2, Source3<->Source2)",
    ("Source 2", "CH No. (full)"): "identity / company_id construction",
    ("Source 2", "Organisation Name"): "identity / company_id fallback + name-normalised join fallback",
}


def _strip_feature_name(raw_name: str) -> str:
    """Strips ColumnTransformer prefixes (plain_numeric__/log_numeric__/
    categorical__) and one-hot category suffixes (e.g.
    categorical__sic_code_1_61900 -> sic_code_1), so a feature can be
    matched against FEATURE_TO_RAW regardless of which pipeline's naming
    convention produced it."""
    name = raw_name
    for prefix in ("plain_numeric__", "log_numeric__", "categorical__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    for base in ["sic_code_1", "company_size", "value_stream", "employee_count_source"]:
        if name.startswith(base + "_") and name != base:
            return base
    return name


def get_estimation_top_features() -> dict[str, list[str]]:
    """Returns {mission: [base feature names, nonzero-weight, up to TOP_N]}
    from the already-saved feature_weights_{mission}.csv files."""
    results = {}
    for mission in REAL_MISSIONS:
        slug = mission.lower().replace(" ", "_")
        path = OUTPUT_DIR / f"feature_weights_{slug}.csv"
        df = pd.read_csv(path, comment="#")
        if "coefficient" in df.columns:
            df["magnitude"] = df["coefficient"].abs()
        else:
            df["magnitude"] = df["importance"]
        nonzero = df[df["magnitude"] > NONZERO_EPS].sort_values("magnitude", ascending=False)
        top = nonzero.head(TOP_N)["feature"].apply(_strip_feature_name).tolist()
        results[mission] = list(dict.fromkeys(top))  # de-dup within a mission, preserve order
    return results


def get_forecasting_top_features() -> dict[str, list[str]]:
    """Refits each mission's best non-benchmark ML candidate at horizon=1
    (by MAE_mean, the bake-off's own scoring metric) and extracts its top-15
    features. Persistence wins the actual deployment in every mission (see
    PROJECT_NOTES.md), so it has no feature weights of its own — this is
    the most informative stand-in for "which covariates carry real signal"."""
    results = {}
    for mission in REAL_MISSIONS:
        slug = mission.lower().replace(" ", "_")
        summary = pd.read_csv(OUTPUT_DIR / f"forecast_bakeoff_{slug}_summary.csv")
        h1 = summary[summary["horizon"] == "1"]
        ml_only = h1[~h1["Model"].isin(BENCHMARK_MODELS)].sort_values("MAE_mean")
        best_model_name = ml_only.iloc[0]["Model"]

        mission_df = get_mission_training_data(mission)
        model, _params = fit_final_ml_model(mission_df, best_model_name)
        preprocessor = model.named_steps["preprocess"]
        regressor = model.named_steps["model"].regressor_
        feature_names = preprocessor.get_feature_names_out()

        if hasattr(regressor, "coef_"):
            magnitude = np.abs(regressor.coef_)
        else:
            magnitude = regressor.feature_importances_
        weights = pd.DataFrame({"feature": feature_names, "magnitude": magnitude})
        nonzero = weights[weights["magnitude"] > NONZERO_EPS].sort_values("magnitude", ascending=False)
        top = nonzero.head(TOP_N)["feature"].apply(_strip_feature_name).tolist()
        results[mission] = (best_model_name, list(dict.fromkeys(top)))
    return results


def build_report() -> pd.DataFrame:
    estimation_top = get_estimation_top_features()
    forecasting_top = get_forecasting_top_features()

    usage_count: dict[str, int] = {}
    usage_detail: dict[str, list[str]] = {}
    for mission, feats in estimation_top.items():
        for f in feats:
            usage_count[f] = usage_count.get(f, 0) + 1
            usage_detail.setdefault(f, []).append(f"{mission} (estimation)")
    for mission, (model_name, feats) in forecasting_top.items():
        for f in feats:
            usage_count[f] = usage_count.get(f, 0) + 1
            usage_detail.setdefault(f, []).append(f"{mission} (forecasting, {model_name})")

    rows = []
    for feature, count in usage_count.items():
        source_file, raw_col, note = FEATURE_TO_RAW.get(feature, ("UNMAPPED", feature, "not yet mapped — check FEATURE_TO_RAW"))
        rows.append(
            {
                "source_file": source_file,
                "raw_column_or_base_field": raw_col,
                "engineered_feature": feature,
                "status": "used_top15",
                "n_mission_models_top15": count,
                "used_in": "; ".join(sorted(usage_detail[feature])),
                "notes": note,
            }
        )

    for (source_file, raw_col), note in STRUCTURAL_COLUMNS.items():
        rows.append(
            {
                "source_file": source_file,
                "raw_column_or_base_field": raw_col,
                "engineered_feature": "(structural — target/join key, not a model feature)",
                "status": "structural_required",
                "n_mission_models_top15": np.nan,
                "used_in": "all",
                "notes": note,
            }
        )

    for raw_col, reason in SOURCE1_DROPPED.items():
        rows.append({"source_file": "Source 1", "raw_column_or_base_field": raw_col, "engineered_feature": "", "status": "dropped_leakage_or_definitional", "n_mission_models_top15": 0, "used_in": "", "notes": reason})
    rows.append({"source_file": "Source 1", "raw_column_or_base_field": "~72 non-Financial-Statement 'company info' columns (gender pay gap, valuations, banking provider, charges & mortgages, tracking reasons, etc.)", "engineered_feature": "", "status": "never_considered", "n_mission_models_top15": 0, "used_in": "", "notes": SOURCE1_NEVER_CONSIDERED_NOTE})

    for raw_col, reason in SOURCE2_DROPPED.items():
        rows.append({"source_file": "Source 2", "raw_column_or_base_field": raw_col, "engineered_feature": "", "status": "dropped_leakage_redundant_or_sparse", "n_mission_models_top15": 0, "used_in": "", "notes": reason})
    for raw_col, reason in SOURCE2_NEVER_CONSIDERED.items():
        rows.append({"source_file": "Source 2", "raw_column_or_base_field": raw_col, "engineered_feature": "", "status": "never_considered_internal", "n_mission_models_top15": 0, "used_in": "", "notes": reason})

    for raw_col, reason in SOURCE3_DROPPED.items():
        rows.append({"source_file": "Source 3", "raw_column_or_base_field": raw_col, "engineered_feature": "", "status": "dropped_redundant_or_sparse", "n_mission_models_top15": 0, "used_in": "", "notes": reason})

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["source_file", "n_mission_models_top15"], ascending=[True, False], na_position="last"
    ).reset_index(drop=True)
    return df


def main() -> None:
    df = build_report()
    out_path = OUTPUT_DIR / "feature_source_mapping.csv"
    df.to_csv(out_path, index=False)

    print(f"Wrote {out_path} ({len(df)} rows)\n")

    # "Source 2 + Source 1" rows (turnover_t and everything derived from it)
    # depend on BOTH files genuinely — cross-listed under each individual
    # source's "used" section below (contains(), not ==), so pulling only
    # Source 2's Total Turnover columns without Source 1's weeks field (or
    # vice versa) doesn't silently look complete in either section alone.
    for source in ["Source 1", "Source 2", "Source 3"]:
        sub = df[df["source_file"].str.contains(source, regex=False)]
        used = sub[sub["status"] == "used_top15"].sort_values("n_mission_models_top15", ascending=False)
        skip = df[(df["source_file"] == source) & df["status"].str.startswith(("dropped", "never_considered"), na=False)]
        print(f"=== {source} ===")
        print(f"Used (appears in >=1 mission's top-{TOP_N}), ranked by mission-model count:")
        print(used[["raw_column_or_base_field", "n_mission_models_top15", "engineered_feature", "source_file"]].to_string(index=False))
        print(f"\nSafe to skip ({len(skip)} raw columns/groups — dropped or never considered):")
        print(skip[["raw_column_or_base_field", "status"]].to_string(index=False))
        print()

    unmapped = df[df["source_file"] == "UNMAPPED"]
    if len(unmapped):
        print("WARNING: features found in a top-15 list with no raw-column mapping yet:")
        print(unmapped[["engineered_feature", "used_in"]].to_string(index=False))


if __name__ == "__main__":
    main()
