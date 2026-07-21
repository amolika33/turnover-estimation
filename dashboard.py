"""Streamlit dashboard for exploring the completed 2030 turnover forecast,
company by company. Reads only already-computed CSVs (forecast_assemble.py
and forecast_reporting.py's outputs) — no model fitting or recomputation
happens here.

Run with:
    streamlit run dashboard.py

No setup beyond `pip install -r requirements.txt` — every input file this
reads is a plain CSV already produced by the forecast_src/ pipeline.
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data" / "processed"

st.set_page_config(page_title="UK Space Sector Turnover Forecast", layout="wide")


@st.cache_data
def load_data():
    missing = [
        p
        for p in [
            "forecast_full_trajectories.csv",
            "forecast_2030_summary.csv",
            "forecast_baseline_validated.csv",
            "forecast_gazelle_10pct.csv",
            "forecast_gazelle_20pct.csv",
            "forecast_gazelle_50m_intersection.csv",
            "forecast_operational_scaling.csv",
            "forecast_10m_crossings.csv",
        ]
        if not (DATA_DIR / p).exists()
    ]
    if missing:
        st.error(
            "Missing required data file(s): "
            + ", ".join(missing)
            + ". Run the forecast_src/ pipeline (forecast_data_prep.py through "
            "forecast_reporting.py, plus forecast_prediction_intervals.py) first."
        )
        st.stop()

    trajectories = pd.read_csv(DATA_DIR / "forecast_full_trajectories.csv")
    summary = pd.read_csv(DATA_DIR / "forecast_2030_summary.csv")
    baseline = pd.read_csv(DATA_DIR / "forecast_baseline_validated.csv")
    gazelle_10 = set(pd.read_csv(DATA_DIR / "forecast_gazelle_10pct.csv")["company_id"])
    gazelle_20 = set(pd.read_csv(DATA_DIR / "forecast_gazelle_20pct.csv")["company_id"])
    intersection = pd.read_csv(DATA_DIR / "forecast_gazelle_50m_intersection.csv")
    operational_scaling = set(pd.read_csv(DATA_DIR / "forecast_operational_scaling.csv")["company_id"])
    crossings_10m = set(pd.read_csv(DATA_DIR / "forecast_10m_crossings.csv")["company_id"])

    return trajectories, summary, baseline, gazelle_10, gazelle_20, intersection, operational_scaling, crossings_10m


(
    trajectories,
    summary,
    baseline,
    gazelle_10,
    gazelle_20,
    intersection_df,
    operational_scaling,
    crossings_10m,
) = load_data()

intersection_lookup = intersection.set_index("company_id")["credibility_status"] if len(intersection) else pd.Series(dtype=object)

st.title("UK Space Sector Turnover Forecast — Company Explorer")
st.caption(
    "Historical observed turnover, recursive forecast to 2030, and evidence-stratified confidence bands. "
    "All data from the turnover-estimation + forecasting pipelines — nothing is recomputed here."
)

# --- Company selector ---
company_options = baseline[["company_id", "company_name", "CH No. (full)", "mission"]].copy()
company_options["CH No. (full)"] = company_options["CH No. (full)"].fillna("—")
company_options["label"] = (
    company_options["company_name"] + "  (" + company_options["CH No. (full)"].astype(str) + ")  — " + company_options["mission"]
)
company_options = company_options.sort_values("company_name")

selected_label = st.selectbox(
    "Search by company name or Companies House number",
    options=company_options["label"].tolist(),
    index=None,
    placeholder="Start typing a company name or CH number...",
)

if selected_label is None:
    st.info("Select a company above to see its turnover history and forecast.")
    st.stop()

company_id = company_options.loc[company_options["label"] == selected_label, "company_id"].iloc[0]

# --- Data for the selected company ---
company_traj = trajectories[trajectories["company_id"] == company_id].sort_values("accounting_year")
company_summary = summary[summary["company_id"] == company_id].iloc[0]

real_rows = company_traj[company_traj["data_type"].isin(["observed", "estimated_baseline"])]
predicted_rows = company_traj[company_traj["data_type"] == "predicted"]

# --- Chart ---
fig = go.Figure()

if len(predicted_rows) and predicted_rows["turnover_upper"].notna().any():
    fig.add_trace(
        go.Scatter(
            x=pd.concat([predicted_rows["accounting_year"], predicted_rows["accounting_year"][::-1]]),
            y=pd.concat([predicted_rows["turnover_upper"], predicted_rows["turnover_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% confidence band",
            showlegend=True,
        )
    )

if len(real_rows):
    fig.add_trace(
        go.Scatter(
            x=real_rows["accounting_year"],
            y=real_rows["turnover"],
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            name="Observed / baseline turnover",
        )
    )

if len(predicted_rows):
    # Connect the line from the last real point so the dashed segment isn't visually disjoint.
    bridge = pd.concat([real_rows.tail(1), predicted_rows]) if len(real_rows) else predicted_rows
    fig.add_trace(
        go.Scatter(
            x=bridge["accounting_year"],
            y=bridge["turnover"],
            mode="lines+markers",
            line=dict(color="#d62728", width=2, dash="dash"),
            name="Forecast to 2030",
        )
    )

fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Turnover (£)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=500,
)
st.plotly_chart(fig, use_container_width=True)

# --- Summary panel ---
st.subheader("Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Identity**")
    st.write(f"Mission: **{company_summary['mission']}**")
    st.write(f"Evidence group: **{company_summary['forecast_evidence_group']}**")
    st.write(f"Baseline source: **{company_summary['turnover_source']}**")
    n_real = int((company_traj['data_type'] == 'observed').sum())
    st.write(f"Real observed years: **{n_real}**")

with col2:
    st.markdown("**2030 forecast**")
    st.write(f"Baseline ({int(company_summary['baseline_year'])}): **£{company_summary['baseline_turnover']:,.0f}**")
    st.write(f"2030 forecast: **£{company_summary['turnover_2030']:,.0f}**")
    st.write(f"Growth multiple: **{company_summary['growth_multiple_2030']:.2f}x**")
    st.write(f"Annualized rate: **{company_summary['annualized_growth_rate_to_2030'] * 100:.1f}%/yr**")

with col3:
    st.markdown("**Flags**")

    def flag(label: str, is_set: bool) -> str:
        return f"{'✅' if is_set else '⬜'} {label}"

    st.write(flag("£10M-by-2030 crossing", company_id in crossings_10m))
    st.write(flag("Gazelle (≥10% YoY, 3+ yrs)", company_id in gazelle_10))
    st.write(flag("Gazelle (≥20% YoY, 3+ yrs)", company_id in gazelle_20))
    if company_id in intersection_lookup.index:
        status = intersection_lookup.loc[company_id]
        st.write(flag(f"£50M intersection ({status})", True))
    else:
        st.write(flag("£50M intersection", False))
    st.write(flag("Operational scaling (employees/assets)", company_id in operational_scaling))

# --- Data provenance ---
st.subheader("Data provenance")
provenance_counts = company_traj["data_type"].value_counts().rename_axis("Data Type").reset_index(name="Years")
st.dataframe(provenance_counts, hide_index=True, use_container_width=False)

with st.expander("Full year-by-year data for this company"):
    display_cols = ["accounting_year", "turnover", "data_type", "model_used", "growth_classification", "turnover_lower", "turnover_upper"]
    display_cols = [c for c in display_cols if c in company_traj.columns]
    st.dataframe(company_traj[display_cols], hide_index=True, use_container_width=True)
