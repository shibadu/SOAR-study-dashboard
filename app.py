"""
SOAR Study Enrollment & Follow-up Tracker
=========================================
A Streamlit dashboard for the SOAR Study REDCap project.
Displays enrollment funnel, visit adherence, safety screening
(MINI-S / HHDS / AUDIT), upcoming appointments, and generates
shareable summary reports.

Setup:
    pip install streamlit plotly pandas pycap
    streamlit run app.py

Environment variables:
    REDCAP_API_URL   = https://your-server.redcap/api/
    REDCAP_API_TOKEN = your-token-here
"""

import os
import base64
from datetime import datetime, timedelta
from io import BytesIO

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from redcap import Project

# ───────────────────────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SOAR Study Tracker",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────────────────────────────
# CUSTOM CSS
# ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1f4e79; }
    .sub-header { font-size: 1.1rem; color: #555; margin-bottom: 1rem; }
    .metric-card { background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #1f4e79; }
    .alert-overdue { background: #fff3cd; padding: 0.5rem 1rem; border-radius: 6px; border-left: 4px solid #ffc107; }
    .alert-missed { background: #f8d7da; padding: 0.5rem 1rem; border-radius: 6px; border-left: 4px solid #dc3545; }
    .share-box { background: #e7f3ff; padding: 1rem; border-radius: 8px; border: 1px solid #b3d9ff; }
    .stDataFrame { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# CONFIG & CONNECTION
# ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def connect_redcap():
    """Establish REDCap API connection. Returns None if not configured."""
    api_url = os.getenv("REDCAP_API_URL", "")
    api_token = os.getenv("REDCAP_API_TOKEN", "")

    # Only fall back to st.secrets if env vars are missing
    if not api_url or not api_token:
        try:
            api_url = st.secrets.get("REDCAP_API_URL", "")
            api_token = st.secrets.get("REDCAP_API_TOKEN", "")
        except Exception:
            # st.secrets raises if no secrets.toml exists at all
            pass

    if not api_url or not api_token:
        return None

    try:
        proj = Project(api_url, api_token)
        return proj
    except Exception as e:
        st.error(f"Failed to connect to REDCap: {e}")
        return None

@st.cache_data(ttl=1800, show_spinner="Pulling data from REDCap...")
def load_data(_proj):
    """Export all records from REDCap. Returns empty DataFrame on failure."""
    if _proj is None:
        return pd.DataFrame()

    try:
        # PyCap 3.x returns a list of dicts; older versions may return a DataFrame
        records = _proj.export_records()
        if isinstance(records, pd.DataFrame):
            return records
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"Data export failed: {e}")
        return pd.DataFrame()

# ───────────────────────────────────────────────────────────────
# DATA PROCESSING
# ───────────────────────────────────────────────────────────────
def _n_unique(df, id_col):
    """Count unique participants (by ID column) rather than raw rows.

    REDCap longitudinal/repeating exports return one row per
    event/instrument instance, so len(df) can double- or triple-count
    a participant who has, e.g., a repeated pre-screening attempt or
    rows exported for multiple events. Counting distinct IDs avoids
    that inflation.
    """
    if df.empty:
        return 0
    if id_col not in df.columns:
        return len(df)
    return df[id_col].nunique(dropna=True)


def process_enrollment(df_prescreen, df_clinical):
    """Compute enrollment funnel metrics using SOAR study fields."""
    pre_id_col = "record_id" if "record_id" in df_prescreen.columns else (
        df_prescreen.columns[0] if not df_prescreen.empty else None
    )
    clin_id_col = "record_id" if "record_id" in df_clinical.columns else (
        df_clinical.columns[0] if not df_clinical.empty else None
    )

    total_screened = _n_unique(df_prescreen, pre_id_col) if not df_prescreen.empty else 0

    eligible_referred = (
        df_prescreen[df_prescreen["prescreening_outcome"] == "1"]
        if not df_prescreen.empty and "prescreening_outcome" in df_prescreen.columns
        else pd.DataFrame()
    )

    consented = (
        eligible_referred[eligible_referred["consenting"] == "1"]
        if not eligible_referred.empty and "consenting" in eligible_referred.columns
        else pd.DataFrame()
    )

    enrolled = (
        consented[consented["screen_outcome"] == "1"]
        if not consented.empty and "screen_outcome" in consented.columns
        else pd.DataFrame()
    )

    clinically_eligible = pd.DataFrame()
    if not df_clinical.empty and "ce_enrollment_decision" in df_clinical.columns:
        clinically_eligible = df_clinical[df_clinical["ce_enrollment_decision"] == "1"]

    stratified = pd.DataFrame()
    if not df_clinical.empty and "ce_assigned_strata" in df_clinical.columns:
        stratified = df_clinical[
            df_clinical["ce_assigned_strata"].notna()
            & (df_clinical["ce_assigned_strata"] != "")
        ]

    study_id_assigned = pd.DataFrame()
    if not df_clinical.empty and "study_id" in df_clinical.columns:
        study_id_assigned = df_clinical[
            df_clinical["study_id"].notna() & (df_clinical["study_id"] != "")
        ]

    return {
        "total_screened": total_screened,
        "eligible_referred": _n_unique(eligible_referred, pre_id_col),
        "declined": (
            _n_unique(
                df_prescreen[df_prescreen["prescreening_outcome"] == "2"], pre_id_col
            )
            if not df_prescreen.empty and "prescreening_outcome" in df_prescreen.columns
            else 0
        ),
        "not_eligible": (
            _n_unique(
                df_prescreen[df_prescreen["prescreening_outcome"] == "3"], pre_id_col
            )
            if not df_prescreen.empty and "prescreening_outcome" in df_prescreen.columns
            else 0
        ),
        "consented": _n_unique(consented, pre_id_col),
        "enrolled": _n_unique(enrolled, pre_id_col),
        "clinically_eligible": _n_unique(clinically_eligible, clin_id_col),
        "stratified": _n_unique(stratified, clin_id_col),
        "study_id_assigned": _n_unique(study_id_assigned, clin_id_col),
    }

def build_visit_matrix(df_visits):
    """Build a participant × visit window matrix from the visit_log form."""
    if df_visits.empty or "visit_window" not in df_visits.columns:
        return pd.DataFrame()

    visit_map = {
        "1": "Week 1",
        "2": "Week 2",
        "3": "Week 3",
        "4": "Week 12",
        "5": "Week 36",
    }

    df = df_visits.copy()
    df["visit_label"] = df["visit_window"].astype(str).map(visit_map)
    df = df[df["visit_label"].notna()]

    id_col = "record_id" if "record_id" in df.columns else df.columns[0]

    matrix = df.pivot_table(
        index=id_col,
        columns="visit_label",
        values="visit_status",
        aggfunc="first",
    )

    for col in ["Week 1", "Week 2", "Week 3", "Week 12", "Week 36"]:
        if col not in matrix.columns:
            matrix[col] = None

    matrix = matrix[["Week 1", "Week 2", "Week 3", "Week 12", "Week 36"]]

    status_map = {
        "1": "Completed",
        "2": "Missed",
        "3": "Rescheduled",
        "4": "Early Term",
    }
    matrix = matrix.map(
        lambda x: status_map.get(str(x), "Pending") if pd.notna(x) else "Pending"
    )

    return matrix

def get_upcoming_visits(df_clinical, days_ahead=7):
    """List participants with visits due in the next N days."""
    today = pd.Timestamp.now().normalize()
    end_window = today + timedelta(days=days_ahead)

    upcoming = []
    due_cols = {
        "Week 1": "week_1_due",
        "Week 2": "week_2_due",
        "Week 3": "week_3_due",
        "Week 12": "week_12_due",
        "Week 36": "week_36_due",
    }

    if df_clinical.empty:
        return pd.DataFrame(upcoming)

    id_col = "record_id" if "record_id" in df_clinical.columns else df_clinical.columns[0]
    study_id_col = "study_id" if "study_id" in df_clinical.columns else None

    for _, row in df_clinical.iterrows():
        for visit_name, col in due_cols.items():
            if col in row and pd.notna(row.get(col)):
                try:
                    due_date = pd.to_datetime(row[col])
                    if today <= due_date <= end_window:
                        upcoming.append(
                            {
                                "Record ID": row.get(id_col),
                                "Participant ID": row.get(study_id_col) if study_id_col else "N/A",
                                "Visit": visit_name,
                                "Due Date": due_date.strftime("%Y-%m-%d"),
                                "Days Left": (due_date - today).days,
                            }
                        )
                except Exception:
                    continue

    return (
        pd.DataFrame(upcoming).sort_values("Days Left")
        if upcoming
        else pd.DataFrame(upcoming)
    )

def get_overdue_visits(df_clinical, df_visits):
    """Find participants whose due date has passed but no completed visit."""
    today = pd.Timestamp.now().normalize()

    completed = pd.DataFrame()
    if not df_visits.empty and "visit_status" in df_visits.columns:
        completed = df_visits[df_visits["visit_status"] == "1"].copy()
        visit_window_map = {
            "1": "Week 1",
            "2": "Week 2",
            "3": "Week 3",
            "4": "Week 12",
            "5": "Week 36",
        }
        if "visit_window" in completed.columns:
            completed["visit_label"] = completed["visit_window"].astype(str).map(
                visit_window_map
            )

    overdue = []
    due_cols = {
        "Week 1": "week_1_due",
        "Week 2": "week_2_due",
        "Week 3": "week_3_due",
        "Week 12": "week_12_due",
        "Week 36": "week_36_due",
    }

    if df_clinical.empty:
        return pd.DataFrame(overdue)

    id_col = "record_id" if "record_id" in df_clinical.columns else df_clinical.columns[0]
    study_id_col = "study_id" if "study_id" in df_clinical.columns else None

    for _, row in df_clinical.iterrows():
        pid = row.get(id_col)
        for visit_name, col in due_cols.items():
            if col in row and pd.notna(row.get(col)):
                try:
                    due_date = pd.to_datetime(row[col])
                    if due_date < today:
                        already_done = pd.DataFrame()
                        if not completed.empty and "visit_label" in completed.columns:
                            match_id = completed[id_col] == pid if id_col in completed.columns else False
                            already_done = completed[
                                match_id & (completed["visit_label"] == visit_name)
                            ]

                        if already_done.empty:
                            overdue.append(
                                {
                                    "Record ID": pid,
                                    "Participant ID": row.get(study_id_col) if study_id_col else "N/A",
                                    "Visit": visit_name,
                                    "Due Date": due_date.strftime("%Y-%m-%d"),
                                    "Days Overdue": (today - due_date).days,
                                }
                            )
                except Exception:
                    continue

    return (
        pd.DataFrame(overdue).sort_values("Days Overdue", ascending=False)
        if overdue
        else pd.DataFrame(overdue)
    )

def build_stratification_summary(df_clinical):
    """Build a stratum-level summary table: counts + % of stratified participants."""
    if df_clinical.empty or "ce_assigned_strata" not in df_clinical.columns:
        return pd.DataFrame()

    strata_map = {
        "1": "Low Smoking + Low Alcohol",
        "2": "Low Smoking + High Alcohol",
        "3": "High Smoking + Low Alcohol",
        "4": "High Smoking + High Alcohol",
    }

    strata = df_clinical[
        df_clinical["ce_assigned_strata"].notna()
        & (df_clinical["ce_assigned_strata"] != "")
    ]["ce_assigned_strata"].astype(str)

    if strata.empty:
        return pd.DataFrame()

    counts = strata.value_counts()
    total = counts.sum()

    summary = pd.DataFrame(
        {
            "Stratum": [strata_map.get(k, k) for k in counts.index],
            "Count": counts.values,
            "% of Stratified": (counts.values / total * 100).round(1),
        }
    )
    return summary.sort_values("Stratum").reset_index(drop=True)


def get_safety_screening_summary(df_clinical):
    """Summarize safety screening results (MINI-S, HHDS, AUDIT)."""
    if df_clinical.empty:
        return {}

    summary = {}

    if "ce_mini_eligibility" in df_clinical.columns:
        summary["mini_eligible"] = len(df_clinical[df_clinical["ce_mini_eligibility"] == "1"])
        summary["mini_screenout"] = len(df_clinical[df_clinical["ce_mini_eligibility"] == "0"])

    if "ce_mini_total_score" in df_clinical.columns:
        scores = pd.to_numeric(df_clinical["ce_mini_total_score"], errors="coerce")
        summary["mini_mean_score"] = round(scores.mean(), 1) if not scores.empty else 0
        summary["mini_high_risk"] = len(scores[scores >= 10])

    if "ce_hhds_eligibility" in df_clinical.columns:
        summary["hhds_eligible"] = len(df_clinical[df_clinical["ce_hhds_eligibility"] == "1"])
        summary["hhds_screenout"] = len(df_clinical[df_clinical["ce_hhds_eligibility"] == "0"])

    if "ce_audit_total_score" in df_clinical.columns:
        scores = pd.to_numeric(df_clinical["ce_audit_total_score"], errors="coerce")
        summary["audit_mean"] = round(scores.mean(), 1) if not scores.empty else 0
        summary["audit_high"] = len(scores[scores >= 16])

    if "ce_assigned_strata" in df_clinical.columns:
        strata_counts = df_clinical["ce_assigned_strata"].value_counts().to_dict()
        summary["strata_counts"] = strata_counts

    return summary

# ───────────────────────────────────────────────────────────────
# SHAREABLE REPORT GENERATOR
# ───────────────────────────────────────────────────────────────
def generate_shareable_link(view_mode, filters=None):
    """Encode current view state into query parameters for sharing."""
    params = {"view": view_mode}
    if filters:
        params.update(filters)

    base_url = "https://your-app-url.streamlit.app"
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    return f"{base_url}/?{query_string}"

def create_summary_html(enrollment, visit_matrix, upcoming, overdue, safety=None):
    """Generate a static HTML summary report for sharing."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SOAR Study Summary Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 2rem auto; padding: 1rem; }}
            h1 {{ color: #1f4e79; border-bottom: 2px solid #1f4e79; padding-bottom: 0.5rem; }}
            .metric {{ display: inline-block; margin: 1rem; padding: 1rem 2rem; background: #f0f4f8; border-radius: 8px; text-align: center; }}
            .metric-value {{ font-size: 2rem; font-weight: bold; color: #1f4e79; }}
            .metric-label {{ font-size: 0.9rem; color: #555; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
            th, td {{ padding: 0.6rem; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #1f4e79; color: white; }}
            .alert {{ background: #fff3cd; padding: 1rem; border-radius: 6px; margin-top: 1rem; }}
            .footer {{ margin-top: 2rem; font-size: 0.8rem; color: #888; text-align: center; }}
        </style>
    </head>
    <body>
        <h1>SOAR Study Enrollment & Follow-up Summary</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>

        <h2>Enrollment Funnel</h2>
        <div>
            <div class="metric"><div class="metric-value">{enrollment.get("total_screened", 0)}</div><div class="metric-label">Pre-Screened</div></div>
            <div class="metric"><div class="metric-value">{enrollment.get("eligible_referred", 0)}</div><div class="metric-label">Eligible & Referred</div></div>
            <div class="metric"><div class="metric-value">{enrollment.get("consented", 0)}</div><div class="metric-label">Consented</div></div>
            <div class="metric"><div class="metric-value">{enrollment.get("clinically_eligible", 0)}</div><div class="metric-label">Clinically Eligible</div></div>
            <div class="metric"><div class="metric-value">{enrollment.get("stratified", 0)}</div><div class="metric-label">Stratified</div></div>
        </div>

        <h2>Safety Screening</h2>
        {f"<p>MINI-S Eligible: {safety.get('mini_eligible', 0)} | Screened Out: {safety.get('mini_screenout', 0)}</p>" if safety else "<p>No safety data available.</p>"}
        {f"<p>HHDS Eligible: {safety.get('hhds_eligible', 0)} | Screened Out: {safety.get('hhds_screenout', 0)}</p>" if safety else ""}

        <h2>Overdue Visits ({len(overdue)})</h2>
        {overdue.to_html(index=False) if not overdue.empty else "<p>No overdue visits. Great job!</p>"}

        <h2>Upcoming Visits (Next 7 Days)</h2>
        {upcoming.to_html(index=False) if not upcoming.empty else "<p>No upcoming visits in the next 7 days.</p>"}

        <div class="footer">
            Generated by SOAR Study Tracker | REDCap Longitudinal Dashboard
        </div>
    </body>
    </html>
    """
    return html

# ───────────────────────────────────────────────────────────────
# MAIN APP
# ───────────────────────────────────────────────────────────────
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### SOAR Study Tracker")
        st.markdown("---")

        view_mode = st.radio(
            "Select View",
            [
                "Dashboard",
                "Enrollment Funnel",
                "Visit Matrix",
                "Safety Screening",
                "Alerts",
                "Shareable Summary",
            ],
            index=0,
        )

        st.markdown("---")

        st.markdown("**Data refresh:** Every 30 min")
        if st.button("Refresh Data Now"):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.markdown(
            "<small>Powered by PyCap + Streamlit</small>", unsafe_allow_html=True
        )

    # ── Load data from REDCap ──
    proj = connect_redcap()
    if proj is None:
        st.warning("REDCap credentials not found.")
        st.markdown(
            """
            **To connect to your REDCap project:**
            1. Set environment variables:
               - `REDCAP_API_URL`
               - `REDCAP_API_TOKEN`
            2. Or create `.streamlit/secrets.toml` next to this script:
               ```toml
               REDCAP_API_URL = "https://your-server.redcap/api/"
               REDCAP_API_TOKEN = "your-api-token"
               ```
            """
        )
        st.stop()

    df_all = load_data(proj)

    if df_all.empty:
        st.warning("No data returned from REDCap. Check API permissions.")
        st.stop()

    df_all = df_all.reset_index()

    df_prescreen = (
        df_all[df_all["prescreening_outcome"].notna()].copy()
        if "prescreening_outcome" in df_all.columns
        else pd.DataFrame()
    )

    df_clinical = (
        df_all[df_all["ce_date"].notna()].copy()
        if "ce_date" in df_all.columns
        else pd.DataFrame()
    )

    df_visits = (
        df_all[df_all["visit_window"].notna()].copy()
        if "visit_window" in df_all.columns
        else pd.DataFrame()
    )

    if df_prescreen.empty and df_clinical.empty and df_visits.empty:
        st.warning(
            "Could not identify SOAR study forms in REDCap data. "
            "Please verify field names match the data dictionary."
        )
        st.stop()

    # ── Compute metrics ──
    enrollment = process_enrollment(df_prescreen, df_clinical)
    visit_matrix = build_visit_matrix(df_visits)
    upcoming = get_upcoming_visits(df_clinical, days_ahead=7)
    overdue = get_overdue_visits(df_clinical, df_visits)
    safety = get_safety_screening_summary(df_clinical)
    strata_summary = build_stratification_summary(df_clinical)

    # ═══════════════════════════════════════════════════════════
    # VIEW: DASHBOARD
    # ═══════════════════════════════════════════════════════════
    if view_mode == "Dashboard":
        st.markdown(
            '<div class="main-header">SOAR Study Enrollment & Follow-up Dashboard</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sub-header">Real-time data from REDCap | Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>',
            unsafe_allow_html=True,
        )

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Pre-Screened", enrollment["total_screened"])
        kpi2.metric(
            "Eligible & Referred",
            enrollment["eligible_referred"],
            f"{enrollment['eligible_referred'] / max(enrollment['total_screened'], 1) * 100:.0f}%",
        )
        kpi3.metric(
            "Consented",
            enrollment["consented"],
            f"{enrollment['consented'] / max(enrollment['eligible_referred'], 1) * 100:.0f}%",
        )
        kpi4.metric("Clinically Eligible", enrollment["clinically_eligible"])
        kpi5.metric("Stratified", enrollment["stratified"])

        st.markdown("---")

        st.subheader("Enrollment Funnel")
        funnel_data = pd.DataFrame(
            {
                "Stage": [
                    "Pre-Screened",
                    "Eligible & Referred",
                    "Consented",
                    "Clinically Eligible",
                    "Stratified",
                ],
                "Count": [
                    enrollment["total_screened"],
                    enrollment["eligible_referred"],
                    enrollment["consented"],
                    enrollment["clinically_eligible"],
                    enrollment["stratified"],
                ],
            }
        )
        fig_funnel = go.Figure(
            go.Funnel(
                y=funnel_data["Stage"],
                x=funnel_data["Count"],
                textposition="inside",
                textinfo="value+percent initial",
                marker={
                    "color": ["#1f4e79", "#2e7d32", "#f57c00", "#7b1fa2", "#c62828"]
                },
            )
        )
        fig_funnel.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=400)
        st.plotly_chart(fig_funnel, use_container_width=True)

        st.markdown("---")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Stratification Summary")
            if not strata_summary.empty:
                st.dataframe(
                    strata_summary, use_container_width=True, hide_index=True
                )
                fig_strata = px.bar(
                    strata_summary,
                    x="Stratum",
                    y="Count",
                    color="Stratum",
                    text="Count",
                    labels={"Count": "Participants"},
                    height=280,
                )
                fig_strata.update_layout(
                    showlegend=False, margin=dict(l=20, r=20, t=10, b=20)
                )
                st.plotly_chart(fig_strata, use_container_width=True)
            else:
                st.info("No stratification data available yet.")

        with col_right:
            st.subheader("Visit Adherence Overview")
            if not visit_matrix.empty:
                status_counts = (
                    visit_matrix.apply(pd.Series.value_counts).fillna(0).astype(int)
                )
                for status in ["Completed", "Pending", "Missed", "Rescheduled", "Early Term"]:
                    if status not in status_counts.index:
                        status_counts.loc[status] = 0

                fig_adherence = px.bar(
                    status_counts.T,
                    barmode="stack",
                    color_discrete_map={
                        "Completed": "#2e7d32",
                        "Pending": "#ffc107",
                        "Missed": "#dc3545",
                        "Rescheduled": "#17a2b8",
                        "Early Term": "#6c757d",
                    },
                    labels={"value": "Participants", "index": "Visit Window"},
                    height=400,
                )
                fig_adherence.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig_adherence, use_container_width=True)
            else:
                st.info("No visit data available yet.")

        st.markdown("---")

        st.subheader("Action Required")
        alert_col1, alert_col2 = st.columns(2)

        with alert_col1:
            st.markdown(f"**{len(overdue)} Overdue Visits**")
            if not overdue.empty:
                st.markdown('<div class="alert-overdue">', unsafe_allow_html=True)
                st.dataframe(overdue.head(10), use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.success("No overdue visits!")

        with alert_col2:
            st.markdown(f"**{len(upcoming)} Upcoming Visits (Next 7 Days)**")
            if not upcoming.empty:
                st.dataframe(upcoming, use_container_width=True, hide_index=True)
            else:
                st.info("No visits scheduled in the next 7 days.")

    # ═══════════════════════════════════════════════════════════
    # VIEW: ENROLLMENT FUNNEL
    # ═══════════════════════════════════════════════════════════
    elif view_mode == "Enrollment Funnel":
        st.markdown(
            '<div class="main-header">Enrollment Funnel Details</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Not Eligible (Pre-screen)", enrollment["not_eligible"])
        col2.metric("Eligible but Declined", enrollment["declined"])
        col3.metric(
            "Screened Out (Behavioral)",
            max(0, enrollment["enrolled"] - enrollment["clinically_eligible"]),
        )

        st.markdown("---")

        if not df_prescreen.empty and "prescreening_outcome" in df_prescreen.columns:
            st.subheader("Pre-Screening Outcomes")
            outcome_counts = (
                df_prescreen["prescreening_outcome"].value_counts().reset_index()
            )
            outcome_map = {
                "1": "Eligible & Referred",
                "2": "Eligible but Declined",
                "3": "Not Eligible",
            }
            outcome_counts["Outcome"] = outcome_counts["prescreening_outcome"].map(
                outcome_map
            )
            fig = px.pie(
                outcome_counts,
                values="count",
                names="Outcome",
                color_discrete_sequence=px.colors.sequential.Blues_r,
            )
            st.plotly_chart(fig, use_container_width=True)

        if not df_prescreen.empty and "motivation_quit" in df_prescreen.columns:
            st.subheader("Motivation to Quit Distribution")
            df_prescreen["motivation_quit_num"] = pd.to_numeric(
                df_prescreen["motivation_quit"], errors="coerce"
            )
            fig = px.histogram(
                df_prescreen,
                x="motivation_quit_num",
                nbins=10,
                labels={"motivation_quit_num": "Motivation Score (1-10)"},
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════
    # VIEW: VISIT MATRIX
    # ═══════════════════════════════════════════════════════════
    elif view_mode == "Visit Matrix":
        st.markdown(
            '<div class="main-header">Participant Visit Matrix</div>',
            unsafe_allow_html=True,
        )
        st.markdown("Color-coded status for each participant across all visit windows.")

        if not visit_matrix.empty:
            def color_status(val):
                colors = {
                    "Completed": "background-color: #2e7d32; color: white",
                    "Pending": "background-color: #d39e00; color: white",
                    "Missed": "background-color: #a71d2a; color: white",
                    "Rescheduled": "background-color: #117a8b; color: white",
                    "Early Term": "background-color: #495057; color: white",
                }
                return colors.get(val, "")

            st.dataframe(
                visit_matrix.style.map(color_status),
                use_container_width=True,
            )

            csv = visit_matrix.to_csv().encode("utf-8")
            st.download_button(
                label="Download Matrix as CSV",
                data=csv,
                file_name="soar_visit_matrix.csv",
                mime="text/csv",
            )
        else:
            st.info("No visit data to display.")

    # ═══════════════════════════════════════════════════════════
    # VIEW: SAFETY SCREENING
    # ═══════════════════════════════════════════════════════════
    elif view_mode == "Safety Screening":
        st.markdown(
            '<div class="main-header">Safety Screening Summary</div>',
            unsafe_allow_html=True,
        )

        if safety:
            c1, c2, c3 = st.columns(3)
            c1.metric("MINI-S Eligible", safety.get("mini_eligible", 0))
            c2.metric("MINI-S Screen Out", safety.get("mini_screenout", 0))
            c3.metric("MINI High Risk (≥10)", safety.get("mini_high_risk", 0))

            c4, c5, c6 = st.columns(3)
            c4.metric("HHDS Eligible", safety.get("hhds_eligible", 0))
            c5.metric("HHDS Screen Out", safety.get("hhds_screenout", 0))
            c6.metric("AUDIT High (≥16)", safety.get("audit_high", 0))

            st.markdown("---")

            if "strata_counts" in safety and safety["strata_counts"]:
                st.subheader("Stratification Distribution")
                strata_df = pd.DataFrame(
                    {
                        "Stratum": list(safety["strata_counts"].keys()),
                        "Count": list(safety["strata_counts"].values()),
                    }
                )
                strata_map = {
                    "1": "Low Smoking + Low Alcohol",
                    "2": "Low Smoking + High Alcohol",
                    "3": "High Smoking + Low Alcohol",
                    "4": "High Smoking + High Alcohol",
                }
                strata_df["Stratum Label"] = strata_df["Stratum"].astype(str).map(strata_map)

                fig = px.bar(
                    strata_df,
                    x="Stratum Label",
                    y="Count",
                    color="Stratum Label",
                    labels={"Count": "Participants"},
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

            if (
                not df_clinical.empty
                and "ce_mini_total_score" in df_clinical.columns
            ):
                st.subheader("MINI-S Total Score Distribution")
                df_clinical["mini_score_num"] = pd.to_numeric(
                    df_clinical["ce_mini_total_score"], errors="coerce"
                )
                fig = px.histogram(
                    df_clinical,
                    x="mini_score_num",
                    nbins=20,
                    labels={"mini_score_num": "MINI-S Total Score"},
                    color_discrete_sequence=["#1f4e79"],
                )
                fig.add_vline(
                    x=6, line_dash="dash", line_color="orange", annotation_text="Moderate"
                )
                fig.add_vline(
                    x=10, line_dash="dash", line_color="red", annotation_text="High"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No clinical evaluation data available.")

    # ═══════════════════════════════════════════════════════════
    # VIEW: ALERTS
    # ═══════════════════════════════════════════════════════════
    elif view_mode == "Alerts":
        st.markdown(
            '<div class="main-header">Alerts & Protocol Deviations</div>',
            unsafe_allow_html=True,
        )

        tab1, tab2, tab3 = st.tabs(
            ["Overdue Visits", "Upcoming Visits", "Protocol Deviations"]
        )

        with tab1:
            st.subheader(f"Overdue Visits ({len(overdue)})")
            if not overdue.empty:
                st.dataframe(overdue, use_container_width=True, hide_index=True)
            else:
                st.success("All visits are on track!")

        with tab2:
            st.subheader(f"Upcoming Visits - Next 7 Days ({len(upcoming)})")
            if not upcoming.empty:
                st.dataframe(upcoming, use_container_width=True, hide_index=True)
            else:
                st.info("No visits due in the next 7 days.")

        with tab3:
            st.subheader("Protocol Deviations")
            if not df_visits.empty and "protocol_deviation" in df_visits.columns:
                deviations = df_visits[df_visits["protocol_deviation"] == "1"]
                if not deviations.empty:
                    display_cols = [
                        c
                        for c in ["record_id", "visit_date", "visit_window", "comment"]
                        if c in deviations.columns
                    ]
                    st.dataframe(deviations[display_cols], use_container_width=True)
                else:
                    st.success("No protocol deviations recorded.")
            else:
                st.info("No protocol deviation data available.")

    # ═══════════════════════════════════════════════════════════
    # VIEW: SHAREABLE SUMMARY
    # ═══════════════════════════════════════════════════════════
    elif view_mode == "Shareable Summary":
        st.markdown(
            '<div class="main-header">Shareable Summary</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "Download a static HTML report."
        )

        st.markdown("---")

        st.subheader("Download Static HTML Report")
        st.markdown(
            "Generate a self-contained HTML file that can be emailed or shared offline."
        )

        if st.button("Generate HTML Report"):
            html_content = create_summary_html(
                enrollment, visit_matrix, upcoming, overdue, safety
            )

            b64 = base64.b64encode(html_content.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="soar_study_summary_{datetime.now().strftime("%Y%m%d")}.html">Click here to download HTML report</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success("Report generated! Click the link above to download.")

        st.markdown("---")

        st.subheader("3. Quick Copy-Paste Stats")
        stats_text = f"""
SOAR Study Summary - {datetime.now().strftime("%Y-%m-%d")}
----------------------------------------
Enrollment:
  - Pre-Screened:           {enrollment['total_screened']}
  - Eligible & Referred:    {enrollment['eligible_referred']} ({enrollment['eligible_referred'] / max(enrollment['total_screened'], 1) * 100:.0f}%)
  - Consented:              {enrollment['consented']} ({enrollment['consented'] / max(enrollment['eligible_referred'], 1) * 100:.0f}%)
  - Clinically Eligible:    {enrollment['clinically_eligible']}
  - Stratified:             {enrollment['stratified']}

Alerts:
  - Overdue visits:         {len(overdue)}
  - Upcoming (7 days):      {len(upcoming)}
        """
        st.text_area("Copy these stats to email/Slack/Teams", stats_text, height=300)

if __name__ == "__main__":
    main()