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

    /* Hide Streamlit Community Cloud's Fork / GitHub / Deploy toolbar for viewers */
    .stDeployButton { display: none !important; }
    [data-testid="stToolbarActionButton"] { display: none !important; }
    [data-testid="stToolbar"] { visibility: hidden !important; height: 0 !important; }
    #MainMenu { visibility: hidden !important; }
    [class*="viewerBadge"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# SHARED CONSTANTS
# ───────────────────────────────────────────────────────────────
STRATA_MAP = {
    "1": "Low Smoking + Low Alcohol",
    "2": "Low Smoking + High Alcohol",
    "3": "High Smoking + Low Alcohol",
    "4": "High Smoking + High Alcohol",
}

# Field names confirmed against the SOAR Study data dictionary
# (SOARStudyScreeningTool_DataDictionary_2026-05-11):
#   - prescreen_age: calc field on the pre_screening form, "Age (years)"
#   - gender: radio field on the pre_screening form, "Gender:" -> 1=Male, 2=Female
AGE_FIELD_CANDIDATES = ["prescreen_age", "age", "participant_age", "ce_age", "demo_age", "age_years"]
SEX_FIELD_CANDIDATES = ["gender", "sex", "participant_sex", "ce_sex", "demo_sex"]
SEX_MAP = {"1": "Male", "2": "Female"}

VISIT_WINDOW_ORDER = ["Week 1", "Week 2", "Week 3", "Week 12", "Week 36"]

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

    for col in VISIT_WINDOW_ORDER:
        if col not in matrix.columns:
            matrix[col] = None

    matrix = matrix[VISIT_WINDOW_ORDER]

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
            "Stratum": [STRATA_MAP.get(k, k) for k in counts.index],
            "Count": counts.values,
            "% of Stratified": (counts.values / total * 100).round(1),
        }
    )
    return summary.sort_values("Stratum").reset_index(drop=True)


def build_stratified_demographics(df_clinical):
    """Age & sex data for participants who have an assigned stratum.

    Looks for the field names in AGE_FIELD_CANDIDATES / SEX_FIELD_CANDIDATES
    (see SHARED CONSTANTS) and uses the first one present in the data.
    Returns an empty DataFrame if neither field is found.

    Note: the "Stratum" column is kept in the output for reference, but the
    Age/Sex charts built from this data are plain (not split by stratum).
    """
    if df_clinical.empty or "ce_assigned_strata" not in df_clinical.columns:
        return pd.DataFrame()

    stratified = df_clinical[
        df_clinical["ce_assigned_strata"].notna()
        & (df_clinical["ce_assigned_strata"] != "")
    ].copy()

    if stratified.empty:
        return pd.DataFrame()

    age_col = next((c for c in AGE_FIELD_CANDIDATES if c in stratified.columns), None)
    sex_col = next((c for c in SEX_FIELD_CANDIDATES if c in stratified.columns), None)

    if age_col is None and sex_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["Stratum"] = stratified["ce_assigned_strata"].astype(str).map(
        lambda k: STRATA_MAP.get(k, k)
    )

    if age_col is not None:
        out["Age"] = pd.to_numeric(stratified[age_col], errors="coerce")
    if sex_col is not None:
        out["Sex"] = stratified[sex_col].astype(str).map(lambda k: SEX_MAP.get(k, k))

    return out


def build_weekly_enrollment_trends(df_clinical):
    """Weekly enrollment count and cumulative stratified-enrollment trend.

    Uses `ce_date` (clinical evaluation date) as the enrollment-event date
    since it's the date field carried on the clinical eligibility form.
    - "Enrolled" = rows where ce_enrollment_decision == "1"
    - "Stratified" = rows where ce_assigned_strata is populated
    Returns (weekly_enrollment_df, cumulative_stratified_df), each empty if
    the required fields aren't present.
    """
    empty = (pd.DataFrame(), pd.DataFrame())
    if df_clinical.empty or "ce_date" not in df_clinical.columns:
        return empty

    df = df_clinical.copy()
    df["ce_date_parsed"] = pd.to_datetime(df["ce_date"], errors="coerce")
    df = df[df["ce_date_parsed"].notna()]
    if df.empty:
        return empty

    df["week"] = df["ce_date_parsed"].dt.to_period("W").apply(lambda p: p.start_time)

    weekly_enrollment = pd.DataFrame()
    if "ce_enrollment_decision" in df.columns:
        enrolled = df[df["ce_enrollment_decision"] == "1"]
        if not enrolled.empty:
            weekly_enrollment = (
                enrolled.groupby("week").size().reset_index(name="Enrollments")
                .sort_values("week")
            )
            weekly_enrollment["Week"] = weekly_enrollment["week"].dt.strftime("%Y-%m-%d")

    cumulative_stratified = pd.DataFrame()
    if "ce_assigned_strata" in df.columns:
        stratified = df[
            df["ce_assigned_strata"].notna() & (df["ce_assigned_strata"] != "")
        ]
        if not stratified.empty:
            weekly_counts = (
                stratified.groupby("week").size().reset_index(name="Count")
                .sort_values("week")
            )
            weekly_counts["Cumulative Stratified"] = weekly_counts["Count"].cumsum()
            weekly_counts["Week"] = weekly_counts["week"].dt.strftime("%Y-%m-%d")
            cumulative_stratified = weekly_counts

    return weekly_enrollment, cumulative_stratified


def build_weekly_retention(visit_matrix):
    """Retention rate per visit week: Completed / (Completed + Missed + Rescheduled).

    Pending and Early Term rows are excluded from both numerator and
    denominator since they aren't yet a completed-or-failed outcome.
    """
    if visit_matrix.empty:
        return pd.DataFrame()

    rows = []
    for week in VISIT_WINDOW_ORDER:
        if week not in visit_matrix.columns:
            continue
        counts = visit_matrix[week].value_counts()
        completed = int(counts.get("Completed", 0))
        missed = int(counts.get("Missed", 0))
        rescheduled = int(counts.get("Rescheduled", 0))
        denominator = completed + missed + rescheduled
        retention_pct = (completed / denominator * 100) if denominator > 0 else None
        rows.append(
            {
                "Week": week,
                "Completed": completed,
                "Missed": missed,
                "Rescheduled": rescheduled,
                "Denominator": denominator,
                "Retention %": round(retention_pct, 1) if retention_pct is not None else None,
            }
        )

    return pd.DataFrame(rows)


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
                "Visit Matrix",
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
    strata_demographics = build_stratified_demographics(df_clinical)
    weekly_enrollment, cumulative_stratified = build_weekly_enrollment_trends(df_clinical)
    weekly_retention = build_weekly_retention(visit_matrix)

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
        viridis_colors = px.colors.sample_colorscale(
            "Viridis", [i / (len(funnel_data) - 1) for i in range(len(funnel_data))]
        )
        fig_funnel = go.Figure(
            go.Funnel(
                y=funnel_data["Stage"],
                x=funnel_data["Count"],
                textposition="inside",
                textinfo="value+percent initial",
                marker={"color": viridis_colors},
            )
        )
        fig_funnel.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=400,
            xaxis_title="Participants",
            yaxis_title="Enrollment Stage",
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

        st.markdown("---")

        st.subheader("Stratification Summary")
        col_left, col_right = st.columns([1, 1])

        with col_left:
            if not strata_summary.empty:
                st.dataframe(
                    strata_summary, use_container_width=True, hide_index=True
                )
            else:
                st.info("No stratification data available yet.")

        with col_right:
            if not strata_summary.empty:
                fig_strata = px.bar(
                    strata_summary,
                    x="Count",
                    y="Stratum",
                    color="Stratum",
                    text="Count",
                    orientation="h",
                    labels={"Count": "Participants"},
                    height=280,
                )
                fig_strata.update_layout(
                    showlegend=False,
                    margin=dict(l=20, r=20, t=10, b=20),
                    yaxis=dict(categoryorder="total ascending"),
                )
                st.plotly_chart(fig_strata, use_container_width=True)

        st.markdown("---")

        st.subheader("Stratified Participant Demographics")
        if not strata_demographics.empty and (
            "Age" in strata_demographics.columns or "Sex" in strata_demographics.columns
        ):
            demo_col1, demo_col2 = st.columns(2)

            with demo_col1:
                if "Age" in strata_demographics.columns:
                    ages = strata_demographics["Age"].dropna()
                    fig_age = px.histogram(
                        strata_demographics,
                        x="Age",
                        nbins=15,
                        labels={"Age": "Age (years)", "count": "Number of Participants"},
                        height=340,
                    )
                    if not ages.empty:
                        median_age = ages.median()
                        q1, q3 = ages.quantile(0.25), ages.quantile(0.75)
                        fig_age.add_vrect(
                            x0=q1, x1=q3,
                            fillcolor="gray", opacity=0.2, line_width=0,
                            annotation_text="IQR", annotation_position="top left",
                        )
                        fig_age.add_vline(
                            x=median_age, line_dash="dash", line_color="white",
                            annotation_text=f"Median: {median_age:.0f}",
                            annotation_position="top",
                        )
                    fig_age.update_layout(
                        margin=dict(l=20, r=20, t=30, b=20),
                        yaxis_title="Number of Participants",
                    )
                    st.plotly_chart(fig_age, use_container_width=True)
                    if not ages.empty:
                        st.caption(
                            f"Median age: {median_age:.0f} years "
                            f"(IQR: {q1:.0f}\u2013{q3:.0f}, n={len(ages)})"
                        )
                else:
                    st.info("No age field found — check AGE_FIELD_CANDIDATES.")

            with demo_col2:
                if "Sex" in strata_demographics.columns:
                    sex_counts = (
                        strata_demographics.groupby("Sex")
                        .size()
                        .reset_index(name="Count")
                    )
                    fig_sex = px.pie(
                        sex_counts,
                        names="Sex",
                        values="Count",
                        labels={"Sex": "Gender"},
                        height=340,
                    )
                    fig_sex.update_traces(
                        textinfo="label+percent+value",
                        textposition="inside",
                    )
                    fig_sex.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig_sex, use_container_width=True)
                else:
                    st.info("No sex field found — check SEX_FIELD_CANDIDATES.")
        else:
            st.info(
                "No age/sex fields detected for stratified participants. "
                "Update AGE_FIELD_CANDIDATES / SEX_FIELD_CANDIDATES to match your data dictionary."
            )

        st.markdown("---")

        trend_col1, trend_col2 = st.columns(2)

        with trend_col1:
            st.subheader("Cumulative Stratified Enrollment (by Week)")
            if not cumulative_stratified.empty:
                fig_cum = px.line(
                    cumulative_stratified,
                    x="Week",
                    y="Cumulative Stratified",
                    markers=True,
                    text="Cumulative Stratified",
                    labels={
                        "Week": "Week Starting",
                        "Cumulative Stratified": "Cumulative Participants Stratified",
                    },
                    height=340,
                    color_discrete_sequence=[px.colors.sequential.Viridis[4]],
                )
                fig_cum.update_traces(textposition="top center")
                fig_cum.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_cum, use_container_width=True)
            else:
                st.info("No stratification date data available yet.")

        with trend_col2:
            st.subheader("Enrollments by Week")
            if not weekly_enrollment.empty:
                fig_weekly = px.bar(
                    weekly_enrollment,
                    x="Week",
                    y="Enrollments",
                    text="Enrollments",
                    color="Enrollments",
                    color_continuous_scale="Viridis",
                    labels={"Week": "Week Starting", "Enrollments": "Participants Enrolled"},
                    height=340,
                )
                fig_weekly.update_traces(textposition="outside")
                fig_weekly.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_weekly, use_container_width=True)
            else:
                st.info("No enrollment date data available yet.")

        st.markdown("---")

        st.subheader("Visit Adherence Overview")
        if not visit_matrix.empty:
            status_counts = (
                visit_matrix.apply(pd.Series.value_counts).fillna(0).astype(int)
            )
            for status in ["Completed", "Pending", "Missed", "Rescheduled", "Early Termination"]:
                if status not in status_counts.index:
                    status_counts.loc[status] = 0

            status_long = (
                status_counts.T.reset_index()
                .melt(id_vars="index", var_name="Status", value_name="Participants")
                .rename(columns={"index": "Visit Window"})
            )
            status_long = status_long[status_long["Participants"] > 0]

            fig_adherence = px.bar(
                status_long,
                x="Visit Window",
                y="Participants",
                color="Status",
                text="Participants",
                barmode="stack",
                color_discrete_map={
                    "Completed": "#2e7d32",
                    "Pending": "#ffc107",
                    "Missed": "#dc3545",
                    "Rescheduled": "#17a2b8",
                    "Early Termination": "#6c757d",
                },
                height=420,
            )
            fig_adherence.update_traces(textposition="inside")
            fig_adherence.update_layout(margin=dict(l=20, r=20, t=30, b=20))
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
                    "Early Termination": "background-color: #495057; color: white",
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

            st.markdown("---")

            st.subheader("Retention by Week")
            st.markdown(
                "Retention % = Completed ÷ (Completed + Missed + Rescheduled) "
                "for each visit window."
            )
            if not weekly_retention.empty:
                fig_retention = px.line(
                    weekly_retention,
                    x="Week",
                    y="Retention %",
                    markers=True,
                    text="Retention %",
                    height=380,
                )
                fig_retention.update_traces(textposition="top center")
                fig_retention.update_layout(
                    yaxis=dict(range=[0, 105]),
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig_retention, use_container_width=True)
                st.dataframe(
                    weekly_retention, use_container_width=True, hide_index=True
                )
            else:
                st.info("Not enough visit outcome data to compute retention yet.")
        else:
            st.info("No visit data to display.")

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
        st.text_area("Copy these stats to email/Teams", stats_text, height=300)

if __name__ == "__main__":
    main()