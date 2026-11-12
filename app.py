"""
SOAR Study Enrollment & Follow-up Tracker
=========================================
A Streamlit dashboard for the SOAR Study REDCap project.
Displays enrollment funnel, retention, safety screening
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
import re
from datetime import datetime, timedelta

import numpy as np
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
    /* Garamond everywhere. EB Garamond is loaded as a web-font fallback for
       machines that do not have Garamond installed. Streamlit's icon font is
       excluded so icons (expander arrows, metric deltas) keep rendering. */
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;500;600;700;800&display=swap');
    .stApp, .stApp div, .stApp p, .stApp label, .stApp li, .stApp a, .stApp small,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp button, .stApp input, .stApp textarea, .stApp select,
    .stApp table, .stApp th, .stApp td,
    .stApp span:not([data-testid="stIconMaterial"]) {
        font-family: 'EB Garamond', Garamond, 'Cormorant Garamond', 'Times New Roman', serif !important;
    }
    [data-testid="stIconMaterial"] {
        font-family: "Material Symbols Rounded", "Material Icons" !important;
    }

    /* Theme-adaptive headers: var(--text-color) tracks Streamlit's active
       light/dark theme, and the accent blue + shadow keep contrast strong
       against both a near-white and a near-black page background. */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #4a90d9;
        letter-spacing: 0.01em;
        text-shadow: 0 1px 3px rgba(0,0,0,0.35);
    }
    .sub-header {
        font-size: 1.1rem;
        font-weight: 500;
        color: var(--text-color, #555);
        opacity: 0.85;
        margin-bottom: 1rem;
    }
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

    /* Hide the "Made with Streamlit" / GitHub footer badges at the bottom of the page */
    footer { visibility: hidden !important; height: 0 !important; }
    [data-testid="stFooter"] { visibility: hidden !important; height: 0 !important; }
    a[href*="streamlit.io"],
    a[href*="github.com"],
    [data-testid="stToolbarActions"],
    [data-testid="stAppDeployButton"],
    [class*="viewerBadge"],
    a[href*="github.com"][class*="viewerBadge"],
    .viewerBadge_link__qRIco,
    .viewerBadge_container__1QSob { display: none !important; }
</style>
""", unsafe_allow_html=True)

PLOT_FONT_FAMILY = "EB Garamond, Garamond, Cormorant Garamond, Times New Roman, serif"


def show_plot(fig, **kwargs):
    """st.plotly_chart with Garamond applied. Streamlit's chart theme would
    otherwise override the font, so it is set explicitly on every figure."""
    fig.update_layout(font=dict(family=PLOT_FONT_FAMILY, size=14))
    return st.plotly_chart(fig, **kwargs)


# ───────────────────────────────────────────────────────────────
# SHARED CONSTANTS
# ───────────────────────────────────────────────────────────────
STRATA_MAP = {
    "1": "Low Smoking + Low Alcohol",
    "2": "Low Smoking + High Alcohol",
    "3": "High Smoking + Low Alcohol",
    "4": "High Smoking + High Alcohol",
}

# Age and gender fields, per the SOAR data dictionary (pre_screening form):
#   prescreen_age (calc, years) — round(datediff([prescreen_dob], "today", "d") / 365.25, 0)
#   gender (radio) — 1=Male, 2=Female
AGE_FIELD_CANDIDATES = ["prescreen_age", "age", "participant_age"]
SEX_FIELD_CANDIDATES = ["gender", "sex", "participant_sex"]
SEX_MAP = {"1": "Male", "2": "Female"}

VISIT_WINDOW_ORDER = ["Day 0", "Week 1", "Week 2", "Week 3", "Week 12", "Week 36"]
# Follow-up windows only (Day 0 is the medication-start visit, not a follow-up),
# used for the retention table.
FOLLOWUP_WINDOWS = [w for w in VISIT_WINDOW_ORDER if w != "Day 0"]

# visit_log.visit_window coding. ASSUMPTION: Day 0 was added as code "0" and
# the existing codes 1-5 were left unchanged. If the data dictionary recoded
# the field (e.g. Day 0 = 1, Week 1 = 2, ...), change this one map. Label text
# is also accepted so a labelled export works too.
VISIT_WINDOW_CODE_MAP = {
    "0": "Day 0",
    "1": "Week 1",
    "2": "Week 2",
    "3": "Week 3",
    "4": "Week 12",
    "5": "Week 36",
    **{w: w for w in VISIT_WINDOW_ORDER},
}

# Medication visits used for retention (Day 0 = medication start, so it forms
# the denominator rather than a follow-up). Due dates come from the
# week_#_due fields; if one is missing for a participant it falls back to
# Day 0 visit date + the nominal offset below.
MED_FOLLOWUP_WINDOWS = ["Week 1", "Week 2", "Week 3", "Week 12", "Week 36"]
MED_DUE_FIELDS = {
    "Week 1": "week_1_due", "Week 2": "week_2_due", "Week 3": "week_3_due",
    "Week 12": "week_12_due", "Week 36": "week_36_due",
}
MED_DUE_FALLBACK_DAYS = {
    "Week 1": 7, "Week 2": 14, "Week 3": 21, "Week 12": 84, "Week 36": 252,
}

# Date field on the visit_log form, tried in order. A Day 0 visit-log row with
# a date in this field = participant started medication. If none of these
# exist, any visit-related *date* column in the visit data is auto-detected.
VISIT_DATE_FIELD_CANDIDATES = [
    "visit_date", "vl_visit_date", "visit_log_date", "actual_visit_date",
    "visit_actual_date", "date_of_visit",
]

# Screening date field (pre_screening form), tried in order. ASSUMPTION: the
# exact name isn't in this file, so if none of these exist the Enrollments view
# auto-detects a "screen*date" column and lets you pick another from a dropdown.
SCREENING_DATE_FIELD_CANDIDATES = [
    "prescreen_date", "prescreening_date", "pre_screening_date",
    "screening_date", "screen_date", "date_screened", "prescreen_visit_date",
    "ps_date",
]

# Target rate of stratified (randomized) enrollment, used both for the
# "Enrollments by Week" reference line and the "Cumulative Enrollments"
# target line.
WEEKLY_ENROLLMENT_TARGET = 4

AGE_GROUP_LABELS = ["18-30", "31-40", "41-50", "51-60", "60+"]
AGE_GROUP_BINS = [17, 30, 40, 50, 60, float("inf")]

# Behavioral Intervention Assigned field, from the clinical_eval (CE) form.
# Raw REDCap field: bt_intervention_type, coded 1 = PSF, 2 = BA. Mapped
# case-insensitively so it also matches if the export ever returns the
# label text ("PSF"/"BA") directly instead of the numeric code.
BEHAVIORAL_INTERVENTION_FIELD = "bt_intervention_type"
BEHAVIORAL_INTERVENTION_MAP = {"1": "PSF", "2": "BA", "PSF": "PSF", "BA": "BA"}

PSF_SESSION_STATUS_FIELDS = [f"bt_psf_s{i}_status" for i in range(1, 9)]

PSF_SESSION_DAY_OFFSETS = [0, 8, 15, 22, 36, 50, 64, 78]

# Field holding Day 0 (medication/intervention date), tried in order.
# "day_0_due" is already used elsewhere in this app, so it is the fallback.
DAY0_FIELD_CANDIDATES = [
    "day_0_date", "day0_date", "medication_date", "med_date",
    "intervention_date", "ce_medication_date", "day_0_due",
]

PSF_SESSION_STATUS_LABELS = {
    field: f"Session {i}" for i, field in enumerate(PSF_SESSION_STATUS_FIELDS, start=1)
}

# Adverse Event Log fields, from the repeating `adverse_event_log` instrument
# (data dictionary: ae_severity, ae_outcome). AEs are captured one row per
# event, so a participant with multiple events contributes multiple rows —
# summaries below count events, not unique participants.
#
# NOTE: a raw export returns the instrument's *unique name*
# ("adverse_event_log") in redcap_repeat_instrument, while a labelled export
# returns the *label* ("Adverse Event Log"). Comparisons go through
# _norm_instrument() so either form matches.
AE_REPEAT_INSTRUMENT_LABEL = "adverse_event_log"
AE_SEVERITY_FIELD = "ae_severity"
AE_SEVERITY_MAP = {"1": "Mild", "2": "Moderate", "3": "Severe"}
AE_SEVERITY_ORDER = ["Mild", "Moderate", "Severe"]
AE_OUTCOME_FIELD = "ae_outcome"
AE_OUTCOME_MAP = {
    "1": "Resolved",
    "2": "Recovering",
    "3": "Ongoing",
    "4": "Resolved with sequelae",
    "5": "Fatal",
    "6": "Unknown",
    # A labelled export returns the old label text; map it to the new wording.
    "Recovered": "Resolved",
    "Recovered with sequelae": "Resolved with sequelae",
}


def _norm_instrument(value):
    """Normalize a REDCap instrument name/label for comparison.

    "Adverse Event Log", "adverse event log" and "adverse_event_log" all
    normalize to "adverse_event_log", so the same filter works whether the
    export was raw or labelled.
    """
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _is_blank(series):
    """True where a REDCap value is effectively empty.

    A REDCap/PyCap export returns unfilled fields as empty strings rather
    than NaN, so .notna() is True for every row and cannot be used to tell
    populated rows from blank ones.
    """
    return series.astype(str).str.strip().replace({"nan": "", "None": ""}) == ""


# Rocket colorscale stops (seaborn's "rocket" palette, sampled 0→1),
# used in place of Viridis for all standard bar/line/pie charts.
#ROCKET_COLORSCALE = [
#    "#03051a", "#221331", "#451c47", "#691f55", "#921c5b", "#b91657",
#    "#d92847", "#ed503e", "#f47d57", "#f6a47c", "#f7c9aa", "#faebdd",
#]

ROCKET_COLORSCALE = [
    "#9e0142", "#d0384e", "#ee6445", "#fa9b58", "#fece7c", "#fff1a8",
    "#f4faad", "#d1ed9c", "#97d5a4", "#5cb7aa", "#3682ba", "#5e4fa2",
]
# Distinct 5-color palette (grey, blue, purple, orange, green) used only
# for the Enrollment Funnel, which keeps its own scheme apart from Rocket.
FUNNEL_COLORS = ["#6c757d", "#2980b9", "#8e44ad", "#e67e22", "#27ae60"]


def rocket_colors(n):
    """Return n colors sampled evenly across the Rocket colorscale."""
    if n <= 0:
        return []
    if n == 1:
        return [px.colors.sample_colorscale(ROCKET_COLORSCALE, [0.5])[0]]
    return px.colors.sample_colorscale(ROCKET_COLORSCALE, [i / (n - 1) for i in range(n)])


def funnel_palette(n):
    """Return n colors for the funnel: grey/blue/purple/orange/green,
    extended by sampling the same hues as a scale if more than 5 stages."""
    if n <= 0:
        return []
    if n <= len(FUNNEL_COLORS):
        return FUNNEL_COLORS[:n]
    return px.colors.sample_colorscale(FUNNEL_COLORS, [i / (n - 1) for i in range(n)])

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

    # Optional CA bundle for servers that don't send the full certificate
    # chain (e.g. missing GoDaddy intermediate). Resolution order:
    #   1. REDCAP_CA_BUNDLE env var / Streamlit secret
    #   2. redcap_ca_bundle.pem sitting next to app.py
    #   3. default system/certifi trust store
    # Once the server serves the full chain, remove the bundle and this
    # falls back to normal verification. Never disable verification: this
    # connection carries the API token and participant-level data.
    ca_bundle = os.getenv("REDCAP_CA_BUNDLE", "")
    if not ca_bundle:
        try:
            ca_bundle = st.secrets.get("REDCAP_CA_BUNDLE", "")
        except Exception:
            pass
    if not ca_bundle:
        local_bundle = os.path.join(os.path.dirname(os.path.abspath(__file__)), "redcap_ca_bundle.pem")
        if os.path.isfile(local_bundle):
            ca_bundle = local_bundle

    try:
        if ca_bundle:
            proj = Project(api_url, api_token, verify_ssl=ca_bundle)
        else:
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


@st.cache_data(ttl=1800, show_spinner="Pulling behavioral intervention assignments...")
def load_behavioral_tracking(_proj):
    """Export the behavioral_tracking form directly.

    The intervention field and the 8 PSF session status fields all live on
    the separate `behavioral_tracking` instrument. A project-wide export can
    omit a form/field when the API user's export permissions do not include
    it, so do not depend on the main export containing `bt_intervention_type`
    or the `bt_psf_s#_status` fields.
    """
    if _proj is None:
        return pd.DataFrame(), "REDCap connection is not available."

    field = BEHAVIORAL_INTERVENTION_FIELD
    all_fields = ["record_id", field] + PSF_SESSION_STATUS_FIELDS

    # Preferred: explicitly request the form and fields.
    try:
        records = _proj.export_records(
            fields=all_fields,
            forms=["behavioral_tracking"],
            raw_or_label="raw",
        )
        if isinstance(records, pd.DataFrame):
            df = records.reset_index()
        else:
            df = pd.DataFrame(records)

        if field in df.columns:
            return df, None
    except Exception as e:
        first_error = str(e)
    else:
        first_error = "The requested field was not returned."

    # Fallback: request the fields directly without restricting by form.
    try:
        records = _proj.export_records(
            fields=all_fields,
            raw_or_label="raw",
        )
        if isinstance(records, pd.DataFrame):
            df = records.reset_index()
        else:
            df = pd.DataFrame(records)

        if field in df.columns:
            return df, None
    except Exception as e:
        second_error = str(e)
    else:
        second_error = "The field was not returned."

    return (
        pd.DataFrame(),
        f"REDCap did not return '{field}' from the Behavioral Tracking form. "
        "The data dictionary confirms this field belongs to the "
        "'behavioral_tracking' instrument and is coded 1=PSF, 2=BA. "
        "Check the API token/user's Data Export rights for that instrument. "
        f"API details: {first_error}; fallback: {second_error}.",
    )


@st.cache_data(ttl=1800, show_spinner="Pulling adverse event log...")
def load_adverse_events(_proj):
    """Export the adverse_event_log form directly.

    ae_severity and ae_outcome live on the separate, repeating
    `adverse_event_log` instrument, just like bt_intervention_type lives on
    behavioral_tracking above. A project-wide export can omit a form/field
    the API user's export permissions don't include, so do not depend on
    the main export containing them.
    """
    if _proj is None:
        return pd.DataFrame(), "REDCap connection is not available."

    ae_fields = [
        "record_id",
        "ae_participant_id",
        "ae_date_onset",
        AE_SEVERITY_FIELD,
        AE_OUTCOME_FIELD,
    ]

    # Preferred: explicitly request the form and fields.
    try:
        records = _proj.export_records(
            fields=ae_fields,
            forms=["adverse_event_log"],
            raw_or_label="raw",
        )
        if isinstance(records, pd.DataFrame):
            df = records.reset_index()
        else:
            df = pd.DataFrame(records)

        if AE_SEVERITY_FIELD in df.columns:
            return df, None
    except Exception as e:
        first_error = str(e)
    else:
        first_error = "The requested field was not returned."

    # Fallback: request the fields directly without restricting by form.
    try:
        records = _proj.export_records(
            fields=ae_fields,
            raw_or_label="raw",
        )
        if isinstance(records, pd.DataFrame):
            df = records.reset_index()
        else:
            df = pd.DataFrame(records)

        if AE_SEVERITY_FIELD in df.columns:
            return df, None
    except Exception as e:
        second_error = str(e)
    else:
        second_error = "The field was not returned."

    return (
        pd.DataFrame(),
        f"REDCap did not return '{AE_SEVERITY_FIELD}' from the Adverse Event "
        "Log form. The data dictionary confirms this field belongs to the "
        "'adverse_event_log' instrument. Check the API token/user's Data "
        "Export rights for that instrument. "
        f"API details: {first_error}; fallback: {second_error}.",
    )


def build_psf_ba_distribution(df_behavioral):
    """Count + percentage of participants assigned to PSF vs BA.

    Source: behavioral_tracking.bt_intervention_type
    REDCap coding: 1 = PSF, 2 = BA.
    """
    empty = pd.DataFrame(columns=["Group", "Count", "Percent"])
    if df_behavioral.empty:
        return empty, "No Behavioral Tracking records were returned."

    field = BEHAVIORAL_INTERVENTION_FIELD
    if field not in df_behavioral.columns:
        return empty, f"Field '{field}' was not returned by the Behavioral Tracking export."

    id_col = "record_id" if "record_id" in df_behavioral.columns else df_behavioral.columns[0]

    values = df_behavioral[[id_col, field]].copy()

    # Normalize both raw REDCap codes (1/2) and labels (PSF/BA).
    values[field] = values[field].astype(str).str.strip().str.upper()
    values["Label"] = values[field].map(BEHAVIORAL_INTERVENTION_MAP)
    values = values.dropna(subset=["Label"])

    # If the behavioral_tracking form is repeated, retain one assignment per participant.
    values = values.drop_duplicates(subset=[id_col], keep="first")

    if values.empty:
        return (
            empty,
            f"Field '{field}' was returned but contains no valid PSF/BA values yet.",
        )

    counts = (
        values["Label"]
        .value_counts()
        .reindex(["PSF", "BA"], fill_value=0)
        .astype(int)
    )
    total = int(counts.sum())

    dist = pd.DataFrame(
        {
            "Group": counts.index,
            "Count": counts.values,
            "Percent": (counts.values / total * 100).round(1) if total else 0,
        }
    )
    return dist, None


def get_day0_dates(df_all):
    """Return (Series record_id -> Day 0 date, field_name used or None)."""
    empty = pd.Series(dtype="datetime64[ns]")
    if df_all is None or df_all.empty:
        return empty, None
    id_col = "record_id" if "record_id" in df_all.columns else df_all.columns[0]
    field = next((c for c in DAY0_FIELD_CANDIDATES if c in df_all.columns), None)
    if field is None:
        return empty, None
    d = df_all[[id_col, field]].copy()
    d[id_col] = d[id_col].astype(str)
    d[field] = pd.to_datetime(d[field], errors="coerce")
    d = d.dropna(subset=[field])
    if d.empty:
        return empty, field
    return d.groupby(id_col)[field].min(), field


def build_psf_session_summary(df_behavioral, day0_dates=None):
    """Per-session status for PSF-assigned participants.

    Source: behavioral_tracking.bt_psf_s1_status ... bt_psf_s8_status
    (1 = Done, 0 = Not Done) plus Day 0 and PSF_SESSION_DAY_OFFSETS.

    For each session:
      Due        = today >= Day 0 + offset (or the session is documented Done)
      Completed  = Done documented
      Missed     = Due and (Not Done, or not filled)
      Pending    = not yet due   (Due = Completed + Missed)
    Participants with no Day 0 date cannot be dated, so anything not
    completed for them is counted as Pending.
    Returns (summary_df, total_psf, warning).
    """
    cols = ["Session", "Due", "Completed", "Missed", "Pending", "Total"]
    empty = pd.DataFrame(columns=cols)
    if df_behavioral.empty:
        return empty, 0, "No Behavioral Tracking records were returned."

    field = BEHAVIORAL_INTERVENTION_FIELD
    if field not in df_behavioral.columns:
        return empty, 0, f"Field '{field}' was not returned by the Behavioral Tracking export."

    id_col = "record_id" if "record_id" in df_behavioral.columns else df_behavioral.columns[0]

    raw = df_behavioral.copy()
    raw[id_col] = raw[id_col].astype(str)

    df = raw.copy()
    df[field] = df[field].astype(str).str.strip().str.upper()
    df["Label"] = df[field].map(BEHAVIORAL_INTERVENTION_MAP)
    # Drop rows with no valid Label BEFORE deduplicating (same ordering as
    # build_psf_ba_distribution, which this must match).
    df = df.dropna(subset=["Label"])
    df = df.drop_duplicates(subset=[id_col], keep="first")
    psf_ids = pd.Index(df.loc[df["Label"] == "PSF", id_col].unique())

    total_psf = int(len(psf_ids))
    if total_psf == 0:
        return empty, 0, "No participants have been assigned to PSF yet."

    present = [f for f in PSF_SESSION_STATUS_FIELDS if f in raw.columns]
    if not present:
        return (
            empty,
            total_psf,
            "None of the PSF session status fields (bt_psf_s1_status ... "
            "bt_psf_s8_status) were returned by the export.",
        )

    # Session statuses may sit on a different row from the assignment row,
    # so combine every row per participant: any "1" wins, then any "0".
    def _combine(series):
        v = series.astype(str).str.strip()
        if (v == "1").any():
            return "1"
        if (v == "0").any():
            return "0"
        return ""

    sub = raw[raw[id_col].isin(psf_ids)]
    status = sub.groupby(id_col)[present].agg(_combine).reindex(psf_ids).fillna("")

    if day0_dates is not None and len(day0_dates) > 0:
        day0 = day0_dates.reindex(psf_ids)
    else:
        day0 = pd.Series(pd.NaT, index=psf_ids, dtype="datetime64[ns]")

    today = pd.Timestamp.now().normalize()
    rows = []
    for i, (sess_field, offset) in enumerate(
        zip(PSF_SESSION_STATUS_FIELDS, PSF_SESSION_DAY_OFFSETS), start=1
    ):
        if sess_field in status.columns:
            completed = status[sess_field] == "1"
        else:
            completed = pd.Series(False, index=psf_ids)
        due_date = day0 + pd.Timedelta(days=offset)
        due = (due_date.notna() & (today >= due_date)) | completed
        missed = due & ~completed
        pending = ~due
        rows.append(
            {
                "Session": f"Session {i}",
                "Due": int(due.sum()),
                "Completed": int(completed.sum()),
                "Missed": int(missed.sum()),
                "Pending": int(pending.sum()),
                "Total": total_psf,
            }
        )

    summary = pd.DataFrame(rows)
    notes = []
    missing_fields = [f for f in PSF_SESSION_STATUS_FIELDS if f not in raw.columns]
    if missing_fields:
        notes.append(
            f"{len(missing_fields)} of 8 session status fields were not returned "
            "and are treated as not filled: " + ", ".join(missing_fields)
        )
    n_no_day0 = int(day0.isna().sum())
    if n_no_day0 == total_psf:
        notes.append(
            "No Day 0 (medication/intervention) date was found for any PSF "
            "participant, so due dates cannot be calculated. Set "
            "DAY0_FIELD_CANDIDATES to the right field name."
        )
    elif n_no_day0:
        notes.append(
            f"{n_no_day0} PSF participant(s) have no Day 0 date; their "
            "uncompleted sessions are counted as Pending."
        )
    return summary, total_psf, (" ".join(notes) if notes else None)


def build_ae_summary(df_all):
    """Summarize Adverse Event Log entries: total count, severity
    breakdown (Mild/Moderate/Severe), and outcome breakdown.

    Source: the repeating `adverse_event_log` instrument (ae_severity,
    ae_outcome), normally passed in as the dedicated export from
    load_adverse_events() rather than the project-wide export, since export
    permissions can omit this form from the latter. AEs are counted as
    events (rows), regardless of which
    participant/record_id they belong to — a participant with several
    logged events contributes several rows here, by design.
    """
    empty_severity = pd.DataFrame(columns=["Severity", "Count"])
    empty_outcome = pd.DataFrame(columns=["Outcome", "Count"])
    empty_ae = pd.DataFrame()

    if df_all.empty or AE_SEVERITY_FIELD not in df_all.columns:
        return empty_ae, empty_severity, empty_outcome, (
            "Adverse Event Log fields were not returned by the export."
        )

    df = df_all.copy()

    def _populated_ae_rows(frame):
        """Rows carrying actual AE content.

        Keyed on identity/onset first (populated the moment an AE row is
        created) so an event still awaiting severity or outcome grading is
        not silently dropped from the count.
        """
        mask = None
        for col in ("ae_participant_id", "ae_date_onset", AE_SEVERITY_FIELD, AE_OUTCOME_FIELD):
            if col in frame.columns:
                filled = ~_is_blank(frame[col])
                mask = filled if mask is None else (mask | filled)
        if mask is None:
            return frame.iloc[0:0]
        return frame[mask]

    if "redcap_repeat_instrument" in df.columns:
        ae = df[
            df["redcap_repeat_instrument"].map(_norm_instrument)
            == _norm_instrument(AE_REPEAT_INSTRUMENT_LABEL)
        ]
        # Fall back to rows with populated AE content, in case the
        # repeat-instrument marker column is blank or absent for this
        # export configuration.
        if ae.empty:
            ae = _populated_ae_rows(df)
    else:
        ae = _populated_ae_rows(df)

    total_ae = int(len(ae))
    if total_ae == 0:
        return ae, empty_severity, empty_outcome, "No Adverse Event Log entries recorded yet."

    def _label_counts(series, code_map, order=None):
        vals = series.astype(str).str.strip()
        # Raw REDCap codes (e.g. "1", "2") map through code_map; values that
        # are already label text (e.g. a labeled export returning "Mild"
        # directly) pass through unchanged.
        mapped = vals.map(code_map).fillna(vals)
        mapped = mapped.replace({"nan": None, "": None, "None": None}).dropna()
        counts = mapped.value_counts()
        if order:
            ordered_index = [c for c in order if c in counts.index] + [
                c for c in counts.index if c not in order
            ]
            counts = counts.reindex(ordered_index)
        return counts

    severity_counts = _label_counts(ae[AE_SEVERITY_FIELD], AE_SEVERITY_MAP, AE_SEVERITY_ORDER)
    severity_summary = pd.DataFrame(
        {"Severity": severity_counts.index, "Count": severity_counts.values.astype(int)}
    )

    if AE_OUTCOME_FIELD in ae.columns:
        outcome_counts = _label_counts(ae[AE_OUTCOME_FIELD], AE_OUTCOME_MAP)
        outcome_summary = pd.DataFrame(
            {"Outcome": outcome_counts.index, "Count": outcome_counts.values.astype(int)}
        )
    else:
        outcome_summary = empty_outcome

    return ae, severity_summary, outcome_summary, None

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

    df = df_visits.copy()
    df["visit_label"] = df["visit_window"].astype(str).str.strip().map(VISIT_WINDOW_CODE_MAP)
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
        "4": "Withdrawn/Discontinued",
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
        "Day 0": "day_0_due",
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
        if "visit_window" in completed.columns:
            completed["visit_label"] = completed["visit_window"].astype(str).str.strip().map(
                VISIT_WINDOW_CODE_MAP
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

def _started_medication_rows(df_visits):
    """Day 0 visit-log rows that carry a date (= medication started).

    Returns (dated_rows_df, id_col, date_field_used, warning).
    """
    if df_visits.empty or "visit_window" not in df_visits.columns:
        return None, None, None, "No visit log data available to count medication starts."

    field = next((c for c in VISIT_DATE_FIELD_CANDIDATES if c in df_visits.columns), None)
    if field is None:
        field = next(
            (
                c for c in df_visits.columns
                if "date" in c.lower() and "visit" in c.lower()
                and not any(x in c.lower() for x in ("due", "next", "window"))
            ),
            None,
        )
    if field is None:
        return None, None, None, (
            "Could not find the visit date field on the visit log. Set "
            "VISIT_DATE_FIELD_CANDIDATES to its name to count medication starts."
        )

    id_col = "record_id" if "record_id" in df_visits.columns else df_visits.columns[0]
    day0 = df_visits[
        df_visits["visit_window"].astype(str).str.strip().map(VISIT_WINDOW_CODE_MAP) == "Day 0"
    ]
    dated = day0[pd.to_datetime(day0[field], errors="coerce").notna()]
    return dated, id_col, field, None


def count_started_medication(df_visits):
    """Participants who started medication = unique participants with a date
    on their Day 0 visit-log row.

    Returns (count, date_field_used, warning).
    """
    dated, id_col, field, warning = _started_medication_rows(df_visits)
    if warning:
        return 0, None, warning
    return int(dated[id_col].astype(str).nunique()), field, None


def get_started_medication_ids(df_visits):
    """Set of record IDs (as str) that started medication (same rule as above)."""
    dated, id_col, _, warning = _started_medication_rows(df_visits)
    if warning:
        return set()
    return set(dated[id_col].astype(str).str.strip())


def get_excluded_participants(df_visits, df_clinical):
    """Participants to take out of the Action Required lists.

    - Terminated: any visit-log row with visit_status = 4 (Withdrawn/Discontinued).
    - Missed Day 0: Day 0 visit logged as Missed (status 2) with no Day 0 visit
      logged as Completed. Anyone already listed as terminated is not repeated.
    Returns (terminated_df, missed_day0_df).
    """
    term_cols = ["Record ID", "Participant ID", "Withdrawn/Discontinued At"]
    miss_cols = ["Record ID", "Participant ID", "Day 0 Due Date"]
    empty_t = pd.DataFrame(columns=term_cols)
    empty_m = pd.DataFrame(columns=miss_cols)
    if (
        df_visits.empty
        or "visit_window" not in df_visits.columns
        or "visit_status" not in df_visits.columns
    ):
        return empty_t, empty_m

    id_col = "record_id" if "record_id" in df_visits.columns else df_visits.columns[0]
    v = df_visits.copy()
    v[id_col] = v[id_col].astype(str)
    v["visit_label"] = v["visit_window"].astype(str).str.strip().map(VISIT_WINDOW_CODE_MAP)
    v["status"] = v["visit_status"].astype(str).str.strip()
    v = v[v["visit_label"].notna()]

    study_ids, day0_due = {}, {}
    if not df_clinical.empty:
        cid = "record_id" if "record_id" in df_clinical.columns else df_clinical.columns[0]
        c = df_clinical.copy()
        c[cid] = c[cid].astype(str)
        if "study_id" in c.columns:
            sid = c[~_is_blank(c["study_id"])].drop_duplicates(cid)
            study_ids = sid.set_index(cid)["study_id"].to_dict()
        if "day_0_due" in c.columns:
            d = c.assign(_d=pd.to_datetime(c["day_0_due"], errors="coerce"))
            d = d.dropna(subset=["_d"]).drop_duplicates(cid)
            day0_due = d.set_index(cid)["_d"].dt.strftime("%Y-%m-%d").to_dict()

    order = {w: i for i, w in enumerate(VISIT_WINDOW_ORDER)}
    term = v[v["status"] == "4"].copy()
    term["_o"] = term["visit_label"].map(order)
    term = term.sort_values("_o").drop_duplicates(id_col)
    terminated_df = pd.DataFrame(
        {
            "Record ID": term[id_col].values,
            "Participant ID": [study_ids.get(r, "N/A") for r in term[id_col]],
            "Withdrawn/Discontinued At": term["visit_label"].values,
        }
    ) if not term.empty else empty_t

    day0 = v[v["visit_label"] == "Day 0"]
    completed_ids = set(day0.loc[day0["status"] == "1", id_col])
    terminated_ids = set(terminated_df["Record ID"])
    missed_ids = [
        r for r in day0.loc[day0["status"] == "2", id_col].drop_duplicates()
        if r not in completed_ids and r not in terminated_ids
    ]
    missed_day0_df = pd.DataFrame(
        {
            "Record ID": missed_ids,
            "Participant ID": [study_ids.get(r, "N/A") for r in missed_ids],
            "Day 0 Due Date": [day0_due.get(r, "") for r in missed_ids],
        }
    ) if missed_ids else empty_m

    return terminated_df, missed_day0_df


def exclude_records(df, record_ids):
    """Drop rows whose "Record ID" is in record_ids (no-op on empty input)."""
    if df.empty or "Record ID" not in df.columns or not record_ids:
        return df
    return df[~df["Record ID"].astype(str).isin(record_ids)].reset_index(drop=True)


def render_excluded_lists(terminated_df, missed_day0_df):
    """Show the Terminated and Missed Day 0 lists side by side."""
    col_t, col_m = st.columns(2)
    with col_t:
        st.markdown(f"**{len(terminated_df)} Terminated from Study**")
        if not terminated_df.empty:
            st.dataframe(terminated_df, use_container_width=True, hide_index=True)
        else:
            st.info("No participants terminated from the study.")
    with col_m:
        st.markdown(f"**{len(missed_day0_df)} Missed Day 0**")
        if not missed_day0_df.empty:
            st.dataframe(missed_day0_df, use_container_width=True, hide_index=True)
        else:
            st.success("No participants have missed Day 0.")


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


def build_stratified_demographics(df_clinical, df_prescreen):
    """Age & sex breakdown for participants who have an assigned stratum.
    """
    if df_clinical.empty or "ce_assigned_strata" not in df_clinical.columns:
        return pd.DataFrame()

    stratified = df_clinical[
        df_clinical["ce_assigned_strata"].notna()
        & (df_clinical["ce_assigned_strata"] != "")
    ].copy()

    if stratified.empty:
        return pd.DataFrame()

    clin_id_col = "record_id" if "record_id" in stratified.columns else stratified.columns[0]

    age_col = next((c for c in AGE_FIELD_CANDIDATES if c in df_prescreen.columns), None)
    sex_col = next((c for c in SEX_FIELD_CANDIDATES if c in df_prescreen.columns), None)

    if age_col is not None or sex_col is not None:
        demo_source = df_prescreen
        demo_id_col = (
            "record_id" if "record_id" in df_prescreen.columns else df_prescreen.columns[0]
        )
    else:
        # Fall back to df_clinical itself in case forms share rows.
        age_col = next((c for c in AGE_FIELD_CANDIDATES if c in stratified.columns), None)
        sex_col = next((c for c in SEX_FIELD_CANDIDATES if c in stratified.columns), None)
        demo_source = stratified
        demo_id_col = clin_id_col

    if age_col is None and sex_col is None:
        return pd.DataFrame()

    demo_cols = [demo_id_col] + [c for c in [age_col, sex_col] if c is not None]
    demo = (
        demo_source[demo_cols]
        .dropna(subset=[demo_id_col])
        .drop_duplicates(subset=[demo_id_col], keep="first")
        .rename(columns={demo_id_col: clin_id_col})
    )

    merged = stratified[[clin_id_col, "ce_assigned_strata"]].merge(
        demo, on=clin_id_col, how="left"
    )

    out = pd.DataFrame()
    out["Stratum"] = merged["ce_assigned_strata"].astype(str).map(
        lambda k: STRATA_MAP.get(k, k)
    )

    if age_col is not None:
        out["Age"] = pd.to_numeric(merged[age_col], errors="coerce")
    if sex_col is not None:
        out["Sex"] = merged[sex_col].astype(str).map(lambda k: SEX_MAP.get(k, k))


    return out


def build_age_group_summary(ages):
    """Bin ages into standard groups (18-30, 31-40, 41-50, 51-60, 60+) and
    compute median/IQR summary stats. Returns (summary_df, stats_dict);
    both empty/blank if there's no valid numeric age data.
    """
    ages_num = pd.to_numeric(ages, errors="coerce").dropna()
    if ages_num.empty:
        return pd.DataFrame(), {}

    binned = pd.cut(ages_num, bins=AGE_GROUP_BINS, labels=AGE_GROUP_LABELS, right=True)
    counts = binned.value_counts().reindex(AGE_GROUP_LABELS, fill_value=0)
    total = int(counts.sum())

    summary = pd.DataFrame(
        {
            "Age Group": AGE_GROUP_LABELS,
            "Count": counts.values,
            "Percent": (counts.values / total * 100).round(1) if total > 0 else 0,
        }
    )
    summary["Label"] = summary.apply(
        lambda r: f"n={int(r['Count'])} ({r['Percent']:.0f}%)", axis=1
    )

    stats = {
        "total": total,
        "median": round(ages_num.median(), 1),
        "q1": round(ages_num.quantile(0.25), 1),
        "q3": round(ages_num.quantile(0.75), 1),
    }
    return summary, stats


def build_enrollment_trends(df_clinical, view_by="Weekly"):
    """Enrollment count and cumulative stratified-enrollment trend, by week or month.

    Uses `ce_date` (clinical evaluation date) as the enrollment-event date
    since it's the date field carried on the clinical eligibility form.
    - "Enrolled" = rows where ce_enrollment_decision == "1"
    - "Stratified" = rows where ce_assigned_strata is populated

    view_by: "Weekly" (weeks starting Monday) or "Monthly" (calendar months).
    Targets scale with the period: WEEKLY_ENROLLMENT_TARGET per week, or
    WEEKLY_ENROLLMENT_TARGET * (days in month / 7) per month.

    Returns (enrollment_df, cumulative_stratified_df), each empty if the
    required fields aren't present. Both carry a "Period" label column.
    """
    empty = (pd.DataFrame(), pd.DataFrame())
    if df_clinical.empty or "ce_date" not in df_clinical.columns:
        return empty

    monthly = view_by == "Monthly"
    freq = "M" if monthly else "W"
    label_fmt = "%b %Y" if monthly else "%Y-%m-%d"

    def _target_for(period_start):
        """Enrollment target for the single period starting at period_start."""
        if monthly:
            return WEEKLY_ENROLLMENT_TARGET * period_start.days_in_month / 7
        return WEEKLY_ENROLLMENT_TARGET

    df = df_clinical.copy()
    df["ce_date_parsed"] = pd.to_datetime(df["ce_date"], errors="coerce")
    df = df[df["ce_date_parsed"].notna()]
    if df.empty:
        return empty

    df["period"] = df["ce_date_parsed"].dt.to_period(freq).apply(lambda p: p.start_time)

    period_enrollment = pd.DataFrame()
    if "ce_enrollment_decision" in df.columns:
        enrolled = df[df["ce_enrollment_decision"] == "1"]
        if not enrolled.empty:
            counts = enrolled.groupby("period").size()
            # Fill periods with zero enrollments so gaps show as empty bars.
            full_range = pd.date_range(
                counts.index.min(), counts.index.max(),
                freq="MS" if monthly else "7D",
            )
            counts = counts.reindex(full_range, fill_value=0)
            period_enrollment = counts.rename("Enrollments").rename_axis("period").reset_index()
            period_enrollment["Period"] = period_enrollment["period"].dt.strftime(label_fmt)
            period_enrollment["Target"] = period_enrollment["period"].apply(_target_for)

    cumulative_stratified = pd.DataFrame()
    if "ce_assigned_strata" in df.columns:
        stratified = df[
            df["ce_assigned_strata"].notna() & (df["ce_assigned_strata"] != "")
        ]
        if not stratified.empty:
            period_counts = (
                stratified.groupby("period").size().reset_index(name="Count")
                .sort_values("period")
            )
            period_counts["Cumulative Enrolled"] = period_counts["Count"].cumsum()

            # Cumulative Target grows for every calendar period elapsed since
            # the first stratified enrollment, so gaps with zero enrollments
            # still count against the target.
            first_period = period_counts["period"].min()
            if monthly:
                # Sum the (days-in-month / 7) * weekly target of every month
                # from the first enrollment month through each row's month.
                def _cum_target(p):
                    months = pd.date_range(first_period, p, freq="MS")
                    return sum(_target_for(m) for m in months)
                period_counts["Cumulative Target"] = period_counts["period"].apply(_cum_target)
            else:
                weeks_elapsed = ((period_counts["period"] - first_period).dt.days // 7) + 1
                period_counts["Cumulative Target"] = weeks_elapsed * WEEKLY_ENROLLMENT_TARGET

            period_counts["Period"] = period_counts["period"].dt.strftime(label_fmt)
            cumulative_stratified = period_counts

    return period_enrollment, cumulative_stratified


def build_weekly_retention(visit_matrix):
    """Retention rate per visit week: Completed / (Completed + Missed + Rescheduled).

    Pending and Withdrawn/Discontinued rows are excluded from both numerator and
    denominator since they aren't yet a completed-or-failed outcome.
    """
    if visit_matrix.empty:
        return pd.DataFrame()

    rows = []
    for week in FOLLOWUP_WINDOWS:
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


def _last_true_index(mask_2d):
    """Column index of the last True in each row of a boolean array (-1 if none)."""
    arr = np.asarray(mask_2d, dtype=bool)
    if arr.size == 0:
        return np.full(arr.shape[0], -1)
    last = arr.shape[1] - 1 - np.argmax(arr[:, ::-1], axis=1)
    return np.where(arr.any(axis=1), last, -1)


def build_medication_retention(df_all, df_visits):
    """Overall + per-week retention among participants who started medication.

    Initiated  = participants with a dated Day 0 (week 0) visit.
    Latest due = the latest of Week 1/2/3/12/36 whose due date is on or
                 before today (none yet due -> only Day 0 is required).
    Retained   = honored their latest due visit (late, on time or early), or
                 honored any later visit. Otherwise not active.
    Honored    = visit status Completed, or a visit date recorded on a visit
                 that is not Missed / Rescheduled / Withdrawn.

    Weekly rows: Due = initiated and due date <= today; Attended = of those,
    honored that week's visit.

    Returns (summary_dict, weekly_df, warning).
    """
    weekly_cols = ["Week", "Due", "Attended", "Retention %"]
    empty_weekly = pd.DataFrame(columns=weekly_cols)
    empty = {"initiated": 0, "retained": 0, "pct": None, "due_fields_found": False}

    dated, id_col, field, warning = _started_medication_rows(df_visits)
    if warning:
        return empty, empty_weekly, warning

    d0 = dated[[id_col, field]].copy()
    d0[id_col] = d0[id_col].astype(str).str.strip()
    d0["_d"] = pd.to_datetime(d0[field], errors="coerce")
    day0 = d0.groupby(id_col)["_d"].min()
    started_ids = day0.index
    if len(started_ids) == 0:
        return empty, empty_weekly, "No participants have started medication yet."

    # Honored matrix (participant x week)
    v = df_visits.copy()
    v[id_col] = v[id_col].astype(str).str.strip()
    v["_label"] = v["visit_window"].astype(str).str.strip().map(VISIT_WINDOW_CODE_MAP)
    v = v[v["_label"].isin(MED_FOLLOWUP_WINDOWS)]
    if "visit_status" in v.columns:
        status = v["visit_status"].astype(str).str.strip()
    else:
        status = pd.Series("", index=v.index)
    has_date = pd.to_datetime(v[field], errors="coerce").notna()
    v["_honored"] = (status == "1") | (has_date & ~status.isin(["2", "3", "4"]))
    if v.empty:
        honored = pd.DataFrame(False, index=started_ids, columns=MED_FOLLOWUP_WINDOWS)
    else:
        honored = (
            v.groupby([id_col, "_label"])["_honored"].any().unstack()
            .reindex(index=started_ids, columns=MED_FOLLOWUP_WINDOWS)
            .fillna(False).astype(bool)
        )

    # Due dates (participant x week)
    all_id = "record_id" if "record_id" in df_all.columns else df_all.columns[0]
    today = pd.Timestamp.now().normalize()
    due = pd.DataFrame(index=started_ids, columns=MED_FOLLOWUP_WINDOWS, dtype="datetime64[ns]")
    due_fields_found = False
    for w in MED_FOLLOWUP_WINDOWS:
        col = MED_DUE_FIELDS[w]
        s = pd.Series(pd.NaT, index=started_ids, dtype="datetime64[ns]")
        if col in df_all.columns:
            t = df_all[[all_id, col]].copy()
            t[all_id] = t[all_id].astype(str).str.strip()
            t[col] = pd.to_datetime(t[col], errors="coerce")
            t = t.dropna(subset=[col])
            if not t.empty:
                due_fields_found = True
                s = t.groupby(all_id)[col].min().reindex(started_ids)
        due[w] = s.fillna(day0 + pd.Timedelta(days=MED_DUE_FALLBACK_DAYS[w]))

    is_due = due.le(today)

    latest_due_idx = _last_true_index(is_due.values)
    last_honored_idx = _last_true_index(honored.values)
    retained = last_honored_idx >= latest_due_idx  # none due -> Day 0 is enough

    rows = []
    for w in MED_FOLLOWUP_WINDOWS:
        n_due = int(is_due[w].sum())
        n_att = int((is_due[w] & honored[w]).sum())
        rows.append({
            "Week": w,
            "Due": n_due,
            "Attended": n_att,
            "Retention %": round(n_att / n_due * 100, 1) if n_due else None,
        })

    initiated = int(len(started_ids))
    n_ret = int(retained.sum())
    summary = {
        "initiated": initiated,
        "retained": n_ret,
        "pct": n_ret / initiated * 100 if initiated else None,
        "due_fields_found": due_fields_found,
    }
    return summary, pd.DataFrame(rows, columns=weekly_cols), None


def build_psf_retention(df_behavioral, day0_dates=None):
    """Behavioral (PSF) retention = done latest due PSF session / all randomised to PSF.

    Latest due = latest session whose due date (Day 0 + PSF_SESSION_DAY_OFFSETS)
    is on or before today. Retained = that session is Done, or any later one is.
    A PSF participant with no Day 0 date and no session done has nothing due
    and nothing done, so counts as not retained (reported in the caption).

    Returns (summary_dict, warning).
    """
    empty = {"randomised": 0, "retained": 0, "pct": None, "n_no_day0": 0}
    if df_behavioral.empty:
        return empty, "No Behavioral Tracking records were returned."
    field = BEHAVIORAL_INTERVENTION_FIELD
    if field not in df_behavioral.columns:
        return empty, f"Field '{field}' was not returned by the Behavioral Tracking export."

    id_col = "record_id" if "record_id" in df_behavioral.columns else df_behavioral.columns[0]
    raw = df_behavioral.copy()
    raw[id_col] = raw[id_col].astype(str)

    df = raw.copy()
    df["Label"] = df[field].astype(str).str.strip().str.upper().map(BEHAVIORAL_INTERVENTION_MAP)
    df = df.dropna(subset=["Label"]).drop_duplicates(subset=[id_col], keep="first")
    psf_ids = pd.Index(df.loc[df["Label"] == "PSF", id_col].unique())
    if len(psf_ids) == 0:
        return empty, "No participants have been assigned to PSF yet."

    present = [f for f in PSF_SESSION_STATUS_FIELDS if f in raw.columns]
    if not present:
        return {**empty, "randomised": int(len(psf_ids))}, (
            "None of the PSF session status fields (bt_psf_s1_status ... "
            "bt_psf_s8_status) were returned by the export."
        )

    def _combine(series):
        v = series.astype(str).str.strip()
        if (v == "1").any():
            return "1"
        if (v == "0").any():
            return "0"
        return ""

    sub = raw[raw[id_col].isin(psf_ids)]
    status = sub.groupby(id_col)[present].agg(_combine).reindex(psf_ids).fillna("")
    done = pd.DataFrame(
        {f: (status[f] == "1") if f in status.columns else pd.Series(False, index=psf_ids)
         for f in PSF_SESSION_STATUS_FIELDS},
        index=psf_ids,
    )

    if day0_dates is not None and len(day0_dates) > 0:
        day0 = day0_dates.reindex(psf_ids)
    else:
        day0 = pd.Series(pd.NaT, index=psf_ids, dtype="datetime64[ns]")

    today = pd.Timestamp.now().normalize()
    is_due = pd.DataFrame(
        {f: day0.notna() & (today >= day0 + pd.Timedelta(days=off))
         for f, off in zip(PSF_SESSION_STATUS_FIELDS, PSF_SESSION_DAY_OFFSETS)},
        index=psf_ids,
    )

    latest_due_idx = _last_true_index(is_due.values)
    last_done_idx = _last_true_index(done.values)
    retained = (last_done_idx >= latest_due_idx) & (
        (latest_due_idx >= 0) | (last_done_idx >= 0)
    )

    total = int(len(psf_ids))
    n_ret = int(retained.sum())
    n_no_day0 = int((day0.isna().values & ~retained).sum())
    return {
        "randomised": total,
        "retained": n_ret,
        "pct": n_ret / total * 100,
        "n_no_day0": n_no_day0,
    }, None


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
def detect_screening_date_field(df_all):
    """Best-guess screening-date column in the REDCap export (or None)."""
    if df_all is None or df_all.empty:
        return None
    field = next((c for c in SCREENING_DATE_FIELD_CANDIDATES if c in df_all.columns), None)
    if field:
        return field
    return next(
        (
            c for c in df_all.columns
            if "screen" in c.lower() and "date" in c.lower()
            and not any(x in c.lower() for x in ("due", "next", "window"))
        ),
        None,
    )


def get_eligible_ids(df_clinical):
    """Record IDs (str) with ce_enrollment_decision == 1 (the 'Eligible' KPI)."""
    if df_clinical.empty or "ce_enrollment_decision" not in df_clinical.columns:
        return set()
    id_col = "record_id" if "record_id" in df_clinical.columns else df_clinical.columns[0]
    rows = df_clinical[df_clinical["ce_enrollment_decision"] == "1"]
    return set(rows[id_col].astype(str).str.strip())


def get_randomised_ids(df_behavioral, df_clinical):
    """Record IDs (str) that are randomised, matching the 'Randomised' KPI.

    Primary rule: a valid PSF/BA behavioral intervention is assigned. If that
    form is unavailable, fall back to participants with an assigned stratum.
    """
    if (
        not df_behavioral.empty
        and BEHAVIORAL_INTERVENTION_FIELD in df_behavioral.columns
    ):
        id_col = "record_id" if "record_id" in df_behavioral.columns else df_behavioral.columns[0]
        labels = (
            df_behavioral[BEHAVIORAL_INTERVENTION_FIELD]
            .astype(str).str.strip().str.upper()
            .map(BEHAVIORAL_INTERVENTION_MAP)
        )
        ids = set(df_behavioral.loc[labels.notna(), id_col].astype(str).str.strip())
        if ids:
            return ids
    if not df_clinical.empty and "ce_assigned_strata" in df_clinical.columns:
        id_col = "record_id" if "record_id" in df_clinical.columns else df_clinical.columns[0]
        rows = df_clinical[
            df_clinical["ce_assigned_strata"].notna()
            & (df_clinical["ce_assigned_strata"] != "")
        ]
        return set(rows[id_col].astype(str).str.strip())
    return set()


def build_enrollment_by_month(df_all, date_field, eligible_ids, randomised_ids, started_ids):
    """Eligible / Randomised / Started Medication counts by SCREENING month.

    Each participant is placed in the month of their (earliest) screening date
    and counted once per stage, so a participant screened in March who starts
    medication in May counts under March for every stage.

    Returns (monthly_df, participants_df, n_undated):
      monthly_df      one row per calendar month from first to last screening
                      month (zero-filled), columns: Month, Eligible,
                      Randomised, Started Medication
      participants_df one row per screened participant with stage flags
      n_undated       participants in any stage with no screening date (these
                      cannot be placed in a month and are left out of monthly_df)
    """
    empty_monthly = pd.DataFrame(
        columns=["Month", "Eligible", "Randomised", "Started Medication"]
    )
    if df_all.empty or date_field not in df_all.columns:
        return empty_monthly, pd.DataFrame(), 0

    id_col = "record_id" if "record_id" in df_all.columns else df_all.columns[0]
    d = df_all[[id_col, date_field]].copy()
    d[id_col] = d[id_col].astype(str).str.strip()
    d["_d"] = pd.to_datetime(d[date_field], errors="coerce")
    d = d.dropna(subset=["_d"])
    screen_dates = d.groupby(id_col)["_d"].min()

    staged_ids = set(eligible_ids) | set(randomised_ids) | set(started_ids)
    n_undated = len(staged_ids - set(screen_dates.index))

    if screen_dates.empty:
        return empty_monthly, pd.DataFrame(), n_undated

    participants = pd.DataFrame({"Record ID": screen_dates.index})
    participants["Screening Date"] = screen_dates.dt.strftime("%Y-%m-%d").values
    participants["_month"] = screen_dates.dt.to_period("M").dt.start_time.values
    participants["Eligible"] = participants["Record ID"].isin(eligible_ids).astype(int)
    participants["Randomised"] = participants["Record ID"].isin(randomised_ids).astype(int)
    participants["Started Medication"] = participants["Record ID"].isin(started_ids).astype(int)

    months = pd.date_range(participants["_month"].min(), participants["_month"].max(), freq="MS")
    monthly = (
        participants.groupby("_month")[["Eligible", "Randomised", "Started Medication"]]
        .sum()
        .reindex(months, fill_value=0)
        .astype(int)
    )
    monthly.insert(0, "Month", monthly.index.strftime("%b %Y"))
    monthly = monthly.reset_index(drop=True)

    participants["Screening Month"] = pd.to_datetime(participants["_month"]).dt.strftime("%Y-%m")
    participants = (
        participants.drop(columns="_month")
        .sort_values(["Screening Date", "Record ID"])
        .reset_index(drop=True)
    )
    participants = participants[
        ["Record ID", "Screening Date", "Screening Month",
         "Eligible", "Randomised", "Started Medication"]
    ]
    return monthly, participants, n_undated


def generate_shareable_link(view_mode, filters=None):
    """Encode current view state into query parameters for sharing."""
    params = {"view": view_mode}
    if filters:
        params.update(filters)

    base_url = "https://your-app-url.streamlit.app"
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    return f"{base_url}/?{query_string}"

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
                "Study Overview",
                "Enrollments",
                "Visit Matrix",
                "Alerts - AE/Safety",
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

    # ae_severity / ae_outcome live on the separate, repeating
    # adverse_event_log instrument, so pull that form directly instead of
    # relying on the project-wide export (same reasoning as
    # behavioral_tracking below).
    df_ae, ae_export_error = load_adverse_events(proj)
    ae_records, ae_severity_summary, ae_outcome_summary, ae_warning = build_ae_summary(df_ae)
    if ae_export_error and ae_records.empty:
        ae_warning = ae_export_error
    total_ae = int(len(ae_records))
    strata_demographics = build_stratified_demographics(df_clinical, df_prescreen)
    weekly_retention = build_weekly_retention(visit_matrix)

    # ── Behavioral Intervention Assigned (PSF vs BA) ──
    # bt_intervention_type is on the separate behavioral_tracking instrument,
    # so pull that form directly instead of relying on the project-wide export.
    df_behavioral, behavioral_export_error = load_behavioral_tracking(proj)
    psf_ba_distribution, scheduler_error = build_psf_ba_distribution(df_behavioral)
    day0_dates, day0_field = get_day0_dates(df_all)
    psf_session_summary, total_psf_assigned, psf_session_warning = build_psf_session_summary(
        df_behavioral, day0_dates
    )

    # Prefer the specific behavioral-export error because it is more actionable.
    if behavioral_export_error and psf_ba_distribution.empty:
        scheduler_error = behavioral_export_error
    if behavioral_export_error and psf_session_summary.empty:
        psf_session_warning = behavioral_export_error

    # Randomised = PSF + BA (participants with a behavioral intervention assigned).
    if not psf_ba_distribution.empty:
        randomised_n = int(psf_ba_distribution["Count"].sum())
        randomised_note = None
    else:
        randomised_n = enrollment["stratified"]
        randomised_note = (
            "PSF/BA assignments were unavailable, so Randomised is showing the "
            "count of participants with an assigned stratum instead."
        )

    # Retention (medication visits + behavioral PSF)
    med_retention, med_weekly_retention, med_retention_warning = build_medication_retention(
        df_all, df_visits
    )
    psf_retention, psf_retention_warning = build_psf_retention(df_behavioral, day0_dates)

    # Started medication = participants with a date on their Day 0 visit log.
    started_med_n, started_med_field, started_med_warning = count_started_medication(df_visits)

    # Action Required: terminated and missed-Day-0 participants are listed
    # separately and removed from the overdue/upcoming lists.
    terminated_df, missed_day0_df = get_excluded_participants(df_visits, df_clinical)
    excluded_ids = set(terminated_df["Record ID"].astype(str)) | set(
        missed_day0_df["Record ID"].astype(str)
    )
    overdue = exclude_records(overdue, excluded_ids)
    upcoming = exclude_records(upcoming, excluded_ids)

    # ═══════════════════════════════════════════════════════════
    # VIEW: DASHBOARD
    # ═══════════════════════════════════════════════════════════
    if view_mode == "Study Overview":
        st.markdown(
            '<div class="main-header">SOAR Study Enrollment & Follow-up Dashboard</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sub-header">Real-time data from REDCap | Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>',
            unsafe_allow_html=True,
        )

        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
        kpi1.metric("Pre-Screened", enrollment["total_screened"])
        kpi2.metric(
            "Met Preliminary Eligibility criteria",
            enrollment["eligible_referred"],
            f"{enrollment['eligible_referred'] / max(enrollment['total_screened'], 1) * 100:.0f}%",
        )
        kpi3.metric(
            "Consented",
            enrollment["consented"],
            f"{enrollment['consented'] / max(enrollment['eligible_referred'], 1) * 100:.0f}%",
        )
        kpi4.metric("Eligible", enrollment["clinically_eligible"])
        kpi5.metric("Randomised", randomised_n)
        kpi6.metric(
            "Started Medication",
            started_med_n,
            f"{started_med_n / max(randomised_n, 1) * 100:.0f}% of randomised",
        )
        if randomised_note:
            st.caption(randomised_note)
        if started_med_warning:
            st.warning(started_med_warning)
        elif started_med_field:
            st.caption(
                f"Started Medication = participants who honored "
                f"Day 0 visit."
            )

        st.markdown("---")

        st.subheader("Enrollment Funnel")
        funnel_data = pd.DataFrame(
            {
                "Stage": [
                    "Pre-Screened",
                    "Met Preliminary Eligibility criteria",
                    "Consented",
                    "Eligible",
                    "Randomised",
                    "Started Medication",
                ],
                "Count": [
                    enrollment["total_screened"],
                    enrollment["eligible_referred"],
                    enrollment["consented"],
                    enrollment["clinically_eligible"],
                    randomised_n,
                    started_med_n,
                ],
            }
        )
        funnel_colors = funnel_palette(len(funnel_data))
        fig_funnel = go.Figure(
            go.Funnel(
                y=funnel_data["Stage"],
                x=funnel_data["Count"],
                textposition="inside",
                textinfo="value+percent initial",
                marker={"color": funnel_colors},
            )
        )
        fig_funnel.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=400)
        show_plot(fig_funnel, use_container_width=True)

        st.markdown("---")

        st.subheader("Behavioral Intervention Assigned: PSF vs BA")
        if scheduler_error:
            st.warning(scheduler_error)
        else:
            total_assigned = int(psf_ba_distribution["Count"].sum())
            sched_col1, sched_col2 = st.columns([1, 1])
            with sched_col1:
                fig_psf_ba = px.pie(
                    psf_ba_distribution,
                    names="Group",
                    values="Count",
                    title=f"Number Assigned (N={total_assigned})",
                    height=318,
                    color_discrete_sequence=rocket_colors(len(psf_ba_distribution)),
                )
                fig_psf_ba.update_traces(
                    text=psf_ba_distribution.apply(
                        lambda r: f"{r['Group']}: {int(r['Count'])} ({r['Percent']:.0f}%)",
                        axis=1,
                    ),
                    textinfo="text",
                    textposition="outside",
                )
                fig_psf_ba.update_layout(
                    showlegend=True,
                    margin=dict(l=20, r=20, t=20, b=20),
                    title_font=dict(size=18, color="#4a90d9"),
                )
                show_plot(fig_psf_ba, use_container_width=True)
            with sched_col2:
                st.dataframe(
                    psf_ba_distribution, use_container_width=True, hide_index=True
                )
                st.caption(
                    "Count and % of participants by Behavioral Intervention "
                    "Assigned (PSF vs BA)"
                )

        st.markdown("---")

        st.subheader("Stratification Summary")
        if not strata_summary.empty:
            strata_col_left, strata_col_right = st.columns([1, 1])

            with strata_col_left:
                st.dataframe(
                    strata_summary, use_container_width=True, hide_index=True
                )

            with strata_col_right:
                fig_strata = px.bar(
                    strata_summary,
                    x="Count",
                    y="Stratum",
                    text="Count",
                    orientation="h",
                    labels={"Count": "Participants"},
                    height=280,
                )
                fig_strata.update_traces(
                    marker_color=rocket_colors(len(strata_summary)),
                    textposition="outside",
                )
                fig_strata.update_layout(
                    showlegend=False,
                    margin=dict(l=20, r=20, t=10, b=20),
                    yaxis=dict(categoryorder="total ascending"),
                )
                show_plot(fig_strata, use_container_width=True)
        else:
            st.info("No stratification data available yet.")

        st.markdown("---")

        st.subheader("Retention")
        ret_left, ret_right = st.columns([1, 2])

        with ret_left:
            st.markdown("**Overall Retention**")
            if med_retention_warning:
                st.warning(med_retention_warning)
            elif med_retention["initiated"] > 0:
                st.metric(
                    "Retained (Medication visit)",
                    f"{med_retention['pct']:.0f}%",
                    f"{med_retention['retained']} of {med_retention['initiated']} initiated",
                    delta_color="off",
                )
            else:
                st.info("No participants have started medication yet.")

            if psf_retention_warning:
                st.warning(psf_retention_warning)
            elif psf_retention["randomised"] > 0:
                st.metric(
                    "Retained behavioral (PSF)",
                    f"{psf_retention['pct']:.0f}%",
                    f"{psf_retention['retained']} of {psf_retention['randomised']} randomised in PSF",
                    delta_color="off",
                )

            overall_rows = []
            if not med_retention_warning and med_retention["initiated"] > 0:
                overall_rows.append(("Medication visit", med_retention["initiated"], med_retention["retained"]))
            if not psf_retention_warning and psf_retention["randomised"] > 0:
                overall_rows.append(("Behavioral (PSF)", psf_retention["randomised"], psf_retention["retained"]))
            if overall_rows:
                fig_overall = go.Figure()
                fig_overall.add_trace(go.Bar(
                    y=[r[0] for r in overall_rows],
                    x=[r[1] for r in overall_rows],
                    name="Denominator",
                    orientation="h",
                    marker_color="#2980b9",
                    text=[r[1] for r in overall_rows],
                    textposition="outside",
                ))
                fig_overall.add_trace(go.Bar(
                    y=[r[0] for r in overall_rows],
                    x=[r[2] for r in overall_rows],
                    name="Retained",
                    orientation="h",
                    marker_color="#2e7d32",
                    text=[f"{r[2]} ({r[2] / r[1] * 100:.0f}%)" for r in overall_rows],
                    textposition="outside",
                ))
                fig_overall.update_layout(
                    barmode="group",
                    height=240,
                    margin=dict(l=20, r=40, t=30, b=20),
                    xaxis=dict(title="Participants", range=[0, max(r[1] for r in overall_rows) * 1.35]),
                    yaxis=dict(title="", autorange="reversed"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )
                show_plot(fig_overall, use_container_width=True)

        with ret_right:
            st.markdown("**Weekly Retention (Medication visits)**")
            if med_retention_warning:
                st.info("Weekly retention needs the visit log date field (see warning on the left).")
            elif not med_weekly_retention.empty and med_weekly_retention["Due"].sum() > 0:
                wk = med_weekly_retention
                pct_labels = [
                    f"{int(a)} ({p:.0f}%)" if pd.notna(p) else "-"
                    for a, p in zip(wk["Attended"], wk["Retention %"])
                ]
                fig_weekly_ret = go.Figure()
                fig_weekly_ret.add_trace(go.Bar(
                    x=wk["Week"],
                    y=wk["Due"],
                    name="Due for week",
                    marker_color="#2980b9",
                    text=wk["Due"],
                    textposition="outside",
                ))
                fig_weekly_ret.add_trace(go.Bar(
                    x=wk["Week"],
                    y=wk["Attended"],
                    name="Attended visit (%)",
                    marker_color="#2e7d32",
                    text=pct_labels,
                    textposition="outside",
                ))
                fig_weekly_ret.update_layout(
                    barmode="group",
                    height=420,
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis=dict(title="Visit window", type="category"),
                    yaxis=dict(title="Participants", range=[0, max(int(wk["Due"].max()), 1) * 1.2]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )
                show_plot(fig_weekly_ret, use_container_width=True)
            else:
                st.info("No medication follow-up visits are due yet.")

        st.caption(
            "Retained (Medication visit) = participants who honored their latest due "
            "medication visit (Week 1, 2, 3, 12 or 36; late, on time or early) ÷ all who "
            "started medication (Day 0 visit date recorded). Weekly: Due = started "
            "medication and the week's due date has passed; Attended = honored that visit. "
            "Retained behavioral (PSF) = participants who completed their latest due PSF "
            "session ÷ all randomised to PSF."
        )
        if not med_retention_warning and not med_retention["due_fields_found"]:
            st.caption(
                "No week_#_due dates were found, so due dates are Day 0 + 7/14/21/84/252 days."
            )
        if not psf_retention_warning and psf_retention.get("n_no_day0"):
            st.caption(
                f"{psf_retention['n_no_day0']} PSF participant(s) have no Day 0 date and no "
                "completed session, so they are counted as not retained."
            )

        st.markdown("---")

        st.subheader("Randomized Participant Demographics")
        if not strata_demographics.empty and (
            "Age" in strata_demographics.columns or "Sex" in strata_demographics.columns
        ):
            demo_col1, demo_col2 = st.columns(2)

            with demo_col1:
                if "Age" in strata_demographics.columns:
                    age_summary, age_stats = build_age_group_summary(
                        strata_demographics["Age"]
                    )
                    if not age_summary.empty:
                        fig_age = px.bar(
                            age_summary,
                            x="Age Group",
                            y="Count",
                            text="Label",
                            labels={"Count": "Participants"},
                            title=f"Age Distribution (N={age_stats['total']})",
                            height=380,
                        )
                        fig_age.update_traces(
                            marker_color=rocket_colors(len(age_summary)),
                            textposition="outside",
                        )
                        fig_age.update_layout(
                            showlegend=False,
                            margin=dict(l=20, r=20, t=60, b=20),
                            title_font=dict(size=18, color="#4a90d9"),
                        )
                        show_plot(fig_age, use_container_width=True)
                        st.caption(
                            f"Median age: {age_stats['median']} years "
                            f"(IQR: {age_stats['q1']}–{age_stats['q3']})"
                        )
                    else:
                        st.info("No valid age values to plot.")
                else:
                    st.info("No age field found — check AGE_FIELD_CANDIDATES.")

            with demo_col2:
                if "Sex" in strata_demographics.columns:
                    sex_counts = (
                        strata_demographics["Sex"].value_counts().reset_index()
                    )
                    sex_counts.columns = ["Sex", "Count"]
                    total_sex = int(sex_counts["Count"].sum())
                    sex_counts["Percent"] = (sex_counts["Count"] / total_sex * 100).round(1)
                    sex_counts["Label"] = sex_counts.apply(
                        lambda r: f"n={int(r['Count'])} ({r['Percent']:.0f}%)", axis=1
                    )

                    fig_sex = px.pie(
                        sex_counts,
                        names="Sex",
                        values="Count",
                        title=f"Sex Distribution (N={total_sex})",
                        height=380,
                    )
                    fig_sex.update_traces(
                        marker=dict(colors=rocket_colors(len(sex_counts))),
                        text=sex_counts["Label"],
                        textinfo="text",
                        textposition="outside",
                    )
                    fig_sex.update_layout(
                        margin=dict(l=20, r=20, t=60, b=20),
                        legend_title_text="Gender",
                        title_font=dict(size=18, color="#4a90d9"),
                    )
                    show_plot(fig_sex, use_container_width=True)
                else:
                    st.info("No sex field found — check SEX_FIELD_CANDIDATES.")
        else:
            st.info(
                "No age/sex fields detected for stratified participants. "
                "Update AGE_FIELD_CANDIDATES / SEX_FIELD_CANDIDATES to match your data dictionary."
            )

        st.markdown("---")

        enroll_view_by = st.radio(
            "View enrollment graphs by",
            ["Weekly", "Monthly"],
            index=0,
            horizontal=True,
            key="enrollment_view_by",
        )
        period_word = "Week" if enroll_view_by == "Weekly" else "Month"
        weekly_enrollment, cumulative_stratified = build_enrollment_trends(
            df_clinical, enroll_view_by
        )
        x_axis_type = "category" if enroll_view_by == "Monthly" else None

        st.subheader(f"Cumulative Enrollments (by {period_word})")
        if not cumulative_stratified.empty:
            achieved_color = rocket_colors(3)[1]
            target_color = "#e63946"

            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=cumulative_stratified["Period"],
                y=cumulative_stratified["Cumulative Enrolled"],
                mode="lines+markers+text",
                name="Cumulative Achieved",
                text=cumulative_stratified["Cumulative Enrolled"],
                textposition="top center",
                line=dict(color=achieved_color, width=3),
                marker=dict(color=achieved_color, size=8),
            ))
            fig_cum.add_trace(go.Scatter(
                x=cumulative_stratified["Period"],
                y=cumulative_stratified["Cumulative Target"],
                mode="lines",
                name="Cumulative Target",
                line=dict(color=target_color, width=2, dash="dash"),
            ))
            fig_cum.update_layout(
                height=380,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title=period_word,
                xaxis_type=x_axis_type,
                yaxis_title="Cumulative Enrolled",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            show_plot(fig_cum, use_container_width=True)

            latest = cumulative_stratified.iloc[-1]
            gap = int(round(latest["Cumulative Target"] - latest["Cumulative Enrolled"]))
            if gap > 0:
                gap_text = f"{gap} behind target"
            elif gap < 0:
                gap_text = f"{abs(gap)} ahead of target"
            else:
                gap_text = "on target"
            st.caption(
                f"Latest {period_word.lower()}: {int(latest['Cumulative Enrolled'])} achieved vs "
                f"{int(round(latest['Cumulative Target']))} targeted ({gap_text})."
            )
        else:
            st.info("No stratification date data available yet.")

        st.markdown("---")

        st.subheader(f"Enrollments by {period_word}")
        if not weekly_enrollment.empty:
            fig_weekly = px.bar(
                weekly_enrollment,
                x="Period",
                y="Enrollments",
                text="Enrollments",
                height=340,
            )
            fig_weekly.update_traces(
                marker_color=rocket_colors(len(weekly_enrollment)),
                textposition="outside",
            )
            if enroll_view_by == "Weekly":
                fig_weekly.add_hline(
                    y=WEEKLY_ENROLLMENT_TARGET,
                    line_dash="dash",
                    line_width=2,
                    line_color="#e63946",
                    annotation_text=f"Target: {WEEKLY_ENROLLMENT_TARGET}/week",
                    annotation_position="top left",
                    annotation_font_color="#e63946",
                )
            else:
                # Months differ in length, so the target is a per-month line.
                fig_weekly.add_trace(go.Scatter(
                    x=weekly_enrollment["Period"],
                    y=weekly_enrollment["Target"],
                    mode="lines",
                    name="Target",
                    line=dict(color="#e63946", width=2, dash="dash"),
                    hovertemplate="Target: %{y:.1f}<extra></extra>",
                ))
                fig_weekly.update_layout(
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )
            fig_weekly.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title=period_word,
                xaxis_type=x_axis_type,
            )
            show_plot(fig_weekly, use_container_width=True)
        else:
            st.info("No enrollment date data available yet.")

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

        st.caption(
            "Terminated and Missed Day 0 participants are excluded from the "
            "lists above and shown separately below."
        )
        render_excluded_lists(terminated_df, missed_day0_df)

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
                    "Withdrawn/Discontinued": "background-color: #495057; color: white",
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
                show_plot(fig_retention, use_container_width=True)
                st.dataframe(
                    weekly_retention, use_container_width=True, hide_index=True
                )
            else:
                st.info("Not enough visit outcome data to compute retention yet.")

            st.markdown("---")

            st.subheader("PSF Session Completion")
            if psf_session_warning:
                st.warning(psf_session_warning)
            if not psf_session_summary.empty:
                st.markdown(f"**Total PSF Assigned: {total_psf_assigned}**")
                if day0_field:
                    st.caption(
                        f"Due dates = Day 0 ({day0_field}) + session offsets "
                        f"{PSF_SESSION_DAY_OFFSETS} days. Due = Completed + Missed; "
                        "Pending = not yet due."
                    )

                series_spec = [
                    ("Due", "Due for Session", "#2980b9"),
                    ("Completed", "Completed", "#2e7d32"),
                    ("Pending", "Pending", "#ffc107"),
                    ("Missed", "Missed", "#dc3545"),
                ]
                fig_psf_sessions = go.Figure()
                for col, label, color in series_spec:
                    fig_psf_sessions.add_trace(go.Bar(
                        y=psf_session_summary["Session"],
                        x=psf_session_summary[col],
                        name=label,
                        orientation="h",
                        marker_color=color,
                        text=psf_session_summary[col],
                        textposition="outside",
                        hovertemplate=f"%{{y}}: %{{x}} {label.lower()}<extra></extra>",
                    ))
                fig_psf_sessions.update_layout(
                    barmode="group",
                    height=620,
                    margin=dict(l=20, r=20, t=50, b=20),
                    title=f"PSF Sessions: Due, Completed, Pending, Missed (N={total_psf_assigned})",
                    title_font=dict(size=18, color="#4a90d9"),
                    xaxis=dict(title="Participants", range=[0, max(total_psf_assigned, 1) * 1.15]),
                    yaxis=dict(title="", autorange="reversed"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )
                show_plot(fig_psf_sessions, use_container_width=True)

                def _n_pct(n, d):
                    return f"{int(n)} ({(n / d * 100) if d else 0:.0f}%)"

                psf_table = pd.DataFrame({
                    "Session": psf_session_summary["Session"],
                    "Due": psf_session_summary["Due"],
                    "Completed n (% of due)": psf_session_summary.apply(
                        lambda r: _n_pct(r["Completed"], r["Due"]), axis=1),
                    "Missed n (% of due)": psf_session_summary.apply(
                        lambda r: _n_pct(r["Missed"], r["Due"]), axis=1),
                    "Pending": psf_session_summary["Pending"],
                    "Total": psf_session_summary["Total"],
                })
                st.dataframe(psf_table, use_container_width=True, hide_index=True)
            elif not psf_session_warning:
                st.info("No PSF session data available yet.")
        else:
            st.info("No visit data to display.")

    # ═══════════════════════════════════════════════════════════
    # VIEW: ENROLLMENTS (by screening month)
    # ═══════════════════════════════════════════════════════════
    elif view_mode == "Enrollments":
        st.markdown(
            '<div class="main-header">Enrollments by Month</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sub-header">Participants grouped by the month they were '
            "screened</div>",
            unsafe_allow_html=True,
        )

        date_cols = sorted(c for c in df_all.columns if "date" in c.lower())
        detected_field = detect_screening_date_field(df_all)
        if not date_cols:
            st.warning("No date fields were returned by REDCap, so enrollments "
                       "can't be grouped by month.")
            st.stop()

        options = date_cols if detected_field else ["Select a date field"] + date_cols
        default_idx = options.index(detected_field) if detected_field else 0
        screening_field = st.selectbox(
            "Screening date field",
            options,
            index=default_idx,
            help="Participants are placed in the month of this date (earliest "
                 "value per participant).",
        )
        if screening_field not in df_all.columns:
            st.info(
                "Pick the screening date field above. To make this permanent, "
                "add its name to SCREENING_DATE_FIELD_CANDIDATES."
            )
            st.stop()

        eligible_ids = get_eligible_ids(df_clinical)
        randomised_ids = get_randomised_ids(df_behavioral, df_clinical)
        started_ids = get_started_medication_ids(df_visits)

        monthly, enroll_participants, n_undated = build_enrollment_by_month(
            df_all, screening_field, eligible_ids, randomised_ids, started_ids
        )

        if monthly.empty:
            st.info(
                f"No valid dates found in '{screening_field}'. Check that it is "
                "the right field."
            )
        else:
            totals = monthly[["Eligible", "Randomised", "Started Medication"]].sum()
            k1, k2, k3 = st.columns(3)
            k1.metric("Eligible", int(totals["Eligible"]))
            k2.metric("Randomised", int(totals["Randomised"]))
            k3.metric(
                "Started Medication",
                int(totals["Started Medication"]),
                f"{totals['Started Medication'] / max(totals['Randomised'], 1) * 100:.0f}% of randomised",
            )
            st.caption(
                f"Grouped by screening date ({screening_field}). Each participant "
                "counts once, in their screening month, even if randomisation or "
                "Day 0 happened later."
            )
            if n_undated:
                st.warning(
                    f"{n_undated} participant(s) at these stages have no screening "
                    "date and are not shown in the monthly counts."
                )

            st.markdown("---")

            stage_colors = {
                "Eligible": "#2980b9",
                "Randomised": "#2e7d32",
                "Started Medication": "#f39c12",
            }
            fig_enroll = go.Figure()
            for stage, color in stage_colors.items():
                fig_enroll.add_trace(go.Bar(
                    x=monthly["Month"],
                    y=monthly[stage],
                    name=stage,
                    marker_color=color,
                    text=monthly[stage],
                    textposition="outside",
                ))
            fig_enroll.update_layout(
                barmode="group",
                height=420,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis=dict(title="Screening month", type="category"),
                yaxis=dict(title="Participants"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            show_plot(fig_enroll, use_container_width=True)

            total_row = pd.DataFrame([{
                "Month": "Total",
                "Eligible": int(totals["Eligible"]),
                "Randomised": int(totals["Randomised"]),
                "Started Medication": int(totals["Started Medication"]),
            }])
            report_df = pd.concat([monthly, total_row], ignore_index=True)
            st.dataframe(report_df, use_container_width=True, hide_index=True)

            dl1, dl2 = st.columns(2)
            stamp = datetime.now().strftime("%Y%m%d")
            with dl1:
                st.download_button(
                    "Download monthly report (CSV)",
                    report_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"soar_enrollment_by_screening_month_{stamp}.csv",
                    mime="text/csv",
                )
            with dl2:
                st.download_button(
                    "Download participant-level list (CSV)",
                    enroll_participants.to_csv(index=False).encode("utf-8"),
                    file_name=f"soar_enrollment_participants_{stamp}.csv",
                    mime="text/csv",
                    help="One row per screened participant with their screening "
                         "month and stage flags (1 = reached stage).",
                )

    # ═══════════════════════════════════════════════════════════
    # VIEW: ALERTS
    # ═══════════════════════════════════════════════════════════
    elif view_mode == "Alerts - AE/Safety":
        st.markdown(
            '<div class="main-header">Alerts</div>',
            unsafe_allow_html=True,
        )

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Overdue Visits",
                "Upcoming Visits",
                "Adverse Events",
                "Terminated / Missed Day 0",
            ]
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
            st.subheader(f"Adverse Events ({total_ae})")
            st.caption(
                "AEs are logged on a repeating form, so counts here are of "
                "events — a participant with more than one AE is counted "
                "once per event, not once per participant."
            )
            if ae_warning:
                st.info(ae_warning)
            else:
                severity_lookup = dict(
                    zip(ae_severity_summary["Severity"], ae_severity_summary["Count"])
                )
                ae_kpi1, ae_kpi2, ae_kpi3, ae_kpi4 = st.columns(4)
                ae_kpi1.metric("Total AEs", total_ae)
                ae_kpi2.metric("Mild", int(severity_lookup.get("Mild", 0)))
                ae_kpi3.metric("Moderate", int(severity_lookup.get("Moderate", 0)))
                ae_kpi4.metric("Severe", int(severity_lookup.get("Severe", 0)))

                st.markdown("---")

                ae_col1, ae_col2 = st.columns(2)

                with ae_col1:
                    st.markdown("**By Severity**")
                    if not ae_severity_summary.empty:
                        fig_ae_severity = px.bar(
                            ae_severity_summary,
                            x="Severity",
                            y="Count",
                            text="Count",
                            height=320,
                        )
                        fig_ae_severity.update_traces(
                            marker_color=rocket_colors(len(ae_severity_summary)),
                            textposition="outside",
                        )
                        fig_ae_severity.update_layout(
                            showlegend=False, margin=dict(l=20, r=20, t=20, b=20)
                        )
                        show_plot(fig_ae_severity, use_container_width=True)
                        st.dataframe(
                            ae_severity_summary, use_container_width=True, hide_index=True
                        )
                    else:
                        st.info("No severity data available for recorded AEs.")

                with ae_col2:
                    st.markdown("**By Outcome**")
                    if not ae_outcome_summary.empty:
                        fig_ae_outcome = px.bar(
                            ae_outcome_summary,
                            x="Count",
                            y="Outcome",
                            text="Count",
                            orientation="h",
                            height=320,
                        )
                        fig_ae_outcome.update_traces(
                            marker_color=rocket_colors(len(ae_outcome_summary)),
                            textposition="outside",
                        )
                        fig_ae_outcome.update_layout(
                            showlegend=False,
                            margin=dict(l=20, r=20, t=20, b=20),
                            yaxis=dict(categoryorder="total ascending"),
                        )
                        show_plot(fig_ae_outcome, use_container_width=True)
                        st.dataframe(
                            ae_outcome_summary, use_container_width=True, hide_index=True
                        )
                    else:
                        st.info("No outcome data available for recorded AEs.")

        with tab4:
            st.caption(
                "These participants are excluded from the Overdue and "
                "Upcoming lists."
            )
            render_excluded_lists(terminated_df, missed_day0_df)

if __name__ == "__main__":
    main()