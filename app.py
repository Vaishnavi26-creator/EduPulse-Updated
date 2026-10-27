

from pathlib import Path

from datetime import date

import json

import hmac



import joblib

import numpy as np

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

import streamlit as st





# ============================================================

# CONFIGURATION

# ============================================================



BASE = Path(__file__).resolve().parent

DATA_PATH = BASE / "data" / "students.csv"

MODEL_PATH = BASE / "models" / "student_risk_model.joblib"

META_PATH = BASE / "models" / "model_metadata.json"

INTERVENTIONS_PATH = BASE / "data" / "interventions.csv"



FACULTY_EMAIL = "faculty@edupulse.demo"

FACULTY_PASSWORD = "Faculty@123"



st.set_page_config(

    page_title="EduPulse | Student Success",

    page_icon="🎓",

    layout="wide",

    initial_sidebar_state="expanded",

)



st.markdown("""

<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');



html, body, [class*="css"] {

    font-family: 'DM Sans', sans-serif;

}

.stApp {

    background: #f4f6fb;

    color: #17233d;

}

.block-container {

    padding-top: 1.2rem;

    padding-bottom: 2.4rem;

    max-width: 1500px;

}

[data-testid="stSidebar"] {

    background: linear-gradient(170deg, #101a33, #1b2c50 68%, #293d68);

}

[data-testid="stSidebar"] * {

    color: #f7f8ff;

}

.hero {

    background: linear-gradient(115deg, #111c38, #243e6c 60%, #6558f5);

    padding: 27px 30px;

    border-radius: 21px;

    color: white;

    margin-bottom: 22px;

    box-shadow: 0 14px 35px #17264a16;

}

.hero .eyebrow {

    font-size: 10px;

    letter-spacing: 1.8px;

    font-weight: 800;

    color: #d4dcff;

    margin-bottom: 8px;

}

.hero h1 {

    font-family: Manrope, sans-serif;

    color: white;

    font-size: clamp(25px, 3vw, 35px);

    font-weight: 800;

    margin: 0 0 8px;

}

.hero p {

    color: #e2e9ff;

    margin: 0;

    font-size: 14px;

    line-height: 1.6;

}

.section-title {

    font-family: Manrope, sans-serif;

    font-size: 20px;

    font-weight: 800;

    margin: 16px 0 4px;

    color: #17233d;

}

.section-sub {

    color: #78849b;

    font-size: 13px;

    margin-bottom: 14px;

}

.metric-card {

    background: white;

    border: 1px solid #e4eaf3;

    border-radius: 15px;

    padding: 16px 17px;

    min-height: 110px;

    box-shadow: 0 4px 14px #17264a08;

    margin-bottom: 12px;

}

.metric-label {

    font-size: 10px;

    letter-spacing: .8px;

    text-transform: uppercase;

    font-weight: 800;

    color: #78849b;

}

.metric-value {

    font-family: Manrope, sans-serif;

    font-size: 27px;

    font-weight: 800;

    color: #17233d;

    margin-top: 8px;

    overflow-wrap: anywhere;

}

.metric-note {

    font-size: 11px;

    color: #8994a6;

    margin-top: 6px;

}

.stButton > button, .stDownloadButton > button {

    border-radius: 10px !important;

    font-weight: 700 !important;

    background: #5145e5 !important;

    color: #ffffff !important;

    border: 1px solid #5145e5 !important;

}

.stButton > button p {

    color: #ffffff !important;

}

.stButton > button:hover, .stDownloadButton > button:hover {

    background: #3730a3 !important;

    color: #ffffff !important;

    border-color: #3730a3 !important;

}

.stButton > button:hover p {

    color: #ffffff !important;

}

[data-testid="stSidebar"] .stButton > button,

[data-testid="stSidebar"] .stButton > button p {

    background: #5145e5 !important;

    color: #ffffff !important;

    border-color: #5145e5 !important;

}

[data-testid="stSidebar"] .stButton > button:hover,

[data-testid="stSidebar"] .stButton > button:hover p {

    background: #3730a3 !important;

    color: #ffffff !important;

    border-color: #3730a3 !important;

}

[data-testid="stDataFrame"] {

    border: 1px solid #e4eaf3;

    border-radius: 12px;

}

.footer {

    font-size: 11px;

    color: #8994a6;

    border-top: 1px solid #e4eaf3;

    padding-top: 12px;

    margin-top: 30px;

}


.login-shell {
    width: 100%;
    max-width: 760px;
    margin: 2.5rem auto 0;
}
.login-shell .hero {
    width: 100%;
    min-height: 155px;
    box-sizing: border-box;
    padding: 32px 36px;
    margin-bottom: 24px;
    border-radius: 22px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.login-shell .hero h1 {
    font-size: clamp(28px, 3.2vw, 40px);
    line-height: 1.2;
}
.login-shell .hero p {
    font-size: 15px;
}
.login-shell [data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #e4eaf3;
    border-radius: 16px;
    padding: 20px 22px 10px;
    box-shadow: 0 8px 28px rgba(23, 38, 74, 0.06);
}
.login-shell [data-testid="stRadio"] {
    margin-bottom: 12px;
}
.login-shell [data-testid="stTextInput"] input {
    min-height: 44px;
    border-radius: 9px;
}
.login-shell .stButton > button,
.login-shell [data-testid="stFormSubmitButton"] > button {
    min-height: 46px;
    border-radius: 10px !important;
}
@media (max-width: 700px) {
    .login-shell {
        margin-top: 1rem;
    }
    .login-shell .hero {
        padding: 25px 22px;
        min-height: 140px;
    }
}

</style>

""", unsafe_allow_html=True)





# ============================================================

# COLUMN DEFINITIONS AND HELPERS

# ============================================================



ID_CANDIDATES = ["Student_ID", "Student ID", "student_id", "ID", "id"]

NAME_CANDIDATES = ["Student_Name", "Student Name", "Name", "name", "Student"]

TARGET_CANDIDATES = ["At_Risk", "At Risk", "at_risk", "Risk", "Target", "target"]



SUBJECT_LABELS = {

    "Mathematics_Marks": "Mathematics",

    "Core_Subject_1_Marks": "Core Subject 1",

    "Core_Subject_2_Marks": "Core Subject 2",

    "Practical_Lab_Marks": "Practical / Lab",

    "Communication_Skills_Marks": "Communication Skills",

    "Subject_Marks_Average": "Subject Average",

}





def first_col(columns, candidates):

    return next((c for c in candidates if c in columns), None)





def hero(title, subtitle, eyebrow="STUDENT SUCCESS INTELLIGENCE"):

    st.markdown(

        f'<div class="hero"><div class="eyebrow">{eyebrow}</div>'

        f'<h1>{title}</h1><p>{subtitle}</p></div>',

        unsafe_allow_html=True,

    )





def section(title, subtitle=""):

    st.markdown(

        f'<div class="section-title">{title}</div>'

        f'<div class="section-sub">{subtitle}</div>',

        unsafe_allow_html=True,

    )





def metric(label, value, note=""):

    st.markdown(

        f'<div class="metric-card"><div class="metric-label">{label}</div>'

        f'<div class="metric-value">{value}</div>'

        f'<div class="metric-note">{note}</div></div>',

        unsafe_allow_html=True,

    )





def fmt_pct(value):

    return "—" if pd.isna(value) else f"{float(value):.1%}"





def num(row, column):

    if not column or column not in row.index:

        return np.nan

    return pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]





def band(probability):

    if pd.isna(probability):

        return "Unavailable"

    if probability >= 0.70:

        return "High"

    if probability >= 0.40:

        return "Moderate"

    return "Low"





def subject_cols(df):

    return [

        c for c in SUBJECT_LABELS

        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])

    ]





def support_actions(row):

    actions = []

    attendance = num(row, "Attendance_Percentage")

    internal = num(row, "Internal_Marks_Out_of_30")

    gpa = num(row, "Previous_Semester_GPA")

    backlogs = num(row, "Backlogs_Count")

    assignment = num(row, "Assignment_Score_Percentage")

    late = num(row, "Late_Submissions_Count")



    if pd.notna(attendance) and attendance < 75:

        actions.append((

            "Attendance support",

            "Review missed classes and agree on a realistic attendance plan."

        ))

    if pd.notna(internal) and internal < 15:

        actions.append((

            "Internal assessment support",

            "Identify difficult topics and schedule regular practice."

        ))

    if pd.notna(gpa) and gpa < 6.5:

        actions.append((

            "Academic mentor meeting",

            "Create achievable weekly goals with a faculty mentor."

        ))

    if pd.notna(backlogs) and backlogs > 0:

        actions.append((

            "Backlog plan",

            "Prioritize backlog subjects and build a revision calendar."

        ))

    if pd.notna(assignment) and assignment < 60:

        actions.append((

            "Assignment support",

            "Review assignment requirements and submission progress."

        ))

    if pd.notna(late) and late > 2:

        actions.append((

            "Deadline planning",

            "Use a weekly calendar with reminders for upcoming submissions."

        ))

    if not actions:

        actions.append((

            "Maintain progress",

            "Continue the study routine and review progress regularly."

        ))

    return actions





# ============================================================

# LOAD DATA

# ============================================================



@st.cache_data(show_spinner=False)

def load_data(path):

    if not Path(path).exists():

        raise FileNotFoundError(f"Dataset not found: {path}")



    df = pd.read_csv(path)

    df.columns = [str(c).strip() for c in df.columns]



    if df.empty:

        raise ValueError("The student dataset is empty.")



    id_col = first_col(df.columns, ID_CANDIDATES)

    if not id_col:

        raise ValueError("Dataset needs a Student_ID column.")



    df[id_col] = df[id_col].astype(str).str.strip()



    if df[id_col].duplicated().any():

        raise ValueError("Student IDs must be unique.")



    target_col = first_col(df.columns, TARGET_CANDIDATES)

    if target_col:

        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")



    name_col = first_col(df.columns, NAME_CANDIDATES)

    return df, id_col, name_col, target_col





# ============================================================

# LOAD SAVED MODEL — NEVER TRAIN A NEW MODEL

# ============================================================



@st.cache_resource(show_spinner="Loading saved prediction service...")

def load_model(model_path, metadata_path):

    if not Path(model_path).exists():

        raise FileNotFoundError(f"Saved model not found: {model_path}")

    if not Path(metadata_path).exists():

        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")



    with open(metadata_path, "r", encoding="utf-8") as file:

        metadata = json.load(file)



    model = joblib.load(model_path)

    features = metadata.get("feature_columns")



    if not isinstance(features, list) or not features:

        raise ValueError(

            "model_metadata.json must contain feature_columns."

        )



    if not hasattr(model, "predict_proba"):

        raise ValueError(

            "The saved model must support predict_proba."

        )



    return model, metadata, features





def positive_class(model, metadata):

    classes = getattr(model, "classes_", None)



    if classes is None and hasattr(model, "named_steps"):

        for step in reversed(list(model.named_steps.values())):

            if hasattr(step, "classes_"):

                classes = step.classes_

                break



    requested = metadata.get("positive_class", 1)



    if classes is None:

        return requested, 1



    for i, cls in enumerate(classes):

        if str(cls) == str(requested):

            return cls, i



    for i, cls in enumerate(classes):

        if str(cls).strip().lower() in {

            "1", "true", "yes", "at risk", "at_risk", "high risk"

        }:

            return cls, i



    if len(classes) == 2:

        return classes[-1], 1



    raise ValueError("Could not identify the At Risk class.")





def predict_row(row, model, metadata, features):

    missing = [feature for feature in features if feature not in row.index]

    if missing:

        raise ValueError(

            "CSV missing model input columns: " + ", ".join(missing)

        )



    X = pd.DataFrame(

        [{feature: row[feature] for feature in features}],

        columns=features,

    )



    pos_class, pos_index = positive_class(model, metadata)

    probabilities = model.predict_proba(X)[0]

    prediction = model.predict(X)[0]



    try:

        is_at_risk = int(prediction) == int(pos_class)

    except (TypeError, ValueError):

        is_at_risk = str(prediction) == str(pos_class)



    return float(probabilities[pos_index]), bool(is_at_risk)





def add_predictions(df, model, metadata, features, id_col, name_col):

    results = []



    for index, row in df.iterrows():

        try:

            probability, at_risk = predict_row(

                row, model, metadata, features

            )

            status = "At Risk" if at_risk else "Lower Risk"

            error = ""

        except Exception as exc:

            probability = np.nan

            status = "Unavailable"

            error = str(exc)



        name = (

            str(row[name_col])

            if name_col and pd.notna(row[name_col])

            else "—"

        )



        results.append({

            "Student_ID": str(row[id_col]),

            "Student_Name": name,

            "Risk probability": probability,

            "Predicted status": status,

            "Prediction issue": error,

            "_index": index,

        })



    return pd.DataFrame(results).set_index("_index")





def load_interventions():

    if INTERVENTIONS_PATH.exists():

        try:

            return pd.read_csv(INTERVENTIONS_PATH).to_dict("records")

        except Exception:

            return []

    return []





def save_interventions(records):

    INTERVENTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(records).to_csv(INTERVENTIONS_PATH, index=False)





# ============================================================

# STARTUP

# ============================================================



try:

    data, ID_COL, NAME_COL, TARGET_COL = load_data(str(DATA_PATH))

    MODEL, META, FEATURES = load_model(

        str(MODEL_PATH), str(META_PATH)

    )



    missing = [f for f in FEATURES if f not in data.columns]

    if missing:

        st.error(

            "Your CSV does not match the saved model inputs: "

            + ", ".join(missing)

        )

        st.info(

            "Use the same feature schema as the training notebook."

        )

        st.stop()



       # Calculate predictions once per session.
    # Reuse the cached results when navigating between features.
    prediction_cache_key = (
        DATA_PATH.stat().st_mtime_ns,
        MODEL_PATH.stat().st_mtime_ns,
        META_PATH.stat().st_mtime_ns,
    )

    if (
        st.session_state.get("_prediction_cache_key")
        != prediction_cache_key
    ):
        st.session_state["_prediction_cache"] = add_predictions(
            data, MODEL, META, FEATURES, ID_COL, NAME_COL
        )
        st.session_state["_prediction_cache_key"] = (
            prediction_cache_key
        )

    PRED = st.session_state["_prediction_cache"]

except Exception as exc:

    st.error(f"EduPulse could not start: {exc}")

    st.markdown("Required files:")

    st.code(

        "data/students.csv\n"

        "models/student_risk_model.joblib\n"

        "models/model_metadata.json"

    )

    st.stop()





# ============================================================

# LOGIN

# ============================================================



if "auth" not in st.session_state:

    st.session_state.auth = None



if "interventions" not in st.session_state:

    st.session_state.interventions = load_interventions()



if "goals" not in st.session_state:

    st.session_state.goals = {}





def login():
    # Wider center column keeps the complete welcome card and form visually balanced.
    _, middle, _ = st.columns([1, 2.2, 1])

    with middle:
        st.markdown('<div class="login-shell">', unsafe_allow_html=True)

        hero(
            "Welcome to EduPulse",
            "Academic early warning and practical student support.",
            "EDUPULSE • STUDENT SUCCESS",
        )

        role = st.radio(
            "Continue as",
            ["Student", "Faculty"],
            horizontal=True,
            key="login_role",
        )

        with st.form("login_form"):
            if role == "Student":
                sid = st.text_input(
                    "Student ID",
                    placeholder="e.g. STU0001",
                ).strip()
                email = password = ""
            else:
                email = st.text_input(
                    "Faculty email",
                    placeholder="faculty@edupulse.demo",
                ).strip().lower()
                password = st.text_input("Password", type="password")
                sid = ""

            submit = st.form_submit_button(
                "Continue",
                type="primary",
                use_container_width=True,
            )

        if submit:
            if role == "Student":
                found = data[data[ID_COL].astype(str).str.strip() == sid]

                if not sid:
                    st.error("Enter your Student ID.")
                elif found.empty:
                    st.error("Student ID not found in the dataset.")
                else:
                    record = found.iloc[0]
                    name = (
                        str(record[NAME_COL])
                        if NAME_COL and pd.notna(record[NAME_COL])
                        else sid
                    )
                    st.session_state.auth = {
                        "role": "Student",
                        "id": sid,
                        "name": name,
                    }
                    st.rerun()

            elif (
                hmac.compare_digest(email, FACULTY_EMAIL)
                and hmac.compare_digest(password, FACULTY_PASSWORD)
            ):
                st.session_state.auth = {
                    "role": "Faculty",
                    "id": email,
                    "name": "Faculty",
                }
                st.rerun()
            else:
                st.error("Incorrect faculty email or password.")

        with st.expander("Faculty demo access"):
            st.code("faculty@edupulse.demo / Faculty@123")

        st.caption(
            "Student ID-only access is for demonstrations. "
            "Use secure institutional authentication before handling real records."
        )
        st.markdown("</div>", unsafe_allow_html=True)



if st.session_state.auth is None:

    login()

    st.stop()





user = st.session_state.auth

role = user["role"]



FACULTY_PAGES = [

    "Overview & risk distribution",

    "At-risk student list",


    "Intervention & follow-up",


    "Reports & notifications",

]



STUDENT_PAGES = [

    "My dashboard",

    "Personal academic profile",

    "Risk report",


    "Semester progress tracker",


    "What-if simulator",

    "Notifications & personal report",

]





# ============================================================

# SIDEBAR

# ============================================================



with st.sidebar:

    st.markdown(

        '<div style="font-size:28px">🎓</div>'

        '<div style="font-family:Manrope;font-size:23px;font-weight:800">'

        'EduPulse</div>'

        '<div style="font-size:10px;letter-spacing:1.4px;'

        'color:#c7d2fe;margin:4px 0 22px">'

        'ACADEMIC EARLY WARNING</div>',

        unsafe_allow_html=True,

    )



    st.markdown(f"**{user['name']}**")

    st.caption(f"{role} workspace")

    st.divider()



    page = st.radio(

        "WORKSPACE",

        FACULTY_PAGES if role == "Faculty" else STUDENT_PAGES,

        label_visibility="collapsed",

    )



    st.divider()

    st.success("Prediction service active")

    st.caption("Academic support workspace")



    if st.button(

        "Log out",

        key="logout",

        use_container_width=True,

        type="primary",

    ):

        st.session_state.auth = None

        st.rerun()





# ============================================================

# FACULTY WORKSPACE

# ============================================================



if role == "Faculty":



    if page == "Overview & risk distribution":

        hero(

            "Faculty success command center",

            "A clear overview of students who may benefit from timely support.",

            "FACULTY WORKSPACE",

        )



        valid = PRED[PRED["Predicted status"] != "Unavailable"]

        at_risk = valid[valid["Predicted status"] == "At Risk"]



        high = (valid["Risk probability"] >= 0.70).sum()

        moderate = (

            (valid["Risk probability"] >= 0.40)

            & (valid["Risk probability"] < 0.70)

        ).sum()



        attendance_col = next(

            (

                c for c in [

                    "Attendance_Percentage",

                    "Attendance",

                    "Attendance_Percent",

                ]

                if c in data.columns

            ),

            None,

        )



        avg_attendance = (

            pd.to_numeric(data[attendance_col], errors="coerce").mean()

            if attendance_col else np.nan

        )



        a, b, c, d = st.columns(4)



        with a:

            metric("Students monitored", f"{len(data):,}", "Loaded student records")

        with b:

            metric("Predicted at risk", f"{len(at_risk):,}", "Saved model predictions")

        with c:

            metric("High-priority signals", f"{high:,}", "Probability ≥ 70%")

        with d:

            metric(

                "Average attendance",

                f"{avg_attendance:.1f}%" if pd.notna(avg_attendance) else "—",

                "Where available",

            )



        left, right = st.columns(2)



        with left:

            section(

                "Predicted risk distribution",

                "Based on saved-model predictions.",

            )



            counts = (

                valid["Predicted status"]

                .value_counts()

                .reindex(["At Risk", "Lower Risk"], fill_value=0)

                .reset_index()

            )

            counts.columns = ["Status", "Students"]



            fig = px.pie(

                counts,

                names="Status",

                values="Students",

                hole=0.62,

                color="Status",

                color_discrete_map={

                    "At Risk": "#e96879",

                    "Lower Risk": "#14a995",

                },

            )

            fig.update_layout(

                height=330,

                margin=dict(t=15, b=10),

                paper_bgcolor="white",

            )

            st.plotly_chart(fig, use_container_width=True)



        with right:

            section(

                "Cohort academic indicators",

                "Average available subject marks.",

            )



            cols = subject_cols(data)



            if cols:

                averages = pd.DataFrame({

                    "Subject": [

                        SUBJECT_LABELS.get(c, c.replace("_", " "))

                        for c in cols

                    ],

                    "Average marks": [

                        pd.to_numeric(data[c], errors="coerce").mean()

                        for c in cols

                    ],

                })



                fig = px.bar(

                    averages,

                    x="Subject",

                    y="Average marks",

                    color="Average marks",

                    color_continuous_scale=[

                        "#b8b5ff", "#5145e5", "#14a995"

                    ],

                )

                fig.update_layout(

                    height=330,

                    margin=dict(t=15, b=70),

                    paper_bgcolor="white",

                    plot_bgcolor="white",

                    coloraxis_showscale=False,

                )

                st.plotly_chart(fig, use_container_width=True)

            else:

                st.info("No subject-mark columns are available.")



        section("Students to review", "Highest model-estimated probabilities.")



        top = PRED.sort_values(

            "Risk probability",

            ascending=False,

            na_position="last",

        ).head(10).copy()



        top["Risk probability"] = top["Risk probability"].map(fmt_pct)



        st.dataframe(

            top[[

                "Student_ID",

                "Student_Name",

                "Predicted status",

                "Risk probability",

            ]],

            use_container_width=True,

            hide_index=True,

        )



    elif page == "At-risk student list":

        hero(

            "At-risk student list",

            "Search, filter, and export a focused early-warning queue.",

            "FACULTY WORKSPACE",

        )



        left, right = st.columns(2)



        with left:

            query = st.text_input("Search by student ID or name")



        with right:

            priorities = st.multiselect(

                "Priority band",

                ["High", "Moderate", "Low"],

                default=["High", "Moderate"],

            )



        view = PRED.copy()

        view["Priority"] = view["Risk probability"].map(band)

        view = view[view["Priority"].isin(priorities)]



        if query:

            view = view[

                view["Student_ID"].str.contains(

                    query, case=False, na=False

                )

                | view["Student_Name"].str.contains(

                    query, case=False, na=False

                )

            ]



        view = view.sort_values(

            "Risk probability",

            ascending=False,

            na_position="last",

        )



        st.caption(f"{len(view)} students match your filters.")



        display = view[[

            "Student_ID",

            "Student_Name",

            "Predicted status",

            "Priority",

            "Risk probability",

        ]].copy()



        display["Risk probability"] = display["Risk probability"].map(fmt_pct)



        st.dataframe(

            display,

            use_container_width=True,

            hide_index=True,

        )



        st.download_button(

            "Download at-risk list",

            view.to_csv(index=False).encode(),

            "edupulse_at_risk_students.csv",

            "text/csv",

            type="primary",

        )



    elif page == "Individual student analysis":

        hero(

            "Individual student analysis",

            "Review the student's record, risk estimate, and possible support steps.",

            "FACULTY WORKSPACE",

        )



        sid = st.selectbox(

            "Choose student ID",

            data[ID_COL].astype(str).tolist(),

        )



        index = data.index[data[ID_COL].astype(str) == sid][0]

        row = data.loc[index]

        pred = PRED.loc[index]



        a, b, c = st.columns(3)



        with a:

            metric(

                "Student",

                str(row[NAME_COL]) if NAME_COL and pd.notna(row[NAME_COL]) else sid,

                "Selected record",

            )

        with b:

            metric(

                "Estimated risk",

                fmt_pct(pred["Risk probability"]),

                "Saved model estimate",

            )

        with c:

            metric(

                "Support priority",

                band(pred["Risk probability"]),

                "For faculty review",

            )



        profile_cols = [

            col for col in [

                "Department", "Year_of_Study", "Semester",

                "Previous_Semester_GPA", "Attendance_Percentage",

                "Classes_Missed", "Internal_Marks_Out_of_30",

                "Assignment_Score_Percentage", "Quiz_Average_Percentage",

                "Backlogs_Count", "Late_Submissions_Count",

                "LMS_Login_Count_Per_Month",

            ]

            if col in data.columns

        ]



        section("Academic profile", "Available recorded indicators.")



        st.dataframe(

            pd.DataFrame({

                "Indicator": [col.replace("_", " ") for col in profile_cols],

                "Value": [row[col] for col in profile_cols],

            }),

            use_container_width=True,

            hide_index=True,

        )



        cols = subject_cols(data)



        if cols:

            section("Subject-wise performance")



            marks = pd.DataFrame({

                "Subject": [SUBJECT_LABELS.get(col, col) for col in cols],

                "Marks": [num(row, col) for col in cols],

            }).dropna()



            fig = px.bar(

                marks,

                x="Subject",

                y="Marks",

                text_auto=".1f",

                color="Marks",

                color_continuous_scale=[

                    "#f3b7c0", "#5145e5", "#14a995"

                ],

            )

            fig.update_layout(

                height=330,

                margin=dict(b=70),

                paper_bgcolor="white",

                plot_bgcolor="white",

                coloraxis_showscale=False,

            )

            st.plotly_chart(fig, use_container_width=True)



        section("Suggested supportive actions")



        for title, description in support_actions(row):

            st.markdown(f"- **{title}:** {description}")



        st.caption(

            "The model score is a prioritization signal, not a causal explanation "

            "or automatic academic decision."

        )



    elif page == "Intervention & follow-up":

        hero(

            "Intervention & follow-up",

            "Record supportive actions and maintain a follow-up register.",

            "FACULTY WORKSPACE",

        )



        with st.form("intervention_form", clear_on_submit=True):

            sid = st.selectbox(

                "Student ID",

                data[ID_COL].astype(str).tolist(),

            )



            action = st.selectbox(

                "Support action",

                [

                    "Attendance discussion",

                    "Subject tutoring",

                    "Backlog planning",

                    "Mentor meeting",

                    "Assignment support",

                    "General check-in",

                    "Other",

                ],

            )



            owner = st.text_input("Assigned faculty / mentor")

            due = st.date_input("Follow-up date", value=date.today())



            status = st.selectbox(

                "Status",

                ["Planned", "In progress", "Completed", "Follow-up needed"],

            )



            notes = st.text_area("Notes")

            submitted = st.form_submit_button(

                "Save follow-up",

                type="primary",

            )



        if submitted:

            st.session_state.interventions.append({

                "Student_ID": sid,

                "Support action": action,

                "Assigned to": owner,

                "Follow-up date": str(due),

                "Status": status,

                "Notes": notes,

            })



            try:

                pd.DataFrame(

                    st.session_state.interventions

                ).to_csv(INTERVENTIONS_PATH, index=False)

                st.success("Follow-up saved.")

            except Exception as exc:

                st.error(f"Could not save follow-up: {exc}")



        section(

            "Follow-up register",

            "Recorded follow-ups are saved locally.",

        )



        interventions = pd.DataFrame(st.session_state.interventions)



        if interventions.empty:

            st.info("No follow-ups recorded yet.")

        else:

            statuses = st.multiselect(

                "Filter status",

                ["Planned", "In progress", "Completed", "Follow-up needed"],

                default=["Planned", "In progress", "Follow-up needed"],

            )



            filtered = (

                interventions[interventions["Status"].isin(statuses)]

                if "Status" in interventions.columns

                else interventions

            )



            st.dataframe(

                filtered,

                use_container_width=True,

                hide_index=True,

            )



            st.download_button(

                "Download follow-up register",

                interventions.to_csv(index=False).encode(),

                "edupulse_followups.csv",

                "text/csv",

            )



    elif page == "Semester & cohort analytics":

        hero(

            "Semester & cohort analytics",

            "Explore academic indicators and recorded outcomes across groups.",

            "FACULTY WORKSPACE",

        )



        group_options = [

            col for col in ["Department", "Year_of_Study", "Semester"]

            if col in data.columns

        ]



        if not group_options:

            st.info("No cohort grouping columns are available.")

        else:

            group_col = st.selectbox("Group students by", group_options)



            grouped = (

                data.groupby(group_col, dropna=False)

                .agg(

                    Students=(ID_COL, "count"),

                    Recorded_At_Risk_Rate=(

                        TARGET_COL,

                        "mean",

                    ) if TARGET_COL else (ID_COL, "count"),

                )

                .reset_index()

            )



            if TARGET_COL:

                grouped["Recorded_At_Risk_Rate"] *= 100

            else:

                grouped["Recorded_At_Risk_Rate"] = np.nan



            grouped["Group"] = grouped[group_col].astype(str)



            fig = px.bar(

                grouped,

                x="Group",

                y="Students",

                color="Group",

                title="Students by cohort",

            )

            fig.update_layout(

                height=350,

                showlegend=False,

                paper_bgcolor="white",

                plot_bgcolor="white",

            )

            st.plotly_chart(fig, use_container_width=True)



            if TARGET_COL:

                fig = px.bar(

                    grouped,

                    x="Group",

                    y="Recorded_At_Risk_Rate",

                    color="Recorded_At_Risk_Rate",

                    title="Recorded At_Risk label rate (%)",

                    color_continuous_scale=["#14a995", "#6558f5"],

                )

                fig.update_layout(

                    height=350,

                    paper_bgcolor="white",

                    plot_bgcolor="white",

                )

                st.plotly_chart(fig, use_container_width=True)



            st.dataframe(

                grouped,

                use_container_width=True,

                hide_index=True,

            )



        st.warning(

            "Historical semester trends require dated records from multiple semesters."

        )



    elif page == "Reports & notifications":

        hero(

            "Reports & notifications",

            "Review support signals and export cohort reports.",

            "FACULTY WORKSPACE",

        )



        report = PRED.copy()

        report["Priority"] = report["Risk probability"].map(band)



        a, b, c = st.columns(3)



        with a:

            metric(

                "High priority",

                int((report["Priority"] == "High").sum()),

                "Estimated probability ≥ 70%",

            )

        with b:

            metric(

                "Moderate priority",

                int((report["Priority"] == "Moderate").sum()),

                "Estimated probability 40–69.9%",

            )

        with c:

            metric(

                "Students monitored",

                len(report),

                "Loaded records",

            )



        st.dataframe(

            report.sort_values(

                "Risk probability",

                ascending=False,

                na_position="last",

            ),

            use_container_width=True,

            hide_index=True,

        )



        st.download_button(

            "Download cohort risk report",

            report.to_csv(index=False).encode(),

            "edupulse_cohort_risk_report.csv",

            "text/csv",

            type="primary",

        )



        st.caption(

            "This prototype does not send email or SMS notifications."

        )





# ============================================================

# STUDENT WORKSPACE

# ============================================================



else:

    sid = user["id"]



    matches = data[data[ID_COL].astype(str) == str(sid)]



    if matches.empty:

        st.error("Your account is not linked to a student record.")

        st.stop()



    row = matches.iloc[0]

    index = row.name

    pred = PRED.loc[index]



    prob = pred["Risk probability"]

    status = pred["Predicted status"]



    if page == "My dashboard":

        hero(

            f"Welcome back, {user['name']}",

            "Your academic snapshot and recommended next steps.",

            "YOUR LEARNING SPACE",

        )



        gpa = num(row, "Previous_Semester_GPA")

        attendance = num(row, "Attendance_Percentage")



        a, b, c, d = st.columns(4)



        with a:

            metric("Student ID", sid, "Your academic record")

        with b:

            metric(

                "Previous GPA",

                f"{gpa:.2f}" if pd.notna(gpa) else "—",

                "Previous semester",

            )

        with c:

            metric(

                "Attendance",

                f"{attendance:.1f}%" if pd.notna(attendance) else "—",

                "Recorded attendance",

            )

        with d:

            metric(

                "Risk estimate",

                fmt_pct(prob),

                f"{band(prob)} support priority",

            )



        left, right = st.columns(2)



        with left:

            section("Subject performance", "Your recorded marks.")



            cols = subject_cols(data)



            if cols:

                marks = pd.DataFrame({

                    "Subject": [SUBJECT_LABELS.get(col, col) for col in cols],

                    "Marks": [num(row, col) for col in cols],

                }).dropna()



                fig = px.bar(

                    marks,

                    x="Subject",

                    y="Marks",

                    text_auto=".1f",

                    color="Marks",

                    color_continuous_scale=[

                        "#f3b7c0", "#6558f5", "#14a995"

                    ],

                )

                fig.update_layout(

                    height=340,

                    margin=dict(b=75),

                    paper_bgcolor="white",

                    plot_bgcolor="white",

                    coloraxis_showscale=False,

                )

                st.plotly_chart(fig, use_container_width=True)

            else:

                st.info("Subject marks are not available.")



        with right:

            section("Your action plan", "Suggestions based on recorded indicators.")



            for title, description in support_actions(row):

                st.markdown(f"**{title}**\n\n{description}\n")



            st.info(

                "Discuss academic concerns with faculty. The model is a support signal, "

                "not a final judgment."

            )



    elif page == "Personal academic profile":

        hero(

            "Personal academic profile",

            "Review the academic information recorded for your student ID.",

            "MY PROFILE",

        )



        cols = [

            col for col in [

                "Department", "Year_of_Study", "Semester",

                "Previous_Semester_GPA", "Attendance_Percentage",

                "Classes_Missed", "Internal_Marks_Out_of_30",

                "Assignment_Score_Percentage", "Quiz_Average_Percentage",

                "Assignment_Submission_Rate_Percentage",

                "Late_Submissions_Count", "Backlogs_Count",

                "Study_Hours_Per_Week", "LMS_Login_Count_Per_Month",

                "Attendance_Trend",

            ]

            if col in data.columns

        ]



        st.dataframe(

            pd.DataFrame({

                "Indicator": [col.replace("_", " ") for col in cols],

                "Your value": [row[col] for col in cols],

            }),

            use_container_width=True,

            hide_index=True,

        )



        cols = subject_cols(data)



        if cols:

            section("Subject marks")



            st.dataframe(

                pd.DataFrame({

                    "Subject": [SUBJECT_LABELS.get(col, col) for col in cols],

                    "Marks": [row[col] for col in cols],

                }),

                use_container_width=True,

                hide_index=True,

            )



    elif page == "Risk report":

        hero(

            "My risk report",

            "Understand your model estimate and indicators worth reviewing.",

            "PERSONAL RISK INSIGHT",

        )



        a, b, c = st.columns(3)



        with a:

            metric("Risk status", status, "Saved model prediction")

        with b:

            metric("Estimated probability", fmt_pct(prob), "Model estimate")

        with c:

            metric(

                "Recorded dataset label",

                str(row[TARGET_COL]) if TARGET_COL else "Not available",

                "Observed label, if supplied",

            )



        st.markdown("### What this means")



        st.write(

            "The probability is an estimate from the trained model. "

            "It is not a diagnosis, certainty, or explanation of why an outcome may occur."

        )



        section("Indicators to review")



        cols = [

            col for col in [

                "Attendance_Percentage", "Previous_Semester_GPA",

                "Internal_Marks_Out_of_30", "Assignment_Score_Percentage",

                "Quiz_Average_Percentage", "Backlogs_Count",

                "Classes_Missed", "Late_Submissions_Count",

            ]

            if col in data.columns

        ]



        st.dataframe(

            pd.DataFrame({

                "Indicator": [col.replace("_", " ") for col in cols],

                "Recorded value": [row[col] for col in cols],

            }),

            use_container_width=True,

            hide_index=True,

        )



        section("Suggested support actions")



        for title, description in support_actions(row):

            st.markdown(f"- **{title}:** {description}")



    elif page == "Subject-wise performance":

        hero(

            "Subject-wise performance",

            "Compare your marks and identify subjects to review.",

            "ACADEMIC PERFORMANCE",

        )



        cols = subject_cols(data)



        if cols:

            marks = pd.DataFrame({

                "Subject": [SUBJECT_LABELS.get(col, col) for col in cols],

                "Marks": [num(row, col) for col in cols],

            }).dropna()



            fig = px.bar(

                marks,

                x="Subject",

                y="Marks",

                text_auto=".1f",

                color="Marks",

                color_continuous_scale=[

                    "#f3b7c0", "#6558f5", "#14a995"

                ],

            )

            fig.update_layout(

                height=400,

                margin=dict(b=75),

                paper_bgcolor="white",

                plot_bgcolor="white",

                coloraxis_showscale=False,

            )

            st.plotly_chart(fig, use_container_width=True)



            st.dataframe(

                marks.sort_values("Marks", ascending=False),

                use_container_width=True,

                hide_index=True,

            )



            if not marks.empty:

                lowest = marks.loc[marks["Marks"].idxmin()]

                metric(

                    "Subject to review",

                    lowest["Subject"],

                    f"Recorded mark: {lowest['Marks']}",

                )

        else:

            st.info("No numeric subject-mark columns were found.")



    elif page == "Semester progress tracker":

        hero(

            "Semester progress tracker",

            "Review available academic indicators.",

            "MY PROGRESS",

        )



        columns = st.columns(3)



        for column, (label, key) in zip(

            columns,

            [

                ("Department", "Department"),

                ("Year of study", "Year_of_Study"),

                ("Semester", "Semester"),

            ],

        ):

            with column:

                metric(

                    label,

                    str(row[key]) if key in data.columns else "—",

                    "From your record",

                )



        progress = [

            ("Attendance_Percentage", "Attendance", 100),

            ("Assignment_Submission_Rate_Percentage", "Assignment submission", 100),

            ("Assignment_Score_Percentage", "Assignment score", 100),

            ("Internal_Marks_Out_of_30", "Internal marks", 30),

            ("Quiz_Average_Percentage", "Quiz average", 100),

        ]



        found = False



        for key, label, maximum in progress:

            value = num(row, key)



            if pd.notna(value):

                found = True

                st.write(f"**{label}:** {value:g} / {maximum:g}")

                st.progress(float(np.clip(value / maximum, 0, 1)))



        if not found:

            st.info("No progress indicators available.")



        st.warning(

            "The current dataset is a snapshot. Historical progress requires "

            "dated records from multiple points in time."

        )



    elif page == "Personalized action plan":

        hero(

            "Personalized action plan",

            "Choose practical goals and review suggested support actions.",

            "YOUR NEXT STEPS",

        )



        for title, description in support_actions(row):

            st.markdown(f"**{title}**\n\n{description}")



        section("My weekly goals", "Track your goals for this session.")



        goals = [

            "Create a weekly study timetable",

            "Practise difficult topics",

            "Ask faculty for support when needed",

            "Review progress at the end of the week",

        ]



        for goal in goals:

            key = f"{sid}:{goal}"



            st.session_state.goals[key] = st.checkbox(

                goal,

                value=st.session_state.goals.get(key, False),

                key=f"goal_{sid}_{goal}",

            )



    elif page == "What-if simulator":

        hero(

            "What-if simulator",

            "Explore hypothetical changes to numeric model inputs.",

            "SCENARIO EXPLORER",

        )



        st.info(

            "A what-if scenario is hypothetical. It does not change your official "

            "record or guarantee a real-world outcome."

        )



        candidates = [

            col for col in FEATURES

            if col in data.columns

            and pd.api.types.is_numeric_dtype(data[col])

            and pd.to_numeric(data[col], errors="coerce").notna().any()

        ]



        preferred = [

            "Attendance_Percentage",

            "Internal_Marks_Out_of_30",

            "Assignment_Score_Percentage",

            "Quiz_Average_Percentage",

            "Study_Hours_Per_Week",

        ]



        selected = [col for col in preferred if col in candidates][:4]



        if not selected:

            selected = candidates[:4]



        altered = row.copy()

        left, right = st.columns(2)



        for i, col in enumerate(selected):

            values = pd.to_numeric(data[col], errors="coerce").dropna()



            if values.empty:

                continue



            low = float(values.min())

            high = float(values.max())

            current = num(row, col)



            if pd.isna(current):

                current = float(values.median())



            with (left if i % 2 == 0 else right):

                altered[col] = st.slider(

                    col.replace("_", " "),

                    min_value=low,

                    max_value=high,

                    value=float(np.clip(current, low, high)),

                    key=f"sim_{sid}_{col}",

                )



        try:

            new_probability, _ = predict_row(

                altered, MODEL, META, FEATURES

            )



            a, b, c = st.columns(3)



            with a:

                metric("Current probability", fmt_pct(prob), "Saved model")

            with b:

                metric("Scenario probability", fmt_pct(new_probability), "Hypothetical values")

            with c:

                metric(

                    "Change",

                    f"{(new_probability - prob) * 100:+.1f} pp"

                    if pd.notna(prob) else "—",

                    "Percentage points",

                )



            fig = go.Figure(

                go.Bar(

                    x=["Current", "What-if"],

                    y=[prob * 100, new_probability * 100],

                    marker_color=["#6558f5", "#14a995"],

                )

            )

            fig.update_layout(

                height=300,

                yaxis_title="Estimated probability (%)",

                yaxis_range=[0, 100],

                paper_bgcolor="white",

                plot_bgcolor="white",

            )

            st.plotly_chart(fig, use_container_width=True)



        except Exception as exc:

            st.error(f"Could not calculate scenario: {exc}")



    elif page == "Notifications & personal report":

        hero(

            "Notifications & personal report",

            "Review your support signal and export an academic summary.",

            "MY REPORTS",

        )



        if band(prob) == "High":

            st.warning(

                "Your model estimate indicates a high-priority signal. "

                "Consider talking with faculty or a mentor."

            )

        elif band(prob) == "Moderate":

            st.info(

                "Your model estimate indicates a moderate support signal. "

                "Review your indicators and goals."

            )

        elif band(prob) == "Low":

            st.success(

                "No high-priority model signal at this time. "

                "Continue monitoring your progress."

            )

        else:

            st.warning(

                "A risk estimate is not available. Contact faculty to review your record."

            )



        report_fields = [

            col for col in data.columns

            if col in [

                ID_COL, NAME_COL, "Department", "Year_of_Study",

                "Semester", "Previous_Semester_GPA",

                "Attendance_Percentage", "Backlogs_Count",

            ]

            or col in SUBJECT_LABELS

        ]



        report = {col: row[col] for col in report_fields}

        report["Model-estimated risk probability"] = prob

        report["Model-predicted status"] = status



        st.download_button(

            "Download my academic report",

            pd.DataFrame([report]).to_csv(index=False).encode(),

            f"edupulse_report_{sid}.csv",

            "text/csv",

            type="primary",

        )



        st.download_button(

            "Download my action plan",

            "\n".join(

                f"{title}: {description}"

                for title, description in support_actions(row)

            ),

            f"edupulse_action_plan_{sid}.txt",

            "text/plain",

        )



        st.caption(

            "This prototype displays in-app notices and downloadable reports. "

            "It does not send email or SMS."

        )





# ============================================================

# FOOTER

# ============================================================



st.markdown(

    '<div class="footer">EduPulse • Academic early-warning support. '

    'Predictions should inform supportive human review, not replace it.</div>',

    unsafe_allow_html=True,

)
