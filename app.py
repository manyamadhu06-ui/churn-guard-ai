
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="ChurnGuard AI | Customer Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #080F1C;
    color: #EAF1FC;
}

[data-testid="stSidebar"] {
    background: #0D1728;
    border-right: 1px solid #1C2B42;
}

[data-testid="stSidebar"] * {
    color: #DCE7F8;
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif;
    color: #F4F7FD;
}

.hero {
    background: linear-gradient(115deg, #10223C, #122F52);
    border: 1px solid #23476D;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 25px;
}

.hero-label {
    color: #63B3FF;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}

.hero-title {
    font-family: 'Manrope', sans-serif;
    font-size: 34px;
    font-weight: 800;
    color: #FFFFFF;
    margin-top: 8px;
}

.hero-subtitle {
    color: #AFC2DC;
    font-size: 14px;
    margin-top: 8px;
}

.metric-card {
    background: #111E31;
    border: 1px solid #23354F;
    border-radius: 13px;
    padding: 20px;
    min-height: 118px;
}

.metric-label {
    color: #91A6C2;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.7px;
    text-transform: uppercase;
}

.metric-value {
    color: #F5F8FF;
    font-family: 'Manrope', sans-serif;
    font-size: 29px;
    font-weight: 800;
    margin-top: 9px;
}

.metric-note {
    color: #7E9ABC;
    font-size: 11px;
    margin-top: 4px;
}

.section-heading {
    font-family: 'Manrope', sans-serif;
    color: #F0F5FC;
    font-size: 19px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 13px;
}

.info-panel {
    background: #111E31;
    border: 1px solid #23354F;
    border-radius: 13px;
    padding: 19px;
}

div[data-testid="stTabs"] button {
    color: #AFC2DC;
}

.stButton > button {
    background: #287FE5;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    padding: 10px 20px;
}

.stButton > button:hover {
    background: #4599FA;
    color: white;
    border: none;
}

.stTextInput input, .stNumberInput input {
    background: #111E31;
    color: white;
}

div[data-baseweb="select"] > div {
    background: #111E31;
    border-color: #2A3E5C;
}

hr {
    border-color: #23354F;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------

@st.cache_resource
def load_model():
    return joblib.load("churn_guard_model.pkl")

@st.cache_data
def load_data():
    data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )
    data = data.dropna(subset=["TotalCharges"]).copy()
    data["SeniorCitizen"] = data["SeniorCitizen"].astype(str)
    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})
    return data

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Unable to load the model or dataset: {e}")
    st.stop()

# ---------------- CHART THEME ----------------

CHART_LAYOUT = dict(
    paper_bgcolor="#111E31",
    plot_bgcolor="#111E31",
    font=dict(color="#C7D5E8", family="DM Sans"),
    margin=dict(l=20, r=20, t=45, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor="#26364D", zerolinecolor="#26364D"),
    yaxis=dict(gridcolor="#26364D", zerolinecolor="#26364D")
)

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 25px 0;">
        <div style="font-size:12px;letter-spacing:2px;color:#63B3FF;font-weight:700;">
        CHURNGUARD
        </div>
        <div style="font-family:Manrope;font-size:23px;font-weight:800;color:white;">
        AI PLATFORM
        </div>
        <div style="font-size:11px;color:#7F98B8;margin-top:5px;">
        CUSTOMER INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Churn Insights",
            "Customer Prediction",
            "Model Performance"
        ],
        label_visibility="visible"
    )

    st.markdown("---")
    st.caption("MACHINE LEARNING PROJECT")
    st.caption("Telecommunications Customer Analytics")
    st.caption("ChurnGuard AI · 2026")

# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <div class="hero-label">CUSTOMER INTELLIGENCE PLATFORM</div>
    <div class="hero-title">Predict. Prevent. Retain.</div>
    <div class="hero-subtitle">
        Transform customer data into actionable retention intelligence
        using machine learning.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- KPI FUNCTION ----------------

def metric_card(label, value, note):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)

total = len(df)
churned = int(df["Churn"].sum())
retained = total - churned
churn_rate = churned / total * 100

# ---------------- OVERVIEW ----------------

if page == "Overview":

    st.markdown('<div class="section-heading">Business Overview</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Total Customers", f"{total:,}", "Customers analysed")

    with c2:
        metric_card("Churned Customers", f"{churned:,}", "Customers who left")

    with c3:
        metric_card("Retained Customers", f"{retained:,}", "Customers who stayed")

    with c4:
        metric_card("Churn Rate", f"{churn_rate:.2f}%", "Overall customer churn")

    st.markdown('<div class="section-heading">Customer Retention Analysis</div>',
                unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        churn_data = pd.DataFrame({
            "Customer Status": ["Retained", "Churned"],
            "Customers": [retained, churned]
        })

        fig = px.pie(
            churn_data,
            names="Customer Status",
            values="Customers",
            hole=0.68,
            color="Customer Status",
            color_discrete_map={
                "Retained": "#3289E8",
                "Churned": "#F07878"
            },
            title="Customer Distribution"
        )

        fig.update_layout(**CHART_LAYOUT)
        fig.update_traces(
            textposition="outside",
            textinfo="percent+label"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        contract_data = (
            df.groupby("Contract")["Churn"]
            .mean().mul(100).reset_index()
        )
        contract_data.columns = ["Contract", "Churn Rate"]

        fig = px.bar(
            contract_data,
            x="Contract",
            y="Churn Rate",
            color="Contract",
            color_discrete_sequence=["#3289E8", "#50B6C9", "#8C79DF"],
            title="Churn Rate by Contract Type",
            text_auto=".1f"
        )

        fig.update_layout(**CHART_LAYOUT, showlegend=False)
        fig.update_yaxes(title="Churn Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-heading">Customer Behaviour</div>',
                unsafe_allow_html=True)

    fig = px.histogram(
        df,
        x="tenure",
        color="Churn",
        nbins=30,
        barmode="overlay",
        opacity=0.78,
        color_discrete_map={0: "#3289E8", 1: "#F07878"},
        title="Customer Tenure Distribution",
        labels={"tenure": "Tenure (Months)", "Churn": "Churn Status"}
    )

    fig.update_layout(**CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- CHURN INSIGHTS ----------------

elif page == "Churn Insights":

    st.markdown('<div class="section-heading">Customer Churn Insights</div>',
                unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        fig = px.box(
            df,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            color_discrete_map={0: "#3289E8", 1: "#F07878"},
            title="Monthly Charges vs Churn",
            labels={"Churn": "Churn Status", "MonthlyCharges": "Monthly Charges"}
        )
        fig.update_layout(**CHART_LAYOUT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.histogram(
            df,
            x="tenure",
            color="Churn",
            marginal="box",
            color_discrete_map={0: "#3289E8", 1: "#F07878"},
            title="Tenure and Churn Patterns",
            labels={"tenure": "Tenure (Months)"}
        )
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    contract_rate = (
        df.groupby("Contract")["Churn"]
        .mean().mul(100).round(2)
        .reset_index(name="Churn Rate (%)")
    )

    st.markdown('<div class="section-heading">Contract Analysis</div>',
                unsafe_allow_html=True)
    st.dataframe(contract_rate, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="info-panel">
        <b>Business interpretation</b><br><br>
        These visualisations help identify customer groups and service
        characteristics associated with churn. They can guide further
        investigation and targeted customer retention strategies.
    </div>
    """, unsafe_allow_html=True)

# ---------------- CUSTOMER PREDICTION ----------------

elif page == "Customer Prediction":

    st.markdown('<div class="section-heading">Individual Customer Risk Assessment</div>',
                unsafe_allow_html=True)

    st.write(
        "Enter customer details below to generate a churn prediction "
        "using the trained machine learning model."
    )

    feature_columns = [
        c for c in df.columns
        if c not in ["customerID", "Churn"]
    ]

    categorical_columns = [
        c for c in feature_columns
        if c not in ["tenure", "MonthlyCharges", "TotalCharges"]
    ]

    numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

    with st.form("prediction_form"):

        st.markdown("#### Customer Profile")
        col1, col2, col3 = st.columns(3)

        inputs = {}

        for i, column in enumerate(feature_columns):
            container = [col1, col2, col3][i % 3]

            with container:
                if column in categorical_columns:
                    options = sorted(
                        df[column].dropna().astype(str).unique().tolist()
                    )
                    default_index = 0

                    inputs[column] = st.selectbox(
                        column,
                        options,
                        index=default_index
                    )

                elif column == "tenure":
                    inputs[column] = st.number_input(
                        "Tenure (Months)", 0, 72, 12
                    )

                elif column == "MonthlyCharges":
                    inputs[column] = st.number_input(
                        "Monthly Charges",
                        min_value=0.0,
                        max_value=float(df[column].max()),
                        value=float(df[column].median())
                    )

                elif column == "TotalCharges":
                    inputs[column] = st.number_input(
                        "Total Charges",
                        min_value=0.0,
                        max_value=float(df[column].max()),
                        value=float(df[column].median())
                    )

        submitted = st.form_submit_button(
            "GENERATE CHURN PREDICTION",
            use_container_width=True
        )

    if submitted:

        input_df = pd.DataFrame([inputs], columns=feature_columns)

        prediction = int(model.predict(input_df)[0])

        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.named_steps["model"].classes_)
            churn_index = classes.index(1)
            probability = float(probabilities[churn_index]) * 100

        st.markdown("---")
        st.markdown("### Prediction Result")

        if prediction == 1:
            st.error("HIGH PRIORITY: MODEL PREDICTS CUSTOMER CHURN")
        else:
            st.success("MODEL PREDICTS CUSTOMER RETENTION")

        if probability is not None:
            st.metric("Estimated Churn Probability", f"{probability:.2f}%")
            st.progress(min(max(probability / 100, 0.0), 1.0))

        st.caption(
            "This is a model prediction based on the supplied customer "
            "information, not a guarantee of future behaviour."
        )

# ---------------- MODEL PERFORMANCE ----------------

elif page == "Model Performance":

    st.markdown('<div class="section-heading">Machine Learning Performance</div>',
                unsafe_allow_html=True)

    try:
        results = pd.read_csv("model_comparison.csv")

        st.markdown("#### Model Comparison")
        st.dataframe(
            results.style.format({
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1 Score": "{:.3f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        metric_columns = [
            c for c in ["Accuracy", "Precision", "Recall", "F1 Score"]
            if c in results.columns
        ]

        if metric_columns:
            chart_data = results.melt(
                id_vars="Model",
                value_vars=metric_columns,
                var_name="Metric",
                value_name="Score"
            )

            fig = px.bar(
                chart_data,
                x="Metric",
                y="Score",
                color="Model",
                barmode="group",
                title="Cross-Validation Model Comparison",
                color_discrete_sequence=["#3289E8", "#50B6C9"],
                text_auto=".3f"
            )
            fig.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.warning("Model comparison CSV not found.")

    st.markdown("#### Feature Importance")

    try:
        importance = pd.read_csv("feature_importance.csv").head(10)

        fig = px.bar(
            importance.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Features Influencing Model Performance",
            color="Importance",
            color_continuous_scale=["#285D9C", "#50B6FF"]
        )

        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.warning("Feature importance CSV not found.")

# ---------------- FOOTER ----------------

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#7186A3;font-size:11px;padding:15px;">
    CHURNGUARD AI · CUSTOMER CHURN PREDICTION SYSTEM<br>
    MACHINE LEARNING PROTOTYPE · 2026
</div>
""", unsafe_allow_html=True)
