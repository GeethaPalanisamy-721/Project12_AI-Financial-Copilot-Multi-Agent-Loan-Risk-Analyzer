# app.py
"""
AI Financial Intelligence Copilot
SaaS-Style Streamlit UI — Blue Gradient Theme

Page 1: Loan Risk Predictor
Page 2: Model Analytics Dashboard
"""

import streamlit as st
import sys
import os
import json
import joblib
import pandas as pd
import numpy as nps
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    classification_report
)

sys.path.append(os.path.abspath("."))
from src.orchestrator import run_pipeline


# ==============================
# 🎨 PAGE CONFIG + GLOBAL CSS
# ==============================
st.set_page_config(
    page_title="AI Financial Copilot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global CSS — Blue Gradient SaaS Theme
st.markdown("""
<style>

/* ============================= */
/* 🌌 BACKGROUND */
/* ============================= */
.stApp {
    background: radial-gradient(circle at 20% 20%, #0f172a, #020617 80%);
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}

/* ============================= */
/* 📦 GLASS CARD */
/* ============================= */
.glass {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
}

.glass:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(59,130,246,0.3);
}

/* ============================= */
/* 📊 METRICS */
/* ============================= */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 16px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(59,130,246,0.2);
}

/* ============================= */
/* 🎯 BUTTON */
/* ============================= */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    border-radius: 12px;
    padding: 14px;
    font-weight: 600;
    border: none;
    color: white;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 20px rgba(99,102,241,0.6);
}

/* ============================= */
/* 🎚 INPUTS */
/* ============================= */
input, select {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
    color: white !important;
}

/* ============================= */
/* 📢 BANNERS */
/* ============================= */
.approve-banner {
    background: rgba(16,185,129,0.1);
    border: 1px solid #10b981;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 25px rgba(16,185,129,0.3);
}

.reject-banner {
    background: rgba(239,68,68,0.1);
    border: 1px solid #ef4444;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 25px rgba(239,68,68,0.3);
}

.review-banner {
    background: rgba(245,158,11,0.1);
    border: 1px solid #f59e0b;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 25px rgba(245,158,11,0.3);
}

/* ============================= */
/* 📌 HEADINGS */
/* ============================= */
.section-header {
    font-size: 1.2rem;
    font-weight: 600;
    color: #60a5fa;
    margin-bottom: 10px;
}

/* ============================= */
/* ✨ FADE-IN ANIMATION */
/* ============================= */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px);}
    to { opacity: 1; transform: translateY(0);}
}

.fade-in {
    animation: fadeIn 0.6s ease-in-out;
}

/* ============================= */
/* 🚫 HIDE STREAMLIT DEFAULT */
/* ============================= */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)



# ==============================
# 🔹 SIDEBAR NAVIGATION
# ==============================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0'>
        <h1 style='color:#60a5fa; font-size:1.8rem'>🏦</h1>
        <h2 style='color:white; font-size:1.1rem; margin:0'>
            AI Financial<br>Copilot
        </h2>
        <p style='color:#93c5fd; font-size:0.8rem; margin-top:4px'>
            Powered by XGBoost + Llama3
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    selected = option_menu(
        menu_title=None,
        options=["Loan Predictor", "Model Analytics"],
        icons=["search-heart", "bar-chart-line"],
        default_index=0,
        styles={
            "container": {
                "background-color": "transparent",
                "padding": "0"
            },
            "icon": {"color": "#60a5fa", "font-size": "16px"},
            "nav-link": {
                "color": "#93c5fd",
                "font-size": "14px",
                "border-radius": "8px",
                "margin": "4px 0"
            },
            "nav-link-selected": {
                "background": "linear-gradient(90deg, #1d4ed8, #3b82f6)",
                "color": "white",
                "font-weight": "600"
            }
        }
    )

    st.divider()

    # Sidebar info
    st.markdown("""
    <div style='padding: 12px; background: rgba(255,255,255,0.05);
                border-radius: 10px; font-size: 0.8rem; color: #93c5fd'>
        <b style='color:#60a5fa'>📊 Model Info</b><br><br>
        Algorithm: XGBoost<br>
        ROC-AUC: 0.756<br>
        Features: 60<br>
        Threshold: 0.40<br>
        Dataset: Home Credit<br>
        LLM: Llama3 (Local)
    </div>
    """, unsafe_allow_html=True)


# ==============================
# 📄 PAGE 1 — LOAN PREDICTOR
# ==============================
if selected == "Loan Predictor":

    # Header
    st.markdown("""
<div style='padding: 24px 0 16px 0'>

<h1 style='
    background: linear-gradient(90deg,#60a5fa,#a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    margin:0;
'>
🏦 AI Financial Copilot
</h1>

<p style='color:#93c5fd; margin-top:6px'>
Loan Risk Predictor — Multi-Agent AI System
</p>

</div>
""", unsafe_allow_html=True)


    st.divider()

    # ── Input Form ──
    st.markdown("""
    <div class='glass fade-in'>
        <div class='section-header'>📋 Customer Details</div>
    </div>
    """, unsafe_allow_html=True)


    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**💰 Financial**")
        amt_credit = st.number_input(
            "Loan Amount (₹)",
            min_value=0, value=500000, step=10000)
        amt_annuity = st.number_input(
            "Annual Payment (₹)",
            min_value=0, value=25000, step=1000)
        goods_price = st.number_input(
            "Goods Price (₹)",
            min_value=0, value=450000, step=10000)

    with col2:
        st.markdown("**📊 Credit Scores**")
        ext2 = st.slider(
            "External Score 2",
            0.0, 1.0, 0.5,
            help="Credit bureau score 2 (0=poor, 1=excellent)")
        ext3 = st.slider(
            "External Score 3",
            0.0, 1.0, 0.4,
            help="Credit bureau score 3 (0=poor, 1=excellent)")

        st.markdown("**🏠 Assets**")
        own_car = st.selectbox(
            "Owns a Car?", [0, 1],
            format_func=lambda x: "Yes ✅" if x == 1 else "No ❌")

    with col3:
        st.markdown("**👤 Personal**")
        days_employed = st.number_input(
            "Days Employed",
            value=-1000,
            help="Negative = employed (e.g. -1000 = 1000 days)")
        days_birth = st.number_input(
            "Days Birth",
            value=-15000,
            help="Negative days from today (e.g. -15000 ≈ 41 years)")
        gender = st.selectbox(
            "Gender", [0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male")
        education = st.selectbox(
            "Education Level",
            [0, 1, 2, 3, 4],
            format_func=lambda x: [
                "Lower Secondary",
                "Secondary",
                "Incomplete Higher",
                "Higher Education",
                "Academic Degree"][x])

    st.divider()

    # ── Analyze Button ──
    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        analyze = st.button("🔍 Analyze Customer Risk")

    # ── Results ──
    if analyze:
        customer_data = {
            "AMT_CREDIT"         : amt_credit,
            "AMT_ANNUITY"        : amt_annuity,
            "EXT_SOURCE_2"       : ext2,
            "EXT_SOURCE_3"       : ext3,
            "DAYS_EMPLOYED"      : days_employed,
            "DAYS_BIRTH"         : days_birth,
            "AMT_GOODS_PRICE"    : goods_price,
            "CODE_GENDER"        : gender,
            "FLAG_OWN_CAR"       : own_car,
            "NAME_EDUCATION_TYPE": education
        }

        with st.spinner("🤖 Running Multi-Agent AI Pipeline..."):
            result = run_pipeline(customer_data)

        st.divider()

        # ── Decision Banner ──
        decision = result["final_decision"]
        if decision == "APPROVE":
            st.markdown("""
            <div class='approve-banner'>
                <h2 style='color:#10b981; margin:0'>
                    ✅ LOAN APPROVED
                </h2>
                <p style='color:#6ee7b7; margin:4px 0 0 0'>
                    Customer presents low credit risk
                </p>
            </div>""", unsafe_allow_html=True)
        elif decision == "REJECT":
            st.markdown("""
            <div class='reject-banner'>
                <h2 style='color:#ef4444; margin:0'>
                    ❌ LOAN REJECTED
                </h2>
                <p style='color:#fca5a5; margin:4px 0 0 0'>
                    Customer presents high credit risk
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='review-banner'>
                <h2 style='color:#f59e0b; margin:0'>
                    ⚠️ MANUAL REVIEW REQUIRED
                </h2>
                <p style='color:#fde68a; margin:4px 0 0 0'>
                    Borderline case — human review recommended
                </p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Metrics Row ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Default Probability",
                  f"{result['probability']:.2%}")
        m2.metric("Model Decision",
                  result["model_decision"])
        m3.metric("Risk Level",
                  result["risk_level"])
        m4.metric("DTI Ratio",
                  result["dti_ratio"])

        st.divider()

        # ── Probability Gauge Chart ──
        st.markdown(
            "<div class='section-header'>📊 Risk Gauge</div>",
            unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result["probability"] * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Default Probability %",
                   "font": {"color": "white"}},
            delta={"reference": 40,
                   "increasing": {"color": "#ef4444"},
                   "decreasing": {"color": "#10b981"}},
            gauge={
                "axis": {"range": [0, 100],
                         "tickcolor": "white",
                         "tickfont": {"color": "white"}},
                "bar": {"color": "#3b82f6"},
                "steps": [
                    {"range": [0, 35],
                     "color": "rgba(34,197,94,0.3)"},
                    {"range": [35, 60],
                     "color": "rgba(234,179,8,0.3)"},
                    {"range": [60, 100],
                     "color": "rgba(239,68,68,0.3)"}
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": 40
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=300,
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.divider()

        # ── AI Explanation ──
        st.markdown(
            "<div class='section-header'>💬 AI Explanation</div>",
            unsafe_allow_html=True)

        st.markdown(f"""
        <div class='glass fade-in'>
            <p style='color:#e2e8f0; line-height:1.8; margin:0'>
                {result['explanation']}
            </p>
        </div>
        """, unsafe_allow_html=True)


# ==============================
# 📊 PAGE 2 — MODEL ANALYTICS
# ==============================
elif selected == "Model Analytics":

    st.markdown("""
    <div style='padding: 24px 0 16px 0'>
        <h1 style='color:white; font-size:2rem; margin:0'>
            📊 Model Analytics Dashboard
        </h1>
        <p style='color:#93c5fd; margin-top:4px'>
            XGBoost model performance & explainability insights
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Load Data ──
    @st.cache_data
    def load_analytics_data():
        model   = joblib.load("models/xgboost_model.pkl")
        X_test  = pd.read_csv("data/processed/X_test.csv")
        y_test  = pd.read_csv("data/processed/y_test.csv").values.ravel()
        with open("models/top_shap_features.json") as f:
            shap_features = json.load(f)
        feat_imp = pd.read_csv("reports/feature_importance.csv")
        return model, X_test, y_test, shap_features, feat_imp

    with st.spinner("Loading model analytics..."):
        model, X_test, y_test, shap_features, feat_imp = \
            load_analytics_data()
        y_prob  = model.predict_proba(X_test)[:, 1]
        y_pred  = (y_prob >= 0.4).astype(int)

    # ── Metrics Row ──
    st.markdown(
        "<div class='section-header'>🎯 Model Performance</div>",
        unsafe_allow_html=True)

    from sklearn.metrics import (precision_score, recall_score,
                                  f1_score, accuracy_score)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC Score",  "0.7565")
    m2.metric("Precision",
              f"{precision_score(y_test, y_pred):.3f}")
    m3.metric("Recall",
              f"{recall_score(y_test, y_pred):.3f}")
    m4.metric("F1 Score",
              f"{f1_score(y_test, y_pred):.3f}")

    st.divider()

    # ── Row 1: SHAP + Confusion Matrix ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "<div class='section-header'>🔥 SHAP Feature Importance</div>",
            unsafe_allow_html=True)

        features = list(shap_features.keys())
        values   = list(shap_features.values())

        fig_shap = go.Figure(go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker=dict(
                color=values,
                colorscale="Viridis",
                showscale=True
            )
        ))
        fig_shap.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=400,
            xaxis=dict(
                title="Mean |SHAP Value|",
                gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(
                autorange="reversed",
                gridcolor="rgba(255,255,255,0.1)"),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with col2:
        st.markdown(
            "<div class='section-header'>📊 Confusion Matrix</div>",
            unsafe_allow_html=True)

        cm = confusion_matrix(y_test, y_pred)
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted Repay", "Predicted Default"],
            y=["Actual Repay", "Actual Default"],
            colorscale="YlGnBu",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 18, "color": "Black"},
            showscale=False
        ))
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.divider()

    # ── Row 2: ROC Curve + Distribution ──
    col3, col4 = st.columns(2)

    with col3:
        st.markdown(
            "<div class='section-header'>📈 ROC-AUC Curve</div>",
            unsafe_allow_html=True)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score   = roc_auc_score(y_test, y_prob)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"XGBoost (AUC={auc_score:.3f})",
            line=dict(color="green", width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1],
            mode="lines",
            name="Random Baseline",
            line=dict(color="red",
                      width=2, dash="dash")
        ))
        fig_roc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=380,
            xaxis=dict(
                title="False Positive Rate",
                gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(
                title="True Positive Rate",
                gridcolor="rgba(255,255,255,0.1)"),
            legend=dict(
                bgcolor="rgba(0,0,0,0.3)",
                bordercolor="#3b82f6"),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col4:
        st.markdown(
            "<div class='section-header'>📊 Model Metrics Table</div>",
            unsafe_allow_html=True)

        report = classification_report(
            y_test, y_pred,
            target_names=["Repay", "Default"],
            output_dict=True)

        report_df = pd.DataFrame(report).transpose().round(3)
        report_df = report_df.drop(
            ["accuracy", "macro avg", "weighted avg"],
            errors="ignore")

        st.dataframe(
            report_df.style.background_gradient(
                cmap="Blues",
                subset=["precision","recall","f1-score"]
            ),
            use_container_width=True,
            height=180
        )

        st.divider()

        # Top features table
        st.markdown(
            "<div class='section-header'>🏆 Top Features</div>",
            unsafe_allow_html=True)

        top_df = pd.DataFrame({
            "Feature"   : list(shap_features.keys()),
            "SHAP Score": [round(v, 4)
                          for v in shap_features.values()]
        })
        st.dataframe(
            top_df.style.background_gradient(
                cmap="Blues",
                subset=["SHAP Score"]),
            use_container_width=True,
            height=350
        )