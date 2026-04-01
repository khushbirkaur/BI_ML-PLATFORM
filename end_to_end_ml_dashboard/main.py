"""
main.py
=======
BI + ML Platform — Streamlit entry point.

Run with:
    streamlit run main.py
"""

import streamlit as st

st.set_page_config(
    page_title="BI + ML Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Sidebar dark theme */
    [data-testid="stSidebar"] {
        background: #0f172a;
    }
    [data-testid="stSidebar"] section {
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] * {
        color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #cbd5e1 !important;
        font-size: 0.9rem;
    }

    /* KPI metric cards */
    [data-testid="metric-container"] {
        background: #1e293b;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }

    /* Page titles */
    h1 { color: #f1f5f9; }
    h2, h3 { color: #cbd5e1; }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='color:#f1f5f9;margin-bottom:0'>📊 BI + ML</h2>"
        "<p style='color:#64748b;font-size:0.75rem;margin-top:4px'>Business Intelligence Platform</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigate",
        options=[
            "🏠  Home",
            "📂  Upload Data",
            "🧹  Data Cleaning",
            "🔍  EDA Analysis",
            "📈  BI Dashboard",
            "🤖  ML Models",
            "🎯  Predictions",
            "📉  Forecasting",
            "💡  Insights Report",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<p style='color:#475569;font-size:0.7rem'>v1.0 · Python + Streamlit<br>"
        "scikit-learn · Plotly · statsmodels</p>",
        unsafe_allow_html=True,
    )

# ── Route to page modules ─────────────────────────────────────────────────────
key = page.split("  ", 1)[-1].strip()   # strip icon prefix

if key == "Home":
    from dashboard.home import render
elif key == "Upload Data":
    from dashboard.upload import render
elif key == "Data Cleaning":
    from dashboard.cleaning import render
elif key == "EDA Analysis":
    from dashboard.eda import render
elif key == "BI Dashboard":
    from dashboard.bi import render
elif key == "ML Models":
    from dashboard.ml import render
elif key == "Predictions":
    from dashboard.predictions import render
elif key == "Forecasting":
    from dashboard.forecasting import render
elif key == "Insights Report":
    from dashboard.insights import render
else:
    from dashboard.home import render

render()
