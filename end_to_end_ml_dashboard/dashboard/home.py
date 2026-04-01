"""
dashboard/home.py
=================
Landing page showing platform overview and current session stats.
"""

import streamlit as st
from utils.data_manager import get_df, get_clean_df, get_models


def render():
    st.title("📊 BI + ML Platform")
    st.markdown(
        "Welcome! Upload any **CSV or Excel** dataset and the platform will "
        "clean it, analyse it, train machine-learning models, forecast trends, "
        "and surface actionable business insights — all in one place."
    )

    st.markdown("---")

    # ── Session-state KPIs ────────────────────────────────────────────────────
    df     = get_df()
    cdf    = get_clean_df()
    models = get_models() or {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Dataset rows",    f"{len(df):,}"        if df     is not None else "—")
    c2.metric("🧱 Columns",         len(df.columns)       if df     is not None else "—")
    c3.metric("🤖 Models trained",  len(models)           if models else "—")

    if models:
        best = max(models, key=lambda k: models[k]["accuracy"])
        c4.metric("🏆 Best accuracy", f"{models[best]['accuracy']}%", help=best)
    else:
        c4.metric("🏆 Best accuracy", "—")

    st.markdown("---")

    # ── Workflow steps ────────────────────────────────────────────────────────
    st.subheader("How to use this platform")

    steps = [
        ("📂", "Upload Data",     "Load a CSV/Excel file or use the built-in sample dataset."),
        ("🧹", "Data Cleaning",   "Auto-fix missing values, duplicates, outliers & encode categories."),
        ("🔍", "EDA Analysis",    "Explore distributions, scatter plots, and correlation heatmaps."),
        ("📈", "BI Dashboard",    "View KPI cards, time-series trends, and segment breakdowns."),
        ("🤖", "ML Models",       "Train Random Forest, Gradient Boosting, Logistic Regression & more."),
        ("🎯", "Predictions",     "Get real-time predictions for individual rows or bulk CSV files."),
        ("📉", "Forecasting",     "Generate ARIMA / linear forecasts with confidence intervals."),
        ("💡", "Insights Report", "Download auto-generated findings and business recommendations."),
    ]

    # Display in a 4-column grid
    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i % 4]:
            st.markdown(f"### {icon} {title}")
            st.caption(desc)
            st.markdown("")   # spacing

    st.markdown("---")
    st.info("👉 **Start here:** go to **Upload Data** in the sidebar to load your dataset.")
