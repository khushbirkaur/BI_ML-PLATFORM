"""
dashboard/cleaning.py
=====================
Automated data-cleaning page.
"""

import streamlit as st
import plotly.express as px
from utils.data_manager import (
    get_df, get_clean_df,
    clean_dataframe,
    set_clean_df,
)


def render():
    st.title("🧹 Data Cleaning")

    raw_df = get_df()
    if raw_df is None:
        st.warning("⚠️ Please upload a dataset first (Upload Data page).")
        return

    st.markdown(
        "The cleaning pipeline handles **missing values**, **duplicates**, "
        "**outliers**, **categorical encoding**, and **date feature engineering** automatically."
    )

    if st.button("▶ Run auto-cleaning pipeline", type="primary"):
        with st.spinner("Cleaning data…"):
            clean_df, report = clean_dataframe(raw_df)
        set_clean_df(clean_df)
        st.session_state["clean_report"] = report
        st.success("✅ Cleaning complete — cleaned dataset is now active across all pages.")

    # ── Show results if cleaning has been run ─────────────────────────────────
    report = st.session_state.get("clean_report")
    if report is None:
        st.info("Click the button above to run the cleaning pipeline.")
        return

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Empty rows dropped",    report["empty_rows_dropped"])
    c2.metric("Missing values filled", report["missing_values_filled"])
    c3.metric("Duplicates removed",    report["duplicates_removed"])
    c4.metric("Outliers capped",       report["outliers_capped"])
    c5.metric("Quality score",         f"{report['quality_score']}%")

    st.markdown("---")

    # Missing-values bar chart
    st.subheader("Missing values by column (before cleaning)")
    mv = report.get("missing_by_column", {})
    if mv:
        fig = px.bar(
            x=list(mv.keys()),
            y=list(mv.values()),
            labels={"x": "Column", "y": "Missing count"},
            color_discrete_sequence=["#3b82f6"],
            title="Missing cell counts per column",
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values found in the original dataset.")

    # Encoding & FE report
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Categorical columns encoded")
        encoded = report.get("categorical_encoded", [])
        if encoded:
            for col in encoded:
                st.markdown(f"- `{col}` → `{col}_enc`")
        else:
            st.write("None")

    with col_b:
        st.subheader("Features engineered from dates")
        fe = report.get("features_engineered", [])
        if fe:
            for col in fe:
                st.markdown(f"- `{col}`")
        else:
            st.write("None")

    # Preview of cleaned data
    clean_df = get_clean_df()
    if clean_df is not None:
        st.markdown("---")
        st.subheader("Cleaned dataset preview")
        st.dataframe(clean_df.head(10), use_container_width=True)
        st.download_button(
            "⬇️ Download cleaned data (CSV)",
            clean_df.to_csv(index=False).encode(),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
