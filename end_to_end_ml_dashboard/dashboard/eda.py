"""
dashboard/eda.py
================
Exploratory Data Analysis page.
"""

import streamlit as st
import plotly.express as px
import numpy as np

from utils.data_manager import (
    get_clean_df,
    get_df,
    get_numeric_cols,
    get_cat_cols,
    get_summary_stats,
    get_correlation_matrix,
)


def render():
    st.title("🔍 EDA Analysis")

    # Get cleaned dataset first
    df = get_clean_df()

    # If cleaned dataset doesn't exist, use original dataset
    if df is None or df.empty:
        df = get_df()

    # If still no dataset
    if df is None or df.empty:
        st.warning("⚠️ Please upload a dataset first (Upload Data page).")
        st.stop()

    # Get column types
    num_cols = get_numeric_cols(df)
    cat_cols = get_cat_cols(df)

    # ── Summary statistics ────────────────────────────────────────────────────
    with st.expander("📋 Summary statistics", expanded=False):
        st.dataframe(get_summary_stats(df), use_container_width=True)

    st.markdown("---")

    # ── Distribution ──────────────────────────────────────────────────────────
    st.subheader("Distribution")

    if len(num_cols) > 0:

        dist_col = st.selectbox("Select column", num_cols, key="eda_dist")

        fig_dist = px.histogram(
            df,
            x=dist_col,
            nbins=40,
            marginal="box",
            color_discrete_sequence=["#3b82f6"],
            title=f"Distribution of {dist_col}",
        )

        fig_dist.update_layout(template="plotly_white")

        st.plotly_chart(fig_dist, use_container_width=True)

    else:
        st.info("No numeric columns available for distribution plot.")

    st.markdown("---")

    # ── Scatter plot ──────────────────────────────────────────────────────────
    st.subheader("Scatter plot")

    if len(num_cols) >= 2:

        sc1, sc2, sc3 = st.columns(3)

        x_col = sc1.selectbox(
            "X axis",
            num_cols,
            index=0,
            key="eda_x",
        )

        y_col = sc2.selectbox(
            "Y axis",
            num_cols,
            index=min(1, len(num_cols) - 1),
            key="eda_y",
        )

        color_col = sc3.selectbox(
            "Colour by",
            ["None"] + cat_cols,
            key="eda_c",
        )

        fig_sc = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=None if color_col == "None" else color_col,
            opacity=0.55,
            color_discrete_sequence=px.colors.qualitative.Safe,
            title=f"{x_col} vs {y_col}",
        )

        fig_sc.update_layout(template="plotly_white")

        st.plotly_chart(fig_sc, use_container_width=True)

    else:
        st.info("Need at least two numeric columns for scatter plot.")

    st.markdown("---")

    # ── Box plots by category ─────────────────────────────────────────────────
    if len(cat_cols) > 0 and len(num_cols) > 0:

        st.subheader("Box plot by category")

        bx_val = st.selectbox("Value column", num_cols, key="eda_bx_val")

        bx_cat = st.selectbox("Group by", cat_cols, key="eda_bx_cat")

        fig_bx = px.box(
            df,
            x=bx_cat,
            y=bx_val,
            color=bx_cat,
            color_discrete_sequence=px.colors.qualitative.Safe,
            title=f"{bx_val} by {bx_cat}",
        )

        fig_bx.update_layout(template="plotly_white", showlegend=False)

        st.plotly_chart(fig_bx, use_container_width=True)

        st.markdown("---")

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.subheader("Correlation heatmap")

    corr = get_correlation_matrix(df)

    if corr is not None and not corr.empty:

        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="Blues",
            title="Pearson correlation matrix",
            aspect="auto",
        )

        fig_corr.update_layout(template="plotly_white")

        st.plotly_chart(fig_corr, use_container_width=True)

    else:
        st.info("Not enough numeric columns to compute correlation.")

    # ── Top correlations table ────────────────────────────────────────────────
    if corr is not None and not corr.empty:

        st.subheader("Top correlated pairs")

        pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .reset_index()
        )

        pairs.columns = ["Feature A", "Feature B", "Correlation"]

        pairs["abs"] = pairs["Correlation"].abs()

        top_pairs = (
            pairs.sort_values("abs", ascending=False)
            .drop(columns="abs")
            .head(10)
        )

        st.dataframe(top_pairs, use_container_width=True)