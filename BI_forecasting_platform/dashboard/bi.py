"""
dashboard/bi.py
===============
Business Intelligence dashboard page.
KPI cards, time-series trends, category breakdowns, and segment share.
"""

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from utils.data_manager import (
    get_clean_df, get_df,
    get_numeric_cols, get_cat_cols, get_date_cols,
)


def render():
    st.title("📈 BI Dashboard")

    df = get_clean_df()

df = get_clean_df()

if df is None:
    df = get_df()

if df is None or df.empty:
    st.warning("⚠️ Please upload a dataset first (Upload Data page).")
    return

    num_cols  = get_numeric_cols(df)
    cat_cols  = get_cat_cols(df)
    date_cols = get_date_cols(df)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.subheader("Key Performance Indicators")

    kpi_cols = num_cols[:4]   # show up to 4 KPIs
    cols = st.columns(len(kpi_cols))
    for i, col in enumerate(kpi_cols):
        total = df[col].sum()
        mean  = df[col].mean()
        label = col.replace("_", " ").title()
        cols[i].metric(label, f"{total:,.0f}", f"Avg {mean:,.1f}")

    st.markdown("---")

    # ── Time-series trend ─────────────────────────────────────────────────────
    if date_cols:
        st.subheader("Time-series trend")
        ts_c1, ts_c2, ts_c3 = st.columns(3)
        date_col = ts_c1.selectbox("Date column", date_cols, key="bi_date")
        val_col  = ts_c2.selectbox("Value column", num_cols, key="bi_val")
        freq     = ts_c3.radio(
            "Frequency", ["D", "W", "MS"],
            horizontal=True,
            format_func=lambda x: {"D": "Daily", "W": "Weekly", "MS": "Monthly"}[x],
            key="bi_freq",
        )

        tmp = df.copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col])
        ts_agg = (
            tmp.groupby(pd.Grouper(key=date_col, freq=freq))[val_col]
               .sum()
               .reset_index()
        )
        fig_ts = px.area(
            ts_agg, x=date_col, y=val_col,
            color_discrete_sequence=["#3b82f6"],
            title=f"{val_col} over time ({freq})",
        )
        fig_ts.update_layout(template="plotly_white")
        st.plotly_chart(fig_ts, use_container_width=True)
        st.markdown("---")

    # ── Category bar chart ────────────────────────────────────────────────────
    if cat_cols:
        st.subheader("Revenue / metric by category")
        cc1, cc2 = st.columns(2)
        grp_col  = cc1.selectbox("Group by",  cat_cols, key="bi_grp")
        met_col  = cc2.selectbox("Metric",    num_cols, key="bi_met")

        grp_df = (
            df.groupby(grp_col)[met_col]
              .sum()
              .reset_index()
              .sort_values(met_col, ascending=False)
        )
        fig_bar = px.bar(
            grp_df, x=grp_col, y=met_col, color=grp_col,
            color_discrete_sequence=px.colors.qualitative.Safe,
            title=f"{met_col} by {grp_col}",
        )
        fig_bar.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("---")

    # ── Donut — segment share ─────────────────────────────────────────────────
    if cat_cols:
        st.subheader("Segment share")
        d1, d2 = st.columns(2)
        seg_col = d1.selectbox("Segment column", cat_cols, key="bi_seg")
        seg_val = d2.selectbox("Value",          num_cols, key="bi_segval")

        pie_df = df.groupby(seg_col)[seg_val].sum().reset_index()
        fig_pie = px.pie(
            pie_df, names=seg_col, values=seg_val,
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Safe,
            title=f"{seg_val} share by {seg_col}",
        )
        fig_pie.update_layout(template="plotly_white")
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.download_button(
        "⬇️ Download dataset (CSV)",
        df.to_csv(index=False).encode(),
        file_name="bi_export.csv",
        mime="text/csv",
    )
