"""
dashboard/forecasting.py
========================
Time-series forecasting page (ARIMA + linear fallback).
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from utils.data_manager import (
    get_clean_df, get_df,
    get_numeric_cols, get_date_cols,
    run_forecast,
)


def render():
    st.title("📉 Forecasting")

    df = get_clean_df() or get_df()
    if df is None:
        st.warning("⚠️ Please upload a dataset first (Upload Data page).")
        return

    date_cols = get_date_cols(df)
    num_cols  = get_numeric_cols(df)

    if not date_cols:
        st.error(
            "No date column detected. "
            "Ensure your dataset has a column containing 'date' in its name."
        )
        return

    # ── Controls ──────────────────────────────────────────────────────────────
    st.subheader("Configuration")
    fc1, fc2, fc3, fc4 = st.columns(4)
    date_col  = fc1.selectbox("Date column",        date_cols,                    key="fc_date")
    value_col = fc2.selectbox("Value to forecast",  num_cols,                     key="fc_val")
    steps     = fc3.slider("Periods ahead",         3, 36, 12,                    key="fc_steps")
    method    = fc4.radio("Method",                 ["auto", "arima", "linear"],  key="fc_method",
                          horizontal=True)

    if st.button("▶ Run forecast", type="primary"):
        with st.spinner("Fitting model and generating forecast…"):
            result = run_forecast(
                df, date_col=date_col, value_col=value_col,
                freq="MS", steps=steps, method=method,
            )

        if "error" in result:
            st.error(result["error"])
            return

        # ── Metrics ───────────────────────────────────────────────────────────
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        fc_vals = result["forecast_values"]
        m1.metric("Next period",          f"{fc_vals[0]:,.0f}")
        m2.metric(f"{steps}-period total", f"{sum(fc_vals):,.0f}")
        m3.metric("MAPE",                 f"{result['mape']}%" if result["mape"] else "N/A")
        m4.metric("Method used",          result["method_used"])

        # ── Forecast chart ────────────────────────────────────────────────────
        fig = go.Figure()

        # Historical line
        fig.add_trace(go.Scatter(
            x=result["historical_dates"],
            y=result["historical_values"],
            name="Historical",
            line=dict(color="#3b82f6", width=2),
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=result["forecast_dates"],
            y=result["forecast_values"],
            name="Forecast",
            line=dict(color="#10b981", width=2, dash="dash"),
        ))

        # 95 % confidence band
        fig.add_trace(go.Scatter(
            x=result["forecast_dates"] + result["forecast_dates"][::-1],
            y=result["upper"] + result["lower"][::-1],
            fill="toself",
            fillcolor="rgba(16,185,129,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
        ))

        fig.update_layout(
            title=f"{value_col} — {steps}-period forecast",
            xaxis_title="Date",
            yaxis_title=value_col,
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", y=1.02, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Download ──────────────────────────────────────────────────────────
        fc_df = pd.DataFrame({
            "date":       result["forecast_dates"],
            "forecast":   result["forecast_values"],
            "lower_95":   result["lower"],
            "upper_95":   result["upper"],
        })
        st.download_button(
            "⬇️ Download forecast (CSV)",
            fc_df.to_csv(index=False).encode(),
            file_name="forecast.csv",
            mime="text/csv",
        )
