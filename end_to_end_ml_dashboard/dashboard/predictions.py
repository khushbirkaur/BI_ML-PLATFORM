"""
dashboard/predictions.py
========================
Real-time single-row predictions and batch CSV predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.data_manager import (
    get_models, get_clean_df, get_df,
    select_features, predict_single,
    get_numeric_cols,
)


def render():
    st.title("🎯 Predictions")

    models = get_models()
    if not models:
        st.warning("⚠️ Train models first (ML Models page).")
        return

    df           = get_clean_df() or get_df()
    num_cols     = get_numeric_cols(df)
    target       = num_cols[-1]              # last numeric = assumed target
    feature_cols = select_features(df, target)

    tab_single, tab_batch = st.tabs(["🔢 Single prediction", "📋 Batch predictions"])

    # ── Single prediction ─────────────────────────────────────────────────────
    with tab_single:
        st.subheader("Enter feature values")
        model_name = st.selectbox("Model", list(models.keys()), key="pred_model")

        # Build input form — up to 6 features per row
        values = []
        cols_per_row = 3
        rows = [feature_cols[i:i+cols_per_row]
                for i in range(0, len(feature_cols), cols_per_row)]

        for row in rows:
            form_cols = st.columns(len(row))
            for j, fc in enumerate(row):
                default = float(df[fc].median()) if fc in df.columns else 0.0
                v = form_cols[j].number_input(fc, value=round(default, 2), key=f"pred_{fc}")
                values.append(v)

        if st.button("▶ Predict", type="primary"):
            pipe      = models[model_name]["model"]
            pred, prob = predict_single(pipe, values)

            st.markdown("---")
            r1, r2 = st.columns(2)
            r1.metric("Predicted class", pred)
            if prob is not None:
                pct = round(prob * 100, 1)
                r2.metric("Probability (class 1)", f"{pct}%")
                st.progress(prob)

                if prob > 0.6:
                    st.error("🔴 High risk — recommended action: immediate intervention.")
                elif prob > 0.35:
                    st.warning("🟡 Medium risk — monitor closely.")
                else:
                    st.success("🟢 Low risk — customer appears stable.")

    # ── Batch prediction ──────────────────────────────────────────────────────
    with tab_batch:
        st.subheader("Upload a CSV for bulk predictions")
        st.caption(
            f"Your file must contain these columns: "
            f"`{', '.join(feature_cols[:6])}{'…' if len(feature_cols) > 6 else ''}`"
        )
        batch_file = st.file_uploader("Choose CSV", type=["csv"], key="batch_upload")

        if batch_file:
            batch_df = pd.read_csv(batch_file)
            available = [c for c in feature_cols if c in batch_df.columns]

            if not available:
                st.error(
                    f"No matching feature columns found in uploaded file. "
                    f"Expected: {feature_cols}"
                )
                return

            model_name2 = st.selectbox("Model", list(models.keys()), key="batch_model")
            pipe        = models[model_name2]["model"]

            X_batch               = batch_df[available].fillna(0).values
            batch_df["prediction"] = pipe.predict(X_batch)

            if hasattr(pipe["clf"], "predict_proba"):
                batch_df["probability"] = (
                    pipe.predict_proba(X_batch)[:, 1].round(3)
                )

            st.success(f"✅ Predictions generated for {len(batch_df):,} rows.")
            st.dataframe(batch_df.head(50), use_container_width=True)

            st.download_button(
                "⬇️ Download predictions (CSV)",
                batch_df.to_csv(index=False).encode(),
                file_name="predictions.csv",
                mime="text/csv",
            )
