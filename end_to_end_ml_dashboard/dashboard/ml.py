"""
dashboard/ml.py
===============
Machine Learning Models page.
Supports classification, regression, and K-Means clustering.
"""

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from utils.data_manager import (
    get_clean_df, get_df,
    get_numeric_cols,
    select_features,
    train_all_classifiers,
    train_regressor,
    run_kmeans,
    get_best_model_name,
    set_models,
)


def render():
    st.title("🤖 ML Models")

    df = get_clean_df() or get_df()
    if df is None:
        st.warning("⚠️ Please upload a dataset first (Upload Data page).")
        return

    num_cols = get_numeric_cols(df)

    # ── Configuration ─────────────────────────────────────────────────────────
    st.subheader("Configuration")
    cfg1, cfg2 = st.columns(2)
    target = cfg1.selectbox("Target column", num_cols, help="Column to predict.")
    task   = cfg2.radio(
        "Task type",
        ["Classification", "Regression", "Clustering"],
        horizontal=True,
    )

    feature_cols = select_features(df, target)
    st.caption(f"Auto-selected {len(feature_cols)} numeric feature(s): "
               f"{', '.join(feature_cols[:8])}{'…' if len(feature_cols) > 8 else ''}")

    k = 4
    if task == "Clustering":
        k = st.slider("Number of clusters (k)", 2, 10, 4)

    # ── Train ─────────────────────────────────────────────────────────────────
    if st.button("▶ Train models", type="primary"):
        with st.spinner("Training… this may take a few seconds."):
            try:
                if task == "Classification":
                    results = train_all_classifiers(df, feature_cols, target)
                    set_models(results)
                    _show_classifier_results(results, feature_cols)

                elif task == "Regression":
                    result = train_regressor(df, feature_cols, target)
                    st.success(
                        f"✅ Linear Regression trained — "
                        f"**R² = {result['r2']}%** | RMSE = {result['rmse']:,.2f}"
                    )

                else:  # Clustering
                    clustered = run_kmeans(df, feature_cols, k)
                    x_c = feature_cols[0]
                    y_c = feature_cols[1] if len(feature_cols) > 1 else feature_cols[0]
                    fig = px.scatter(
                        clustered, x=x_c, y=y_c,
                        color=clustered["cluster"].dropna().astype(int).astype(str),
                        color_discrete_sequence=px.colors.qualitative.Safe,
                        title=f"K-Means clustering (k={k})",
                        labels={"color": "Cluster"},
                    )
                    fig.update_layout(template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    st.success(f"✅ K-Means clustering complete — {k} clusters identified.")

            except Exception as exc:
                st.error(f"Training failed: {exc}")


# ── Classifier results helper ─────────────────────────────────────────────────
def _show_classifier_results(results: dict, feature_cols: list[str]):
    st.success("✅ All classifiers trained.")

    # Comparison table
    st.subheader("Model comparison")
    rows = [
        {
            "Model":       name,
            "Accuracy %":  r["accuracy"],
            "F1 %":        r["f1"],
            "AUC %":       r["auc"] if r["auc"] else "—",
        }
        for name, r in results.items()
    ]
    st.dataframe(
        pd.DataFrame(rows).sort_values("Accuracy %", ascending=False),
        use_container_width=True,
    )

    best = get_best_model_name(results)
    st.info(f"🏆 Best model: **{best}** ({results[best]['accuracy']}% accuracy)")

    # Feature importance
    fi = results[best]["feature_importance"]
    if fi is not None:
        st.subheader(f"Feature importance — {best}")
        fig_fi = px.bar(
            fi.reset_index(),
            x="index",
            y=fi.name if fi.name else fi.values,
            labels={"index": "Feature", fi.name or 0: "Importance"},
            color_discrete_sequence=["#3b82f6"],
            title="Feature importance scores",
        )
        fig_fi.update_layout(template="plotly_white")
        st.plotly_chart(fig_fi, use_container_width=True)

    # Confusion matrix
    st.subheader(f"Confusion matrix — {best}")
    cm = results[best]["cm"]
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "Actual"},
        title="Confusion matrix",
        aspect="auto",
    )
    fig_cm.update_layout(template="plotly_white")
    st.plotly_chart(fig_cm, use_container_width=True)
