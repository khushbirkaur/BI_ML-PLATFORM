"""
dashboard/upload.py
===================
Data upload page. Supports CSV, Excel, or the built-in sample dataset.
"""

import streamlit as st
import pandas as pd
from utils.data_manager import load_uploaded_file as load_file, generate_sample_data, set_df, get_df

def render():

    st.title("📂 Upload Data")

    st.markdown(
        "Upload a **CSV or Excel** file, or load the built-in sample dataset."
    )

    tab_file, tab_sample = st.tabs(
        ["📁 Upload file", "🧪 Sample dataset"]
    )

    # ── Tab 1 : Upload file ───────────────────────────────────────────────
    with tab_file:

        uploaded = st.file_uploader(
            "Choose a file (CSV or Excel, max 200 MB)",
            type=["csv", "xlsx", "xls"],
        )

        if uploaded is not None:

            try:

                df = load_file(uploaded)

                set_df(df)

                st.success(
                    f"✅ **{uploaded.name}** loaded — "
                    f"{len(df):,} rows × {len(df.columns)} columns"
                )

                _show_preview(df)

            except Exception as exc:

                st.error(f"Could not read file: {exc}")

    # ── Tab 2 : Sample dataset ───────────────────────────────────────────
    with tab_sample:

        st.markdown(
            "No dataset yet? Load a synthetic **sales / churn dataset** "
            "to explore every feature of the platform."
        )

        n_rows = st.slider(
            "Number of rows",
            200,
            5000,
            1200,
            step=100,
        )

        if st.button("Load sample dataset", type="primary"):

            df = generate_sample_data(n=n_rows)

            set_df(df)

            st.success(
                f"✅ Sample dataset loaded — "
                f"{len(df):,} rows × {len(df.columns)} columns"
            )

            _show_preview(df)

    # ── Show existing dataset if already loaded ───────────────────────────
    df = get_df()

    if df is not None and not df.empty and uploaded is None:

        st.markdown("---")

        st.subheader("Currently loaded dataset")

        _show_preview(df)


# ── Helper Function ──────────────────────────────────────────────────────
def _show_preview(df: pd.DataFrame):

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing cells", int(df.isnull().sum().sum()))
    c4.metric("Duplicates", int(df.duplicated().sum()))

    st.markdown("### First 10 rows")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    with st.expander("Column data types"):

        dtype_df = (
            df.dtypes.rename("dtype")
            .reset_index()
            .rename(columns={"index": "column"})
        )

        dtype_df["non_null"] = df.notnull().sum().values

        dtype_df["null_%"] = (
            (df.isnull().mean() * 100)
            .round(1)
            .values
        )

        st.dataframe(dtype_df, use_container_width=True)