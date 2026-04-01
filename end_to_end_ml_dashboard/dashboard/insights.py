"""
dashboard/insights.py
=====================
Auto-generated insights report page.
"""

import streamlit as st
import pandas as pd
from utils.data_manager import (
    get_clean_df, get_df,
    get_numeric_cols,
    generate_insights,
    PRIORITY_COLOR, CATEGORY_ICON, BORDER_COLOR,
)


def render():
    st.title("💡 Insights Report")

    df = get_clean_df() or get_df()
    if df is None:
        st.warning("⚠️ Please upload a dataset first (Upload Data page).")
        return

    num_cols = get_numeric_cols(df)
    target   = st.selectbox(
        "Target variable (optional — improves driver analysis)",
        ["None"] + num_cols,
    )
    target_col = None if target == "None" else target

    if st.button("▶ Generate insights", type="primary"):
        with st.spinner("Analysing dataset…"):
            insights = generate_insights(df, target_col)

        st.success(f"✅ Found **{len(insights)}** insight(s).")
        st.markdown("---")

        for ins in insights:
            icon   = PRIORITY_COLOR.get(ins["priority"], "⚪")
            cicon  = CATEGORY_ICON.get(ins["category"], "")
            border = BORDER_COLOR.get(ins["priority"], "#6b7280")

            st.markdown(
                f"""
                <div style="
                    border-left: 3px solid {border};
                    background: #1e293b;
                    border-radius: 0 10px 10px 0;
                    padding: 0.8rem 1.1rem;
                    margin-bottom: 0.8rem;
                ">
                    <div style="font-weight:600; color:#f1f5f9; margin-bottom:5px;">
                        {icon} {cicon} {ins['title']}
                    </div>
                    <div style="font-size:0.875rem; color:#94a3b8; line-height:1.65;">
                        {ins['description']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Export ─────────────────────────────────────────────────────────────
        st.markdown("---")
        report_df = pd.DataFrame(insights)
        st.download_button(
            "⬇️ Download insights report (CSV)",
            report_df.to_csv(index=False).encode(),
            file_name="insights_report.csv",
            mime="text/csv",
        )
