"""Sidebar layout and navigation for the Streamlit app."""

from __future__ import annotations

import streamlit as st


def render_sidebar() -> str:
    """Render the premium fintech sidebar and the page navigation radio selector."""
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-box">
                <div class="brand-dot"></div>
                <div>
                    <div class="brand-name">Daily Expense</div>
                    <div class="brand-subtitle">AI Finance Copilot</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pages = [
            "Home",
            "Dashboard",
            "Expense Analytics",
            "AI Category Prediction",
            "Expense Amount Prediction",
            "Spending Forecast",
            "Anomaly Detection",
            "Financial Insights",
        ]
        selection = st.radio("Navigation", pages, label_visibility="collapsed")
        st.markdown("---")
        st.caption("Model status")
        st.success("Live ML pipeline connected")

    return selection
