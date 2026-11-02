"""Reusable KPI card component for the dashboard view."""

from __future__ import annotations

import streamlit as st


def render_kpi_card(title: str, value: str, delta: str | None = None, accent: str = "blue") -> None:
    """Render a premium KPI card with a subtle glow and compact typography."""
    accent_colors = {
        "blue": "#6ea8fe",
        "purple": "#7c5cff",
        "cyan": "#6ae3ff",
        "green": "#63f2b1",
        "orange": "#ffb86c",
    }
    color = accent_colors.get(accent, "#6ea8fe")

    st.markdown(
        f"""
        <div class="kpi-card" style="border-color: {color};">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta">{delta or 'Updated from live data'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
