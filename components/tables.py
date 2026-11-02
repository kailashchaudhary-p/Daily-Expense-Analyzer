"""Reusable tables for recent transactions and anomaly results."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_recent_transactions(df: pd.DataFrame, limit: int = 10) -> None:
    """Display a compact recent transaction dataset for the dashboard."""
    if df.empty:
        st.info("No transactions match the selected filters.")
        return

    table = df.sort_values("Date", ascending=False).head(limit).copy()
    table["Date"] = pd.to_datetime(table["Date"]).dt.strftime("%d %b %Y")
    st.dataframe(table[["Date", "Mode", "Category", "Note", "Amount"]], use_container_width=True, hide_index=True)


def render_anomaly_table(df: pd.DataFrame) -> None:
    """Display suspicious transactions and the anomaly score for review."""
    if df.empty:
        st.info("No transactions were flagged for review.")
        return

    anomaly_view = df[["Date", "Category", "Mode", "Amount", "is_anomaly", "anomaly_score"]].copy()
    anomaly_view["Date"] = pd.to_datetime(anomaly_view["Date"]).dt.strftime("%d %b %Y")
    anomaly_view = anomaly_view[anomaly_view["is_anomaly"] == True].sort_values("anomaly_score")

    if anomaly_view.empty:
        st.info("No transaction exceeded the anomaly threshold. The model did not flag any unusual activity.")
        return

    st.dataframe(anomaly_view, use_container_width=True, hide_index=True)
