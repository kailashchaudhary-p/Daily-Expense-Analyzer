"""Reusable chart-building helpers for the dashboard and analytics pages."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def monthly_spending_chart(df: pd.DataFrame):
    """Render a trend line for monthly spend totals."""
    monthly = df.groupby(pd.Grouper(key="Date", freq="ME"), as_index=False)["Amount"].sum()
    monthly["Month"] = monthly["Date"].dt.strftime("%b %Y")
    fig = px.line(monthly, x="Month", y="Amount", markers=True, template="plotly_dark")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    return fig


def category_breakdown_chart(df: pd.DataFrame):
    """Render a horizontal bar chart of category totals."""
    breakdown = df.groupby("Category")["Amount"].sum().sort_values(ascending=False).reset_index()
    breakdown.columns = ["Category", "Total"]
    fig = px.bar(
        breakdown,
        x="Total",
        y="Category",
        orientation="h",
        template="plotly_dark",
        color="Total",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360)
    return fig


def weekday_weekend_chart(df: pd.DataFrame):
    """Render the weekday vs weekend comparison chart."""
    comparison = df.groupby("Day_type")["Amount"].sum().reset_index(name="Total")
    fig = px.pie(comparison, names="Day_type", values="Total", template="plotly_dark")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
    return fig


def top_payment_mode_chart(df: pd.DataFrame):
    """Render a summary of payment mode totals."""
    mode_totals = df.groupby("Mode")["Amount"].sum().sort_values(ascending=False).reset_index()
    mode_totals.columns = ["Mode", "Total"]
    fig = px.bar(mode_totals, x="Mode", y="Total", template="plotly_dark")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    return fig
