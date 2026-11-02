"""Daily Expense Analyzer - premium Streamlit fintech dashboard.

The app is organized into small modules so the UI, analytics logic, and machine
learning code stay maintainable and easy to follow.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components.cards import render_kpi_card
from components.charts import (
    category_breakdown_chart,
    monthly_spending_chart,
    top_payment_mode_chart,
    weekday_weekend_chart,
)
from components.sidebar import render_sidebar
from components.tables import render_anomaly_table, render_recent_transactions
from src.analytics import (
    compute_category_breakdown,
    compute_insights,
    compute_kpis,
    compute_monthly_spending,
    compute_recent_transactions,
    compute_weekday_weekend,
)
from src.data_loader import load_transactions, prepare_expense_dataframe
from src.forecasting import forecast_next_months
from src.predictions import detect_anomalies, predict_amount, predict_category
from src.preprocessing import ensure_models_exist


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load the cleaned expense data from disk and cache it for the app lifetime."""
    df = load_transactions()
    return prepare_expense_dataframe(df)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar date and category filters so all dashboard sections stay consistent."""
    if df.empty:
        return df

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    with st.sidebar:
        st.markdown("### Filters")
        start_date, end_date = st.date_input(
            "Date range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date,
        )

        categories = sorted(df["Category"].unique())
        selected_categories = st.multiselect("Category focus", categories, default=categories)

    if isinstance(start_date, tuple):
        start_date, end_date = start_date

    filtered = df[(df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)]

    if selected_categories:
        filtered = filtered[filtered["Category"].isin(selected_categories)]

    return filtered.reset_index(drop=True)


def render_home_page() -> None:
    """Build the premium landing page and quick overview for the project."""
    st.markdown(
        """
        <div class="hero-box">
            <div class="main-title">Daily Expense Analyzer</div>
            <div class="subtle-text">AI-powered spend intelligence for smarter financial decisions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-panel">
            <h3>Futuristic finance cockpit</h3>
            <p class="subtle-text">
                This tool combines transaction analytics, category prediction, amount estimation,
                anomaly detection, and forecasting to turn raw expenses into actionable business insight.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    feature_cols = st.columns(4)
    features = [
        ("Live dashboard", "Real transaction KPIs and charts"),
        ("AI category prediction", "Predict expense category with the saved Random Forest"),
        ("Expense regression", "Estimate amount from transaction context"),
        ("Anomaly scoring", "Detect unusually high or irregular expenses"),
    ]
    for idx, (title, text) in enumerate(features):
        with feature_cols[idx]:
            st.markdown(
                f"""
                <div class="glass-panel">
                    <div class="feature-pill">{title}</div>
                    <div class="subtle-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_dashboard_page(df: pd.DataFrame) -> None:
    """Render the KPI summary and interactive charts for the main dashboard."""
    st.header("Executive dashboard")
    kpis = compute_kpis(df)

    cols = st.columns(5)
    values = [
        ("Total Expenses", f"₹{kpis['total_expenses']:,.2f}", "Total spend", "purple"),
        ("Average Transaction", f"₹{kpis['average_transaction']:,.2f}", "Mean expense", "blue"),
        ("Highest Expense", f"₹{kpis['highest_expense']:,.2f}", "Maximum single transaction", "cyan"),
        ("Total Transactions", f"{kpis['total_transactions']}", "Expense entries", "green"),
        ("Most Used Payment Mode", str(kpis['most_used_payment_mode']), "Top channel", "orange"),
    ]
    for col, (title, value, delta, accent) in zip(cols, values):
        with col:
            render_kpi_card(title, value, delta, accent)

    monthly = compute_monthly_spending(df)
    breakdown = compute_category_breakdown(df)
    weekday = compute_weekday_weekend(df)
    recent = compute_recent_transactions(df, limit=8)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.subheader("Monthly spending")
        st.plotly_chart(monthly_spending_chart(df), use_container_width=True)
    with chart_cols[1]:
        st.subheader("Category-wise spending")
        st.plotly_chart(category_breakdown_chart(df), use_container_width=True)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.subheader("Weekday vs weekend")
        st.plotly_chart(weekday_weekend_chart(df), use_container_width=True)
    with chart_cols[1]:
        st.subheader("Payment mode totals")
        st.plotly_chart(top_payment_mode_chart(df), use_container_width=True)

    st.subheader("Recent transactions")
    render_recent_transactions(recent, limit=8)


def render_analytics_page(df: pd.DataFrame) -> None:
    """Render deeper analytics for daily, weekly, and monthly comparison views."""
    st.header("Expense analytics")
    category_filter = st.selectbox("Focus category", ["All"] + sorted(df["Category"].unique().tolist()))

    if category_filter != "All":
        filtered = df[df["Category"] == category_filter].copy()
    else:
        filtered = df.copy()

    st.subheader("Monthly trend")
    st.plotly_chart(monthly_spending_chart(filtered), use_container_width=True)

    st.subheader("Category breakdown")
    st.plotly_chart(category_breakdown_chart(filtered), use_container_width=True)

    st.subheader("Payment mode analysis")
    st.plotly_chart(top_payment_mode_chart(filtered), use_container_width=True)

    st.subheader("Weekday vs weekend")
    st.plotly_chart(weekday_weekend_chart(filtered), use_container_width=True)


def render_category_prediction_page(df: pd.DataFrame) -> None:
    """Allow the user to enter a transaction and get a category prediction from the saved model."""
    st.header("AI category prediction")
    with st.form("category_prediction_form"):
        amount = st.number_input("Amount", min_value=1.0, value=250.0, step=10.0)
        mode = st.selectbox("Payment mode", sorted(df["Mode"].unique().tolist()))
        day = st.slider("Day of month", 1, 31, 15)
        month = st.slider("Month", 1, 12, 6)
        day_type = st.selectbox("Day type", ["Weekday", "Weekend"])
        submitted = st.form_submit_button("Predict category")

    if submitted:
        payload = {"amount": amount, "mode": mode, "day": day, "month": month, "day_type": day_type}
        predicted = predict_category(payload)
        st.success(f"Predicted category: {predicted}")
        st.json({
            "amount": amount,
            "payment_mode": mode,
            "day": day,
            "month": month,
            "day_type": day_type,
            "predicted_category": predicted,
        })


def render_amount_prediction_page(df: pd.DataFrame) -> None:
    """Predict transaction amount from category, mode, and date context."""
    st.header("Expense amount prediction")
    with st.form("amount_prediction_form"):
        category = st.selectbox("Category", sorted(df["Category"].unique().tolist()))
        mode = st.selectbox("Payment mode", sorted(df["Mode"].unique().tolist()))
        day = st.slider("Day of month", 1, 31, 15)
        month = st.slider("Month", 1, 12, 6)
        day_type = st.selectbox("Day type", ["Weekday", "Weekend"])
        submitted = st.form_submit_button("Predict amount")

    if submitted:
        payload = {"category": category, "mode": mode, "day": day, "month": month, "day_type": day_type}
        prediction = predict_amount(payload)
        st.success(f"Estimated amount: ₹{prediction:,.2f}")
        st.json({
            "category": category,
            "payment_mode": mode,
            "day": day,
            "month": month,
            "day_type": day_type,
            "estimated_amount": round(prediction, 2),
        })


def render_forecast_page(df: pd.DataFrame) -> None:
    """Show the historical spend trend and forecasted future totals."""
    st.header("Spending forecast")
    forecast_months = st.slider("Forecast horizon (months)", 1, 12, 6)

    historical = compute_monthly_spending(df)
    future = forecast_next_months(df, periods=forecast_months)

    historical_chart = historical.copy()
    historical_chart["Month"] = pd.to_datetime(historical_chart["Month"]).dt.strftime("%b %Y")
    future_chart = future.copy()
    future_chart["Month"] = pd.to_datetime(future_chart["Month"]).dt.strftime("%b %Y")

    chart_df = pd.concat(
        [
            historical_chart.assign(Type="Historical"),
            future_chart.assign(Type="Forecast"),
        ],
        ignore_index=True,
    )

    fig = px.line(chart_df, x="Month", y="Total", color="Type", markers=True, template="plotly_dark")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast horizon")
    st.dataframe(future.assign(Month=lambda x: x["Month"].dt.strftime("%b %Y")), use_container_width=True, hide_index=True)


def render_anomaly_page(df: pd.DataFrame) -> None:
    """Display the anomaly score, unusual transactions, and review guidance."""
    st.header("Anomaly detection")
    anomalies = detect_anomalies(df)
    flagged = anomalies[anomalies["is_anomaly"]].copy()

    st.write(
        "This model highlights unusual spend patterns and should be treated as a review signal rather than automatic fraud detection."
    )

    st.metric("Flagged transactions", len(flagged))
    render_anomaly_table(anomalies)


def render_insights_page(df: pd.DataFrame) -> None:
    """Display insights grounded in the actual expense data."""
    st.header("Financial insights")
    insights = compute_insights(df)

    for insight in insights:
        st.markdown(f"- {insight}")

    category_total = compute_category_breakdown(df).head(5)
    st.subheader("Top spending categories")
    st.dataframe(category_total, use_container_width=True, hide_index=True)

    monthly = compute_monthly_spending(df)
    if not monthly.empty:
        st.subheader("Latest monthly totals")
        st.dataframe(monthly.tail(6), use_container_width=True, hide_index=True)


def main() -> None:
    """Entry point for the app and page dispatch."""
    st.set_page_config(page_title="Daily Expense Analyzer", layout="wide", initial_sidebar_state="expanded")

    with open("assets/styles.css", "r", encoding="utf-8") as style_file:
        st.markdown(f"<style>{style_file.read()}</style>", unsafe_allow_html=True)

    page = render_sidebar()
    dataset = load_dataset()
    ensure_models_exist(dataset)
    filtered = apply_filters(dataset)

    if page == "Home":
        render_home_page()
    elif page == "Dashboard":
        render_dashboard_page(filtered)
    elif page == "Expense Analytics":
        render_analytics_page(filtered)
    elif page == "AI Category Prediction":
        render_category_prediction_page(filtered)
    elif page == "Expense Amount Prediction":
        render_amount_prediction_page(filtered)
    elif page == "Spending Forecast":
        render_forecast_page(filtered)
    elif page == "Anomaly Detection":
        render_anomaly_page(filtered)
    elif page == "Financial Insights":
        render_insights_page(filtered)


if __name__ == "__main__":
    main()
