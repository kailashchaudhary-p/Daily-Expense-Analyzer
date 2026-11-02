"""Dashboard metrics and analytical summaries used by the Streamlit pages."""

from __future__ import annotations

import pandas as pd


def compute_kpis(df: pd.DataFrame) -> dict:
    """Return the main transaction metrics used in the dashboard cards."""
    total_expenses = float(df["Amount"].sum())
    avg_amount = float(df["Amount"].mean())
    highest_expense = float(df["Amount"].max())
    total_transactions = int(len(df))

    most_used_mode = (
        df.groupby("Mode")["Amount"].sum().sort_values(ascending=False).head(1).index[0]
        if not df.empty
        else "N/A"
    )
    highest_category = (
        df.groupby("Category")["Amount"].sum().sort_values(ascending=False).head(1)
        if not df.empty
        else pd.Series(dtype=float)
    )
    category_name = highest_category.index[0] if not highest_category.empty else "N/A"

    return {
        "total_expenses": total_expenses,
        "average_transaction": avg_amount,
        "highest_expense": highest_expense,
        "total_transactions": total_transactions,
        "most_used_payment_mode": most_used_mode,
        "highest_spending_category": category_name,
    }


def compute_monthly_spending(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate spending by month for trend charts and analysis."""
    return (
        df.groupby(pd.Grouper(key="Date", freq="ME"), as_index=False)["Amount"]
        .sum()
        .rename(columns={"Date": "Month", "Amount": "Total"})
    )


def compute_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Return category totals for the spending breakdown chart."""
    return (
        df.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index(name="Total")
    )


def compute_weekday_weekend(df: pd.DataFrame) -> pd.DataFrame:
    """Return weekday and weekend totals for the comparison chart."""
    return (
        df.groupby("Day_type")["Amount"]
        .sum()
        .reset_index(name="Total")
        .rename(columns={"Day_type": "Day Type"})
    )


def compute_recent_transactions(df: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    """Show the most recent transactions in the dashboard table."""
    recent = df.sort_values("Date", ascending=False).head(limit).copy()
    return recent[["Date", "Mode", "Category", "Note", "Amount"]].reset_index(drop=True)


def compute_insights(df: pd.DataFrame) -> list[str]:
    """Build explanation-style spending insights from the actual dataset."""
    insights: list[str] = []

    if df.empty:
        return ["No transaction data is available for analysis."]

    monthly = compute_monthly_spending(df)
    if len(monthly) >= 2:
        current = monthly["Total"].iloc[-1]
        previous = monthly["Total"].iloc[-2]
        delta = ((current - previous) / previous * 100) if previous else 0
        direction = "up" if delta >= 0 else "down"
        insights.append(
            f"Monthly spend is {direction} by {abs(delta):.1f}% compared with the previous month."
        )

    category_totals = compute_category_breakdown(df)
    if not category_totals.empty:
        top_category = category_totals.iloc[0]
        insights.append(
            f"The most expensive category is {top_category['Category']} with ₹{top_category['Total']:.2f} in total spending."
        )

    weekday_weekend = compute_weekday_weekend(df)
    if len(weekday_weekend) == 2:
        weekend_value = weekday_weekend[weekday_weekend["Day Type"] == "Weekend"]["Total"].sum()
        weekday_value = weekday_weekend[weekday_weekend["Day Type"] == "Weekday"]["Total"].sum()
        if weekday_value > 0:
            ratio = weekend_value / weekday_value
            insights.append(
                f"Weekend spending is {ratio:.2f}x the weekday amount, highlighting a strong end-of-week spend pattern."
            )

    abnormal = df[df["Amount"] > df["Amount"].quantile(0.95)]
    if not abnormal.empty:
        insights.append(
            f"There are {len(abnormal)} unusually high transactions above the 95th percentile, which may deserve a closer review."
        )

    mode_totals = df.groupby("Mode")["Amount"].sum().sort_values(ascending=False)
    if not mode_totals.empty:
        top_mode = mode_totals.index[0]
        insights.append(f"The most active payment channel is {top_mode}, which accounts for the largest share of spending volume.")

    return insights
