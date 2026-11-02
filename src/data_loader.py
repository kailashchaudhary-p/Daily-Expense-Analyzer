"""Data ingestion and preparation helpers.

This module keeps the raw CSV loading logic separate from the dashboard and ML code,
which helps the app stay readable and makes the dataset easier to validate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATA_PATH, RAW_DATA_PATH


def load_transactions() -> pd.DataFrame:
    """Load the expense dataset from the project data directory if available."""
    data_path = DATA_PATH if DATA_PATH.exists() else RAW_DATA_PATH

    if not data_path.exists():
        raise FileNotFoundError(
            f"No transaction data was found. Expected {DATA_PATH} or {RAW_DATA_PATH}."
        )

    df = pd.read_csv(data_path)
    return df


def prepare_expense_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset and create the date-derived fields used throughout the app."""
    prepared = df.copy()

    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce", dayfirst=True, format="mixed")
    prepared["Amount"] = pd.to_numeric(prepared["Amount"], errors="coerce")
    prepared["Mode"] = prepared["Mode"].fillna("Unknown").astype(str).str.strip()
    prepared["Category"] = prepared["Category"].fillna("Other").astype(str).str.strip()
    prepared["Income/Expense"] = prepared["Income/Expense"].fillna("Expense").astype(str).str.strip()

    prepared = prepared.dropna(subset=["Date", "Amount"]).copy()
    prepared = prepared[prepared["Income/Expense"].str.lower() == "expense"].copy()

    prepared["Month"] = prepared["Date"].dt.month
    prepared["Day"] = prepared["Date"].dt.day
    prepared["Year"] = prepared["Date"].dt.year
    prepared["Month_Name"] = prepared["Date"].dt.month_name()
    prepared["Day_type"] = np.where(prepared["Date"].dt.dayofweek >= 5, "Weekend", "Weekday")

    return prepared.reset_index(drop=True)
