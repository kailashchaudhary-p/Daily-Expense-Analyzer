"""Forecasting helpers for the spending forecast page.

This wrapper keeps the forecasting logic separated from the Streamlit UI and makes
it easier to reuse the trained model across the dashboard and prediction screens.
"""

from __future__ import annotations

import pandas as pd

from src.predictions import generate_forecast


def forecast_next_months(df: pd.DataFrame, periods: int = 6) -> pd.DataFrame:
    """Return monthly spending estimates for the future period requested by the user."""
    return generate_forecast(df, periods=periods)
