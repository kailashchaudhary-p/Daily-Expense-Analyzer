"""Prediction entry points for the product pages.

These functions load the trained models, align user-entered features to the saved
feature order, and return predictions without retraining on every request.
"""

from __future__ import annotations

import joblib
import pandas as pd

from config import (
    AMOUNT_MODEL_PATH,
    ANOMALY_MODEL_PATH,
    CATEGORY_MODEL_PATH,
    FORECAST_MODEL_PATH,
)
from src.preprocessing import (
    ensure_models_exist,
    prepare_amount_input,
    prepare_anomaly_input,
    prepare_category_input,
)


def load_category_model():
    """Load the saved category classifier."""
    return joblib.load(CATEGORY_MODEL_PATH)


def load_amount_model():
    """Load the saved amount regressor."""
    return joblib.load(AMOUNT_MODEL_PATH)


def load_anomaly_model():
    """Load the saved anomaly detector."""
    return joblib.load(ANOMALY_MODEL_PATH)


def load_forecast_bundle():
    """Load the saved monthly forecast bundle."""
    return joblib.load(FORECAST_MODEL_PATH)


def predict_category(data: dict) -> str:
    """Predict the expense category for a user-entered transaction."""
    ensure_models_exist()
    model = load_category_model()
    input_frame = prepare_category_input(data)
    prediction = model.predict(input_frame)[0]
    return str(prediction)


def predict_amount(data: dict) -> float:
    """Predict the transaction amount using the saved regression model."""
    ensure_models_exist()
    model = load_amount_model()
    input_frame = prepare_amount_input(data)
    prediction = model.predict(input_frame)[0]
    return float(prediction)


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Score each transaction with the anomaly detector and flag unusual spending."""
    ensure_models_exist()
    detector = load_anomaly_model()
    anomaly_input = prepare_anomaly_input(df)
    labels = detector.predict(anomaly_input)
    scores = detector.score_samples(anomaly_input)

    result = df.copy()
    result["is_anomaly"] = labels == -1
    result["anomaly_score"] = scores
    return result.sort_values("anomaly_score").reset_index(drop=True)


def generate_forecast(df: pd.DataFrame, periods: int = 6) -> pd.DataFrame:
    """Forecast future expense totals from the monthly trend model."""
    ensure_models_exist()
    bundle = load_forecast_bundle()
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    monthly = (
        df.groupby(pd.Grouper(key="Date", freq="ME"), as_index=False)["Amount"]
        .sum()
        .rename(columns={"Date": "Month", "Amount": "Total"})
    )

    history = monthly["Total"].tolist()
    if not history:
        return pd.DataFrame(columns=["Month", "Total"])

    if len(history) < 3:
        last_value = history[-1]
        history = history + [last_value] * (3 - len(history))

    future_dates = pd.date_range(
        start=monthly["Month"].iloc[-1] + pd.DateOffset(months=1),
        periods=periods,
        freq="ME",
    )

    window = history[-3:]
    results = []
    for idx in range(periods):
        row = {"lag_1": window[-1], "lag_2": window[-2], "lag_3": window[-3], "month_number": future_dates[idx].month}
        prediction = model.predict(pd.DataFrame([row], columns=feature_names))[0]
        results.append({"Month": future_dates[idx], "Total": max(prediction, 0)})
        window.append(max(prediction, 0))

    return pd.DataFrame(results)
