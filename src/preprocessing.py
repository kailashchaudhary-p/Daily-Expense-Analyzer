"""Train, validate, and reuse the project’s ML models.

The original notebook trained a balanced random forest for category prediction,
Linear Regression for amount estimation, Isolation Forest for anomaly scoring,
and a Ridge forecaster for monthly spend trends. This module rebuilds the same
logic if the saved model files are missing and keeps the feature order aligned
with the training pipeline.
"""

from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from config import (
    AMOUNT_MODEL_PATH,
    ANOMALY_FEATURES_PATH,
    ANOMALY_MODEL_PATH,
    CATEGORY_MODEL_PATH,
    CLASSIFIER_FEATURES_PATH,
    FORECAST_MODEL_PATH,
    MODELS_DIR,
    REGRESSOR_FEATURES_PATH,
)
from src.data_loader import prepare_expense_dataframe


def _build_category_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create the feature table used by the category classifier."""
    X = df[["Amount", "Mode", "Month", "Day", "Day_type"]].copy()
    X = pd.get_dummies(X, columns=["Mode", "Day_type"], dtype=int)
    y = df["Category"].astype(str)
    return X, y


def _build_amount_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create the regression features for amount prediction."""
    X = df[["Category", "Mode", "Month", "Day", "Day_type"]].copy()
    X = pd.get_dummies(X, columns=["Category", "Mode", "Day_type"], dtype=int)
    y = df["Amount"].astype(float)
    return X, y


def _build_anomaly_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the numeric feature matrix for the anomaly detector."""
    X = df[["Amount", "Month", "Day", "Mode", "Day_type"]].copy()
    X = pd.get_dummies(X, columns=["Mode", "Day_type"], dtype=int)
    return X


def train_category_model(df: pd.DataFrame) -> None:
    """Train the expense category classifier and persist it with the exact feature columns."""
    X, y = _build_category_frame(df)
    if y.value_counts().min() >= 2:
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        warnings.warn(
            "Some category values appear only once in the dataset; using a non-stratified split to avoid training failure.",
            RuntimeWarning,
            stacklevel=2,
        )
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    joblib.dump(clf, CATEGORY_MODEL_PATH)
    joblib.dump(list(X.columns), CLASSIFIER_FEATURES_PATH)


def train_amount_model(df: pd.DataFrame) -> None:
    """Train the amount regressor and store the model alongside its ordered feature list."""
    X, y = _build_amount_frame(df)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, AMOUNT_MODEL_PATH)
    joblib.dump(list(X.columns), REGRESSOR_FEATURES_PATH)


def train_anomaly_model(df: pd.DataFrame) -> None:
    """Train an Isolation Forest and persist it so anomaly scores can be computed later."""
    X = _build_anomaly_frame(df)
    detector = IsolationForest(contamination=0.03, random_state=42, n_estimators=200)
    detector.fit(X)

    joblib.dump(detector, ANOMALY_MODEL_PATH)
    joblib.dump(list(X.columns), ANOMALY_FEATURES_PATH)


def train_forecast_model(df: pd.DataFrame) -> None:
    """Train a lightweight Ridge forecaster on monthly totals for future spend estimation."""
    monthly = (
        df.groupby(pd.Grouper(key="Date", freq="ME"), as_index=False)["Amount"]
        .sum()
        .rename(columns={"Amount": "total_amount"})
    )

    monthly["month_number"] = monthly["Date"].dt.month
    for lag in [1, 2, 3]:
        monthly[f"lag_{lag}"] = monthly["total_amount"].shift(lag)

    if monthly.empty:
        training = monthly.copy()
    else:
        training = monthly.copy()
        for lag in [1, 2, 3]:
            training[f"lag_{lag}"] = training[f"lag_{lag}"].fillna(training["total_amount"].iloc[0])
        training["month_number"] = training["month_number"].fillna(training["month_number"].iloc[0])

    features = [f"lag_{lag}" for lag in [1, 2, 3]] + ["month_number"]
    X = training[features]
    y = training["total_amount"]

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    bundle = {
        "model": model,
        "feature_names": features,
        "history": monthly,
    }
    joblib.dump(bundle, FORECAST_MODEL_PATH)


def ensure_models_exist(df: pd.DataFrame | None = None) -> None:
    """Check for saved models and train them from the dataset when the project has no artifacts yet."""
    if df is None:
        from src.data_loader import load_transactions

        df = prepare_expense_dataframe(load_transactions())

    required_paths = [
        CATEGORY_MODEL_PATH,
        CLASSIFIER_FEATURES_PATH,
        AMOUNT_MODEL_PATH,
        REGRESSOR_FEATURES_PATH,
        ANOMALY_MODEL_PATH,
        ANOMALY_FEATURES_PATH,
        FORECAST_MODEL_PATH,
    ]

    if not all(path.exists() for path in required_paths):
        MODELS_DIR.mkdir(exist_ok=True)

        if not CATEGORY_MODEL_PATH.exists():
            train_category_model(df)
        if not AMOUNT_MODEL_PATH.exists():
            train_amount_model(df)
        if not ANOMALY_MODEL_PATH.exists():
            train_anomaly_model(df)
        if not FORECAST_MODEL_PATH.exists():
            train_forecast_model(df)


def prepare_category_input(data: dict) -> pd.DataFrame:
    """Align one transaction with the category classifier’s stored feature order."""
    record = {
        "Amount": float(data["amount"]),
        "Mode": str(data["mode"]),
        "Month": int(data["month"]),
        "Day": int(data["day"]),
        "Day_type": "Weekend" if data.get("day_type") == "Weekend" else "Weekday",
    }

    frame = pd.DataFrame([record])
    frame = pd.get_dummies(frame, columns=["Mode", "Day_type"], dtype=int)
    feature_columns = joblib.load(CLASSIFIER_FEATURES_PATH)

    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = 0

    return frame[feature_columns]


def prepare_amount_input(data: dict) -> pd.DataFrame:
    """Align one transaction with the amount regressor’s stored feature order."""
    record = {
        "Category": str(data["category"]),
        "Mode": str(data["mode"]),
        "Month": int(data["month"]),
        "Day": int(data["day"]),
        "Day_type": "Weekend" if data.get("day_type") == "Weekend" else "Weekday",
    }

    frame = pd.DataFrame([record])
    frame = pd.get_dummies(frame, columns=["Category", "Mode", "Day_type"], dtype=int)
    feature_columns = joblib.load(REGRESSOR_FEATURES_PATH)

    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = 0

    return frame[feature_columns]


def prepare_anomaly_input(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the anomaly detector feature matrix using the same schema as training."""
    X = df[["Amount", "Month", "Day", "Mode", "Day_type"]].copy()
    X = pd.get_dummies(X, columns=["Mode", "Day_type"], dtype=int)
    feature_columns = joblib.load(ANOMALY_FEATURES_PATH)

    for column in feature_columns:
        if column not in X.columns:
            X[column] = 0

    return X[feature_columns]
