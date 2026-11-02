# Daily Expense Analyzer

A premium fintech-style Streamlit application for expense analysis, predictive modeling, and anomaly detection built from a real transaction dataset.

## Overview

This project turns a daily expense CSV into an interactive AI finance dashboard. It combines:

- automated spending analysis
- transaction trend monitoring
- category prediction using a Random Forest classifier
- amount prediction using a regression model
- anomaly detection with Isolation Forest
- month-ahead spending forecasts using a statistical regressor

The interface is designed to feel like a modern SaaS product while staying fully within Streamlit.

## Project structure

- app.py — main Streamlit application entry point
- config.py — shared paths and model configuration
- src/
  - data_loader.py — CSV loading and date cleaning
  - preprocessing.py — ML training and feature alignment
  - analytics.py — KPI and dashboard calculations
  - predictions.py — model predictions and anomaly scoring
  - forecasting.py — forecast wrappers
- components/
  - sidebar.py — navigation layout
  - cards.py — KPI card component
  - charts.py — Plotly visualizations
  - tables.py — tables for transactions and anomaly review
- assets/styles.css — dark fintech styling
- models/ — trained pickle artifacts generated from the dataset
- data/Transactions.csv — workspace copy of the raw dataset
- Transactions.csv — original data file

## Installation

1. Open a terminal in the project folder.
2. Create and activate a virtual environment if you prefer.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

From the project root:

```bash
streamlit run app.py
```

Or in VS Code:

1. Open the project folder.
2. Use the integrated terminal.
3. Run `streamlit run app.py`.
4. Open the local URL shown in the terminal.

## Included ML models

The project uses a real dataset and trains these models when the saved artifacts are missing:

- Category classifier: Random Forest with balanced class weighting
- Amount regressor: Linear Regression
- Anomaly detector: Isolation Forest
- Forecast model: Ridge regression on monthly spend totals

All models are saved under the `models/` directory in `.pkl` format.

## Features

- premium dark fintech UI
- interactive monthly and category charts
- real KPI metrics from transaction data
- date-range and category filters
- AI category prediction
- expense amount prediction
- anomaly detection review screen
- future spending forecast
- financial insight summary based on actual spending patterns

## Screenshots

Use the app locally and capture screenshots for GitHub or portfolio use.

## Limitations

- The forecasting model is intentionally lightweight and based on historical aggregated totals.
- Model accuracy depends on the quality and size of the transaction dataset.
- The app is built for a full expense dataset rather than a real-time financial feed.

## Future improvements

- add user authentication and profiles
- connect to a database or API
- support more advanced forecasting algorithms
- add budget health alerts and recommendation engines
- create a deployment pipeline for cloud hosting

## Notes

This project is designed for portfolio, learning, and interview-ready demonstration work. The models are trained from the real project dataset and saved locally so the app can run without a retraining step on every refresh.
