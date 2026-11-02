from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DATA_PATH = DATA_DIR / "Transactions.csv"
RAW_DATA_PATH = ROOT_DIR / "Transactions.csv"
MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "assets"

CATEGORY_MODEL_PATH = MODELS_DIR / "category_classifier.pkl"
CLASSIFIER_FEATURES_PATH = MODELS_DIR / "classifier_features.pkl"
AMOUNT_MODEL_PATH = MODELS_DIR / "amount_regressor.pkl"
REGRESSOR_FEATURES_PATH = MODELS_DIR / "regressor_features.pkl"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_detector.pkl"
ANOMALY_FEATURES_PATH = MODELS_DIR / "anomaly_features.pkl"
FORECAST_MODEL_PATH = MODELS_DIR / "forecasting_model.pkl"
