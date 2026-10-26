"""Common paths and settings used by the training scripts, the API and the DAGs."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data" / "raw"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT_DIR / "models"))

# if no tracking server is given we just log to a local sqlite file
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR / 'mlflow.db'}")

FLIGHT_MODEL_PATH = MODELS_DIR / "flight_price_model.joblib"
GENDER_MODEL_PATH = MODELS_DIR / "gender_model.joblib"
RECOMMENDER_PATH = MODELS_DIR / "hotel_recommender.joblib"

RANDOM_STATE = 42
