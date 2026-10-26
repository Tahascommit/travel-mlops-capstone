"""Train the flight price regression model and track it with MLflow.

Trains a few candidate models, logs each one as an MLflow run, keeps the one with
the lowest RMSE, saves it to models/flight_price_model.joblib and registers it in
the MLflow model registry. The "champion" alias only moves to the new version if it
beats the current champion.

Usage:
    python -m src.flight_price.train
    python -m src.flight_price.train --no-register
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import DATA_DIR, FLIGHT_MODEL_PATH, MLFLOW_TRACKING_URI, RANDOM_STATE
from src.data import load, validate_flights
from src.flight_price.features import (CATEGORICAL, FEATURES, TARGET,
                                       build_preprocessor, build_route_table)

EXPERIMENT_NAME = "flight-price-prediction"
REGISTERED_MODEL_NAME = "flight-price-regressor"

# mlflow saves sklearn models with skops, which refuses tree objects unless we say we trust them.
# these are our own models so it's fine
SKOPS_TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


def candidate_models():
    return {
        "linear_regression": LinearRegression(),
        # params below come from the random search in the regression notebook
        "random_forest": RandomForestRegressor(
            n_estimators=100, min_samples_leaf=1, max_features=0.5, n_jobs=-1, random_state=RANDOM_STATE),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.2, max_leaf_nodes=63, max_depth=6,
            l2_regularization=0.1, random_state=RANDOM_STATE),
    }


def evaluate(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def file_md5(path):
    # used as a simple data version tag so we know which data a run was trained on
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def promote_if_better(client, version, rmse):
    """Point the champion alias to this version if it has a lower RMSE than the current one."""
    try:
        current = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
        current_rmse = client.get_run(current.run_id).data.metrics["rmse"]
    except Exception:
        current, current_rmse = None, float("inf")

    if rmse <= current_rmse:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version)
        return True
    print(f"kept champion v{current.version} (rmse {current_rmse:.3f}) over v{version} (rmse {rmse:.3f})")
    return False


def train(data_dir=DATA_DIR, model_path=FLIGHT_MODEL_PATH, register=True):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    flights = load("flights", data_dir)
    problems = validate_flights(flights)
    if problems:
        raise ValueError("data validation failed: " + "; ".join(problems))

    X, y = flights[FEATURES], flights[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    data_version = file_md5(f"{data_dir}/flights.csv")

    results = []
    for name, estimator in candidate_models().items():
        with mlflow.start_run(run_name=name) as run:
            pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
            pipe.fit(X_train, y_train)
            metrics = evaluate(y_test, pipe.predict(X_test))

            mlflow.log_params({"model_type": name, "n_train": len(X_train), "n_test": len(X_test),
                               "features": ",".join(FEATURES)})
            mlflow.log_params({f"model__{k}": v for k, v in estimator.get_params().items()
                               if isinstance(v, (int, float, str, bool)) or v is None})
            mlflow.log_metrics(metrics)
            mlflow.set_tag("data_md5", data_version)

            info = mlflow.sklearn.log_model(
                pipe, name="model",
                signature=infer_signature(X_test.head(100), pipe.predict(X_test.head(100))),
                input_example=X_test.head(3),
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )
            print(f"{name:<24} rmse={metrics['rmse']:.3f}  mae={metrics['mae']:.3f}  r2={metrics['r2']:.5f}")
            results.append({"name": name, "run_id": run.info.run_id,
                            "model_uri": info.model_uri, "pipe": pipe, **metrics})

    best = min(results, key=lambda r: r["rmse"])
    print(f"best model: {best['name']}")

    # this bundle is what the flask api and streamlit app load
    allowed = {c: sorted(flights[c].unique().tolist()) for c in CATEGORICAL}
    bundle = {
        "pipeline": best["pipe"],
        "routes": build_route_table(flights),
        "allowed": allowed,
        "model_name": best["name"],
        "metrics": {k: best[k] for k in ("rmse", "mae", "r2")},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_md5": data_version,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    summary = {k: bundle[k] for k in ("model_name", "metrics", "trained_at", "data_md5")}

    if register:
        client = MlflowClient()
        mv = mlflow.register_model(best["model_uri"], REGISTERED_MODEL_NAME)
        # keep a copy of the joblib bundle with the best run too
        with mlflow.start_run(run_id=best["run_id"]):
            mlflow.log_artifact(str(model_path), artifact_path="bundle")
        summary["registered_version"] = mv.version
        summary["promoted"] = promote_if_better(client, mv.version, best["rmse"])

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-register", action="store_true", help="skip the model registry step")
    args = parser.parse_args()
    print(json.dumps(train(register=not args.no_register), indent=2))
