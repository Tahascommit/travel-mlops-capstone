"""Download the current champion model from the MLflow registry into models/.

Jenkins runs this before building the docker image, so the image always ships
the model version that Airflow (or a manual training run) promoted to champion.

Usage:
    python -m src.flight_price.export_champion
"""
import json
import tempfile

import joblib
import mlflow
from mlflow import MlflowClient

from src.config import FLIGHT_MODEL_PATH, MLFLOW_TRACKING_URI
from src.flight_price.train import REGISTERED_MODEL_NAME


def export_champion(output_path=FLIGHT_MODEL_PATH):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    version = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")

    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(
            run_id=version.run_id, artifact_path="bundle/flight_price_model.joblib", dst_path=tmp)
        bundle = joblib.load(local)

    # remember which registry version this is, the api shows it on /model-info
    bundle["registry_version"] = str(version.version)
    bundle["run_id"] = version.run_id
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)

    return {"registered_model": REGISTERED_MODEL_NAME, "version": version.version,
            "run_id": version.run_id, "model_name": bundle["model_name"],
            "metrics": bundle["metrics"], "saved_to": str(output_path)}


if __name__ == "__main__":
    print(json.dumps(export_champion(), indent=2))
