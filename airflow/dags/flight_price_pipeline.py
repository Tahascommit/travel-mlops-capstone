"""
Flight price model - weekly retraining pipeline

    check_data >> train_and_register >> quality_gate >> is_new_champion >> trigger_deployment

check_data          checks the csv files before we train on them (columns, nulls, bad prices, routes)
train_and_register  trains the candidate models, logs every run to mlflow and registers the best one.
                    the "champion" alias only moves if the new model is at least as good as the old one
quality_gate        stops the pipeline if the model is below the minimum bar we accept
is_new_champion     skips the deployment if the champion didn't change
trigger_deployment  starts the jenkins job, which pulls the champion from mlflow, builds the
                    docker image and does a rolling update on kubernetes

The ML tasks run in /opt/ml-venv (see airflow/Dockerfile) and import the code from the repo,
which is mounted read-only at /opt/airflow/project.
"""
import os
from datetime import datetime, timedelta

from airflow.sdk import dag, task

PROJECT_DIR = os.getenv("PROJECT_DIR", "/opt/airflow/project")
ML_PYTHON = os.getenv("ML_PYTHON", "/opt/ml-venv/bin/python")

# minimum quality we accept before a model can go anywhere near production
MIN_R2 = 0.95
MAX_RMSE = 50.0

default_args = {
    "owner": "travel-mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="flight_price_training_pipeline",
    description="Validate data, retrain the flight price model, register it in MLflow and trigger deployment",
    schedule="@weekly",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    default_args=default_args,
    tags=["flight-price", "training", "mlflow"],
)
def flight_price_training_pipeline():

    @task.external_python(python=ML_PYTHON, expect_airflow=False)
    def check_data(project_dir: str) -> dict:
        import sys
        sys.path.insert(0, project_dir)
        from src.data import load, validate_flights

        flights = load("flights")
        problems = validate_flights(flights)
        if problems:
            raise ValueError("flights data failed validation: " + "; ".join(problems))

        # hotels and users aren't used by this model but a broken file should still be caught early
        hotels, users = load("hotels"), load("users")
        report = {
            "flights_rows": len(flights),
            "hotels_rows": len(hotels),
            "users_rows": len(users),
            "routes": int(flights.groupby(["from", "to"]).ngroups),
            "date_range": [str(flights["date"].min().date()), str(flights["date"].max().date())],
        }
        print(report)
        return report

    @task.external_python(python=ML_PYTHON, expect_airflow=False)
    def train_and_register(project_dir: str, data_report: dict) -> dict:
        import sys
        sys.path.insert(0, project_dir)
        from src.flight_price.train import train

        print(f"training on {data_report['flights_rows']} flights")
        summary = train(register=True)
        print(summary)
        return summary

    @task
    def quality_gate(summary: dict) -> dict:
        metrics = summary["metrics"]
        if metrics["r2"] < MIN_R2 or metrics["rmse"] > MAX_RMSE:
            raise ValueError(f"model below the quality bar (r2 >= {MIN_R2}, rmse <= {MAX_RMSE}): {metrics}")
        print(f"quality gate passed: {metrics}")
        return summary

    @task.short_circuit
    def is_new_champion(summary: dict) -> bool:
        print(f"registered version {summary.get('registered_version')}, promoted = {summary.get('promoted')}")
        return bool(summary.get("promoted"))

    @task
    def trigger_deployment() -> str:
        import requests

        url = os.environ["JENKINS_URL"]
        job = os.getenv("JENKINS_JOB", "flight-price-cicd")

        session = requests.Session()
        session.auth = (os.environ["JENKINS_USER"], os.environ["JENKINS_PASSWORD"])
        # jenkins wants a crumb (csrf token) for POST requests, it's tied to the session
        crumb = session.get(f"{url}/crumbIssuer/api/json", timeout=15)
        crumb.raise_for_status()
        crumb = crumb.json()

        # the job has a RETRAIN parameter, so it has to be started with buildWithParameters.
        # RETRAIN=false because we just trained, jenkins only needs to ship the new champion
        res = session.post(f"{url}/job/{job}/buildWithParameters", params={"RETRAIN": "false"},
                           headers={crumb["crumbRequestField"]: crumb["crumb"]}, timeout=15)
        res.raise_for_status()
        queued = res.headers.get("Location", "")
        print(f"triggered jenkins job {job}, queue item: {queued}")
        return queued

    data_report = check_data(PROJECT_DIR)
    summary = train_and_register(PROJECT_DIR, data_report)
    checked = quality_gate(summary)
    is_new_champion(checked) >> trigger_deployment()


flight_price_training_pipeline()
