# Airflow

Airflow 3 runs in docker compose (see the root `docker-compose.yml`): api server, scheduler and dag processor with the LocalExecutor and a postgres database. The UI is on http://localhost:8080 (user `admin`, password from `.env`).

## The DAG: `flight_price_training_pipeline`

Runs `@weekly` and can be triggered by hand from the UI.

```
check_data >> train_and_register >> quality_gate >> is_new_champion >> trigger_deployment
```

| Task | What it does |
|---|---|
| `check_data` | loads the three csv files and validates the flights (columns, nulls, prices > 0, distance > 0, origin != destination). Fails the run if something is off |
| `train_and_register` | runs `src.flight_price.train`: trains linear regression, random forest and hist gradient boosting, logs every run to MLflow, registers the best one as a new version of `flight-price-regressor` and moves the `champion` alias if it is not worse than the current champion |
| `quality_gate` | fails if R² < 0.95 or RMSE > 50, so a bad model can never be deployed |
| `is_new_champion` | short circuit, skips the deployment when the champion didn't change |
| `trigger_deployment` | starts the Jenkins job (as the `airflow` user), which ships the new champion to Kubernetes |

Tasks retry once after 2 minutes.

## Why a separate virtualenv

The training code needs pinned versions of pandas, scikit-learn and mlflow. Installing those next to Airflow can break Airflow's own dependencies, so `airflow/Dockerfile` creates `/opt/ml-venv` with `requirements-train.txt`, and the ML tasks run there with `@task.external_python`. The repo is mounted read-only at `/opt/airflow/project` so the tasks import the same `src/` code as everything else.

## Commands

```bash
docker compose up -d airflow-apiserver airflow-scheduler airflow-dag-processor
docker exec airflow-scheduler airflow dags list
docker exec airflow-scheduler airflow dags trigger flight_price_training_pipeline
```

Task logs end up in `airflow/logs/` (not committed).
