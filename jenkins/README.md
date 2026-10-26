# Jenkins

Jenkins runs in docker compose on http://localhost:8081 (user `admin`, password from `.env`). Everything is set up as code, there is no setup wizard:

- `Dockerfile`: Jenkins LTS + docker cli/buildx, kubectl and uv, plus the plugins from `plugins.txt`
- `casc.yaml`: Jenkins configuration as code. Creates the `admin` user, an `airflow` user for the retraining DAG, and the `flight-price-cicd` pipeline job
- `kubeconfig.sh`: builds a kubeconfig that works from inside the container (see below)

The pipeline itself is the `Jenkinsfile` in the repo root.

## Pipeline stages

| Stage | What |
|---|---|
| Python env | `uv` creates a python 3.12 venv with the api + training requirements |
| Unit tests | `pytest tests/test_features.py` |
| Get model | downloads the `champion` model from the MLflow registry (`src.flight_price.export_champion`). With `RETRAIN=true` it trains first |
| API tests | `pytest tests/test_api.py` against the exported model |
| Build image | `docker build -t flight-price-api:<build number>` |
| Deploy to Kubernetes | applies the manifests with the new image tag, waits for the rollout and runs `kubectl rollout undo` if the pods don't get healthy |
| Smoke test | calls `/health`, `/model-info` and `/predict` on the live service |

Test results are published with the junit plugin.

## What starts a build

- a new commit on `main` (Jenkins polls GitHub every 5 minutes)
- the Airflow DAG after it promotes a new champion ("Started by user airflow")
- by hand with Build with Parameters

## How Jenkins reaches docker and kubernetes

The docker socket from Docker Desktop is mounted into the container, so `docker build` builds on the same engine that Kubernetes uses. The socket belongs to group 0, so the jenkins user gets that group (`group_add`) instead of running Jenkins as root.

The mac's `~/.kube/config` is mounted read-only. It points to `https://127.0.0.1:<port>`, which inside a container is the container itself, so `kubeconfig.sh` rewrites the server to `host.docker.internal` and sets `tls-server-name: localhost` (a name that is in the API server certificate). TLS verification stays on. The generated kubeconfig is deleted at the end of every build.
