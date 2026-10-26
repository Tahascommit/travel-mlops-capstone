# Flight price API

Flask app that serves the flight price model. It loads `models/flight_price_model.joblib` once at startup (path can be changed with the `MODEL_PATH` env variable).

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/` | small html form to try the model in the browser |
| GET | `/health` | `{"status": "ok", ...}`, used by the docker healthcheck and the kubernetes probes. `host` tells you which pod answered |
| GET | `/model-info` | model name, mlflow registry version, test metrics, and the allowed cities / classes / agencies |
| POST | `/predict` | one flight (json object) or up to 1000 flights (json list) |
| GET | `/metrics` | prometheus metrics: request count per endpoint and status, predict latency, predicted prices |

The client only sends `from`, `to`, `flightType` and `agency`. Distance and flight time are fixed per route, so the API looks them up itself.

## Example

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"from": "Sao Paulo (SP)", "to": "Brasilia (DF)", "flightType": "premium", "agency": "Rainbow"}'
```

```json
{
  "model": "random_forest",
  "predicted_price": 586.06,
  "predictions": [{"from": "Sao Paulo (SP)", "to": "Brasilia (DF)", "flightType": "premium",
                   "agency": "Rainbow", "distance": 257.81, "time": 0.67, "predicted_price": 586.06}]
}
```

For a batch, send a list of the same objects and read `predictions`.

## Errors

Bad input gets a 400 with a message, for example:

- `{"error": "record 0: 'business' is not a valid flightType, use one of ['economic', 'firstClass', 'premium']"}`
- `{"error": "record 0: no route from 'Rio de Janeiro (RJ)' to 'Salvador (BH)'"}`

Unknown paths give 404, wrong methods 405, and anything unexpected is logged and returns a 500 json instead of a stack trace.

## Running

```bash
gunicorn --bind 0.0.0.0:8000 --workers 1 --threads 4 api.app:app     # from the repo root
pytest tests/test_api.py
```

In the container gunicorn runs 1 worker with 4 threads. Scaling is done by running more pods instead of more workers per pod, which also keeps the prometheus counters simple.
