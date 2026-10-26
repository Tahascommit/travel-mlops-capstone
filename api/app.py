"""Flask REST API for the flight price model.

Endpoints
    GET  /            small html form to try the model in the browser
    GET  /health      liveness/readiness check (used by docker and kubernetes)
    GET  /model-info  which model is loaded, its test metrics and the allowed input values
    POST /predict     one flight (json object) or many flights (json list) -> predicted price
    GET  /metrics     prometheus metrics (request count, latency, predictions)

Run locally:
    gunicorn --bind 0.0.0.0:8000 api.app:app
"""
import logging
import os
import socket
import time

import joblib
from flask import Flask, jsonify, render_template_string, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.config import FLIGHT_MODEL_PATH
from src.flight_price.features import make_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("flight-price-api")

REQUESTS = Counter("api_requests_total", "Requests received", ["endpoint", "status"])
LATENCY = Histogram("api_predict_latency_seconds", "Time spent in /predict")
PREDICTED_PRICE = Histogram("api_predicted_price", "Predicted prices",
                            buckets=(250, 500, 750, 1000, 1250, 1500, 1750, 2000))

MAX_BATCH = 1000


def load_model(path):
    bundle = joblib.load(path)
    log.info("loaded %s model trained at %s from %s", bundle["model_name"], bundle["trained_at"], path)
    return bundle


def create_app(model_path=None):
    app = Flask(__name__)
    bundle = load_model(model_path or os.getenv("MODEL_PATH", FLIGHT_MODEL_PATH))
    cities = sorted({key.split("|")[0] for key in bundle["routes"]})

    @app.get("/")
    def index():
        return render_template_string(FORM_HTML, cities=cities, allowed=bundle["allowed"],
                                      model_name=bundle["model_name"])

    @app.get("/health")
    def health():
        REQUESTS.labels("/health", 200).inc()
        # hostname is handy on kubernetes to see which pod answered
        return jsonify(status="ok", model=bundle["model_name"], host=socket.gethostname())

    @app.get("/model-info")
    def model_info():
        REQUESTS.labels("/model-info", 200).inc()
        return jsonify(
            model_name=bundle["model_name"],
            trained_at=bundle["trained_at"],
            data_md5=bundle["data_md5"],
            test_metrics=bundle["metrics"],
            cities=cities,
            flight_types=bundle["allowed"]["flightType"],
            agencies=bundle["allowed"]["agency"],
        )

    @app.post("/predict")
    def predict():
        start = time.perf_counter()
        payload = request.get_json(silent=True)

        if payload is None:
            return error("request body must be json", 400)
        records = payload if isinstance(payload, list) else [payload]
        if not records or not all(isinstance(r, dict) for r in records):
            return error("send a json object or a list of json objects", 400)
        if len(records) > MAX_BATCH:
            return error(f"max {MAX_BATCH} flights per request", 400)

        try:
            X = make_features(records, bundle["routes"], bundle["allowed"])
        except ValueError as e:
            return error(str(e), 400)

        prices = bundle["pipeline"].predict(X)
        results = []
        for rec, row, price in zip(records, X.to_dict("records"), prices):
            PREDICTED_PRICE.observe(price)
            results.append({
                "from": rec["from"], "to": rec["to"],
                "flightType": rec["flightType"], "agency": rec["agency"],
                "distance": row["distance"], "time": row["time"],
                "predicted_price": round(float(price), 2),
            })

        elapsed = time.perf_counter() - start
        LATENCY.observe(elapsed)
        REQUESTS.labels("/predict", 200).inc()
        log.info("predicted %d flight(s) in %.1f ms", len(results), elapsed * 1000)

        body = {"model": bundle["model_name"], "predictions": results}
        if not isinstance(payload, list):
            body["predicted_price"] = results[0]["predicted_price"]
        return jsonify(body)

    @app.get("/metrics")
    def metrics():
        return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

    @app.errorhandler(404)
    def not_found(_):
        return error("not found", 404)

    @app.errorhandler(405)
    def wrong_method(_):
        return error("method not allowed", 405)

    @app.errorhandler(Exception)
    def unhandled(e):
        log.exception("unhandled error")
        return error("internal server error", 500)

    return app


def error(message, status):
    # label by route rule, not raw path, so random 404 urls don't create new metric series
    endpoint = request.url_rule.rule if request.url_rule else "other"
    REQUESTS.labels(endpoint, status).inc()
    return jsonify(error=message), status


FORM_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Flight Price Prediction</title>
  <style>
    body { font-family: -apple-system, Segoe UI, sans-serif; background: #f4f6f9; margin: 0; padding: 40px 16px; }
    .card { max-width: 460px; margin: auto; background: #fff; padding: 28px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,.08); }
    h2 { margin-top: 0; }
    label { display: block; margin-top: 14px; font-size: 14px; color: #444; }
    select, button { width: 100%; padding: 9px; margin-top: 5px; font-size: 15px; border-radius: 6px; border: 1px solid #ccc; }
    button { background: #1f6feb; color: #fff; border: none; margin-top: 22px; cursor: pointer; }
    #result { margin-top: 20px; font-size: 18px; }
    small { color: #777; }
  </style>
</head>
<body>
<div class="card">
  <h2>Flight Price Prediction</h2>
  <small>model: {{ model_name }}</small>
  <label>From <select id="from">{% for c in cities %}<option>{{ c }}</option>{% endfor %}</select></label>
  <label>To <select id="to">{% for c in cities %}<option {% if loop.index == 2 %}selected{% endif %}>{{ c }}</option>{% endfor %}</select></label>
  <label>Flight type <select id="flightType">{% for t in allowed.flightType %}<option>{{ t }}</option>{% endfor %}</select></label>
  <label>Agency <select id="agency">{% for a in allowed.agency %}<option>{{ a }}</option>{% endfor %}</select></label>
  <button onclick="predict()">Predict price</button>
  <div id="result"></div>
</div>
<script>
async function predict() {
  const body = {};
  for (const id of ["from", "to", "flightType", "agency"]) body[id] = document.getElementById(id).value;
  const res = await fetch("/predict", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const data = await res.json();
  const out = document.getElementById("result");
  if (res.ok) {
    const p = data.predictions[0];
    out.innerHTML = "Predicted price: <b>R$ " + p.predicted_price.toFixed(2) + "</b><br><small>" + p.distance + " km, " + p.time + " h</small>";
  } else {
    out.innerHTML = "<span style='color:#c62828'>" + data.error + "</span>";
  }
}
</script>
</body>
</html>"""


app = create_app()
