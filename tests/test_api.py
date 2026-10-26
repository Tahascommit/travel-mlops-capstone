"""API tests. They need models/flight_price_model.joblib, so run the training script first."""
import pytest

from api.app import create_app

FLIGHT = {"from": "Recife (PE)", "to": "Florianopolis (SC)", "flightType": "firstClass", "agency": "FlyingDrops"}


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"


def test_model_info(client):
    info = client.get("/model-info").get_json()
    assert len(info["cities"]) == 9
    assert info["test_metrics"]["r2"] > 0.9


def test_predict_single(client):
    res = client.post("/predict", json=FLIGHT)
    assert res.status_code == 200
    body = res.get_json()
    # this exact flight is in the data at 1434.38, the model should be very close
    assert abs(body["predicted_price"] - 1434.38) < 25


def test_predict_batch(client):
    res = client.post("/predict", json=[FLIGHT, {**FLIGHT, "flightType": "economic", "agency": "CloudFy"}])
    preds = res.get_json()["predictions"]
    assert len(preds) == 2
    assert preds[0]["predicted_price"] > preds[1]["predicted_price"]  # first class costs more than economy


def test_unknown_route_returns_400(client):
    res = client.post("/predict", json={**FLIGHT, "from": "Rio de Janeiro (RJ)", "to": "Salvador (BH)"})
    assert res.status_code == 400
    assert "no route" in res.get_json()["error"]


def test_bad_body_returns_400(client):
    assert client.post("/predict", data="hello", content_type="text/plain").status_code == 400
    assert client.post("/predict", json=[]).status_code == 400
    assert client.post("/predict", json={"from": "Recife (PE)"}).status_code == 400


def test_unknown_path_returns_404(client):
    assert client.get("/does-not-exist").status_code == 404


def test_metrics_endpoint(client):
    assert b"api_requests_total" in client.get("/metrics").data
