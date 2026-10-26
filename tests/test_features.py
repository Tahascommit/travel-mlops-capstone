import pytest

from src.flight_price.features import FEATURES, make_features

ROUTES = {"Recife (PE)|Florianopolis (SC)": {"distance": 676.53, "time": 1.76}}
ALLOWED = {"flightType": ["economic", "firstClass", "premium"],
           "agency": ["CloudFy", "FlyingDrops", "Rainbow"]}
GOOD = {"from": "Recife (PE)", "to": "Florianopolis (SC)", "flightType": "economic", "agency": "CloudFy"}


def test_route_lookup_fills_distance_and_time():
    X = make_features([GOOD], ROUTES, ALLOWED)
    assert list(X.columns) == FEATURES
    assert X.loc[0, "distance"] == 676.53
    assert X.loc[0, "time"] == 1.76


def test_unknown_route():
    with pytest.raises(ValueError, match="no route"):
        make_features([{**GOOD, "to": "Natal (RN)"}], ROUTES, ALLOWED)


def test_bad_flight_type():
    with pytest.raises(ValueError, match="flightType"):
        make_features([{**GOOD, "flightType": "business"}], ROUTES, ALLOWED)


def test_missing_field():
    with pytest.raises(ValueError, match="missing"):
        make_features([{"from": "Recife (PE)", "to": "Florianopolis (SC)"}], ROUTES, ALLOWED)
