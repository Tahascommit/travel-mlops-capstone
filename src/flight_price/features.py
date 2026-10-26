"""Feature handling for the flight price model.

The model only needs from, to, flightType and agency from the user. Distance and
flight time are fixed for every route in the data, so we keep a lookup table of
routes and fill them in ourselves instead of asking the client for them.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL = ["from", "to", "flightType", "agency"]
NUMERIC = ["distance", "time"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "price"


def route_key(origin, destination):
    return f"{origin}|{destination}"


def build_route_table(flights):
    """Map every (from, to) pair to its distance and time."""
    routes = flights.groupby(["from", "to"])[["distance", "time"]].first().reset_index()
    return {
        route_key(r["from"], r["to"]): {"distance": float(r["distance"]), "time": float(r["time"])}
        for _, r in routes.iterrows()
    }


def build_preprocessor():
    return ColumnTransformer(
        # dense output since there are only ~20 one-hot columns and HistGradientBoosting needs dense input
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
        remainder="passthrough",  # distance and time go through as they are
    )


def make_features(records, routes, allowed):
    """Turn raw request dicts into the dataframe the model expects.

    records: list of dicts with from, to, flightType, agency
    routes: output of build_route_table
    allowed: dict of allowed values per categorical column
    Raises ValueError with a readable message when something is wrong.
    """
    rows = []
    for i, rec in enumerate(records):
        missing = [c for c in CATEGORICAL if not rec.get(c)]
        if missing:
            raise ValueError(f"record {i}: missing field(s) {missing}")

        for col in ("flightType", "agency"):
            if rec[col] not in allowed[col]:
                raise ValueError(f"record {i}: '{rec[col]}' is not a valid {col}, use one of {allowed[col]}")

        key = route_key(rec["from"], rec["to"])
        if key not in routes:
            raise ValueError(f"record {i}: no route from '{rec['from']}' to '{rec['to']}'")

        rows.append({c: rec[c] for c in CATEGORICAL} | routes[key])

    return pd.DataFrame(rows, columns=FEATURES)
