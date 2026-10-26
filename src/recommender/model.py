"""Hotel recommender used by the training script and the Streamlit app.

Every hotel gets a score for a user, which is a weighted mix of:
  - hist: the user's own booking history (recent stays can count more)
  - cf:   a collaborative filtering score from TruncatedSVD on the user x hotel matrix
  - pop:  how popular the hotel is with travellers from the same home city
and the hotel in the user's home city gets pushed down, because people almost never
book a hotel in the city they live in.

The saved artifacts are a plain dict of numpy arrays, lists and dicts (no pandas objects
and no custom classes), so the file loads the same way in any environment.
"""
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from src.config import RANDOM_STATE, RECOMMENDER_PATH

# picked with rolling validation in notebooks/Hotel_Recommendation_System.ipynb
DEFAULT_PARAMS = {
    "half_life": 10.0,      # a stay 10 bookings ago counts half as much as the latest one
    "n_components": 2,
    "w_hist": 0.5,
    "w_cf": 0.0,            # the grid search gave collaborative filtering no weight on this data
    "w_pop": 0.5,
    "home_penalty": 1.0,
}

STAY_COLUMNS = ["userCode", "travelCode", "name", "place", "days", "price", "total", "date"]


# ---------- building blocks (same logic as in the notebook) ----------

def hotel_table(hotels):
    """One row per hotel with its city and price per day, sorted by hotel name."""
    return (hotels.groupby("name")
            .agg(place=("place", "first"), price_per_day=("price", "first"))
            .reset_index()
            .rename(columns={"name": "hotel"}))


def home_cities(flights):
    """Home city of each user = the city most of their outbound flights leave from."""
    outbound = flights.sort_values(["travelCode", "date"]).drop_duplicates("travelCode")
    return outbound.groupby("userCode")["from"].agg(lambda s: s.value_counts().index[0])


def history_scores(stays, user_ids, hotel_ids, half_life=None):
    """Share of each user's stays per hotel. With a half life, older stays count less
    (a stay that is `half_life` bookings old counts half as much as the latest one)."""
    age = stays.groupby("userCode").cumcount(ascending=False).to_numpy()
    weight = np.ones(len(stays)) if half_life is None else 0.5 ** (age / half_life)
    m = (pd.DataFrame({"u": stays["userCode"].to_numpy(), "h": stays["name"].map(hotel_ids).to_numpy(), "w": weight})
         .pivot_table(index="u", columns="h", values="w", aggfunc="sum")
         .reindex(index=user_ids, columns=range(len(hotel_ids)))
         .fillna(0.0)
         .to_numpy())
    totals = m.sum(axis=1, keepdims=True)
    return np.divide(m, totals, out=np.zeros_like(m), where=totals > 0)


def cf_scores(stays, user_ids, hotel_ids, n_components=2):
    """Low rank reconstruction of the (row normalised) user x hotel matrix.
    Hotels that users with a similar pattern book get a higher score."""
    counts = history_scores(stays, sorted(stays["userCode"].unique()), hotel_ids)
    svd = TruncatedSVD(n_components=int(n_components), random_state=RANDOM_STATE).fit(counts)
    target = history_scores(stays, user_ids, hotel_ids)
    return svd.inverse_transform(svd.transform(target))


def popularity_by_city(stays, home, n_hotels):
    """Booking share of every hotel overall and per home city."""
    hotel_idx = stays["hotel_id"].to_numpy()
    overall = np.bincount(hotel_idx, minlength=n_hotels) / len(stays)
    by_city = {}
    cities = stays["userCode"].map(home)
    for city, grp in stays.groupby(cities):
        counts = np.bincount(grp["hotel_id"].to_numpy(), minlength=n_hotels)
        by_city[city] = counts / counts.sum()
    return overall, by_city


def blend(hist, cf, pop, home_mask, params):
    return (params["w_hist"] * hist + params["w_cf"] * cf + params["w_pop"] * pop
            - params["home_penalty"] * home_mask)


# ---------- artifacts ----------

def build_artifacts(hotels, flights, params=None, metrics=None):
    """Fit the recommender on all stays and pack everything the app needs into a dict."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    stays = hotels.sort_values(["userCode", "date", "travelCode"]).reset_index(drop=True)
    table = hotel_table(stays)
    hotel_ids = {name: i for i, name in enumerate(table["hotel"])}
    stays["hotel_id"] = stays["name"].map(hotel_ids)

    home = home_cities(flights)
    user_ids = sorted(stays["userCode"].unique())
    overall, by_city = popularity_by_city(stays, home, len(table))

    counts = (pd.crosstab(stays["userCode"], stays["hotel_id"])
              .reindex(index=user_ids, columns=range(len(table)), fill_value=0).to_numpy())

    return {
        "params": params,
        "hotels": {
            "hotel": table["hotel"].tolist(),
            "place": table["place"].tolist(),
            "price_per_day": table["price_per_day"].astype(float).tolist(),
        },
        "user_row": {int(u): i for i, u in enumerate(user_ids)},
        "hist": history_scores(stays, user_ids, hotel_ids, params["half_life"]),
        "cf": cf_scores(stays, user_ids, hotel_ids, params["n_components"]),
        "visit_counts": counts,
        "home": {int(u): str(c) for u, c in home.items()},
        "pop_overall": overall,
        "pop_by_city": by_city,
        # past stays as plain columns so user_history can rebuild a dataframe
        "stays": {
            "userCode": stays["userCode"].to_numpy(dtype=int),
            "travelCode": stays["travelCode"].to_numpy(dtype=int),
            "hotel": stays["name"].to_numpy(dtype=object),
            "place": stays["place"].to_numpy(dtype=object),
            "days": stays["days"].to_numpy(dtype=int),
            "price_per_day": stays["price"].to_numpy(dtype=float),
            "total": stays["total"].to_numpy(dtype=float),
            "date": stays["date"].dt.strftime("%Y-%m-%d").to_numpy(dtype=object),
        },
        "metrics": metrics or {},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save_artifacts(artifacts, path=RECOMMENDER_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, path)
    return path


def load_artifacts(path=RECOMMENDER_PATH):
    return joblib.load(path)


# ---------- serving ----------

def _hotel_frame(artifacts):
    return pd.DataFrame(artifacts["hotels"])


def popular_hotels(artifacts, k=3):
    """Most booked hotels overall, used for users we know nothing about."""
    df = _hotel_frame(artifacts)
    df["score"] = artifacts["pop_overall"]
    df["reason"] = [f"{share:.1%} of all bookings" for share in df["score"]]
    return df.sort_values("score", ascending=False).head(k).reset_index(drop=True)


def recommend(artifacts, user_code, k=3, max_price=None):
    """Top k hotels for a user as a dataframe: hotel, place, price_per_day, score, reason."""
    params = artifacts["params"]
    hotels = _hotel_frame(artifacts)
    n = len(hotels)
    user_code = int(user_code)
    row = artifacts["user_row"].get(user_code)
    home = artifacts["home"].get(user_code)

    if row is None and home is None:
        # cold start, no hotel or flight history at all
        recs = popular_hotels(artifacts, k=n)
        recs["reason"] = "popular choice overall (no history for this user yet)"
    else:
        hist = artifacts["hist"][row] if row is not None else np.zeros(n)
        cf = artifacts["cf"][row] if row is not None else np.zeros(n)
        visits = artifacts["visit_counts"][row] if row is not None else np.zeros(n, dtype=int)
        pop = artifacts["pop_by_city"].get(home, artifacts["pop_overall"])
        home_mask = (np.array(artifacts["hotels"]["place"], dtype=object) == home).astype(float)

        parts = {
            "hist": params["w_hist"] * hist,
            "cf": params["w_cf"] * cf,
            "pop": params["w_pop"] * pop,
        }
        recs = hotels.copy()
        recs["score"] = blend(hist, cf, pop, home_mask, params)

        reasons = []
        for i in range(n):
            if home_mask[i]:
                reasons.append("in your home city, so it is ranked lower")
                continue
            # the part of the score that adds the most decides the main reason
            driver = max(parts, key=lambda p: parts[p][i])
            if driver == "hist":
                msg = "one of your regular hotels"
            elif driver == "cf":
                msg = "travellers with a similar booking history like it"
            else:
                msg = f"popular with travellers from {home}" if home else "popular choice overall"
            msg += f", you stayed here {int(visits[i])} times" if visits[i] else ", you have not tried it yet"
            reasons.append(msg)
        recs["reason"] = reasons

    if max_price is not None:
        recs = recs[recs["price_per_day"] <= max_price]

    recs = recs.sort_values("score", ascending=False).head(k).reset_index(drop=True)
    recs["score"] = recs["score"].round(4)
    return recs[["hotel", "place", "price_per_day", "score", "reason"]]


def user_history(artifacts, user_code):
    """All past hotel stays of a user, newest first."""
    stays = pd.DataFrame(artifacts["stays"])
    mine = stays[stays["userCode"] == int(user_code)].drop(columns="userCode")
    return mine.sort_values("date", ascending=False).reset_index(drop=True)
