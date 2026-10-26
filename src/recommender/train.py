"""Build the hotel recommender and track it with MLflow.

Uses the final settings picked in the notebook (DEFAULT_PARAMS in model.py), checks them
with the same chronological leave-last-out evaluation, logs everything to the
"hotel-recommendation" experiment and saves models/hotel_recommender.joblib for the app.

Usage:
    python -m src.recommender.train
"""
import json

import mlflow
import numpy as np
import pandas as pd

from src.config import MLFLOW_TRACKING_URI, RANDOM_STATE, RECOMMENDER_PATH
from src.data import load
from src.recommender.model import (DEFAULT_PARAMS, blend, build_artifacts, cf_scores,
                                   history_scores, home_cities, hotel_table,
                                   popularity_by_city, save_artifacts)

EXPERIMENT_NAME = "hotel-recommendation"
VAL_LEVELS = [1, 2, 3, 4, 5]   # 2nd to 6th last stay of each user
TEST_LEVEL = 0                 # last stay of each user
MIN_STAYS = 5
K = 3


def rank_metrics(scores, target, n_hotels, k=K):
    top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    hit = top == target[:, None]
    found = hit.any(axis=1)
    ndcg = np.where(found, 1 / np.log2(hit.argmax(axis=1) + 2), 0)
    rng = np.random.default_rng(RANDOM_STATE)
    a, b = rng.integers(0, len(top), 3000), rng.integers(0, len(top), 3000)
    overlap = np.mean([len(set(top[i]) & set(top[j])) / k for i, j in zip(a, b)])
    return {"hit_at_3": found.mean(), "precision_at_3": found.mean() / k, "ndcg_at_3": ndcg.mean(),
            "coverage": len(np.unique(top)) / n_hotels, "personalization": 1 - overlap}


def evaluate(stays, home, hotel_ids, places, params, levels):
    """Average metrics when predicting the stay at each level from the stays before it."""
    n_stays = stays.groupby("userCode").size()
    eligible = set(n_stays[n_stays >= MIN_STAYS].index)
    rows = []
    for level in levels:
        train = stays[stays["pos_from_end"] > level]
        target = stays[(stays["pos_from_end"] == level) & stays["userCode"].isin(eligible)]
        users = target["userCode"].tolist()
        homes = np.array([home.get(u) for u in users], dtype=object)

        overall, by_city = popularity_by_city(train, home, len(hotel_ids))
        pop = np.vstack([by_city.get(c, overall) for c in homes])
        home_mask = (places[None, :] == homes[:, None]).astype(float)
        hist = history_scores(train, users, hotel_ids, params["half_life"])
        cf = cf_scores(train, users, hotel_ids, params["n_components"])

        rows.append(rank_metrics(blend(hist, cf, pop, home_mask, params), target["hotel_id"].to_numpy(), len(hotel_ids)))
    return pd.DataFrame(rows).mean().to_dict()


def train(model_path=RECOMMENDER_PATH, params=None):
    params = {**DEFAULT_PARAMS, **(params or {})}
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    hotels, flights = load("hotels"), load("flights")
    stays = hotels.sort_values(["userCode", "date", "travelCode"]).reset_index(drop=True)
    table = hotel_table(stays)
    hotel_ids = {name: i for i, name in enumerate(table["hotel"])}
    places = table["place"].to_numpy(dtype=object)
    stays["hotel_id"] = stays["name"].map(hotel_ids)
    stays["pos_from_end"] = stays.groupby("userCode").cumcount(ascending=False)
    home = home_cities(flights)

    # popularity in the user's home city is the baseline we compare against
    baseline = {"half_life": None, "n_components": 2, "w_hist": 0.0, "w_cf": 0.0, "w_pop": 1.0, "home_penalty": 0.0}

    with mlflow.start_run(run_name="hybrid_recommender"):
        val = evaluate(stays, home, hotel_ids, places, params, VAL_LEVELS)
        test = evaluate(stays, home, hotel_ids, places, params, [TEST_LEVEL])
        base_test = evaluate(stays, home, hotel_ids, places, baseline, [TEST_LEVEL])

        mlflow.log_params({**{k: str(v) for k, v in params.items()},
                           "min_stays": MIN_STAYS, "k": K, "n_users": stays["userCode"].nunique(),
                           "n_stays": len(stays)})
        mlflow.log_metrics({f"val_{k}": v for k, v in val.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test.items()})
        mlflow.log_metrics({f"baseline_test_{k}": v for k, v in base_test.items()})

        print(f"{'':<22}{'hit@3':>8}{'ndcg@3':>8}{'coverage':>10}{'personal.':>11}")
        for name, m in [("validation", val), ("test", test), ("baseline test", base_test)]:
            print(f"{name:<22}{m['hit_at_3']:>8.4f}{m['ndcg_at_3']:>8.4f}{m['coverage']:>10.3f}{m['personalization']:>11.3f}")

        # final artifacts are built on every stay we have
        metrics = {k: round(float(v), 4) for k, v in test.items()}
        artifacts = build_artifacts(hotels, flights, params=params, metrics=metrics)
        save_artifacts(artifacts, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="recommender")

    return {"params": params, "test_metrics": metrics, "trained_at": artifacts["trained_at"],
            "saved_to": str(model_path)}


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
