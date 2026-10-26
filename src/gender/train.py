"""Train the gender classification model and track it with MLflow.

Only users with gender male/female are used for training ('none' means the gender
was not recorded). The input is the cleaned first name, turned into character
n-grams. Three tuned candidates are trained (params come from the grid / random
search in the gender notebook), each one is logged as an MLflow run, and the one
with the best 5-fold CV accuracy is kept. CV accuracy is used instead of the test
score because the test set is only 180 names. The best model is saved to
models/gender_model.joblib and registered in the MLflow model registry, and the
"champion" alias only moves if it beats the current champion.

Usage:
    python -m src.gender.train
    python -m src.gender.train --no-register
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.config import DATA_DIR, GENDER_MODEL_PATH, MLFLOW_TRACKING_URI, RANDOM_STATE
from src.data import load
from src.gender.features import LABELS, first_name

EXPERIMENT_NAME = "gender-classification"
REGISTERED_MODEL_NAME = "gender-classifier"
POS_LABEL = "female"

# mlflow saves sklearn models with skops, which refuses these types unless we say we trust them.
# these are our own models so it's fine
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree", "numpy.ravel"]


def flatten_step():
    # mlflow's pyfunc wrapper always passes a DataFrame, but the vectorizers want a flat list of
    # names. np.ravel turns a list, Series or one column DataFrame into a 1-D array of names.
    return ("flatten", FunctionTransformer(np.ravel))


def candidate_models():
    # params below come from the hyperparameter search in the gender notebook
    return {
        "logistic_regression": Pipeline([
            flatten_step(),
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), sublinear_tf=False)),
            ("clf", LogisticRegression(C=10, max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "naive_bayes": Pipeline([
            flatten_step(),
            ("vec", CountVectorizer(analyzer="char_wb", ngram_range=(1, 5), binary=True)),
            ("clf", MultinomialNB(alpha=0.5)),
        ]),
        "random_forest": Pipeline([
            flatten_step(),
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
            ("clf", RandomForestClassifier(n_estimators=500, max_depth=40, min_samples_leaf=1,
                                           max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
    }


def evaluate(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, list(model.classes_).index(POS_LABEL)]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, pos_label=POS_LABEL)),
        "recall": float(recall_score(y, pred, pos_label=POS_LABEL)),
        "f1": float(f1_score(y, pred, pos_label=POS_LABEL)),
        "roc_auc": float(roc_auc_score((y == POS_LABEL).astype(int), proba)),
    }


def file_md5(path):
    # used as a simple data version tag so we know which data a run was trained on
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_training_data(data_dir):
    users = load("users", data_dir)
    labeled = users[users["gender"].isin(LABELS)]
    if labeled.empty or set(labeled["gender"]) != set(LABELS):
        raise ValueError("users data needs both male and female labeled users to train")

    X = labeled["name"].map(first_name)
    if (X == "").any():
        raise ValueError("some users have an empty name after cleaning")
    return X, labeled["gender"]


def promote_if_better(client, version, cv_accuracy):
    """Point the champion alias to this version if its CV accuracy is at least as good as the current one."""
    try:
        current = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
        current_acc = client.get_run(current.run_id).data.metrics["cv_accuracy"]
    except Exception:
        current, current_acc = None, float("-inf")

    if cv_accuracy >= current_acc:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version)
        return True
    print(f"kept champion v{current.version} (cv acc {current_acc:.4f}) over v{version} (cv acc {cv_accuracy:.4f})")
    return False


def train(data_dir=DATA_DIR, model_path=GENDER_MODEL_PATH, register=True):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_training_data(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    data_version = file_md5(f"{data_dir}/users.csv")

    results = []
    for name, pipe in candidate_models().items():
        with mlflow.start_run(run_name=name) as run:
            cv_accuracy = float(cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy").mean())
            pipe.fit(X_train, y_train)
            metrics = {"cv_accuracy": cv_accuracy, **evaluate(pipe, X_test, y_test)}

            mlflow.log_params({"model_type": name, "n_train": len(X_train), "n_test": len(X_test),
                               "input": "first_name"})
            mlflow.log_params({k: v for k, v in pipe.get_params().items()
                               if "__" in k and (isinstance(v, (int, float, str, bool, tuple)) or v is None)})
            mlflow.log_metrics(metrics)
            mlflow.set_tag("data_md5", data_version)

            info = mlflow.sklearn.log_model(
                pipe, name="model",
                signature=infer_signature(X_test.head(50).tolist(), pipe.predict(X_test.head(50))),
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )
            print(f"{name:<20} cv_acc={cv_accuracy:.4f}  test_acc={metrics['accuracy']:.4f}  "
                  f"f1={metrics['f1']:.4f}  auc={metrics['roc_auc']:.4f}")
            results.append({"name": name, "run_id": run.info.run_id,
                            "model_uri": info.model_uri, "pipe": pipe, **metrics})

    best = max(results, key=lambda r: r["cv_accuracy"])
    print(f"best model: {best['name']}")

    # this bundle is what the api / streamlit app load. the pipeline takes a list of first names
    bundle = {
        "pipeline": best["pipe"],
        "classes": list(best["pipe"].classes_),
        "model_name": best["name"],
        "metrics": {k: best[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "cv_accuracy")},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_md5": data_version,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    summary = {k: bundle[k] for k in ("model_name", "metrics", "trained_at", "data_md5")}

    if register:
        client = MlflowClient()
        mv = mlflow.register_model(best["model_uri"], REGISTERED_MODEL_NAME)
        # keep a copy of the joblib bundle with the best run too
        with mlflow.start_run(run_id=best["run_id"]):
            mlflow.log_artifact(str(model_path), artifact_path="bundle")
        summary["registered_version"] = mv.version
        summary["promoted"] = promote_if_better(client, mv.version, best["cv_accuracy"])

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-register", action="store_true", help="skip the model registry step")
    args = parser.parse_args()
    print(json.dumps(train(register=not args.no_register), indent=2))
