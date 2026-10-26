# Saved models

| File | Made by | Used by |
|---|---|---|
| `flight_price_model.joblib` | `src.flight_price.train` / `export_champion` | flask api, streamlit |
| `gender_model.joblib` | `src.gender.train` | streamlit |
| `hotel_recommender.joblib` | `src.recommender.train` | streamlit |

Each file is a dict with the fitted pipeline (or for the recommender, plain arrays) plus metadata: test metrics, training time and an md5 of the data it was trained on. The flight model exported from MLflow also has `registry_version` and `run_id`.

The full history of every training run is in MLflow, these files are just the current versions.
