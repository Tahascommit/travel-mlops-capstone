# Streamlit app

```bash
streamlit run streamlit_app/app.py           # from the repo root, http://localhost:8501
# or
docker compose up -d streamlit
```

## Pages

- **Hotel recommendations**: pick a traveller to see their profile, their past stays and the top hotels for the next trip, each with a short reason ("one of your regular hotels", "popular with travellers from Recife (PE)", ...). You can change how many hotels to show and set a max price per day. "New user" shows the cold start case (most popular hotels).
- **Travel insights**: flights per month, average price per agency and class, top routes and hotel stays per hotel. All charts can be filtered by company.
- **Flight price**: sends the request to the Flask API on Kubernetes. If the API is down it uses the local model file and says so. It also shows the real price from the data for the same route/class/agency.
- **Gender from name**: the gender classifier with its confidence. It learned from the (mostly English) names in the dataset, so names from other languages are less reliable.
- **Models & monitoring**: test metrics of the three models, and live numbers from the API (`/health`, `/model-info` and the prometheus `/metrics`): which model version is live, number of predictions, error responses and average latency.

## Settings

| Env variable | Default |
|---|---|
| `API_URL` | http://localhost:8000 |
| `MLFLOW_UI_URL` | http://localhost:5001 |
