"""Streamlit app for the travel project.

Pages
    Hotel recommendations  pick a user, see their past stays and the hotels the recommender suggests
    Travel insights        charts on the flights / hotels / users data, filtered by company
    Flight price           price prediction through the flask api (falls back to the local model)
    Gender from name       try the gender classifier
    Models & monitoring    metrics of the three models and live stats from the api

Run from the repo root:
    streamlit run streamlit_app/app.py
"""
import os
import re
import sys
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import requests
import streamlit as st

# streamlit only puts this file's folder on the path, we need the repo root for src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, FLIGHT_MODEL_PATH, GENDER_MODEL_PATH  # noqa: E402
from src.flight_price.features import make_features  # noqa: E402
from src.gender.features import first_name  # noqa: E402
from src.recommender.model import load_artifacts, recommend, user_history  # noqa: E402

API_URL = os.getenv("API_URL", "http://localhost:8000")
MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL", "http://localhost:5001")

# chart colors: one blue for single series charts, and a fixed color per flight class
BLUE = "#2a78d6"
CLASS_COLORS = {"economic": "#2a78d6", "premium": "#eb6834", "firstClass": "#1baf7a"}

st.set_page_config(page_title="Travel MLOps", layout="wide")


# ---------- loading (cached so it only happens once per session) ----------

@st.cache_data
def load_data():
    users = pd.read_csv(DATA_DIR / "users.csv")
    flights = pd.read_csv(DATA_DIR / "flights.csv", parse_dates=["date"], date_format="%m/%d/%Y")
    hotels = pd.read_csv(DATA_DIR / "hotels.csv", parse_dates=["date"], date_format="%m/%d/%Y")
    return users, flights, hotels


@st.cache_resource
def load_models():
    return {
        "recommender": load_artifacts(),
        "flight": joblib.load(FLIGHT_MODEL_PATH),
        "gender": joblib.load(GENDER_MODEL_PATH),
    }


def bar_chart(df, x, y, title, horizontal=False, color=BLUE, x_title=None, y_title=None, fmt=",.0f"):
    """Simple single series bar chart with tooltips."""
    if horizontal:
        enc = dict(y=alt.Y(f"{x}:N", sort="-x", title=y_title), x=alt.X(f"{y}:Q", title=x_title))
    else:
        enc = dict(x=alt.X(f"{x}:N", sort="-y", title=x_title), y=alt.Y(f"{y}:Q", title=y_title))
    return (
        alt.Chart(df, title=title)
        .mark_bar(color=color, cornerRadiusEnd=4)
        .encode(**enc, tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:Q", format=fmt)])
    )


users, flights, hotels = load_data()
models = load_models()

st.sidebar.title("Travel MLOps")
page = st.sidebar.radio(
    "Page",
    ["Hotel recommendations", "Travel insights", "Flight price", "Gender from name", "Models & monitoring"],
)
st.sidebar.caption(f"API: {API_URL}")
st.sidebar.caption(f"MLflow: {MLFLOW_UI_URL}")


# ---------- hotel recommendations ----------

if page == "Hotel recommendations":
    st.title("Hotel recommendations")
    st.write("Pick a traveller to see where they stayed before and which hotels the model suggests "
             "for their next trip. The score mixes the user's own history with what people from the same "
             "home city book.")

    rec = models["recommender"]
    user_options = {"New user (no history)": None}
    for _, u in users.sort_values("code").iterrows():
        user_options[f"{u['code']} - {u['name']} ({u['company']})"] = int(u["code"])

    col_user, col_k, col_price = st.columns([3, 1, 1])
    label = col_user.selectbox("Traveller", list(user_options), index=1)
    k = col_k.slider("How many hotels", 1, 9, 3)
    max_price = col_price.slider("Max price per day", 60, 320, 320, step=10)
    user_code = user_options[label]

    if user_code is None:
        recs = recommend(rec, -1, k=k, max_price=max_price)
        st.info("No history for a new user, so the recommendations fall back to the most popular hotels.")
    else:
        user = users.set_index("code").loc[user_code]
        stays = user_history(rec, user_code)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Age", int(user["age"]))
        c2.metric("Company", user["company"])
        c3.metric("Home city", rec["home"].get(user_code, "unknown"))
        c4.metric("Hotel stays", len(stays))
        c5.metric("Spent on hotels", f"R$ {stays['total'].sum():,.0f}" if len(stays) else "R$ 0")
        recs = recommend(rec, user_code, k=k, max_price=max_price)

    st.subheader("Recommended hotels")
    if recs.empty:
        st.warning("No hotel fits that price limit.")
    else:
        st.dataframe(
            recs.rename(columns={"hotel": "Hotel", "place": "City", "price_per_day": "Price / day (R$)",
                                 "score": "Score", "reason": "Why"}),
            hide_index=True, width="stretch",
        )

    if user_code is not None and len(stays):
        st.subheader("Past stays")
        left, right = st.columns([1, 1])
        per_hotel = stays.groupby(["hotel", "place"]).size().reset_index(name="stays")
        per_hotel["label"] = per_hotel["hotel"] + " - " + per_hotel["place"]
        left.altair_chart(bar_chart(per_hotel, "label", "stays", "Stays per hotel", horizontal=True,
                                    x_title="Stays", y_title=None), width="stretch")
        right.dataframe(stays.head(15), hide_index=True, width="stretch")


# ---------- travel insights ----------

elif page == "Travel insights":
    st.title("Travel insights")

    companies = sorted(users["company"].unique())
    chosen = st.multiselect("Company", companies, default=companies)
    codes = users.loc[users["company"].isin(chosen), "code"]
    f = flights[flights["userCode"].isin(codes)]
    h = hotels[hotels["userCode"].isin(codes)]

    if f.empty:
        st.warning("Pick at least one company.")
        st.stop()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Travellers", f"{len(codes):,}")
    k2.metric("Flights", f"{len(f):,}")
    k3.metric("Hotel stays", f"{len(h):,}")
    k4.metric("Avg flight price", f"R$ {f['price'].mean():,.0f}")
    k5.metric("Avg hotel bill", f"R$ {h['total'].mean():,.0f}" if len(h) else "-")

    left, right = st.columns(2)

    monthly = f.set_index("date").resample("MS").size().reset_index(name="flights")
    line = (
        alt.Chart(monthly, title="Flights per month")
        .mark_line(color=BLUE, strokeWidth=2)
        .encode(x=alt.X("date:T", title=None), y=alt.Y("flights:Q", title="Flights"),
                tooltip=[alt.Tooltip("date:T", format="%b %Y"), alt.Tooltip("flights:Q", format=",")])
    )
    left.altair_chart(line, width="stretch")

    by_class = f.groupby(["agency", "flightType"])["price"].mean().reset_index()
    grouped = (
        alt.Chart(by_class, title="Average flight price by agency and class")
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("agency:N", title=None),
            xOffset=alt.XOffset("flightType:N", sort=list(CLASS_COLORS)),
            y=alt.Y("price:Q", title="Average price (R$)"),
            color=alt.Color("flightType:N", title="Class", sort=list(CLASS_COLORS),
                            scale=alt.Scale(domain=list(CLASS_COLORS), range=list(CLASS_COLORS.values()))),
            tooltip=["agency", "flightType", alt.Tooltip("price:Q", format=",.2f")],
        )
    )
    right.altair_chart(grouped, width="stretch")

    left, right = st.columns(2)
    routes = (f.assign(route=f["from"] + " -> " + f["to"]).groupby("route").size()
              .nlargest(10).reset_index(name="flights"))
    left.altair_chart(bar_chart(routes, "route", "flights", "Top 10 routes", horizontal=True,
                                x_title="Flights"), width="stretch")

    per_hotel = h.groupby(["name", "place"]).size().reset_index(name="stays")
    per_hotel["hotel"] = per_hotel["name"] + " - " + per_hotel["place"]
    right.altair_chart(bar_chart(per_hotel, "hotel", "stays", "Hotel stays by hotel", horizontal=True,
                                 x_title="Stays"), width="stretch")

    with st.expander("Show the numbers behind the charts"):
        st.dataframe(by_class.round(2), hide_index=True)
        st.dataframe(routes, hide_index=True)


# ---------- flight price ----------

elif page == "Flight price":
    st.title("Flight price prediction")
    st.write("The request goes to the Flask API (the one running on Kubernetes). "
             "If the API can't be reached the app uses the local model file instead.")

    bundle = models["flight"]
    cities = sorted({key.split("|")[0] for key in bundle["routes"]})
    c1, c2, c3, c4 = st.columns(4)
    origin = c1.selectbox("From", cities, index=cities.index("Sao Paulo (SP)"))
    destination = c2.selectbox("To", cities, index=cities.index("Brasilia (DF)"))
    flight_type = c3.selectbox("Class", bundle["allowed"]["flightType"])
    agency = c4.selectbox("Agency", bundle["allowed"]["agency"])

    if st.button("Predict price", type="primary"):
        record = {"from": origin, "to": destination, "flightType": flight_type, "agency": agency}
        try:
            res = requests.post(f"{API_URL}/predict", json=record, timeout=5)
            if res.status_code == 400:
                st.error(res.json()["error"])
                st.stop()
            res.raise_for_status()
            pred = res.json()["predictions"][0]
            source = f"Flask API ({API_URL})"
        except requests.RequestException:
            try:
                X = make_features([record], bundle["routes"], bundle["allowed"])
            except ValueError as e:
                st.error(str(e))
                st.stop()
            pred = {**X.iloc[0].to_dict(), "predicted_price": float(bundle["pipeline"].predict(X)[0])}
            source = "local model file (API not reachable)"

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted price", f"R$ {pred['predicted_price']:,.2f}")
        m2.metric("Distance", f"{pred['distance']:,.0f} km")
        m3.metric("Flight time", f"{pred['time']:.2f} h")
        st.caption(f"Served by: {source}")

        # since prices are fixed per route/class/agency we can show what the data says too
        actual = flights[(flights["from"] == origin) & (flights["to"] == destination)
                         & (flights["flightType"] == flight_type) & (flights["agency"] == agency)]["price"]
        if len(actual):
            st.caption(f"Price for this flight in the data: R$ {actual.iloc[0]:,.2f} ({len(actual)} bookings)")
        else:
            st.caption("This agency doesn't sell this class on this route in the data, so this is an estimate.")


# ---------- gender ----------

elif page == "Gender from name":
    st.title("Gender from name")
    st.write("The classifier only looks at the first name (character patterns like the ending). "
             "It was trained on the names in the users table, which are mostly English names, so it is "
             "much less reliable on names from other languages.")

    gender = models["gender"]
    name = st.text_input("Full name", "Wilma Mcinnis")
    if name.strip():
        first = first_name(name)
        proba = gender["pipeline"].predict_proba([first])[0]
        probs = dict(zip(gender["classes"], proba))
        label = max(probs, key=probs.get)
        c1, c2 = st.columns(2)
        c1.metric("Prediction", label)
        c2.metric("Confidence", f"{probs[label]:.0%}")
        st.caption(f"first name used: '{first}'")
        if probs[label] < 0.7:
            st.warning("Low confidence, the name doesn't look clearly male or female to the model.")


# ---------- models & monitoring ----------

elif page == "Models & monitoring":
    st.title("Models & monitoring")
    st.write(f"Test set metrics saved with each model. Full run history is in MLflow: {MLFLOW_UI_URL}")

    flight, gender, rec = models["flight"], models["gender"], models["recommender"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Flight price")
        st.caption(f"{flight['model_name']}, trained {flight['trained_at']}")
        st.metric("RMSE", f"{flight['metrics']['rmse']:.4f}")
        st.metric("MAE", f"{flight['metrics']['mae']:.4f}")
        st.metric("R²", f"{flight['metrics']['r2']:.4f}")
    with c2:
        st.subheader("Gender")
        st.caption(f"{gender['model_name']}, trained {gender['trained_at']}")
        st.metric("Accuracy", f"{gender['metrics']['accuracy']:.1%}")
        st.metric("F1", f"{gender['metrics']['f1']:.3f}")
        st.metric("ROC AUC", f"{gender['metrics']['roc_auc']:.3f}")
    with c3:
        st.subheader("Hotel recommender")
        st.caption(f"hybrid, trained {rec['trained_at']}")
        st.metric("Hit rate @3", f"{rec['metrics']['hit_at_3']:.1%}")
        st.metric("NDCG @3", f"{rec['metrics']['ndcg_at_3']:.3f}")
        st.metric("Catalog coverage", f"{rec['metrics']['coverage']:.0%}")

    st.subheader("Live API")
    try:
        info = requests.get(f"{API_URL}/model-info", timeout=3).json()
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        prom = requests.get(f"{API_URL}/metrics", timeout=3).text
    except requests.RequestException:
        st.warning(f"API at {API_URL} is not reachable.")
        st.stop()

    def prom_sum(metric, label_filter=""):
        # adds up all samples of a prometheus metric, optionally only lines containing label_filter
        total = 0.0
        for line in prom.splitlines():
            if line.startswith(metric) and label_filter in line:
                total += float(line.rsplit(" ", 1)[1])
        return total

    preds = prom_sum("api_requests_total", 'endpoint="/predict",status="200"')
    errors = sum(prom_sum("api_requests_total", f'status="{code}"') for code in (400, 404, 405, 500))
    count = prom_sum("api_predict_latency_seconds_count")
    avg_ms = prom_sum("api_predict_latency_seconds_sum") / count * 1000 if count else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Status", health["status"])
    m2.metric("Model version", f"v{info['registry_version']}")
    m3.metric("Predictions", f"{preds:,.0f}")
    m4.metric("Error responses", f"{errors:,.0f}")
    m5.metric("Avg latency", f"{avg_ms:.1f} ms")
    st.caption(f"Numbers come from the pod that answered ({health['host']}). Each pod keeps its own counters, "
               "in a real setup Prometheus would scrape all pods and add them up.")
    with st.expander("Raw /model-info"):
        st.json(info)
    with st.expander("Raw /metrics (prometheus format)"):
        st.code("\n".join(line for line in prom.splitlines() if re.match(r"^(# )?(HELP |TYPE )?api_", line)))
