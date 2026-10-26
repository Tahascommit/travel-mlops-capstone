"""Loading and basic checks for the three csv files."""
import pandas as pd

from src.config import DATA_DIR

EXPECTED_COLUMNS = {
    "users": ["code", "company", "name", "gender", "age"],
    "flights": ["travelCode", "userCode", "from", "to", "flightType",
                "price", "time", "distance", "agency", "date"],
    "hotels": ["travelCode", "userCode", "name", "place", "days",
               "price", "total", "date"],
}


def load(name, data_dir=DATA_DIR):
    """Read one of users / flights / hotels and make sure the columns are what we expect."""
    if name not in EXPECTED_COLUMNS:
        raise ValueError(f"unknown dataset '{name}'")

    df = pd.read_csv(f"{data_dir}/{name}.csv")
    missing = set(EXPECTED_COLUMNS[name]) - set(df.columns)
    if missing:
        raise ValueError(f"{name}.csv is missing columns: {sorted(missing)}")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    return df


def validate_flights(df):
    """Some sanity checks before training. Returns a list of problems (empty list = ok)."""
    problems = []
    if df.empty:
        problems.append("flights data is empty")
    if df[EXPECTED_COLUMNS["flights"]].isnull().any().any():
        problems.append("flights data has null values")
    if (df["price"] <= 0).any():
        problems.append("found flights with price <= 0")
    if (df["distance"] <= 0).any():
        problems.append("found flights with distance <= 0")
    if (df["from"] == df["to"]).any():
        problems.append("found flights where origin == destination")
    return problems
