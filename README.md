# Travel MLOps Capstone

Flight price prediction, gender classification and hotel recommendation on the travel dataset (users, flights, hotels), along with the deployment side of things: Flask API, Docker, Kubernetes, Airflow, Jenkins and MLflow.

Work in progress, the full README will be added once all parts are done.

## Data

The three csv files are in `data/raw/`:

- `users.csv` - 1340 users (code, company, name, gender, age)
- `flights.csv` - 271,888 flight bookings
- `hotels.csv` - 40,552 hotel bookings

Flights and hotels link to users through `userCode`, and to each other through `travelCode`.
