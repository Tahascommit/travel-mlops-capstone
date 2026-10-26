# Airflow image with an extra virtualenv for our ML code.
# The training code needs pandas / scikit-learn / mlflow at fixed versions, putting them in their
# own venv means they can't break (or be broken by) airflow's own dependencies.
FROM apache/airflow:3.3.2-python3.12

USER root
RUN python -m venv /opt/ml-venv && chown -R airflow:0 /opt/ml-venv

USER airflow
COPY requirements-train.txt /tmp/requirements-train.txt
# pendulum: airflow checks for it in the venv before running an external_python task
# (it logs an import error otherwise), same version as airflow itself uses
RUN /opt/ml-venv/bin/pip install --no-cache-dir -r /tmp/requirements-train.txt pendulum==3.2.0
