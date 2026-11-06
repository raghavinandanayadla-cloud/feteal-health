# Local Runbook

## Prepare and Train

1. Install the project requirements with `python -m pip install -r requirements.txt`.
2. Verify `data/raw/fetal_health.csv` is present.
3. Run Labs 1–6 in order using the commands in the root README. Lab 4 creates the tracked model and MLflow records; Labs 5 and 6 rely on those records.
4. Run `python -m pipelines.run_lab7_deployment` to create the deployable model bundle.

## Start the API

Run `uvicorn api.main:app --host 127.0.0.1 --port 8000`. Check `GET /health`; a healthy response includes `{"status":"ok","model_loaded":true}`. Interactive OpenAPI docs are available at `/docs`.

Send `POST /predict` with a JSON object containing a `features` object and all 21 named numeric columns. The response includes the integer class, human-readable label, and probabilities. Invalid feature sets return HTTP 422. A missing or unreadable model returns HTTP 503 without exposing internal file paths.

## Monitoring and Recovery

Run `python -m pipelines.run_lab12_monitoring --reference data/raw/fetal_health.csv --current PATH_TO_CURRENT_CSV`. The current CSV must contain all 21 model features. Include `fetal_health` labels to calculate performance metrics. Review `reports/lab12_monitoring_report.json` and inspect `drifted_features`; an alert indicates distribution movement only and does not by itself establish degraded clinical performance.

If the API health check reports unavailable, confirm `deployment/model.pkl` exists and matches the runtime environment, then rerun the Lab 7 packaging step. After retraining, rerun the API tests and end-to-end check before replacing a deployed bundle.

## Container and Kubernetes

Build the image after packaging: `docker build -t fetal-health-api:latest .`. Run it locally with `docker run --rm -p 8000:8000 fetal-health-api:latest`. For a local Kubernetes cluster, make the image available to the cluster and apply `kubectl apply -f k8s/`. The sample manifests are not a production security or high-availability configuration.

This runbook supports educational experimentation only, not medical use. Do not use real identifiable patient data with the sample service.