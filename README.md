# Fetal Health Prediction

A beginner-friendly end-to-end fetal-health classification and MLOps project using pandas, scikit-learn, local MLflow, FastAPI, Docker, and Kubernetes manifests. The workflow continues from model reproducibility through local deployment, automated checks, and data-drift reporting.

This is an educational machine-learning example, not a clinical decision tool.

## Project Layout

```text
Fetal_Health_Prediction/
├── data/
│   ├── raw/fetal_health.csv
│   └── processed/                 # created by Lab 2
├── notebooks/Fetal_Health_Prediction.ipynb
├── src/                           # shared data, model, and evaluation code
├── pipelines/                     # one runnable script per lab
├── api/                           # FastAPI inference service
├── models/                        # fitted model files created by Lab 3 or 4
├── deployment/                    # deployable model bundle from Lab 7
├── tests/                         # API, failure, and monitoring tests
├── k8s/                            # local Kubernetes Deployment and Service
├── .github/workflows/             # CI, failure, monitoring, and runner checks
├── outputs/                       # plots and classification report
├── reports/                       # data quality and evaluation reports
├── artifacts/                     # reserved for generated artifacts
├── mlruns/                        # local MLflow experiment data
├── mlflow.db                      # local SQLite tracking database (created by Lab 4)
├── requirements.txt
└── README.md
```

## Setup (Windows PowerShell)

Run these commands from the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, allow it for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The raw input file must be at `data\raw\fetal_health.csv`.

## Run Each Lab

Run from the project folder. These commands use the project virtual environment directly, so activation is optional:

```powershell
.\.venv\Scripts\python.exe -m pipelines.run_lab1_eda
.\.venv\Scripts\python.exe -m pipelines.run_lab2_preprocessing
.\.venv\Scripts\python.exe -m pipelines.run_lab3_baseline
.\.venv\Scripts\python.exe -m pipelines.run_lab4_tracking
.\.venv\Scripts\python.exe -m pipelines.run_lab5_registry
.\.venv\Scripts\python.exe -m pipelines.run_lab6_reproducibility
```

Lab 1 prints basic summaries and saves the class-distribution and correlation plots. Lab 2 validates the schema, removes exact duplicate records, and writes the cleaned CSV, metadata, and data-quality report. Labs 3 and 4 can also run on their own; both load and clean the raw CSV through the shared code.

Lab 3 selects a baseline by 5-fold stratified cross-validation on the training partition. The test partition is stratified, held out during selection, and used once for final evaluation. Imputation and scaling are part of each model pipeline, so they are fitted within each training fold. The random seed is fixed at 42. Class weighting helps account for the unequal class counts.

Lab 3 and Lab 4 create:

- `models\best_fetal_model.pkl`: fitted preprocessing and classifier pipeline
- `models\preprocessor.pkl`: fitted imputer and scaler
- `outputs\accuracy_comparison.png`: cross-validation macro-F1 comparison
- `outputs\confusion_matrix.png` and `outputs\classification_report.txt`
- `reports\evaluation_report.txt`

Lab 4 logs parameters and metrics to local MLflow. To view the runs, start the UI from the project folder:

```powershell
mlflow ui --backend-store-uri sqlite:///./mlflow.db
```

Open the local URL printed by MLflow (usually `http://127.0.0.1:5000`). The UI is local; no cloud account is needed.

Lab 5 registers the existing `models\best_fetal_model.pkl` pipeline in the local MLflow Model Registry as `FetalHealthPredictionBestModel`. It records a model version, training parameters and metrics, and relevant evaluation artifacts. Run Lab 4 first if the saved model or baseline runs do not exist. Its report is written to `reports\lab5_registry_report.txt`.

Lab 6 snapshots the latest completed baseline metrics, reruns the Lab 4 tracked workflow with the fixed seed, and compares the selected model and test metrics. It then reloads the saved pipeline and verifies predictions on the same reproducible test split. Its report is written to `reports\lab6_reproducibility_report.txt`. Labs 5 and 6 add new MLflow records and do not delete earlier runs.

## Labs 7–13: Serving and Monitoring

Run Labs 3 or 4 before packaging the model. The API expects the same 21 numeric fields in `EXPECTED_FEATURES`; it returns class 1 (Normal), 2 (Suspect), or 3 (Pathological) with probabilities. It rejects missing, extra, or non-finite fields.

```powershell
python -m pipelines.run_lab7_deployment
python -m pytest -q
python -m pytest -q tests/test_failure_handling.py
python -m pipelines.run_lab10_lifecycle
python -m pipelines.run_lab12_monitoring --reference data/raw/fetal_health.csv --current data/raw/fetal_health.csv
python -m pipelines.run_api_performance
python -m pipelines.run_lab13_end_to_end
```

Lab 7 copies the selected fitted pipeline to `deployment\model.pkl` and writes a SHA-256 metadata manifest. Lab 8 runs the full test suite in GitHub Actions. Lab 9 exercises API failure handling. Lab 10 retrains through the existing MLflow workflow and packages the resulting model. Lab 12 compares feature distributions using population stability index (PSI), with a demonstration alert threshold of 0.2; labeled metrics are calculated only when a distinct current dataset and packaged model are supplied. Lab 13 checks the packaged model through `/health` and `/predict`. The monitoring workflow is scheduled weekly and can also be started manually.

The API performance runner records in-process request latency percentiles in `artifacts\api_performance_report.json`. It is a local smoke measurement, not a network benchmark or service-level objective.

Start the local API after Lab 7:

```powershell
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/` for the project dashboard or `http://127.0.0.1:8000/docs` to inspect the interactive API. The service writes request IDs, outcomes, predictions, and latency to `logs\inference\inference_log.jsonl`; it deliberately does not log submitted feature values.

For container deployment, build after Lab 7 so `deployment\model.pkl` exists:

```powershell
docker build -t fetal-health-api:latest .
docker run --rm -p 8000:8000 fetal-health-api:latest
```

Rebuild the image after changing the API or dashboard so the container includes the latest files. If port 8000 is already in use, stop the old container or use another host port, such as `-p 8001:8000`.

The manifests in `k8s/` target a local cluster with the image available. Apply them with `kubectl apply -f k8s/` after building/loading the image into that cluster.

This remains an educational project, not a medical device or clinical decision tool. The example monitoring workflow compares the raw dataset against itself unless a different `--current` dataset is supplied; it demonstrates the pipeline and is not production drift surveillance. Do not send identifiable patient information to the demo API or treat its output as medical advice.

## Notebook

Open `notebooks\Fetal_Health_Prediction.ipynb` in VS Code or Jupyter and run cells from top to bottom. It calls the same modules as the lab scripts to avoid duplicating logic.

## Dataset Target

The target column is `fetal_health`: 1 = Normal, 2 = Suspect, and 3 = Pathological. The pipeline checks the expected feature names and target values before training.
