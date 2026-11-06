# Fetal Health Project Architecture

The project has one shared training contract. `src/preprocess.py` defines the 21 predictors, target column, class codes, and seed. Labs 3 and 4 fit preprocessing and the classifier together in a scikit-learn pipeline, so serving uses the identical imputer and scaler fitted during training.

## Model Lifecycle

Labs 1–6 cover data inspection, cleaning, baseline model selection, MLflow tracking and registry, and reproducibility. Lab 7 packages the selected `models/best_fetal_model.pkl` into `deployment/model.pkl` and records its SHA-256, feature contract, and class labels. Lab 10 reruns tracked training and refreshes the deployment bundle.

## Inference and Observability

`api/main.py` validates requests, creates a one-row DataFrame in training feature order, and loads the packaged pipeline through `api/model_loader.py`. `/health` reports whether the model is available. `/predict` returns the predicted class, display label, and class probabilities. Operational logs contain request identifiers and outcomes, not submitted feature values.

`src/monitoring.py` computes PSI for each input feature using decile boundaries learned from the reference sample. A PSI of 0.2 or greater marks that feature as drifted. This threshold is an educational example, not a clinically validated limit. If labeled data and a model are available, the report also includes accuracy and macro-F1.

## Checks and Deployment

The `tests/` suite covers API contracts, invalid inputs, unavailable models, logging privacy, and drift calculations. GitHub Actions runs unit and failure tests; manual workflows exercise the full trained-model API and scheduled data-monitoring report. Docker packages the inference service and deployment model. Kubernetes manifests provide a basic single-replica local deployment with readiness/liveness checks.

The sample is not intended for clinical use. The repository does not establish clinical validity, patient-level privacy controls, authentication, rate limiting, or production alert handling.