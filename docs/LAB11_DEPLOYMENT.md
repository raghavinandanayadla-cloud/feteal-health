# Lab 11: Container and Kubernetes Deployment

## Build the Model Artifact

Train with Lab 4 or run the Lab 10 lifecycle, then package the selected model:

```powershell
python -m pipelines.run_lab7_deployment
```

The Docker build requires `deployment/model.pkl`. The artifact is intentionally produced locally and excluded from Git; rebuild it from the included training data before building an image.

## Docker

```powershell
docker build -t fetal-health-api:latest .
docker run --rm -p 8000:8000 fetal-health-api:latest
```

Verify `http://127.0.0.1:8000/health` and the interactive endpoint documentation at `http://127.0.0.1:8000/docs`.

## Kubernetes

Make the locally built image available to the selected cluster, then apply the resources:

```powershell
kubectl apply -f k8s/
kubectl get deployments,services
kubectl port-forward service/fetal-health-api 8000:8000
```

The sample uses one replica and basic HTTP probes. It does not configure ingress, TLS, authentication, persistent log storage, autoscaling, secret management, or production alerting. The API is an educational example and is not approved for clinical use.