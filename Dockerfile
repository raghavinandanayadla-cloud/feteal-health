FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FETAL_MODEL_PATH=/app/deployment/model.pkl

WORKDIR /app
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
COPY api ./api
COPY src ./src
COPY deployment/model.pkl ./deployment/model.pkl
COPY deployment/model_metadata.json ./deployment/model_metadata.json
RUN mkdir -p /app/logs/inference

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]