"""Append privacy-conscious inference events as JSON Lines."""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.preprocess import PROJECT_ROOT

LOG_PATH = PROJECT_ROOT / "logs" / "inference" / "inference_log.jsonl"
_WRITE_LOCK = threading.Lock()


def log_event(
    *,
    request_id: str,
    status: str,
    latency_ms: float,
    prediction: int | None = None,
    error_type: str | None = None,
    path: Path = LOG_PATH,
) -> None:
    """Write operational metadata without storing submitted feature values."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "status": status,
        "latency_ms": round(latency_ms, 3),
    }
    if prediction is not None:
        record["prediction"] = prediction
    if error_type is not None:
        record["error_type"] = error_type

    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK, path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, allow_nan=False) + "\n")