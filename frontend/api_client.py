"""Centralized API client for frontend integration with FastAPI backend."""

import os
import io
import logging
from typing import Dict, Any, Tuple, Optional, Union
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_TIMEOUT = 10  # seconds

logger = logging.getLogger(__name__)


class APIClientError(Exception):
    """Custom exception for API interaction errors."""
    pass


def get_health() -> Dict[str, Any]:
    """Check API health and model readiness status."""
    url = f"{API_BASE_URL}/health"
    try:
        response = requests.get(url, timeout=DEFAULT_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            data = response.json()
            return {
                "status": "degraded",
                "prediction_ready": False,
                "details": data.get("details", "Model artifact unavailable."),
            }
        else:
            return {
                "status": "error",
                "prediction_ready": False,
                "details": f"API returned status {response.status_code}",
            }
    except requests.exceptions.RequestException as e:
        return {
            "status": "offline",
            "prediction_ready": False,
            "details": f"Cannot connect to FastAPI backend at {API_BASE_URL}. Ensure backend is running (`uvicorn backend.main:app --port 8000`).",
        }


def predict_severity_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send prediction payload to /predict/severity endpoint."""
    url = f"{API_BASE_URL}/predict/severity"
    try:
        response = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 422:
            detail = response.json().get("detail", "Validation failed.")
            raise APIClientError(f"Validation error: {detail}")
        elif response.status_code == 503:
            detail = response.json().get("detail", "Model service unavailable.")
            raise APIClientError(f"Service unavailable: {detail}")
        else:
            raise APIClientError(f"Prediction failed with status {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        raise APIClientError(
            f"Failed to communicate with FastAPI backend at {API_BASE_URL}. "
            f"Please verify the backend is running. Details: {str(e)}"
        )


def predict_batch_api(file_bytes: bytes, filename: str) -> Tuple[bool, Union[pd.DataFrame, str]]:
    """Send CSV file to /predict/batch endpoint and parse returned CSV response."""
    url = f"{API_BASE_URL}/predict/batch"
    try:
        files = {"file": (filename, file_bytes, "text/csv")}
        response = requests.post(url, files=files, timeout=60)

        if response.status_code == 200:
            csv_str = response.content.decode("utf-8")
            df = pd.read_csv(io.StringIO(csv_str))
            return True, df
        else:
            try:
                err_data = response.json()
                msg = err_data.get("detail", response.text)
            except Exception:
                msg = response.text
            return False, f"API Error ({response.status_code}): {msg}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection failed to {url}: {str(e)}"


def get_model_metrics_api() -> Dict[str, Any]:
    """Retrieve model metrics and comparison reports from /model/metrics."""
    url = f"{API_BASE_URL}/model/metrics"
    try:
        response = requests.get(url, timeout=DEFAULT_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        else:
            raise APIClientError(f"Metrics endpoint error {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        raise APIClientError(f"Failed to fetch model metrics from backend: {str(e)}")
