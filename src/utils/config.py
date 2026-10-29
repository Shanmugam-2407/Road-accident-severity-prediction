"""Configuration management for RoadGuard AI."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve project root directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables from .env if present
load_dotenv(BASE_DIR / ".env")

# API Configuration
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{API_HOST}:{API_PORT}")

# File & Directory Paths
DATASET_PATH = Path(os.getenv("DATASET_PATH", str(BASE_DIR / "data" / "sample" / "sample_accidents.csv")))
if not DATASET_PATH.is_absolute():
    DATASET_PATH = BASE_DIR / DATASET_PATH

RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", str(BASE_DIR / "data" / "raw")))
if not RAW_DATA_DIR.is_absolute():
    RAW_DATA_DIR = BASE_DIR / RAW_DATA_DIR

PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", str(BASE_DIR / "data" / "processed")))
if not PROCESSED_DATA_DIR.is_absolute():
    PROCESSED_DATA_DIR = BASE_DIR / PROCESSED_DATA_DIR

ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_ARTIFACT_PATH = Path(os.getenv("MODEL_ARTIFACT_PATH", str(ARTIFACTS_DIR / "accident_severity_model.joblib")))
if not MODEL_ARTIFACT_PATH.is_absolute():
    MODEL_ARTIFACT_PATH = BASE_DIR / MODEL_ARTIFACT_PATH

MODEL_METADATA_PATH = Path(os.getenv("MODEL_METADATA_PATH", str(ARTIFACTS_DIR / "model_metadata.json")))
if not MODEL_METADATA_PATH.is_absolute():
    MODEL_METADATA_PATH = BASE_DIR / MODEL_METADATA_PATH

EVALUATION_SUMMARY_PATH = Path(os.getenv("EVALUATION_SUMMARY_PATH", str(REPORTS_DIR / "evaluation_summary.json")))
if not EVALUATION_SUMMARY_PATH.is_absolute():
    EVALUATION_SUMMARY_PATH = BASE_DIR / EVALUATION_SUMMARY_PATH

MODEL_COMPARISON_PATH = Path(os.getenv("MODEL_COMPARISON_PATH", str(REPORTS_DIR / "model_comparison.csv")))
if not MODEL_COMPARISON_PATH.is_absolute():
    MODEL_COMPARISON_PATH = BASE_DIR / MODEL_COMPARISON_PATH

MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
APP_ENV = os.getenv("APP_ENV", "development")

# Target schema definition
TARGET_COLUMN = "accident_severity"
TARGET_CLASSES = ["Slight", "Serious", "Fatal"]

# Severity label mappings (normalizes numeric or variant strings)
SEVERITY_MAPPING = {
    "1": "Fatal",
    "2": "Serious",
    "3": "Slight",
    1: "Fatal",
    2: "Serious",
    3: "Slight",
    "fatal": "Fatal",
    "serious": "Serious",
    "slight": "Slight",
    "Fatal": "Fatal",
    "Serious": "Serious",
    "Slight": "Slight",
}

# Features allowed as inputs for prediction (available at time of accident assessment)
NUMERICAL_FEATURES = [
    "number_of_vehicles",
    "speed_limit",
    "accident_hour",
]

CATEGORICAL_FEATURES = [
    "road_type",
    "road_surface_conditions",
    "weather_conditions",
    "light_conditions",
    "urban_or_rural_area",
    "junction_detail",
    "day_of_week",
]

MODEL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Excluded columns to strictly prevent target/post-outcome leakage & ID overfitting
LEAKAGE_AND_EXCLUDED_COLUMNS = [
    "accident_id",
    "accident_reference",
    "number_of_casualties",  # Post-outcome information
    "casualty_severity",
    "did_police_officer_attend",
    TARGET_COLUMN,
    "latitude",
    "longitude",
    "accident_time",
    "accident_date",
    "date",
]

# Standard allowed categorical choices for validation and UI dropdowns
CATEGORICAL_CHOICES = {
    "road_type": [
        "Single carriageway",
        "Dual carriageway",
        "Roundabout",
        "One way street",
        "Slip road",
    ],
    "road_surface_conditions": [
        "Dry",
        "Wet or damp",
        "Snow",
        "Frost or ice",
        "Flood",
    ],
    "weather_conditions": [
        "Fine no high winds",
        "Raining no high winds",
        "Raining + high winds",
        "Fine + high winds",
        "Snowing",
        "Fog or mist",
        "Other",
    ],
    "light_conditions": [
        "Daylight",
        "Darkness - lights lit",
        "Darkness - lights unlit",
        "Darkness - no lighting",
    ],
    "urban_or_rural_area": [
        "Urban",
        "Rural",
    ],
    "junction_detail": [
        "Not at junction",
        "T or staggered junction",
        "Crossroads",
        "Roundabout",
        "Private drive",
        "Other junction",
    ],
    "day_of_week": [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
}

