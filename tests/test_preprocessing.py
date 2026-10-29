"""Automated tests for dataset loading, schema validation, and preprocessing."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    load_dataset,
    validate_dataset_schema,
    summarize_dataset_quality,
    prepare_training_data,
    build_preprocessor,
    DatasetValidationError,
)
from src.features.feature_engineering import (
    extract_accident_hour,
    extract_day_of_week,
    normalize_target_severity,
    engineer_features,
)
from src.utils.config import (
    MODEL_FEATURE_COLUMNS,
    LEAKAGE_AND_EXCLUDED_COLUMNS,
    TARGET_COLUMN,
)


@pytest.fixture
def valid_sample_df():
    """Synthetic minimal test fixture for preprocessing."""
    return pd.DataFrame({
        "accident_id": [f"TEST_{i}" for i in range(30)],
        "accident_time": [f"{10 + (i % 12)}:30" for i in range(30)],
        "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"] * 5,
        "number_of_vehicles": [1, 2, 3] * 10,
        "speed_limit": [30, 40, 60] * 10,
        "road_type": ["Single carriageway", "Dual carriageway", "Roundabout"] * 10,
        "road_surface_conditions": ["Dry", "Wet or damp", "Dry"] * 10,
        "weather_conditions": ["Fine no high winds", "Raining no high winds", "Fine no high winds"] * 10,
        "light_conditions": ["Daylight", "Darkness - lights lit", "Daylight"] * 10,
        "urban_or_rural_area": ["Urban", "Rural", "Urban"] * 10,
        "junction_detail": ["Not at junction", "Crossroads", "Roundabout"] * 10,
        "number_of_casualties": [1, 2, 3] * 10,  # Post-outcome leakage column
        "accident_severity": ["Slight", "Serious", "Fatal"] * 10,
    })


def test_load_dataset_valid(valid_sample_df):
    """Test loading a valid CSV file."""
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        valid_sample_df.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        loaded = load_dataset(temp_path)
        assert len(loaded) == 30
        assert "accident_severity" in loaded.columns
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_load_dataset_missing_file():
    """Test loading non-existent CSV raises DatasetValidationError."""
    with pytest.raises(DatasetValidationError, match="not found"):
        load_dataset("non_existent_directory/invalid.csv")


def test_load_dataset_empty_file():
    """Test loading empty CSV raises DatasetValidationError."""
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        f.write("")
        temp_path = f.name

    try:
        with pytest.raises(DatasetValidationError, match="empty"):
            load_dataset(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_validate_dataset_schema_missing_target(valid_sample_df):
    """Test schema validation fails when target column is missing."""
    no_target = valid_sample_df.drop(columns=["accident_severity"])
    res = validate_dataset_schema(no_target, require_target=True)
    assert not res["is_valid"]
    assert any("accident_severity" in err for err in res["errors"])


def test_validate_dataset_schema_invalid_targets():
    """Test schema validation fails when target has unrecognized categories."""
    invalid_target_df = pd.DataFrame({
        "speed_limit": [30, 40],
        "accident_severity": ["UnknownSeverityA", "UnknownSeverityB"],
    })
    res = validate_dataset_schema(invalid_target_df, require_target=True)
    assert not res["is_valid"]
    assert any("no recognized severity categories" in err for err in res["errors"])


def test_target_leakage_prevention(valid_sample_df):
    """Verify post-outcome and ID columns are strictly omitted from engineered features."""
    engineered = engineer_features(valid_sample_df, is_training=True)

    # Feature columns must match MODEL_FEATURE_COLUMNS exactly
    assert list(engineered.columns) == MODEL_FEATURE_COLUMNS

    # Excluded post-outcome columns must not be in model features
    assert "number_of_casualties" not in engineered.columns
    assert "accident_id" not in engineered.columns
    assert "accident_severity" not in engineered.columns


def test_hour_and_day_extraction():
    """Verify feature engineering extracts hour and standardized day correctly."""
    df = pd.DataFrame({
        "accident_time": ["08:15", "19:45", "invalid"],
        "day_of_week": ["monday", "SATURDAY", None],
    })
    hours = extract_accident_hour(df)
    assert hours.iloc[0] == 8
    assert hours.iloc[1] == 19
    assert hours.iloc[2] == 12  # fallback

    days = extract_day_of_week(df)
    assert days.iloc[0] == "Monday"
    assert days.iloc[1] == "Saturday"


def test_prepare_training_data(valid_sample_df):
    """Verify train_test_split produces properly shaped datasets."""
    X_train, X_test, y_train, y_test = prepare_training_data(valid_sample_df, test_size=0.2, random_state=42)

    assert len(X_train) == 24
    assert len(X_test) == 6
    assert list(X_train.columns) == MODEL_FEATURE_COLUMNS
    assert set(y_train.unique()).issubset({"Slight", "Serious", "Fatal"})


def test_summarize_dataset_quality(valid_sample_df):
    """Verify data quality summary reports row counts and missing attributes."""
    summary = summarize_dataset_quality(valid_sample_df)
    assert summary["total_records"] == 30
    assert summary["total_columns"] == 13
    assert summary["duplicate_records"] == 0
    assert "Slight" in summary["severity_distribution"]
