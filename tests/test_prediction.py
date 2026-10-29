"""Automated tests for prediction inference engine and schema enforcement."""

from pathlib import Path
import pytest
import pandas as pd

from src.models.predict import (
    load_model,
    predict_single,
    predict_batch,
    validate_input_dict,
    ModelArtifactNotFoundError,
    InvalidPredictionInputError,
)
from src.utils.config import MODEL_ARTIFACT_PATH, TARGET_CLASSES


@pytest.fixture
def valid_record():
    """Valid single collision circumstance record."""
    return {
        "number_of_vehicles": 2,
        "speed_limit": 40,
        "accident_hour": 18,
        "day_of_week": "Friday",
        "road_type": "Single carriageway",
        "road_surface_conditions": "Dry",
        "weather_conditions": "Fine no high winds",
        "light_conditions": "Daylight",
        "urban_or_rural_area": "Urban",
        "junction_detail": "T or staggered junction",
    }


def test_validate_input_dict_valid(valid_record):
    """Test validation succeeds on a properly formatted record."""
    validated = validate_input_dict(valid_record)
    assert validated["number_of_vehicles"] == 2
    assert validated["speed_limit"] == 40
    assert validated["accident_hour"] == 18


def test_validate_input_dict_invalid_speed(valid_record):
    """Test validation rejects invalid speed limit."""
    bad_record = valid_record.copy()
    bad_record["speed_limit"] = -10
    with pytest.raises(InvalidPredictionInputError, match="speed_limit"):
        validate_input_dict(bad_record)


def test_validate_input_dict_invalid_hour(valid_record):
    """Test validation rejects out-of-range accident hour."""
    bad_record = valid_record.copy()
    bad_record["accident_hour"] = 28
    with pytest.raises(InvalidPredictionInputError, match="accident_hour"):
        validate_input_dict(bad_record)


def test_load_model_missing_artifact():
    """Test load_model raises ModelArtifactNotFoundError for non-existent file."""
    fake_path = Path("non_existent_folder/missing_model.joblib")
    with pytest.raises(ModelArtifactNotFoundError, match="not found"):
        load_model(artifact_path=fake_path, force_reload=True)


def test_predict_single_schema(valid_record):
    """Test single prediction returns required response keys and valid severity."""
    if not MODEL_ARTIFACT_PATH.is_file():
        pytest.skip("Model artifact not yet trained.")

    res = predict_single(valid_record)
    assert res["status"] == "success"
    assert res["predicted_severity"] in TARGET_CLASSES
    assert isinstance(res["class_probabilities"], dict)
    assert len(res["class_probabilities"]) > 0
    assert 0.0 <= res["confidence"] <= 1.0


def test_predict_batch_df(valid_record):
    """Test batch prediction appends predictions and confidence columns."""
    if not MODEL_ARTIFACT_PATH.is_file():
        pytest.skip("Model artifact not yet trained.")

    batch_df = pd.DataFrame([valid_record, valid_record])
    results = predict_batch(batch_df)

    assert "predicted_severity" in results.columns
    assert len(results) == 2
    assert results["predicted_severity"].iloc[0] in TARGET_CLASSES
    if "prediction_confidence" in results.columns:
        assert 0.0 <= results["prediction_confidence"].iloc[0] <= 1.0
