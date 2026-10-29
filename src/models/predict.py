"""Inference engine and prediction utilities for RoadGuard AI."""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import joblib
import numpy as np
import pandas as pd

from src.features.feature_engineering import engineer_features
from src.utils.config import (
    MODEL_ARTIFACT_PATH,
    MODEL_METADATA_PATH,
    MODEL_FEATURE_COLUMNS,
    CATEGORICAL_CHOICES,
)

logger = logging.getLogger(__name__)

# Cached model instance
_CACHED_PIPELINE = None


class ModelArtifactNotFoundError(FileNotFoundError):
    """Raised when the trained model pipeline artifact is missing on disk."""
    pass


class InvalidPredictionInputError(ValueError):
    """Raised when inference input data is missing required attributes or malformed."""
    pass


def load_model(artifact_path: Optional[Path] = None, force_reload: bool = False):
    """
    Load the trained model pipeline with optional in-memory caching.
    
    Raises:
        ModelArtifactNotFoundError: If artifact file does not exist.
    """
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is not None and not force_reload:
        return _CACHED_PIPELINE

    path = artifact_path or MODEL_ARTIFACT_PATH
    if not Path(path).is_file():
        raise ModelArtifactNotFoundError(
            f"Model artifact not found at '{path}'. Please train the model first by running: "
            f"`python -m src.models.train`"
        )

    try:
        pipeline = joblib.load(path)
        _CACHED_PIPELINE = pipeline
        logger.info(f"Loaded model pipeline successfully from {path}")
        return pipeline
    except Exception as e:
        raise RuntimeError(f"Failed to deserialize model pipeline from '{path}': {str(e)}")


def validate_input_dict(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a single record dictionary against expected ranges and categories."""
    if not isinstance(input_data, dict):
        raise InvalidPredictionInputError("Prediction input must be a dictionary.")

    clean_data = input_data.copy()

    # Validate numerical bounds
    if "number_of_vehicles" in clean_data:
        try:
            val = int(clean_data["number_of_vehicles"])
            if val < 1:
                raise InvalidPredictionInputError("number_of_vehicles must be at least 1.")
            clean_data["number_of_vehicles"] = val
        except (ValueError, TypeError):
            raise InvalidPredictionInputError("number_of_vehicles must be a valid integer.")

    if "speed_limit" in clean_data:
        try:
            val = int(clean_data["speed_limit"])
            if val <= 0:
                raise InvalidPredictionInputError("speed_limit must be a positive integer.")
            clean_data["speed_limit"] = val
        except (ValueError, TypeError):
            raise InvalidPredictionInputError("speed_limit must be a valid integer.")

    if "accident_hour" in clean_data and clean_data["accident_hour"] is not None:
        try:
            val = int(clean_data["accident_hour"])
            if not (0 <= val <= 23):
                raise InvalidPredictionInputError("accident_hour must be between 0 and 23.")
            clean_data["accident_hour"] = val
        except (ValueError, TypeError):
            raise InvalidPredictionInputError("accident_hour must be an integer between 0 and 23.")

    return clean_data


def predict_single(
    record: Dict[str, Any],
    pipeline: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Perform severity prediction for a single accident record.
    
    Returns structured result with predicted label, probability distribution,
    and confidence score.
    """
    validated = validate_input_dict(record)
    df_raw = pd.DataFrame([validated])
    features_df = engineer_features(df_raw, is_training=False)

    model = pipeline or load_model()

    prediction_raw = model.predict(features_df)
    predicted_severity = str(prediction_raw[0])

    probabilities = {}
    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_df)[0]
        classes = list(getattr(model, "classes_", []))
        for cls_name, prob_val in zip(classes, probs):
            probabilities[str(cls_name)] = round(float(prob_val), 4)

        if predicted_severity in probabilities:
            confidence = probabilities[predicted_severity]
        elif len(probs) > 0:
            confidence = round(float(np.max(probs)), 4)

    return {
        "status": "success",
        "predicted_severity": predicted_severity,
        "confidence": confidence,
        "class_probabilities": probabilities,
        "features_used": features_df.iloc[0].to_dict(),
    }


def predict_batch(
    df: pd.DataFrame,
    pipeline: Optional[Any] = None
) -> pd.DataFrame:
    """
    Perform batch severity prediction for a DataFrame of accident records.
    
    Appends 'predicted_severity' and class probabilities as columns.
    """
    if df.empty:
        raise InvalidPredictionInputError("Input DataFrame for batch prediction is empty.")

    model = pipeline or load_model()
    features_df = engineer_features(df, is_training=False)

    preds = model.predict(features_df)

    results_df = df.copy()
    results_df["predicted_severity"] = preds

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_df)
        classes = list(getattr(model, "classes_", []))
        for i, cls_name in enumerate(classes):
            results_df[f"prob_{cls_name.lower()}"] = np.round(probs[:, i], 4)
        results_df["prediction_confidence"] = np.round(np.max(probs, axis=1), 4)

    return results_df

