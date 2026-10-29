"""Backend business logic and prediction service integration."""

import io
import json
import logging
from typing import Dict, Any, Tuple, Optional
import pandas as pd

from src.models.predict import (
    load_model,
    predict_single,
    predict_batch,
    ModelArtifactNotFoundError,
    InvalidPredictionInputError,
)
from src.utils.config import (
    MODEL_ARTIFACT_PATH,
    MODEL_METADATA_PATH,
    EVALUATION_SUMMARY_PATH,
    MODEL_COMPARISON_PATH,
)

logger = logging.getLogger(__name__)


class PredictionService:
    """Service handling prediction logic, model loading, and metrics retrieval."""

    def __init__(self):
        self._model = None
        self._metadata = None
        self._evaluation_summary = None

    def get_model(self):
        """Retrieve or lazy-load model pipeline."""
        if self._model is None:
            self._model = load_model()
        return self._model

    def get_metadata(self) -> Dict[str, Any]:
        """Load and cache model metadata from disk."""
        if self._metadata is None:
            if MODEL_METADATA_PATH.is_file():
                with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
            else:
                self._metadata = {
                    "model_name": "Unknown",
                    "trained_at": None,
                    "selection_criterion": "Not yet trained",
                }
        return self._metadata

    def check_health(self) -> Dict[str, Any]:
        """Inspect health status and model artifact availability."""
        try:
            model = self.get_model()
            meta = self.get_metadata()
            return {
                "status": "healthy",
                "model_loaded": True,
                "model_name": meta.get("model_name", type(model.named_steps["classifier"]).__name__),
                "prediction_ready": True,
                "details": "Model pipeline is ready for inference.",
            }
        except Exception as e:
            logger.warning(f"Health check warning: {e}")
            return {
                "status": "degraded",
                "model_loaded": False,
                "model_name": None,
                "prediction_ready": False,
                "details": str(e),
            }

    def predict_severity(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict severity for a single record payload."""
        model = self.get_model()
        meta = self.get_metadata()

        res = predict_single(input_data, pipeline=model)
        res["model_identifier"] = meta.get("model_name", "RoadGuard ML Pipeline")
        return res

    def process_batch_csv(self, file_bytes: bytes, filename: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process uploaded CSV file for batch inference.
        
        Returns:
            Tuple of (predicted DataFrame, summary dictionary).
        """
        if not file_bytes:
            raise InvalidPredictionInputError("Uploaded CSV file is empty.")

        try:
            df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
        except Exception as e:
            raise InvalidPredictionInputError(f"Failed to parse CSV file: {str(e)}")

        if df.empty:
            raise InvalidPredictionInputError("Uploaded CSV contains no data rows.")

        total_rows = len(df)
        model = self.get_model()

        # Run batch prediction
        results_df = predict_batch(df, pipeline=model)

        summary = {
            "filename": filename,
            "total_records": total_rows,
            "predictions_summary": results_df["predicted_severity"].value_counts().to_dict(),
        }

        return results_df, summary

    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """Fetch actual model evaluation reports and comparison records."""
        if not EVALUATION_SUMMARY_PATH.is_file():
            raise FileNotFoundError(
                "Evaluation summary not found. Train the model first using `python -m src.models.train`."
            )

        with open(EVALUATION_SUMMARY_PATH, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        meta = self.get_metadata()

        return {
            "status": "success",
            "selected_model": eval_data.get("selected_model", meta.get("model_name", "Random Forest")),
            "selection_criterion": eval_data.get("selection_criterion", ""),
            "metrics": eval_data.get("metrics", {}),
            "model_comparison": eval_data.get("all_models_comparison", []),
            "feature_importance": eval_data.get("feature_importance", {}),
            "metadata": meta,
        }


# Global singleton service instance
prediction_service = PredictionService()
