"""Backend business logic and prediction service integration."""

import io
import json
import logging
from typing import Dict, Any, Tuple, Optional, List
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

    def get_dataset_analytics(self) -> Dict[str, Any]:
        """Aggregate statistical breakdowns from the historical dataset."""
        from src.utils.config import DATASET_PATH, TARGET_COLUMN
        if not DATASET_PATH.is_file():
            return {"error": "Dataset not found."}

        df = pd.read_csv(DATASET_PATH, low_memory=False)

        # Severity distribution
        severity_counts = df[TARGET_COLUMN].value_counts().to_dict() if TARGET_COLUMN in df.columns else {}

        # Hourly distribution
        time_col = None
        for cand in ["accident_time", "time", "accident_hour"]:
            if cand in df.columns:
                time_col = cand
                break

        hour_counts = {}
        if time_col:
            if time_col in ["accident_time", "time"]:
                hours = df[time_col].dropna().astype(str).str.extract(r"^(\d{1,2})")[0]
                hours = pd.to_numeric(hours, errors="coerce").dropna().astype(int)
            else:
                hours = pd.to_numeric(df[time_col], errors="coerce").dropna().astype(int)
            hour_counts = {int(k): int(v) for k, v in hours.value_counts().sort_index().items()}

        # Weather distribution
        weather_counts = df["weather_conditions"].value_counts().to_dict() if "weather_conditions" in df.columns else {}

        # Road surface distribution
        surface_counts = df["road_surface_conditions"].value_counts().to_dict() if "road_surface_conditions" in df.columns else {}

        # Light condition distribution
        light_counts = df["light_conditions"].value_counts().to_dict() if "light_conditions" in df.columns else {}

        # Day of week distribution
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_counts = {}
        if "day_of_week" in df.columns:
            counts = df["day_of_week"].value_counts()
            for d in day_order:
                day_counts[d] = int(counts.get(d, 0))

        # Urban/Rural
        area_counts = df["urban_or_rural_area"].value_counts().to_dict() if "urban_or_rural_area" in df.columns else {}

        return {
            "total_records": len(df),
            "severity_distribution": severity_counts,
            "hourly_distribution": hour_counts,
            "weather_distribution": weather_counts,
            "surface_distribution": surface_counts,
            "light_distribution": light_counts,
            "day_distribution": day_counts,
            "area_distribution": area_counts,
        }

    def get_hotspot_data(self, limit: int = 2500) -> List[Dict[str, Any]]:
        """Retrieve coordinate hotspots for interactive mapping."""
        from src.utils.config import DATASET_PATH, TARGET_COLUMN
        if not DATASET_PATH.is_file():
            return []

        df = pd.read_csv(DATASET_PATH, low_memory=False)
        if "latitude" not in df.columns or "longitude" not in df.columns:
            return []

        geo = df.dropna(subset=["latitude", "longitude"]).copy()
        geo = geo[
            (geo["latitude"] >= -90)
            & (geo["latitude"] <= 90)
            & (geo["longitude"] >= -180)
            & (geo["longitude"] <= 180)
        ]

        if len(geo) > limit:
            geo = geo.sample(n=limit, random_state=42)

        records = []
        for _, row in geo.iterrows():
            records.append({
                "id": str(row.get("accident_id", "")),
                "lat": float(row["latitude"]),
                "lng": float(row["longitude"]),
                "severity": str(row.get(TARGET_COLUMN, "Slight")),
                "speed": int(row.get("speed_limit", 30)),
                "weather": str(row.get("weather_conditions", "")),
                "surface": str(row.get("road_surface_conditions", "")),
                "vehicles": int(row.get("number_of_vehicles", 1)),
            })
        return records


# Global singleton service instance
prediction_service = PredictionService()

