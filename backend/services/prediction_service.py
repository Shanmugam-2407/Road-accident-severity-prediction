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

    def get_dataset_analytics(
        self,
        severity: Optional[str] = None,
        weather: Optional[str] = None,
        surface: Optional[str] = None,
        area: Optional[str] = None,
        speed: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Aggregate statistical breakdowns from the historical dataset with optional multi-factor filters."""
        from src.utils.config import DATASET_PATH, TARGET_COLUMN
        if not DATASET_PATH.is_file():
            return {"error": "Dataset not found."}

        df = pd.read_csv(DATASET_PATH, low_memory=False)

        # Baseline unfiltered total count
        raw_total_records = len(df)

        # Apply optional filters
        if severity and severity != "ALL" and TARGET_COLUMN in df.columns:
            df = df[df[TARGET_COLUMN].astype(str).str.lower() == severity.lower()]

        if weather and weather != "ALL" and "weather_conditions" in df.columns:
            df = df[df["weather_conditions"].astype(str) == weather]

        if surface and surface != "ALL" and "road_surface_conditions" in df.columns:
            df = df[df["road_surface_conditions"].astype(str) == surface]

        if area and area != "ALL" and "urban_or_rural_area" in df.columns:
            df = df[df["urban_or_rural_area"].astype(str) == area]

        if speed is not None and "speed_limit" in df.columns:
            df = df[df["speed_limit"] == speed]

        if "accident_date" in df.columns:
            if start_date:
                df = df[df["accident_date"] >= start_date]
            if end_date:
                df = df[df["accident_date"] <= end_date]

        filtered_total = len(df)

        # Severity distribution
        severity_counts = {"Slight": 0, "Serious": 0, "Fatal": 0}
        if TARGET_COLUMN in df.columns:
            counts = df[TARGET_COLUMN].value_counts().to_dict()
            for k, v in counts.items():
                if k in severity_counts:
                    severity_counts[k] = int(v)

        # Hourly distribution (0-23)
        time_col = None
        for cand in ["accident_time", "time", "accident_hour"]:
            if cand in df.columns:
                time_col = cand
                break

        hour_counts = {h: 0 for h in range(24)}
        if time_col and not df.empty:
            if time_col in ["accident_time", "time"]:
                hours = df[time_col].dropna().astype(str).str.extract(r"^(\d{1,2})")[0]
                hours = pd.to_numeric(hours, errors="coerce").dropna().astype(int)
            else:
                hours = pd.to_numeric(df[time_col], errors="coerce").dropna().astype(int)
            for h, v in hours.value_counts().items():
                if 0 <= int(h) <= 23:
                    hour_counts[int(h)] = int(v)

        # Weather distribution
        weather_counts = df["weather_conditions"].value_counts().to_dict() if ("weather_conditions" in df.columns and not df.empty) else {}

        # Road surface distribution
        surface_counts = df["road_surface_conditions"].value_counts().to_dict() if ("road_surface_conditions" in df.columns and not df.empty) else {}

        # Light condition distribution
        light_counts = df["light_conditions"].value_counts().to_dict() if ("light_conditions" in df.columns and not df.empty) else {}

        # Day of week distribution
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_counts = {d: 0 for d in day_order}
        if "day_of_week" in df.columns and not df.empty:
            counts = df["day_of_week"].value_counts()
            for d in day_order:
                day_counts[d] = int(counts.get(d, 0))

        # Urban/Rural
        area_counts = df["urban_or_rural_area"].value_counts().to_dict() if ("urban_or_rural_area" in df.columns and not df.empty) else {}

        # Monthly trends
        monthly_trends = {}
        if "accident_date" in df.columns and not df.empty:
            try:
                df["_month"] = pd.to_datetime(df["accident_date"], errors="coerce").dt.to_period("M").astype(str)
                month_groups = df.groupby(["_month", TARGET_COLUMN]).size().unstack(fill_value=0)
                for month_str, row in month_groups.iterrows():
                    monthly_trends[str(month_str)] = {
                        "Slight": int(row.get("Slight", 0)),
                        "Serious": int(row.get("Serious", 0)),
                        "Fatal": int(row.get("Fatal", 0)),
                        "Total": int(row.sum()),
                    }
            except Exception:
                pass

        # Speed limit distribution
        speed_counts = {}
        if "speed_limit" in df.columns and not df.empty:
            speed_counts = {int(k): int(v) for k, v in df["speed_limit"].value_counts().sort_index().items()}

        # Road type distribution
        road_type_counts = df["road_type"].value_counts().to_dict() if ("road_type" in df.columns and not df.empty) else {}

        # Casualties count
        total_casualties = int(df["number_of_casualties"].sum()) if ("number_of_casualties" in df.columns and not df.empty) else 0

        return {
            "total_records": filtered_total,
            "raw_total_records": raw_total_records,
            "severity_distribution": severity_counts,
            "hourly_distribution": hour_counts,
            "weather_distribution": weather_counts,
            "surface_distribution": surface_counts,
            "light_distribution": light_counts,
            "day_distribution": day_counts,
            "area_distribution": area_counts,
            "monthly_trends": monthly_trends,
            "speed_distribution": speed_counts,
            "road_type_distribution": road_type_counts,
            "total_casualties": total_casualties,
        }

    def get_hotspot_data(
        self,
        severity: Optional[str] = None,
        speed: Optional[int] = None,
        limit: int = 2500
    ) -> List[Dict[str, Any]]:
        """Retrieve coordinate hotspots for interactive mapping with optional filters."""
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

        if severity and severity != "ALL" and TARGET_COLUMN in geo.columns:
            geo = geo[geo[TARGET_COLUMN].astype(str).str.lower() == severity.lower()]

        if speed is not None and "speed_limit" in geo.columns:
            geo = geo[geo["speed_limit"] == speed]

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
                "date": str(row.get("accident_date", "")),
                "time": str(row.get("accident_time", "")),
            })
        return records


# Global singleton service instance
prediction_service = PredictionService()

