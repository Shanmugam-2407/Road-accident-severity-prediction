"""Pydantic schemas for RoadGuard AI backend API."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class AccidentPredictionRequest(BaseModel):
    """Input payload for single accident severity prediction."""
    number_of_vehicles: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Number of vehicles involved in the collision.",
        examples=[2]
    )
    speed_limit: int = Field(
        default=30,
        ge=10,
        le=120,
        description="Posted speed limit in mph.",
        examples=[30]
    )
    accident_hour: Optional[int] = Field(
        default=17,
        ge=0,
        le=23,
        description="Hour of day (0-23) when accident occurred.",
        examples=[17]
    )
    day_of_week: str = Field(
        default="Friday",
        description="Day of week.",
        examples=["Friday"]
    )
    road_type: str = Field(
        default="Single carriageway",
        description="Type of road carriageway.",
        examples=["Single carriageway"]
    )
    road_surface_conditions: str = Field(
        default="Wet or damp",
        description="Road surface physical condition.",
        examples=["Wet or damp"]
    )
    weather_conditions: str = Field(
        default="Raining no high winds",
        description="Prevailing weather conditions.",
        examples=["Raining no high winds"]
    )
    light_conditions: str = Field(
        default="Darkness - lights lit",
        description="Natural and street lighting condition.",
        examples=["Darkness - lights lit"]
    )
    urban_or_rural_area: str = Field(
        default="Urban",
        description="Regional urbanization categorization.",
        examples=["Urban"]
    )
    junction_detail: str = Field(
        default="T or staggered junction",
        description="Junction or intersection layout.",
        examples=["T or staggered junction"]
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "number_of_vehicles": 2,
                "speed_limit": 50,
                "accident_hour": 22,
                "day_of_week": "Saturday",
                "road_type": "Dual carriageway",
                "road_surface_conditions": "Wet or damp",
                "weather_conditions": "Raining + high winds",
                "light_conditions": "Darkness - lights unlit",
                "urban_or_rural_area": "Rural",
                "junction_detail": "Not at junction"
            }
        }
    }


class AccidentPredictionResponse(BaseModel):
    """Response payload returning predicted accident severity and confidence."""
    status: str = Field(description="Execution status of prediction.")
    predicted_severity: str = Field(description="Predicted severity class: Slight, Serious, or Fatal.")
    confidence: Optional[float] = Field(default=None, description="Confidence score of predicted class.")
    class_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Estimated probability distribution over severity classes."
    )
    model_identifier: str = Field(description="Identifier or name of trained model pipeline.")
    disclaimer: str = Field(
        default=(
            "RoadGuard AI provides statistical severity estimates based on historical accident patterns. "
            "It does not guarantee real-world outcomes and is not an officially certified emergency-dispatch system."
        ),
        description="Responsible use disclaimer."
    )


class HealthResponse(BaseModel):
    """Backend and model health readiness status."""
    status: str
    app_name: str
    version: str
    model_loaded: bool
    model_name: Optional[str] = None
    prediction_ready: bool
    details: Optional[str] = None


class ModelMetricsResponse(BaseModel):
    """Evaluation summary and performance metrics response."""
    status: str
    selected_model: str
    selection_criterion: str
    metrics: Dict[str, Any]
    model_comparison: List[Dict[str, Any]]
    feature_importance: Dict[str, Any]
    metadata: Dict[str, Any]
