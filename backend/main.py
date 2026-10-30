"""FastAPI application entrypoint for RoadGuard AI."""

import io
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, HTMLResponse

from backend.schemas.prediction_schema import (
    AccidentPredictionRequest,
    AccidentPredictionResponse,
    HealthResponse,
    ModelMetricsResponse,
)
from backend.services.prediction_service import prediction_service
from src.models.predict import ModelArtifactNotFoundError, InvalidPredictionInputError
from src.utils.config import MAX_UPLOAD_SIZE_MB

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RoadGuard AI - Road Accident Severity Prediction API",
    description=(
        "Machine Learning backend API for real-time and batch road collision severity prediction, "
        "model performance auditing, and risk factor analytics."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


INDEX_HTML_PATH = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@app.get("/", summary="Root Endpoint & Web Interface")
def get_root(request: Request):
    """
    Root endpoint. Returns the rich interactive HTML/CSS/JS frontend when accessed
    via a browser, and JSON application metadata when accessed via API/TestClient.
    """
    accept_header = request.headers.get("accept", "")
    if "text/html" in accept_header and "application/json" not in accept_header:
        if INDEX_HTML_PATH.is_file():
            return HTMLResponse(content=INDEX_HTML_PATH.read_text(encoding="utf-8"))

    return {
        "application": "RoadGuard AI - Road Accident Severity Prediction System",
        "version": "1.0.0",
        "tagline": "Understand Risk. Improve Road Safety.",
        "status": "online",
        "documentation": "/docs",
        "web_ui": "/ui",
    }


@app.get("/ui", response_class=HTMLResponse, summary="Web Interface")
def get_ui():
    """Serves the standalone inline HTML/CSS/JavaScript RoadGuard AI frontend."""
    if INDEX_HTML_PATH.is_file():
        return HTMLResponse(content=INDEX_HTML_PATH.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Frontend index.html not found.")


@app.get("/analytics/summary", summary="Get Historical Analytics Summary")
def get_analytics_summary(
    severity: Optional[str] = None,
    weather: Optional[str] = None,
    surface: Optional[str] = None,
    area: Optional[str] = None,
    speed: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve aggregate historical distributions for the dashboard and analytics charts with optional filtering."""
    return prediction_service.get_dataset_analytics(
        severity=severity,
        weather=weather,
        surface=surface,
        area=area,
        speed=speed,
        start_date=start_date,
        end_date=end_date,
    )


@app.get("/analytics/hotspots", summary="Get Geographic Accident Hotspots")
def get_analytics_hotspots(
    severity: Optional[str] = None,
    speed: Optional[int] = None,
    limit: int = 2500,
) -> List[Dict[str, Any]]:
    """Retrieve validated coordinate clusters for the interactive map with optional filters."""
    return prediction_service.get_hotspot_data(severity=severity, speed=speed, limit=limit)


@app.get("/health", response_model=HealthResponse, summary="Health & Model Readiness Check")
def get_health() -> HealthResponse:
    """
    Check API backend operational status and verify that the machine learning
    pipeline artifact is loaded and ready for predictions.
    """
    health_info = prediction_service.check_health()
    if not health_info["prediction_ready"]:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unavailable",
                "app_name": "RoadGuard AI",
                "version": "1.0.0",
                "model_loaded": False,
                "model_name": None,
                "prediction_ready": False,
                "details": health_info.get("details", "Model artifact not available."),
            },
        )

    return HealthResponse(
        status="healthy",
        app_name="RoadGuard AI",
        version="1.0.0",
        model_loaded=health_info["model_loaded"],
        model_name=health_info.get("model_name"),
        prediction_ready=health_info["prediction_ready"],
        details=health_info.get("details"),
    )


@app.post(
    "/predict/severity",
    response_model=AccidentPredictionResponse,
    summary="Predict Single Accident Severity",
)
def predict_severity(payload: AccidentPredictionRequest) -> AccidentPredictionResponse:
    """
    Predict accident severity category ('Slight', 'Serious', 'Fatal') and return
    class probability distribution for a single collision event.
    """
    try:
        result = prediction_service.predict_severity(payload.model_dump())
        return AccidentPredictionResponse(
            status="success",
            predicted_severity=result["predicted_severity"],
            confidence=result.get("confidence"),
            class_probabilities=result.get("class_probabilities", {}),
            model_identifier=result.get("model_identifier", "RoadGuard ML Pipeline"),
        )
    except ModelArtifactNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except InvalidPredictionInputError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(f"Unexpected prediction failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@app.post("/predict/batch", summary="Batch Severity Prediction via CSV Upload")
async def predict_batch_endpoint(file: UploadFile = File(...)):
    """
    Upload a CSV of road accident circumstances, predict severity for each row,
    and download the enriched dataset containing predictions and confidence scores.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only .csv files are supported.",
        )

    content = await file.read()
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds maximum limit of {MAX_UPLOAD_SIZE_MB} MB.",
        )

    try:
        results_df, summary = prediction_service.process_batch_csv(content, file.filename)
    except ModelArtifactNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except InvalidPredictionInputError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing error: {str(e)}",
        )

    # Convert results to CSV byte stream
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=predicted_{file.filename}",
            "X-Total-Records": str(summary["total_records"]),
        },
    )


@app.get("/model/metrics", response_model=ModelMetricsResponse, summary="Get Model Evaluation Metrics")
def get_model_metrics() -> ModelMetricsResponse:
    """
    Retrieve audited model evaluation metrics, baseline comparison records,
    and permutation feature importance from actual model validation.
    """
    try:
        metrics_data = prediction_service.get_evaluation_metrics()
        return ModelMetricsResponse(**metrics_data)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(f"Failed to fetch model metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics retrieval error: {str(e)}",
        )

