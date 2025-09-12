#!/usr/bin/env python3
"""
FastAPI web service for real-time anomaly detection.
Provides REST endpoints for model inference and management.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import io
import logging
from pathlib import Path
import uvicorn
from datetime import datetime

# Import custom modules
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.model_serialization import ModelRegistry
from utils.config_manager import ConfigManager, load_default_config
from utils.preprocessing import interpolate_missing, zscore, butter_lowpass
from utils.feature_engineering import PhysiologicalFeatureExtractor, build_feature_matrix


# Initialize FastAPI app
app = FastAPI(
    title="Physiological Anomaly Detection API",
    description="Real-time anomaly detection for physiological signals (EDA, HR)",
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

# Global variables
model_registry = ModelRegistry()
config_manager = ConfigManager()
logger = logging.getLogger(__name__)

# Load default configuration
try:
    config = load_default_config()
    config_manager.config = config
    config_manager.setup_logging()
except Exception as e:
    logger.warning(f"Could not load configuration: {e}")
    config = None


# Pydantic models for API
class PhysiologicalData(BaseModel):
    """Input data for anomaly detection."""

    timestamp: List[float] = Field(..., description="Timestamps in seconds")
    eda: List[float] = Field(..., description="EDA values (microsiemens)")
    hr: List[float] = Field(..., description="Heart rate values (BPM)")

    class Config:
        schema_extra = {
            "example": {
                "timestamp": [0.0, 0.25, 0.5, 0.75, 1.0],
                "eda": [2.1, 2.3, 2.2, 2.4, 2.1],
                "hr": [72, 74, 73, 75, 72],
            }
        }


class AnomalyResult(BaseModel):
    """Anomaly detection result."""

    anomaly_detected: bool = Field(..., description="Whether anomaly was detected")
    anomaly_score: float = Field(..., description="Anomaly score (higher = more anomalous)")
    confidence: float = Field(..., description="Confidence in prediction")
    timestamp: str = Field(..., description="Timestamp of prediction")

    class Config:
        schema_extra = {
            "example": {
                "anomaly_detected": True,
                "anomaly_score": 0.85,
                "confidence": 0.92,
                "timestamp": "2024-01-01T12:00:00Z",
            }
        }


class BatchAnomalyResult(BaseModel):
    """Batch anomaly detection result."""

    results: List[AnomalyResult] = Field(..., description="Individual results for each window")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")

    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "anomaly_detected": False,
                        "anomaly_score": 0.23,
                        "confidence": 0.95,
                        "timestamp": "2024-01-01T12:00:00Z",
                    }
                ],
                "summary": {
                    "total_windows": 10,
                    "anomalies_detected": 1,
                    "anomaly_rate": 0.1,
                    "avg_score": 0.35,
                },
            }
        }


class ModelInfo(BaseModel):
    """Model information."""

    model_name: str
    version: str
    model_id: str
    is_production: bool
    registered_timestamp: str
    performance_metrics: Optional[Dict[str, float]] = None


class PredictionRequest(BaseModel):
    """Prediction request configuration."""

    model_name: Optional[str] = Field(None, description="Model to use (default: production)")
    model_version: Optional[str] = Field("production", description="Model version")
    threshold: Optional[float] = Field(None, description="Custom threshold")
    use_features: bool = Field(True, description="Use advanced features")
    preprocessing: Dict[str, Any] = Field(default_factory=dict, description="Preprocessing options")


# Dependency functions
async def get_model(model_name: str = "ensemble", version: str = "production"):
    """Get model from registry."""
    try:
        model, metadata = model_registry.get_model(model_name, version)
        return model, metadata
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model not found: {e}")


async def preprocess_data(
    data: PhysiologicalData, preprocessing_config: Dict[str, Any] = None
) -> np.ndarray:
    """Preprocess physiological data."""
    if preprocessing_config is None:
        preprocessing_config = {}

    # Convert to pandas series (handle mismatched lengths gracefully)
    try:
        df = pd.DataFrame({"time": data.timestamp, "EDA": data.eda, "HR": data.hr})
    except Exception as e:
        # Invalid/mismatched input lengths
        raise HTTPException(status_code=422, detail=str(e))

    # Interpolate missing values
    eda = interpolate_missing(df["EDA"])
    hr = interpolate_missing(df["HR"])

    # Apply EDA filtering if configured
    if preprocessing_config.get("filter_eda", True):
        cutoff = preprocessing_config.get("eda_cutoff", 1.0)
        # Apply filtering only when enough samples are available; otherwise fallback
        try:
            if len(eda) >= 20:  # heuristic to avoid filtfilt padlen issues on tiny inputs
                eda = pd.Series(butter_lowpass(eda.values, fs=4.0, cutoff=cutoff))
        except Exception as e:
            logger.warning(f"EDA filtering skipped due to short input: {e}")

    # Normalize
    eda_z = zscore(eda)
    hr_z = zscore(hr)

    # Combine signals
    X = np.vstack([eda_z.values, hr_z.values]).T

    return X


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    html_content = """
    <html>
        <head>
            <title>Physiological Anomaly Detection API</title>
        </head>
        <body>
            <h1>Physiological Anomaly Detection API</h1>
            <p>Welcome to the Physiological Anomaly Detection API!</p>
            <h2>Available Endpoints:</h2>
            <ul>
                <li><a href="/docs">API Documentation (Swagger)</a></li>
                <li><a href="/redoc">API Documentation (ReDoc)</a></li>
                <li><a href="/health">Health Check</a></li>
                <li><a href="/models">List Models</a></li>
            </ul>
            <h2>Quick Start:</h2>
            <p>1. Upload your data using <code>POST /predict</code></p>
            <p>2. Check model status using <code>GET /models</code></p>
            <p>3. View real-time monitoring at <code>GET /monitor</code></p>
        </body>
    </html>
    """
    return html_content


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "models_available": len(model_registry.list_models()),
    }


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models."""
    models_df = model_registry.list_models()

    if models_df.empty:
        return []

    models = []
    for _, row in models_df.iterrows():
        model_info = ModelInfo(
            model_name=row["model_name"],
            version=row["version"],
            model_id=row["model_id"],
            is_production=row["is_production"],
            registered_timestamp=row["registered_timestamp"],
        )

        # Add performance metrics if available
        perf_metrics = {}
        for metric in ["roc_auc", "f1", "precision", "recall"]:
            if metric in row and pd.notna(row[metric]):
                perf_metrics[metric] = float(row[metric])

        if perf_metrics:
            model_info.performance_metrics = perf_metrics

        models.append(model_info)

    return models


@app.post("/predict/single", response_model=AnomalyResult)
async def predict_single(data: PhysiologicalData):
    """Predict anomaly for a single time window."""
    try:
        # Use production by default to ensure 404 when no production model is set
        model, metadata = await get_model("ensemble", "production")

        # Preprocess data
        X = await preprocess_data(data, {})

        if len(X) < 5:  # Need minimum samples
            raise HTTPException(status_code=400, detail="Insufficient data points")

        # Extract features
        use_features = True
        if use_features:
            try:
                from utils.feature_engineering import build_feature_matrix

                window_size = min(len(X), int(30 * 4))  # 30 seconds at 4Hz
                X_features, _ = build_feature_matrix(X, window_size, window_size)

                if len(X_features) > 0:
                    X = X_features[0:1]  # Use first window
                else:
                    # Fallback to raw data
                    X = X.flatten().reshape(1, -1)
            except Exception as e:
                logger.warning(f"Feature extraction failed, using raw data: {e}")
                X = X.flatten().reshape(1, -1)
        else:
            X = X.flatten().reshape(1, -1)

        # Get anomaly score
        if hasattr(model, "reconstruction_error"):
            score = model.reconstruction_error(X)[0]
        elif hasattr(model, "anomaly_score"):
            score = model.anomaly_score(X)[0]
        elif hasattr(model, "score"):
            score = model.score(X)[0]
        else:
            raise HTTPException(status_code=500, detail="Model doesn't support scoring")

        # Determine threshold
        threshold = metadata.get("threshold", np.percentile([score], 95))

        # Make prediction
        is_anomaly = score >= threshold

        # Calculate confidence (simplified)
        confidence = min(0.99, max(0.51, abs(score - threshold) / threshold + 0.5))

        return AnomalyResult(
            anomaly_detected=bool(is_anomaly),
            anomaly_score=float(score),
            confidence=float(confidence),
            timestamp=datetime.now().isoformat(),
        )

    except HTTPException as e:
        # Preserve intended HTTP status codes (e.g., 404 for missing model, 422 for validation)
        raise e
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchAnomalyResult)
async def predict_batch(
    data: PhysiologicalData, request_config: PredictionRequest = PredictionRequest()
):
    """Predict anomalies for multiple time windows."""
    try:
        model, metadata = await get_model(
            request_config.model_name or "ensemble", request_config.model_version
        )

        # Preprocess data
        X = await preprocess_data(data, request_config.preprocessing)

        if len(X) < 30:  # Need minimum samples
            raise HTTPException(status_code=400, detail="Insufficient data for batch processing")

        # Extract features using sliding windows
        if request_config.use_features:
            try:
                window_size = int(30 * 4)  # 30 seconds at 4Hz
                step_size = int(window_size * 0.5)  # 50% overlap
                X_features, _ = build_feature_matrix(X, window_size, step_size)

                if len(X_features) == 0:
                    raise ValueError("No windows extracted")

                X_processed = X_features
            except Exception as e:
                logger.warning(f"Feature extraction failed: {e}")
                # Fallback to simple windowing
                window_size = min(len(X) // 4, 120)
                X_processed = np.array(
                    [
                        X[i : i + window_size].flatten()
                        for i in range(0, len(X) - window_size + 1, window_size // 2)
                    ]
                )
        else:
            # Simple windowing of raw data
            window_size = min(len(X) // 4, 120)
            X_processed = np.array(
                [
                    X[i : i + window_size].flatten()
                    for i in range(0, len(X) - window_size + 1, window_size // 2)
                ]
            )

        # Get anomaly scores
        if hasattr(model, "reconstruction_error"):
            scores = model.reconstruction_error(X_processed)
        elif hasattr(model, "anomaly_score"):
            scores = model.anomaly_score(X_processed)
        elif hasattr(model, "score"):
            scores = model.score(X_processed)
        else:
            raise HTTPException(status_code=500, detail="Model doesn't support scoring")

        # Determine threshold
        threshold = request_config.threshold
        if threshold is None:
            threshold = metadata.get("threshold", np.percentile(scores, 95))

        # Make predictions
        predictions = scores >= threshold

        # Create results
        results = []
        base_time = datetime.now()

        for i, (score, is_anomaly) in enumerate(zip(scores, predictions)):
            confidence = min(0.99, max(0.51, abs(score - threshold) / threshold + 0.5))

            results.append(
                AnomalyResult(
                    anomaly_detected=bool(is_anomaly),
                    anomaly_score=float(score),
                    confidence=float(confidence),
                    timestamp=(base_time).isoformat(),
                )
            )

        # Summary statistics
        summary = {
            "total_windows": len(results),
            "anomalies_detected": int(np.sum(predictions)),
            "anomaly_rate": float(np.mean(predictions)),
            "avg_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "min_score": float(np.min(scores)),
            "threshold_used": float(threshold),
        }

        return BatchAnomalyResult(results=results, summary=summary)

    except HTTPException as e:
        # Preserve validation and not-found errors
        raise e
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/upload")
async def predict_upload(
    file: UploadFile = File(...), request_config: PredictionRequest = PredictionRequest()
):
    """Upload CSV file and predict anomalies."""
    try:
        # Read uploaded file
        content = await file.read()

        # Parse CSV
        try:
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid CSV file")

        # Validate columns
        required_cols = ["time", "EDA", "HR"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_cols}")

        # Convert to PhysiologicalData
        data = PhysiologicalData(
            timestamp=df["time"].tolist(), eda=df["EDA"].tolist(), hr=df["HR"].tolist()
        )

        # Use batch prediction
        result = await predict_batch(data, request_config)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitor")
async def monitoring_dashboard():
    """Simple monitoring dashboard."""
    html_content = """
    <html>
        <head>
            <title>Anomaly Detection Monitor</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Real-time Anomaly Detection Monitor</h1>
            <div id="status">
                <h2>System Status</h2>
                <p>API Status: <span style="color: green;">Online</span></p>
                <p>Models Available: <span id="model-count">Loading...</span></p>
                <p>Last Updated: <span id="last-update">Loading...</span></p>
            </div>
            
            <script>
                async function updateStatus() {
                    try {
                        const response = await fetch('/models');
                        const models = await response.json();
                        document.getElementById('model-count').textContent = models.length;
                        document.getElementById('last-update').textContent = new Date().toLocaleString();
                    } catch (error) {
                        console.error('Error updating status:', error);
                    }
                }
                
                // Update status every 30 seconds
                updateStatus();
                setInterval(updateStatus, 30000);
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/models/register")
async def register_model(model_name: str, version: str, is_production: bool = False):
    """Register a new model (placeholder - requires actual model file)."""
    # This is a placeholder endpoint for model registration
    # In practice, you'd upload model files and register them
    return {
        "message": "Model registration endpoint - implementation depends on deployment strategy",
        "model_name": model_name,
        "version": version,
        "is_production": is_production,
    }


# Background tasks
@app.on_event("startup")
async def startup_event():
    """Initialize API on startup."""
    logger.info("Starting Physiological Anomaly Detection API")

    # Setup directories
    if config:
        try:
            config_manager.setup_output_directories()
        except Exception as e:
            logger.warning(f"Could not setup directories: {e}")

    # Log available models
    models_df = model_registry.list_models()
    logger.info(f"Loaded {len(models_df)} models")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Physiological Anomaly Detection API")


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    """Handle FileNotFoundError exceptions."""
    return JSONResponse(status_code=404, content={"detail": str(exc)})


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Create FastAPI app with custom configuration."""
    global config, config_manager

    if config_path:
        config_manager = ConfigManager(config_path)
        config = config_manager.config
        config_manager.setup_logging()

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Anomaly Detection API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    # Load configuration if provided
    if args.config:
        app = create_app(args.config)

    # Run server
    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload, log_level="info")
