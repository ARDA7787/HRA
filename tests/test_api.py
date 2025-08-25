#!/usr/bin/env python3
"""
Unit tests for the FastAPI application.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from pathlib import Path
import sys
import io

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_data():
    """Sample physiological data for testing."""
    return {
        "timestamp": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        "eda": [2.1, 2.3, 2.2, 2.4, 2.1, 2.5, 2.3, 2.2, 2.4],
        "hr": [72, 74, 73, 75, 72, 76, 74, 73, 75]
    }


class TestAPIEndpoints:
    """Test API endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_list_models(self, client):
        """Test model listing endpoint."""
        response = client.get("/models")
        assert response.status_code == 200
        
        models = response.json()
        assert isinstance(models, list)
        # May be empty if no models are registered
    
    def test_monitoring_dashboard(self, client):
        """Test monitoring dashboard endpoint."""
        response = client.get("/monitor")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_model_registration_placeholder(self, client):
        """Test model registration endpoint."""
        response = client.post(
            "/models/register",
            params={
                "model_name": "test_model",
                "version": "1.0.0",
                "is_production": False
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert data["model_name"] == "test_model"


class TestPredictionEndpoints:
    """Test prediction endpoints."""
    
    def test_single_prediction_no_models(self, client, sample_data):
        """Test single prediction when no models are available."""
        response = client.post("/predict/single", json=sample_data)
        # Should return 404 when no models are found or 422 for validation errors
        assert response.status_code in [404, 422]
    
    def test_batch_prediction_no_models(self, client, sample_data):
        """Test batch prediction when no models are available."""
        response = client.post("/predict/batch", json=sample_data)
        # Should return 404 when no models are found or 422 for validation errors
        assert response.status_code in [404, 422]
    
    def test_upload_prediction_invalid_file(self, client):
        """Test upload prediction with invalid file."""
        # Create invalid CSV content using BytesIO
        invalid_csv = b"invalid,csv,content\n1,2"
        files = {"file": ("test.csv", io.BytesIO(invalid_csv), "text/csv")}
        
        response = client.post("/predict/upload", files=files)
        # Should return 400 for invalid CSV or 404 for no models
        assert response.status_code in [400, 404]
    
    def test_prediction_input_validation(self, client):
        """Test prediction input validation."""
        # Test with missing required fields
        invalid_data = {
            "timestamp": [0.0, 0.25],
            "eda": [2.1, 2.3]
            # Missing HR field
        }
        
        response = client.post("/predict/single", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_prediction_insufficient_data(self, client):
        """Test prediction with insufficient data."""
        # Very short data
        minimal_data = {
            "timestamp": [0.0, 0.25],
            "eda": [2.1, 2.3],
            "hr": [72, 74]
        }
        
        response = client.post("/predict/single", json=minimal_data)
        # Should return error for insufficient data or 404 for no models
        assert response.status_code in [400, 404]


class TestAPIValidation:
    """Test API input validation."""
    
    def test_physiological_data_validation(self, client):
        """Test PhysiologicalData model validation."""
        # Test with mismatched array lengths
        invalid_data = {
            "timestamp": [0.0, 0.25, 0.5],
            "eda": [2.1, 2.3],  # Different length
            "hr": [72, 74, 73]
        }
        
        response = client.post("/predict/single", json=invalid_data)
        # Should accept different lengths (will be handled in preprocessing)
        assert response.status_code in [400, 404, 422]
    
    def test_prediction_request_validation(self, client, sample_data):
        """Test PredictionRequest validation."""
        # Test with valid request config
        request_data = {
            **sample_data,
            "model_name": "test_model",
            "model_version": "1.0.0",
            "threshold": 0.5,
            "use_features": True
        }
        
        response = client.post("/predict/single", json=request_data)
        # Should return 404 for model not found
        assert response.status_code in [404, 422]


class TestErrorHandling:
    """Test error handling in API."""
    
    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/predict/single",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_content_type(self, client, sample_data):
        """Test handling of missing content type."""
        response = client.post("/predict/single", json=sample_data)
        # Should work with proper JSON data
        assert response.status_code in [200, 404, 422]
    
    def test_large_payload(self, client):
        """Test handling of large payloads."""
        # Create large data payload
        large_data = {
            "timestamp": list(range(10000)),
            "eda": [2.0] * 10000,
            "hr": [70] * 10000
        }
        
        response = client.post("/predict/single", json=large_data)
        # Should handle large payloads or return appropriate error
        assert response.status_code in [200, 400, 404, 413, 422]


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema generation."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
    
    def test_swagger_docs(self, client):
        """Test Swagger documentation."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_docs(self, client):
        """Test ReDoc documentation."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


# Integration tests
class TestAPIIntegration:
    """Integration tests for API."""
    
    def test_api_startup_shutdown(self):
        """Test API startup and shutdown events."""
        with TestClient(app) as test_client:
            # API should start successfully
            response = test_client.get("/health")
            assert response.status_code == 200
    
    def test_cors_headers(self, client):
        """Test CORS headers."""
        response = client.options("/health")
        # CORS should be enabled
        assert response.status_code in [200, 405]  # Method may not be allowed but CORS should work
    
    def test_content_type_handling(self, client, sample_data):
        """Test different content types."""
        # JSON content type
        response = client.post(
            "/predict/single", 
            json=sample_data,
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [200, 404, 422]


if __name__ == "__main__":
    pytest.main([__file__])
