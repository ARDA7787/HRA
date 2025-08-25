#!/usr/bin/env python3
"""
Unit tests for anomaly detection models.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.isolation_forest import IFAnomalyDetector
from models.autoencoder import AEAnomalyDetector
from models.vae_anomaly import VAEAnomalyDetector
from models.lstm_anomaly import LSTMAnomalyDetector
from models.transformer_anomaly import TransformerAnomalyDetector
from models.ensemble import EnsembleAnomalyDetector


class TestIsolationForest:
    """Test cases for Isolation Forest model."""

    def setup_method(self):
        """Set up test fixtures."""
        self.X_train = np.random.randn(100, 10)
        self.X_test = np.random.randn(50, 10)
        self.model = IFAnomalyDetector(contamination=0.1, random_state=42)

    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.model is not None
        assert hasattr(self.model, "fit")
        assert hasattr(self.model, "score")
        assert hasattr(self.model, "predict")

    def test_model_fitting(self):
        """Test model training."""
        self.model.fit(self.X_train)
        # Model should be fitted
        assert hasattr(self.model.model, "decision_function")

    def test_model_scoring(self):
        """Test anomaly scoring."""
        self.model.fit(self.X_train)
        scores = self.model.score(self.X_test)

        assert len(scores) == len(self.X_test)
        assert all(isinstance(score, (int, float)) for score in scores)
        assert not np.any(np.isnan(scores))

    def test_model_prediction(self):
        """Test binary prediction."""
        self.model.fit(self.X_train)
        scores = self.model.score(self.X_test)
        threshold = np.percentile(scores, 90)
        predictions = self.model.predict(self.X_test, threshold)

        assert len(predictions) == len(self.X_test)
        assert all(pred in [0, 1] for pred in predictions)


class TestAutoencoder:
    """Test cases for Autoencoder model."""

    def setup_method(self):
        """Set up test fixtures."""
        self.X_train = np.random.randn(100, 10).astype(np.float32)
        self.X_test = np.random.randn(50, 10).astype(np.float32)
        self.model = AEAnomalyDetector(
            input_dim=10, latent_dim=5, epochs=2, batch_size=32  # Fast for testing
        )

    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.model is not None
        assert hasattr(self.model, "fit")
        assert hasattr(self.model, "reconstruction_error")
        assert hasattr(self.model, "predict")

    def test_model_fitting(self):
        """Test model training."""
        self.model.fit(self.X_train)
        # Model should be in eval mode after training
        assert not self.model.model.training

    def test_reconstruction_error(self):
        """Test reconstruction error computation."""
        self.model.fit(self.X_train)
        errors = self.model.reconstruction_error(self.X_test)

        assert len(errors) == len(self.X_test)
        assert all(error >= 0 for error in errors)
        assert not np.any(np.isnan(errors))

    def test_model_prediction(self):
        """Test binary prediction."""
        self.model.fit(self.X_train)
        errors = self.model.reconstruction_error(self.X_test)
        threshold = np.percentile(errors, 90)
        predictions = self.model.predict(self.X_test, threshold)

        assert len(predictions) == len(self.X_test)
        assert all(pred in [0, 1] for pred in predictions)


class TestVAE:
    """Test cases for VAE model."""

    def setup_method(self):
        """Set up test fixtures."""
        self.X_train = np.random.randn(100, 10).astype(np.float32)
        self.X_test = np.random.randn(50, 10).astype(np.float32)
        self.model = VAEAnomalyDetector(
            input_dim=10, latent_dim=5, epochs=2, batch_size=32  # Fast for testing
        )

    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.model is not None
        assert hasattr(self.model, "fit")
        assert hasattr(self.model, "reconstruction_error")
        assert hasattr(self.model, "anomaly_score")

    def test_model_fitting(self):
        """Test model training."""
        self.model.fit(self.X_train)
        assert not self.model.model.training

    def test_reconstruction_error(self):
        """Test reconstruction error computation."""
        self.model.fit(self.X_train)
        errors = self.model.reconstruction_error(self.X_test)

        assert len(errors) == len(self.X_test)
        assert all(error >= 0 for error in errors)
        assert not np.any(np.isnan(errors))

    def test_anomaly_score(self):
        """Test combined anomaly scoring."""
        self.model.fit(self.X_train)
        scores = self.model.anomaly_score(self.X_test)

        assert len(scores) == len(self.X_test)
        assert all(isinstance(score, (int, float, np.number)) for score in scores)
        assert not np.any(np.isnan(scores))

    def test_sample_generation(self):
        """Test sample generation."""
        self.model.fit(self.X_train)
        generated = self.model.generate_samples(10)

        assert generated.shape == (10, 10)
        assert not np.any(np.isnan(generated))


class TestLSTMAutoencoder:
    """Test cases for LSTM Autoencoder model."""

    def setup_method(self):
        """Set up test fixtures."""
        # LSTM expects windowed data
        self.X_train = np.random.randn(50, 120 * 2).astype(
            np.float32
        )  # 50 windows, 120 timesteps * 2 features
        self.X_test = np.random.randn(20, 120 * 2).astype(np.float32)
        self.model = LSTMAnomalyDetector(
            input_dim=2,
            hidden_dim=32,
            latent_dim=16,
            seq_len=120,
            epochs=2,  # Fast for testing
            batch_size=16,
        )

    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.model is not None
        assert hasattr(self.model, "fit")
        assert hasattr(self.model, "reconstruction_error")

    def test_model_fitting(self):
        """Test model training."""
        self.model.fit(self.X_train)
        assert not self.model.model.training

    def test_reconstruction_error(self):
        """Test reconstruction error computation."""
        self.model.fit(self.X_train)
        errors = self.model.reconstruction_error(self.X_test)

        assert len(errors) == len(self.X_test)
        assert all(error >= 0 for error in errors)
        assert not np.any(np.isnan(errors))


class TestTransformerAnomalyDetector:
    """Test cases for Transformer model."""

    def setup_method(self):
        """Set up test fixtures."""
        # Transformer expects windowed data
        self.X_train = np.random.randn(50, 120 * 2).astype(np.float32)
        self.X_test = np.random.randn(20, 120 * 2).astype(np.float32)
        self.model = TransformerAnomalyDetector(
            input_dim=2,
            d_model=32,
            nhead=4,
            num_layers=2,
            epochs=2,  # Fast for testing
            batch_size=16,
        )

    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.model is not None
        assert hasattr(self.model, "fit")
        assert hasattr(self.model, "reconstruction_error")

    def test_model_fitting(self):
        """Test model training."""
        self.model.fit(self.X_train)
        assert not self.model.model.training

    def test_reconstruction_error(self):
        """Test reconstruction error computation."""
        self.model.fit(self.X_train)
        errors = self.model.reconstruction_error(self.X_test)

        assert len(errors) == len(self.X_test)
        assert all(error >= 0 for error in errors)
        assert not np.any(np.isnan(errors))


class TestEnsemble:
    """Test cases for Ensemble model."""

    def setup_method(self):
        """Set up test fixtures."""
        self.X_train = np.random.randn(100, 10).astype(np.float32)
        self.X_test = np.random.randn(50, 10).astype(np.float32)

        # Create base models
        model1 = IFAnomalyDetector(contamination=0.1, random_state=42)
        model1.fit(self.X_train)

        model2 = AEAnomalyDetector(input_dim=10, epochs=2, batch_size=32)
        model2.fit(self.X_train)

        self.models = [model1, model2]
        self.ensemble = EnsembleAnomalyDetector(
            models=self.models, combination_method="weighted_voting"
        )

    def test_ensemble_initialization(self):
        """Test ensemble initialization."""
        assert len(self.ensemble.models) == 2
        assert hasattr(self.ensemble, "score")
        assert hasattr(self.ensemble, "predict")

    def test_ensemble_scoring(self):
        """Test ensemble scoring."""
        scores = self.ensemble.score(self.X_test)

        assert len(scores) == len(self.X_test)
        assert all(isinstance(score, (int, float)) for score in scores)
        assert not np.any(np.isnan(scores))

    def test_ensemble_prediction(self):
        """Test ensemble prediction."""
        scores = self.ensemble.score(self.X_test)
        threshold = np.percentile(scores, 90)
        predictions = self.ensemble.predict(self.X_test, threshold)

        assert len(predictions) == len(self.X_test)
        assert all(pred in [0, 1] for pred in predictions)


# Integration tests
class TestModelIntegration:
    """Integration tests for the full pipeline."""

    def test_model_compatibility(self):
        """Test that all models have compatible interfaces."""
        X = np.random.randn(50, 10).astype(np.float32)

        # Test isolation forest
        if_model = IFAnomalyDetector(random_state=42)
        if_model.fit(X)
        if_scores = if_model.score(X)

        # Test autoencoder
        ae_model = AEAnomalyDetector(input_dim=10, epochs=1)
        ae_model.fit(X)
        ae_scores = ae_model.reconstruction_error(X)

        # All should return valid scores
        assert len(if_scores) == len(X)
        assert len(ae_scores) == len(X)
        assert not np.any(np.isnan(if_scores))
        assert not np.any(np.isnan(ae_scores))

    def test_ensemble_with_different_models(self):
        """Test ensemble with different types of models."""
        X = np.random.randn(50, 10).astype(np.float32)

        # Create different models
        models = []

        # Isolation Forest
        if_model = IFAnomalyDetector(random_state=42)
        if_model.fit(X)
        models.append(if_model)

        # Autoencoder
        ae_model = AEAnomalyDetector(input_dim=10, epochs=1)
        ae_model.fit(X)
        models.append(ae_model)

        # Create ensemble
        ensemble = EnsembleAnomalyDetector(models=models)
        scores = ensemble.score(X)

        assert len(scores) == len(X)
        assert not np.any(np.isnan(scores))


if __name__ == "__main__":
    pytest.main([__file__])
