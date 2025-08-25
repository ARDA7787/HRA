#!/usr/bin/env python3
"""
Unit tests for utility functions.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.preprocessing import (
    interpolate_missing, zscore, butter_lowpass, sliding_windows, 
    build_window_matrix, inject_synthetic_anomalies
)
from utils.feature_engineering import PhysiologicalFeatureExtractor, build_feature_matrix
from utils.metrics import compute_metrics
from utils.config_manager import ConfigManager, load_default_config


class TestPreprocessing:
    """Test cases for preprocessing functions."""
    
    def test_interpolate_missing(self):
        """Test missing value interpolation."""
        # Create series with missing values
        data = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        result = interpolate_missing(data)
        
        assert not result.isna().any()
        assert len(result) == len(data)
        assert result.iloc[0] == 1.0
        assert result.iloc[2] == 3.0
        assert result.iloc[4] == 5.0
    
    def test_zscore(self):
        """Test z-score normalization."""
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = zscore(data)
        
        assert abs(result.mean()) < 1e-10  # Mean should be ~0
        assert abs(result.std(ddof=0) - 1.0) < 1e-10  # Std should be ~1 (using ddof=0 like zscore function)
        assert len(result) == len(data)
    
    def test_zscore_constant(self):
        """Test z-score with constant values."""
        data = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
        result = zscore(data)
        
        assert all(result == 0.0)
    
    def test_butter_lowpass(self):
        """Test Butterworth low-pass filter."""
        # Create test signal with high frequency noise
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
        
        filtered = butter_lowpass(signal, fs=100, cutoff=5.0)
        
        assert len(filtered) == len(signal)
        assert not np.any(np.isnan(filtered))
        # High frequency component should be reduced
        assert np.var(filtered) < np.var(signal)
    
    def test_sliding_windows(self):
        """Test sliding window generation."""
        data = np.arange(10)
        windows = list(sliding_windows(data, window_size=3, step=2))
        
        assert len(windows) == 4  # (0-3), (2-5), (4-7), (6-9)
        assert windows[0] == (0, 3)
        assert windows[1] == (2, 5)
        assert windows[-1] == (6, 9)
    
    def test_build_window_matrix(self):
        """Test window matrix construction."""
        X = np.random.randn(20, 2)
        window_matrix = build_window_matrix(X, window_size=5, step=3)
        
        assert window_matrix.shape[1] == 5 * 2  # window_size * features
        assert window_matrix.shape[0] > 0
        assert not np.any(np.isnan(window_matrix))
    
    def test_inject_synthetic_anomalies(self):
        """Test synthetic anomaly injection."""
        X = np.random.randn(100, 5)
        X_aug = inject_synthetic_anomalies(X, rate=0.1, scale=2.0, rng=np.random.RandomState(42))
        
        assert X_aug.shape == X.shape
        # Should have some modifications
        assert not np.array_equal(X, X_aug)


class TestFeatureEngineering:
    """Test cases for feature engineering."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = PhysiologicalFeatureExtractor(fs=4.0)
        self.signal = np.random.randn(100)
        self.eda_signal = np.random.randn(100) + 2.0  # Positive EDA values
        self.hr_signal = np.random.randn(100) * 10 + 70  # HR around 70 bpm
    
    def test_statistical_features(self):
        """Test statistical feature extraction."""
        features = self.extractor.extract_statistical_features(self.signal, "test_")
        
        expected_keys = [
            "test_mean", "test_std", "test_min", "test_max", 
            "test_median", "test_iqr", "test_skewness", "test_kurtosis", "test_rms"
        ]
        
        for key in expected_keys:
            assert key in features
            assert isinstance(features[key], (int, float))
            assert not np.isnan(features[key])
    
    def test_frequency_features(self):
        """Test frequency domain feature extraction."""
        features = self.extractor.extract_frequency_features(self.signal, "test_")
        
        expected_keys = [
            "test_dominant_freq", "test_spectral_centroid", "test_spectral_spread",
            "test_power_low", "test_power_high", "test_spectral_entropy"
        ]
        
        for key in expected_keys:
            assert key in features
            assert isinstance(features[key], (int, float))
            assert not np.isnan(features[key])
    
    def test_time_domain_features(self):
        """Test time domain feature extraction."""
        features = self.extractor.extract_time_domain_features(self.signal, "test_")
        
        expected_keys = [
            "test_slope", "test_num_peaks", "test_peak_density",
            "test_zero_crossings", "test_signal_energy", 
            "test_autocorr_lag1", "test_variability"
        ]
        
        for key in expected_keys:
            assert key in features
            assert isinstance(features[key], (int, float))
            assert not np.isnan(features[key])
    
    def test_cross_signal_features(self):
        """Test cross-signal feature extraction."""
        features = self.extractor.extract_cross_signal_features(self.eda_signal, self.hr_signal)
        
        expected_keys = [
            "eda_hr_correlation", "eda_hr_lag_correlation",
            "eda_hr_coherence", "eda_hr_phase_sync"
        ]
        
        for key in expected_keys:
            assert key in features
            assert isinstance(features[key], (int, float))
            assert not np.isnan(features[key])
    
    def test_extract_all_features(self):
        """Test comprehensive feature extraction."""
        features = self.extractor.extract_all_features(self.eda_signal, self.hr_signal)
        
        # Should have features from all categories
        assert len(features) > 20  # Should have many features
        
        # Check for presence of different feature types
        eda_features = [k for k in features.keys() if k.startswith("eda_")]
        hr_features = [k for k in features.keys() if k.startswith("hr_")]
        cross_features = [k for k in features.keys() if "eda_hr" in k]
        
        assert len(eda_features) > 0
        assert len(hr_features) > 0
        assert len(cross_features) > 0
        
        # All features should be numeric
        for value in features.values():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
    
    def test_build_feature_matrix(self):
        """Test feature matrix construction."""
        X = np.random.randn(200, 2)  # 2D signal (EDA, HR)
        window_size = 30
        step = 15
        
        feature_matrix, feature_names = build_feature_matrix(X, window_size, step, fs=4.0)
        
        assert feature_matrix.shape[0] > 0  # Should have windows
        assert feature_matrix.shape[1] > 0  # Should have features
        assert len(feature_names) == feature_matrix.shape[1]
        assert not np.any(np.isnan(feature_matrix))
        assert not np.any(np.isinf(feature_matrix))


class TestMetrics:
    """Test cases for evaluation metrics."""
    
    def test_compute_metrics_binary(self):
        """Test metrics computation with binary classification."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.1, 0.85])
        
        metrics = compute_metrics(y_true, y_scores)
        
        expected_keys = ['precision', 'recall', 'f1', 'roc_auc']
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
            assert 0 <= metrics[key] <= 1  # Metrics should be in [0, 1]
    
    def test_compute_metrics_with_predictions(self):
        """Test metrics with provided predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.1, 0.85])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        
        metrics = compute_metrics(y_true, y_scores, y_pred)
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert metrics['precision'] == 1.0  # Perfect predictions
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
    
    def test_compute_metrics_edge_cases(self):
        """Test metrics with edge cases."""
        # All normal samples
        y_true = np.array([0, 0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4])
        
        metrics = compute_metrics(y_true, y_scores)
        assert 'error' in metrics  # Should return error for invalid case


class TestConfigManager:
    """Test cases for configuration management."""
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_default_config()
        
        assert hasattr(config, 'data')
        assert hasattr(config, 'models')
        assert hasattr(config.data, 'fs')
        assert isinstance(config.data.fs, (int, float))
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization."""
        manager = ConfigManager()
        
        assert hasattr(manager, 'config')
        assert hasattr(manager, 'setup_logging')
        assert hasattr(manager, 'setup_output_directories')
    
    def test_get_enabled_models(self):
        """Test getting enabled models."""
        manager = ConfigManager()
        manager.config = load_default_config()
        
        enabled_models = manager.get_enabled_models()
        assert isinstance(enabled_models, list)


# Performance tests
class TestPerformance:
    """Performance and stress tests."""
    
    def test_feature_extraction_performance(self):
        """Test feature extraction performance with large data."""
        extractor = PhysiologicalFeatureExtractor(fs=4.0)
        
        # Large signals
        eda_signal = np.random.randn(1000)
        hr_signal = np.random.randn(1000)
        
        import time
        start_time = time.time()
        features = extractor.extract_all_features(eda_signal, hr_signal)
        end_time = time.time()
        
        # Should complete in reasonable time (< 1 second)
        assert end_time - start_time < 1.0
        assert len(features) > 0
    
    def test_model_training_performance(self):
        """Test model training performance."""
        from models.isolation_forest import IFAnomalyDetector
        
        # Large dataset
        X = np.random.randn(1000, 20)
        model = IFAnomalyDetector(random_state=42)
        
        import time
        start_time = time.time()
        model.fit(X)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 5.0
        
        # Test scoring performance
        start_time = time.time()
        scores = model.score(X)
        end_time = time.time()
        
        assert end_time - start_time < 1.0
        assert len(scores) == len(X)


if __name__ == "__main__":
    pytest.main([__file__])
