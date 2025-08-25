#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_physiological_data():
    """Generate sample physiological data for testing."""
    np.random.seed(42)
    
    # Generate 10 minutes of data at 4 Hz
    n_samples = 10 * 60 * 4  # 2400 samples
    time = np.arange(n_samples) / 4.0
    
    # Generate realistic EDA signal (2-5 μS range)
    eda_base = 3.0
    eda_trend = 0.5 * np.sin(2 * np.pi * time / 300)  # 5-minute cycles
    eda_noise = 0.1 * np.random.randn(n_samples)
    eda = eda_base + eda_trend + eda_noise
    
    # Generate realistic HR signal (60-90 bpm range)
    hr_base = 75.0
    hr_trend = 10.0 * np.sin(2 * np.pi * time / 120)  # 2-minute cycles
    hr_noise = 2.0 * np.random.randn(n_samples)
    hr = hr_base + hr_trend + hr_noise
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    labels = np.zeros(n_samples, dtype=int)
    labels[anomaly_indices] = 1
    
    # Make anomalies more extreme
    eda[anomaly_indices] += np.random.randn(len(anomaly_indices)) * 1.0
    hr[anomaly_indices] += np.random.randn(len(anomaly_indices)) * 15.0
    
    return pd.DataFrame({
        'time': time,
        'EDA': eda,
        'HR': hr,
        'label': labels
    })


@pytest.fixture
def sample_csv_file(sample_physiological_data, temp_dir):
    """Create sample CSV file for testing."""
    csv_path = temp_dir / "test_data.csv"
    sample_physiological_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_windowed_data():
    """Generate sample windowed data for model testing."""
    np.random.seed(42)
    
    # Window features (50 windows, 100 features each)
    X = np.random.randn(50, 100).astype(np.float32)
    
    # Binary labels (10% anomalies)
    y = np.zeros(50, dtype=int)
    anomaly_indices = np.random.choice(50, size=5, replace=False)
    y[anomaly_indices] = 1
    
    return X, y


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    from utils.config_manager import Config, DataConfig, ModelConfig
    
    config = Config()
    config.data = DataConfig()
    config.data.fs = 4.0
    config.data.window_sec = 30.0
    config.data.overlap = 0.5
    
    # Add model configs
    config.models = {
        'isolation_forest': ModelConfig(
            enabled=True,
            params={'contamination': 0.1, 'random_state': 42}
        ),
        'autoencoder': ModelConfig(
            enabled=True,
            params={'latent_dim': 32, 'epochs': 2, 'batch_size': 32}
        )
    }
    
    return config


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Suppress warnings during tests
    import warnings
    warnings.filterwarnings('ignore')
    
    # Set environment variables for testing
    import os
    os.environ['ANOMALY_DEVICE'] = 'cpu'
    
    yield
    
    # Cleanup
    if 'ANOMALY_DEVICE' in os.environ:
        del os.environ['ANOMALY_DEVICE']


@pytest.fixture
def sample_model_results():
    """Sample model evaluation results."""
    return {
        'isolation_forest': {
            'roc_auc': 0.87,
            'f1': 0.82,
            'precision': 0.85,
            'recall': 0.79
        },
        'autoencoder': {
            'roc_auc': 0.91,
            'f1': 0.86,
            'precision': 0.88,
            'recall': 0.84
        },
        'ensemble': {
            'roc_auc': 0.94,
            'f1': 0.90,
            'precision': 0.92,
            'recall': 0.88
        }
    }


# Test data generators
def generate_sine_wave(frequency=1.0, duration=10.0, fs=100.0, noise_level=0.1):
    """Generate sine wave with noise for testing."""
    t = np.arange(0, duration, 1/fs)
    signal = np.sin(2 * np.pi * frequency * t)
    noise = noise_level * np.random.randn(len(signal))
    return t, signal + noise


def generate_anomalous_signal(normal_signal, anomaly_rate=0.05, anomaly_scale=3.0):
    """Inject anomalies into a normal signal."""
    signal = normal_signal.copy()
    n_anomalies = int(len(signal) * anomaly_rate)
    anomaly_indices = np.random.choice(len(signal), size=n_anomalies, replace=False)
    
    # Add extreme values at anomaly indices
    signal[anomaly_indices] += anomaly_scale * np.random.randn(n_anomalies)
    
    labels = np.zeros(len(signal), dtype=int)
    labels[anomaly_indices] = 1
    
    return signal, labels


# Performance testing helpers
@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed_time()
        
        def elapsed_time(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time
    
    return Timer()


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )
    config.addinivalue_line(
        "markers", "model: marks tests as model tests"
    )


# Skip tests if dependencies are missing
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available dependencies."""
    try:
        import torch
    except ImportError:
        skip_torch = pytest.mark.skip(reason="PyTorch not available")
        for item in items:
            if "torch" in item.keywords or "pytorch" in item.keywords:
                item.add_marker(skip_torch)
    
    try:
        import plotly
    except ImportError:
        skip_plotly = pytest.mark.skip(reason="Plotly not available")
        for item in items:
            if "plotly" in item.keywords or "visualization" in item.keywords:
                item.add_marker(skip_plotly)
