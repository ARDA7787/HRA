#!/usr/bin/env python3
"""
Configuration management utilities.
Handles loading and validation of configuration files.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class DataConfig:
    """Data configuration."""
    fs: float = 4.0
    window_sec: float = 30.0
    overlap: float = 0.5
    interpolation_method: str = "linear"
    normalization_method: str = "zscore"
    eda: Dict[str, Any] = field(default_factory=lambda: {
        "lowpass_filter": True,
        "cutoff_frequency": 1.0,
        "filter_order": 4
    })
    features: Dict[str, Any] = field(default_factory=lambda: {
        "use_advanced_features": True,
        "statistical_features": True,
        "frequency_features": True,
        "time_domain_features": True,
        "cross_signal_features": True
    })


@dataclass
class ModelConfig:
    """Model configuration."""
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    cross_validation: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "method": "stratified",
        "n_splits": 5,
        "gap": 0
    })
    subject_independent: Dict[str, bool] = field(default_factory=lambda: {
        "enabled": True
    })
    uncertainty_quantification: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "n_bootstrap": 100,
        "bootstrap_ratio": 0.8
    })
    metrics: list = field(default_factory=lambda: [
        "roc_auc", "avg_precision", "f1", "precision", "recall", "accuracy"
    ])


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    interactive: bool = True
    save_static: bool = True
    theme: str = "plotly_white"
    plots: Dict[str, bool] = field(default_factory=lambda: {
        "time_series": True,
        "roc_curves": True,
        "confusion_matrix": True,
        "feature_importance": True,
        "latent_space": True,
        "score_distribution": True,
        "dashboard": True
    })


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/anomaly_detection.log"
    console: bool = True


@dataclass
class OutputConfig:
    """Output configuration."""
    base_dir: str = "outputs"
    save_models: bool = True
    save_predictions: bool = True
    save_visualizations: bool = True
    save_reports: bool = True
    models_dir: str = "models"
    plots_dir: str = "plots"
    reports_dir: str = "reports"
    predictions_dir: str = "predictions"


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    ensemble: Dict[str, Any] = field(default_factory=dict)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    paths: Dict[str, str] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    anomaly_injection: Dict[str, Any] = field(default_factory=dict)
    realtime: Dict[str, Any] = field(default_factory=dict)
    api: Dict[str, Any] = field(default_factory=dict)
    database: Dict[str, Any] = field(default_factory=dict)
    experiment_tracking: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Configuration manager for loading and validating configurations."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: Union[str, Path]) -> Config:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            self.config = self._dict_to_config(config_dict)
            self.config_path = config_path
            
            # Validate configuration
            self._validate_config()
            
            self.logger.info(f"Configuration loaded successfully from: {config_path}")
            return self.config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
    
    def save_config(self, config_path: Union[str, Path], config: Optional[Config] = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path to save configuration
            config: Configuration to save (defaults to current config)
        """
        if config is None:
            config = self.config
        
        config_dict = self._config_to_dict(config)
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to: {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model configuration
        """
        if model_name not in self.config.models:
            self.logger.warning(f"Model '{model_name}' not found in configuration")
            return ModelConfig(enabled=False)
        
        return self.config.models[model_name]
    
    def update_model_config(self, model_name: str, params: Dict[str, Any]) -> None:
        """
        Update configuration for a specific model.
        
        Args:
            model_name: Name of the model
            params: Parameters to update
        """
        if model_name not in self.config.models:
            self.config.models[model_name] = ModelConfig()
        
        self.config.models[model_name].params.update(params)
        self.logger.info(f"Updated configuration for model: {model_name}")
    
    def get_enabled_models(self) -> List[str]:
        """Get list of enabled models."""
        return [name for name, config in self.config.models.items() if config.enabled]
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        log_config = self.config.logging
        
        # Create logs directory
        log_file = Path(log_config.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config.level.upper()),
            format=log_config.format,
            handlers=[]
        )
        
        logger = logging.getLogger()
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_config.format))
        logger.addHandler(file_handler)
        
        # Console handler
        if log_config.console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_config.format))
            logger.addHandler(console_handler)
    
    def setup_output_directories(self) -> Dict[str, Path]:
        """
        Setup output directories based on configuration.
        
        Returns:
            Dictionary of directory paths
        """
        output_config = self.config.output
        base_dir = Path(output_config.base_dir)
        
        directories = {
            'base': base_dir,
            'models': base_dir / output_config.models_dir,
            'plots': base_dir / output_config.plots_dir,
            'reports': base_dir / output_config.reports_dir,
            'predictions': base_dir / output_config.predictions_dir
        }
        
        # Create directories
        for name, path in directories.items():
            path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {path}")
        
        return directories
    
    def get_device_config(self, model_name: str) -> str:
        """
        Get device configuration for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Device string ('cuda', 'cpu')
        """
        model_config = self.get_model_config(model_name)
        device = model_config.params.get('device', 'auto')
        
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        return device
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> Config:
        """Convert dictionary to Config object."""
        config = Config()
        
        # Data configuration
        if 'data' in config_dict:
            data_dict = config_dict['data']
            config.data = DataConfig(
                fs=data_dict.get('fs', 4.0),
                window_sec=data_dict.get('window_sec', 30.0),
                overlap=data_dict.get('overlap', 0.5),
                interpolation_method=data_dict.get('interpolation_method', 'linear'),
                normalization_method=data_dict.get('normalization_method', 'zscore'),
                eda=data_dict.get('eda', {}),
                features=data_dict.get('features', {})
            )
        
        # Model configurations
        if 'models' in config_dict:
            for model_name, model_dict in config_dict['models'].items():
                config.models[model_name] = ModelConfig(
                    enabled=model_dict.get('enabled', True),
                    params=model_dict.get('params', {})
                )
        
        # Other configurations
        for key in ['ensemble', 'evaluation', 'visualization', 'logging', 'output',
                   'paths', 'performance', 'thresholds', 'anomaly_injection',
                   'realtime', 'api', 'database', 'experiment_tracking']:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        return config
    
    def _config_to_dict(self, config: Config) -> Dict[str, Any]:
        """Convert Config object to dictionary."""
        config_dict = {}
        
        # Data configuration
        config_dict['data'] = {
            'fs': config.data.fs,
            'window_sec': config.data.window_sec,
            'overlap': config.data.overlap,
            'interpolation_method': config.data.interpolation_method,
            'normalization_method': config.data.normalization_method,
            'eda': config.data.eda,
            'features': config.data.features
        }
        
        # Model configurations
        config_dict['models'] = {}
        for model_name, model_config in config.models.items():
            config_dict['models'][model_name] = {
                'enabled': model_config.enabled,
                'params': model_config.params
            }
        
        # Other configurations
        for key in ['ensemble', 'evaluation', 'visualization', 'logging', 'output',
                   'paths', 'performance', 'thresholds', 'anomaly_injection',
                   'realtime', 'api', 'database', 'experiment_tracking']:
            if hasattr(config, key):
                config_dict[key] = getattr(config, key)
        
        return config_dict
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Validate data configuration
        if self.config.data.fs <= 0:
            raise ValueError("Sampling frequency must be positive")
        
        if not 0 < self.config.data.overlap < 1:
            raise ValueError("Overlap must be between 0 and 1")
        
        if self.config.data.window_sec <= 0:
            raise ValueError("Window size must be positive")
        
        # Validate model configurations
        for model_name, model_config in self.config.models.items():
            if model_config.enabled and not model_config.params:
                self.logger.warning(f"Model '{model_name}' is enabled but has no parameters")
        
        self.logger.info("Configuration validation passed")
    
    def merge_configs(self, other_config: Union[Config, Dict[str, Any]]) -> Config:
        """
        Merge another configuration with the current one.
        
        Args:
            other_config: Configuration to merge
            
        Returns:
            Merged configuration
        """
        if isinstance(other_config, dict):
            other_config = self._dict_to_config(other_config)
        
        merged_config = deepcopy(self.config)
        
        # Merge model configurations
        for model_name, model_config in other_config.models.items():
            if model_name in merged_config.models:
                merged_config.models[model_name].params.update(model_config.params)
            else:
                merged_config.models[model_name] = model_config
        
        return merged_config


def load_default_config() -> Config:
    """Load default configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    
    if config_path.exists():
        manager = ConfigManager(config_path)
        return manager.config
    else:
        return Config()


# Environment variable support
def get_config_from_env() -> Dict[str, Any]:
    """Get configuration overrides from environment variables."""
    env_config = {}
    
    # Data configuration
    if 'ANOMALY_FS' in os.environ:
        env_config.setdefault('data', {})['fs'] = float(os.environ['ANOMALY_FS'])
    
    if 'ANOMALY_WINDOW_SEC' in os.environ:
        env_config.setdefault('data', {})['window_sec'] = float(os.environ['ANOMALY_WINDOW_SEC'])
    
    # Model configuration
    if 'ANOMALY_DEVICE' in os.environ:
        device = os.environ['ANOMALY_DEVICE']
        env_config.setdefault('models', {})
        for model_name in ['autoencoder', 'vae', 'lstm', 'transformer']:
            env_config['models'].setdefault(model_name, {}).setdefault('params', {})['device'] = device
    
    # API configuration
    if 'ANOMALY_API_PORT' in os.environ:
        env_config.setdefault('api', {})['port'] = int(os.environ['ANOMALY_API_PORT'])
    
    return env_config


from typing import List
