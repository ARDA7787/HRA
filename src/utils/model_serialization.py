#!/usr/bin/env python3
"""
Model serialization and persistence utilities.
Handles saving/loading of trained models with metadata.
"""

import pickle
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
from datetime import datetime
import hashlib
import joblib
import warnings

warnings.filterwarnings("ignore")


class ModelSerializer:
    """Handles model serialization and deserialization with metadata."""

    def __init__(self, base_path: Union[str, Path] = "models"):
        """
        Initialize model serializer.

        Args:
            base_path: Base directory for saving models
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _generate_model_id(self, model_name: str, metadata: Dict[str, Any]) -> str:
        """Generate unique model ID based on name and metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_str = json.dumps(metadata, sort_keys=True, default=str)
        hash_obj = hashlib.md5(metadata_str.encode())
        hash_hex = hash_obj.hexdigest()[:8]
        return f"{model_name}_{timestamp}_{hash_hex}"

    def save_model(
        self,
        model: Any,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
    ) -> str:
        """
        Save model with metadata.

        Args:
            model: Trained model to save
            model_name: Name of the model type
            metadata: Additional metadata
            model_id: Custom model ID (optional)

        Returns:
            Model ID
        """
        if metadata is None:
            metadata = {}

        # Add standard metadata
        metadata.update(
            {
                "model_name": model_name,
                "save_timestamp": datetime.now().isoformat(),
                "model_type": type(model).__name__,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "torch_version": (
                    torch.__version__
                    if hasattr(model, "model")
                    and isinstance(getattr(model, "model"), torch.nn.Module)
                    else None
                ),
            }
        )

        if model_id is None:
            model_id = self._generate_model_id(model_name, metadata)

        model_dir = self.base_path / model_id
        model_dir.mkdir(exist_ok=True)

        try:
            # Save model
            if hasattr(model, "model") and isinstance(getattr(model, "model"), torch.nn.Module):
                # PyTorch model
                torch.save(
                    {
                        "model_state_dict": model.model.state_dict(),
                        "model_class": type(model).__name__,
                        "model_init_params": getattr(model, "_init_params", {}),
                        "optimizer_state_dict": (
                            getattr(model, "optimizer", {}).state_dict()
                            if hasattr(getattr(model, "optimizer", {}), "state_dict")
                            else None
                        ),
                    },
                    model_dir / "model.pth",
                )

                # Save the wrapper object separately
                model_copy = model.__class__.__new__(model.__class__)
                for key, value in model.__dict__.items():
                    if key != "model" and key != "optimizer":
                        setattr(model_copy, key, value)

                with open(model_dir / "model_wrapper.pkl", "wb") as f:
                    pickle.dump(model_copy, f)

            else:
                # Scikit-learn or other models
                if hasattr(model, "model") and hasattr(model.model, "get_params"):
                    # Use joblib for sklearn models
                    joblib.dump(model, model_dir / "model.joblib")
                else:
                    # Use pickle for other models
                    with open(model_dir / "model.pkl", "wb") as f:
                        pickle.dump(model, f)

            # Save metadata
            with open(model_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            # Save model info
            model_info = {
                "model_id": model_id,
                "model_name": model_name,
                "save_path": str(model_dir),
                "save_timestamp": metadata["save_timestamp"],
            }

            with open(model_dir / "model_info.json", "w") as f:
                json.dump(model_info, f, indent=2)

            self.logger.info(f"Model saved successfully: {model_id}")
            return model_id

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, model_id: str) -> tuple[Any, Dict[str, Any]]:
        """
        Load model with metadata.

        Args:
            model_id: Model ID to load

        Returns:
            Tuple of (model, metadata)
        """
        model_dir = self.base_path / model_id

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        try:
            # Load metadata
            with open(model_dir / "metadata.json", "r") as f:
                metadata = json.load(f)

            # Load model
            if (model_dir / "model.pth").exists():
                # PyTorch model
                checkpoint = torch.load(model_dir / "model.pth", map_location="cpu")

                # Load wrapper
                with open(model_dir / "model_wrapper.pkl", "rb") as f:
                    model = pickle.load(f)

                # Reconstruct the PyTorch model
                model_class_name = checkpoint["model_class"]

                # Import the appropriate model class
                if model_class_name == "AEAnomalyDetector":
                    from ..models.autoencoder import AEAnomalyDetector, DenseAE

                    # Reconstruct the model
                    init_params = checkpoint.get("model_init_params", {})
                    model.model = DenseAE(
                        init_params.get("input_dim", 2), init_params.get("latent_dim", 32)
                    )
                    model.model.load_state_dict(checkpoint["model_state_dict"])

                elif model_class_name == "VAEAnomalyDetector":
                    from ..models.vae_anomaly import VAEAnomalyDetector, VAE

                    init_params = checkpoint.get("model_init_params", {})
                    model.model = VAE(
                        init_params.get("input_dim", 2), init_params.get("latent_dim", 32)
                    )
                    model.model.load_state_dict(checkpoint["model_state_dict"])

                elif model_class_name == "LSTMAnomalyDetector":
                    from ..models.lstm_anomaly import LSTMAnomalyDetector, LSTMAutoencoder

                    init_params = checkpoint.get("model_init_params", {})
                    model.model = LSTMAutoencoder(
                        input_dim=init_params.get("input_dim", 2),
                        hidden_dim=init_params.get("hidden_dim", 64),
                        latent_dim=init_params.get("latent_dim", 32),
                        seq_len=init_params.get("seq_len", 120),
                    )
                    model.model.load_state_dict(checkpoint["model_state_dict"])

                elif model_class_name == "TransformerAnomalyDetector":
                    from ..models.transformer_anomaly import (
                        TransformerAnomalyDetector,
                        TransformerEncoder,
                    )

                    init_params = checkpoint.get("model_init_params", {})
                    model.model = TransformerEncoder(
                        input_dim=init_params.get("input_dim", 2),
                        d_model=init_params.get("d_model", 64),
                    )
                    model.model.load_state_dict(checkpoint["model_state_dict"])

            elif (model_dir / "model.joblib").exists():
                # Joblib model
                model = joblib.load(model_dir / "model.joblib")

            elif (model_dir / "model.pkl").exists():
                # Pickle model
                with open(model_dir / "model.pkl", "rb") as f:
                    model = pickle.load(f)
            else:
                raise FileNotFoundError("No model file found in directory")

            self.logger.info(f"Model loaded successfully: {model_id}")
            return model, metadata

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def list_models(self) -> pd.DataFrame:
        """
        List all saved models.

        Returns:
            DataFrame with model information
        """
        models = []

        for model_dir in self.base_path.iterdir():
            if model_dir.is_dir():
                try:
                    info_file = model_dir / "model_info.json"
                    metadata_file = model_dir / "metadata.json"

                    if info_file.exists() and metadata_file.exists():
                        with open(info_file, "r") as f:
                            info = json.load(f)

                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)

                        model_entry = {
                            "model_id": info["model_id"],
                            "model_name": info["model_name"],
                            "model_type": metadata.get("model_type", "Unknown"),
                            "save_timestamp": info["save_timestamp"],
                            "save_path": info["save_path"],
                        }

                        # Add key metadata
                        for key in ["roc_auc", "f1", "precision", "recall", "dataset"]:
                            if key in metadata:
                                model_entry[key] = metadata[key]

                        models.append(model_entry)

                except Exception as e:
                    self.logger.warning(f"Error reading model info from {model_dir}: {e}")
                    continue

        return pd.DataFrame(models)

    def delete_model(self, model_id: str) -> bool:
        """
        Delete a saved model.

        Args:
            model_id: Model ID to delete

        Returns:
            True if successful
        """
        model_dir = self.base_path / model_id

        if not model_dir.exists():
            self.logger.warning(f"Model directory not found: {model_dir}")
            return False

        try:
            import shutil

            shutil.rmtree(model_dir)
            self.logger.info(f"Model deleted successfully: {model_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting model: {e}")
            return False

    def export_model(self, model_id: str, export_path: Union[str, Path]) -> bool:
        """
        Export model to a different location.

        Args:
            model_id: Model ID to export
            export_path: Destination path

        Returns:
            True if successful
        """
        model_dir = self.base_path / model_id
        export_path = Path(export_path)

        if not model_dir.exists():
            self.logger.error(f"Model directory not found: {model_dir}")
            return False

        try:
            import shutil

            shutil.copytree(model_dir, export_path)
            self.logger.info(f"Model exported successfully to: {export_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting model: {e}")
            return False


import sys


class ModelRegistry:
    """Registry for managing multiple models and their versions."""

    def __init__(self, registry_path: Union[str, Path] = "model_registry.json"):
        """
        Initialize model registry.

        Args:
            registry_path: Path to registry file
        """
        self.registry_path = Path(registry_path)
        self.serializer = ModelSerializer()

        # Load existing registry
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                self.registry = json.load(f)
        else:
            self.registry = {}

    def register_model(
        self,
        model: Any,
        model_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
        is_production: bool = False,
    ) -> str:
        """
        Register a new model version.

        Args:
            model: Trained model
            model_name: Name of the model
            version: Version string
            metadata: Additional metadata
            is_production: Whether this is a production model

        Returns:
            Model ID
        """
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "version": version,
                "is_production": is_production,
                "registered_timestamp": datetime.now().isoformat(),
            }
        )

        # Save model
        model_id = self.serializer.save_model(model, model_name, metadata)

        # Update registry
        if model_name not in self.registry:
            self.registry[model_name] = {}

        self.registry[model_name][version] = {
            "model_id": model_id,
            "is_production": is_production,
            "registered_timestamp": metadata["registered_timestamp"],
            "metadata": metadata,
        }

        # Save registry
        self._save_registry()

        return model_id

    def get_model(self, model_name: str, version: str = "latest") -> tuple[Any, Dict[str, Any]]:
        """
        Get a specific model version.

        Args:
            model_name: Name of the model
            version: Version string or 'latest' or 'production'

        Returns:
            Tuple of (model, metadata)
        """
        if model_name not in self.registry:
            raise ValueError(f"Model '{model_name}' not found in registry")

        model_versions = self.registry[model_name]

        if version == "latest":
            # Get latest version
            latest_version = max(
                model_versions.keys(), key=lambda v: model_versions[v]["registered_timestamp"]
            )
            model_id = model_versions[latest_version]["model_id"]

        elif version == "production":
            # Get production version
            prod_versions = {k: v for k, v in model_versions.items() if v["is_production"]}
            if not prod_versions:
                raise ValueError(f"No production version found for model '{model_name}'")

            latest_prod = max(
                prod_versions.keys(), key=lambda v: prod_versions[v]["registered_timestamp"]
            )
            model_id = prod_versions[latest_prod]["model_id"]

        else:
            # Get specific version
            if version not in model_versions:
                raise ValueError(f"Version '{version}' not found for model '{model_name}'")
            model_id = model_versions[version]["model_id"]

        return self.serializer.load_model(model_id)

    def promote_to_production(self, model_name: str, version: str) -> bool:
        """
        Promote a model version to production.

        Args:
            model_name: Name of the model
            version: Version to promote

        Returns:
            True if successful
        """
        if model_name not in self.registry or version not in self.registry[model_name]:
            return False

        # Demote existing production models
        for v in self.registry[model_name].values():
            v["is_production"] = False

        # Promote new version
        self.registry[model_name][version]["is_production"] = True

        self._save_registry()
        return True

    def list_models(self) -> pd.DataFrame:
        """List all registered models."""
        models = []

        for model_name, versions in self.registry.items():
            for version, info in versions.items():
                model_entry = {
                    "model_name": model_name,
                    "version": version,
                    "model_id": info["model_id"],
                    "is_production": info["is_production"],
                    "registered_timestamp": info["registered_timestamp"],
                }

                # Add metadata
                metadata = info.get("metadata", {})
                for key in ["roc_auc", "f1", "precision", "recall"]:
                    if key in metadata:
                        model_entry[key] = metadata[key]

                models.append(model_entry)

        return pd.DataFrame(models)

    def _save_registry(self):
        """Save registry to file."""
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2, default=str)
