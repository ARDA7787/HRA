#!/usr/bin/env python3
"""
Ensemble methods for combining multiple anomaly detection models.
Provides weighted voting, stacking, and dynamic ensemble selection.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pickle
import warnings

warnings.filterwarnings('ignore')


class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detector that combines multiple base models.
    Supports various combination strategies including weighted voting and stacking.
    """
    
    def __init__(self, models: List[Any], combination_method: str = 'weighted_voting',
                 weights: Optional[List[float]] = None, meta_model: Optional[Any] = None):
        """
        Initialize ensemble detector.
        
        Args:
            models: List of trained anomaly detection models
            combination_method: How to combine predictions ('weighted_voting', 'stacking', 'dynamic')
            weights: Weights for weighted voting (if None, equal weights used)
            meta_model: Meta-learner for stacking (if None, LogisticRegression used)
        """
        self.models = models
        self.combination_method = combination_method
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if meta_model is None:
            self.meta_model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            self.meta_model = meta_model
            
        self.is_fitted = False
        self.model_performances = {}
        
    def fit_meta_model(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleAnomalyDetector':
        """
        Fit meta-model for stacking ensemble.
        
        Args:
            X: Training data for meta-model
            y: True labels for training
        """
        if self.combination_method == 'stacking':
            # Get predictions from all base models
            base_predictions = self._get_base_predictions(X)
            
            # Train meta-model
            self.meta_model.fit(base_predictions, y)
            
        self.is_fitted = True
        return self
    
    def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all base models."""
        predictions = []
        
        for model in self.models:
            if hasattr(model, 'reconstruction_error'):
                scores = model.reconstruction_error(X)
            elif hasattr(model, 'anomaly_score'):
                scores = model.anomaly_score(X)
            elif hasattr(model, 'score'):
                scores = model.score(X)
            else:
                raise ValueError(f"Model {type(model)} doesn't have a compatible scoring method")
            
            predictions.append(scores)
        
        return np.column_stack(predictions)
    
    def _weighted_voting(self, predictions: np.ndarray) -> np.ndarray:
        """Combine predictions using weighted voting."""
        return np.average(predictions, axis=1, weights=self.weights)
    
    def _stacking_prediction(self, predictions: np.ndarray) -> np.ndarray:
        """Combine predictions using stacking."""
        if not self.is_fitted:
            raise ValueError("Meta-model not fitted. Call fit_meta_model first.")
        
        # Get probabilities from meta-model
        if hasattr(self.meta_model, 'predict_proba'):
            probs = self.meta_model.predict_proba(predictions)[:, 1]
        else:
            probs = self.meta_model.decision_function(predictions)
        
        return probs
    
    def _dynamic_selection(self, predictions: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Dynamic ensemble selection based on local performance.
        Selects the best model(s) for each instance based on local neighborhood.
        """
        if not self.model_performances:
            # Use equal weights if no performance info available
            return self._weighted_voting(predictions)
        
        # Simple dynamic selection: weight by overall performance
        performance_weights = np.array([self.model_performances.get(i, 1.0) 
                                      for i in range(len(self.models))])
        performance_weights = performance_weights / np.sum(performance_weights)
        
        return np.average(predictions, axis=1, weights=performance_weights)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute ensemble anomaly scores.
        
        Args:
            X: Data to score
            
        Returns:
            Ensemble anomaly scores
        """
        base_predictions = self._get_base_predictions(X)
        
        if self.combination_method == 'weighted_voting':
            return self._weighted_voting(base_predictions)
        elif self.combination_method == 'stacking':
            return self._stacking_prediction(base_predictions)
        elif self.combination_method == 'dynamic':
            return self._dynamic_selection(base_predictions, X)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
    
    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """
        Predict anomalies using ensemble.
        
        Args:
            X: Data to predict
            threshold: Anomaly threshold
            
        Returns:
            Binary predictions (1 = anomaly)
        """
        scores = self.score(X)
        return (scores >= threshold).astype(int)
    
    def evaluate_base_models(self, X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
        """
        Evaluate individual base models and store performance.
        
        Args:
            X: Evaluation data
            y: True labels
            
        Returns:
            Dictionary of model index to performance score
        """
        performances = {}
        
        for i, model in enumerate(self.models):
            try:
                if hasattr(model, 'reconstruction_error'):
                    scores = model.reconstruction_error(X)
                elif hasattr(model, 'anomaly_score'):
                    scores = model.anomaly_score(X)
                elif hasattr(model, 'score'):
                    scores = model.score(X)
                else:
                    continue
                
                # Use AUC as performance metric
                if len(np.unique(y)) > 1:
                    auc = roc_auc_score(y, scores)
                    performances[i] = auc
                else:
                    performances[i] = 0.5  # Random performance
                    
            except Exception as e:
                print(f"Error evaluating model {i}: {e}")
                performances[i] = 0.5
        
        self.model_performances = performances
        return performances
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights/importance."""
        if self.combination_method == 'weighted_voting':
            return {f"model_{i}": weight for i, weight in enumerate(self.weights)}
        elif self.combination_method == 'dynamic':
            return {f"model_{i}": self.model_performances.get(i, 1.0) 
                   for i in range(len(self.models))}
        else:
            return {f"model_{i}": 1.0 / len(self.models) for i in range(len(self.models))}


class AdaptiveEnsemble:
    """
    Adaptive ensemble that can add/remove models and update weights dynamically.
    """
    
    def __init__(self, max_models: int = 10, performance_threshold: float = 0.6):
        """
        Initialize adaptive ensemble.
        
        Args:
            max_models: Maximum number of models to keep
            performance_threshold: Minimum performance to keep a model
        """
        self.max_models = max_models
        self.performance_threshold = performance_threshold
        self.models = []
        self.performances = []
        self.weights = []
        
    def add_model(self, model: Any, performance: float = None) -> None:
        """
        Add a new model to the ensemble.
        
        Args:
            model: Trained anomaly detection model
            performance: Performance score (if known)
        """
        self.models.append(model)
        self.performances.append(performance or 0.5)
        
        # Update weights
        self._update_weights()
        
        # Prune if necessary
        if len(self.models) > self.max_models:
            self._prune_models()
    
    def _update_weights(self) -> None:
        """Update model weights based on performance."""
        if not self.performances:
            self.weights = []
            return
        
        # Softmax weighting based on performance
        exp_perfs = np.exp(np.array(self.performances))
        self.weights = exp_perfs / np.sum(exp_perfs)
    
    def _prune_models(self) -> None:
        """Remove poorly performing models."""
        # Remove models below threshold
        keep_indices = [i for i, perf in enumerate(self.performances) 
                       if perf >= self.performance_threshold]
        
        if len(keep_indices) > self.max_models:
            # Keep top performers
            sorted_indices = sorted(keep_indices, 
                                  key=lambda i: self.performances[i], 
                                  reverse=True)
            keep_indices = sorted_indices[:self.max_models]
        
        # Update lists
        self.models = [self.models[i] for i in keep_indices]
        self.performances = [self.performances[i] for i in keep_indices]
        self._update_weights()
    
    def update_performance(self, model_idx: int, performance: float) -> None:
        """Update performance score for a specific model."""
        if 0 <= model_idx < len(self.models):
            self.performances[model_idx] = performance
            self._update_weights()
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute ensemble scores."""
        if not self.models:
            return np.zeros(len(X))
        
        predictions = []
        for model in self.models:
            if hasattr(model, 'reconstruction_error'):
                scores = model.reconstruction_error(X)
            elif hasattr(model, 'anomaly_score'):
                scores = model.anomaly_score(X)
            elif hasattr(model, 'score'):
                scores = model.score(X)
            else:
                scores = np.zeros(len(X))
            
            predictions.append(scores)
        
        predictions = np.column_stack(predictions)
        return np.average(predictions, axis=1, weights=self.weights)
    
    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """Predict anomalies."""
        scores = self.score(X)
        return (scores >= threshold).astype(int)


def create_model_ensemble(models_config: List[Dict], X_train: np.ndarray, 
                         X_val: Optional[np.ndarray] = None, 
                         y_val: Optional[np.ndarray] = None) -> EnsembleAnomalyDetector:
    """
    Create and train an ensemble of anomaly detection models.
    
    Args:
        models_config: List of model configurations
        X_train: Training data
        X_val: Validation data (for meta-model training)
        y_val: Validation labels (for meta-model training)
        
    Returns:
        Trained ensemble detector
    """
    from .isolation_forest import IFAnomalyDetector
    from .autoencoder import AEAnomalyDetector
    from .vae_anomaly import VAEAnomalyDetector
    from .lstm_anomaly import LSTMAnomalyDetector
    from .transformer_anomaly import TransformerAnomalyDetector
    
    trained_models = []
    
    for config in models_config:
        model_type = config.get('type')
        model_params = config.get('params', {})
        
        try:
            if model_type == 'isolation_forest':
                model = IFAnomalyDetector(**model_params)
                model.fit(X_train)
                
            elif model_type == 'autoencoder':
                model = AEAnomalyDetector(input_dim=X_train.shape[1], **model_params)
                model.fit(X_train)
                
            elif model_type == 'vae':
                model = VAEAnomalyDetector(input_dim=X_train.shape[1], **model_params)
                model.fit(X_train)
                
            elif model_type == 'lstm':
                model = LSTMAnomalyDetector(input_dim=2, **model_params)  # Assuming 2D input
                model.fit(X_train)
                
            elif model_type == 'transformer':
                model = TransformerAnomalyDetector(input_dim=2, **model_params)
                model.fit(X_train)
                
            else:
                print(f"Unknown model type: {model_type}")
                continue
                
            trained_models.append(model)
            print(f"Successfully trained {model_type}")
            
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            continue
    
    if not trained_models:
        raise ValueError("No models were successfully trained")
    
    # Create ensemble
    ensemble = EnsembleAnomalyDetector(
        models=trained_models,
        combination_method='weighted_voting'
    )
    
    # Fit meta-model if validation data is provided
    if X_val is not None and y_val is not None:
        ensemble.fit_meta_model(X_val, y_val)
    
    return ensemble
