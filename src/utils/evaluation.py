#!/usr/bin/env python3
"""
Advanced evaluation utilities for anomaly detection models.
Includes cross-validation, subject-independent evaluation, and uncertainty quantification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class TimeSeriesGroupKFold:
    """
    Custom K-Fold for time series with group constraints.
    Ensures no temporal leakage and respects subject boundaries.
    """
    
    def __init__(self, n_splits: int = 5, gap: int = 0):
        """
        Initialize TimeSeriesGroupKFold.
        
        Args:
            n_splits: Number of splits
            gap: Minimum gap between train and test sets (in samples)
        """
        self.n_splits = n_splits
        self.gap = gap
    
    def split(self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None):
        """
        Generate train/test splits respecting temporal order and groups.
        
        Args:
            X: Data array
            y: Target array (optional)
            groups: Group labels (e.g., subject IDs)
            
        Yields:
            train_idx, test_idx: Indices for train and test sets
        """
        n_samples = len(X)
        
        if groups is not None:
            # Group-based splitting (e.g., by subject)
            unique_groups = np.unique(groups)
            n_groups = len(unique_groups)
            
            if n_groups < self.n_splits:
                raise ValueError(f"Number of groups ({n_groups}) < n_splits ({self.n_splits})")
            
            group_fold_size = n_groups // self.n_splits
            
            for i in range(self.n_splits):
                test_start = i * group_fold_size
                test_end = min((i + 1) * group_fold_size, n_groups)
                
                test_groups = unique_groups[test_start:test_end]
                test_mask = np.isin(groups, test_groups)
                test_idx = np.where(test_mask)[0]
                train_idx = np.where(~test_mask)[0]
                
                yield train_idx, test_idx
        else:
            # Time-based splitting
            fold_size = n_samples // self.n_splits
            
            for i in range(self.n_splits):
                test_start = i * fold_size
                test_end = min((i + 1) * fold_size, n_samples)
                
                test_idx = np.arange(test_start, test_end)
                
                # Add gap to prevent leakage
                train_end = max(0, test_start - self.gap)
                train_start = max(0, test_end + self.gap)
                
                train_idx = np.concatenate([
                    np.arange(0, train_end),
                    np.arange(train_start, n_samples)
                ])
                
                if len(train_idx) == 0:
                    continue
                    
                yield train_idx, test_idx


class AnomalyEvaluator:
    """Comprehensive evaluator for anomaly detection models."""
    
    def __init__(self, random_state: int = 42):
        """Initialize evaluator."""
        self.random_state = random_state
        
    def compute_comprehensive_metrics(self, y_true: np.ndarray, y_scores: np.ndarray, 
                                    y_pred: Optional[np.ndarray] = None,
                                    threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True binary labels
            y_scores: Anomaly scores (higher = more anomalous)
            y_pred: Binary predictions (optional)
            threshold: Decision threshold (optional)
            
        Returns:
            Dictionary of metrics
        """
        if len(np.unique(y_true)) < 2:
            return {'error': 'Need both normal and anomalous samples for evaluation'}
        
        metrics = {}
        
        try:
            # ROC AUC
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            
            # Average Precision (PR AUC)
            metrics['avg_precision'] = average_precision_score(y_true, y_scores)
            
            # If predictions not provided, use threshold
            if y_pred is None:
                if threshold is None:
                    threshold = np.percentile(y_scores, 95)
                y_pred = (y_scores >= threshold).astype(int)
            
            # Precision, Recall, F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            
            metrics.update({
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'threshold': float(threshold) if threshold is not None else None
            })
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics.update({
                'true_positive': int(tp),
                'false_positive': int(fp),
                'true_negative': int(tn),
                'false_negative': int(fn),
                'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            })
            
            # Additional metrics
            metrics['accuracy'] = float((tp + tn) / (tp + tn + fp + fn))
            metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
            
            # False positive rate and true positive rate
            metrics['fpr'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            metrics['tpr'] = metrics['sensitivity']
            
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    def cross_validate_model(self, model_class: Any, model_params: Dict, 
                           X: np.ndarray, y: np.ndarray, 
                           cv_method: str = 'stratified', n_splits: int = 5,
                           groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform cross-validation for anomaly detection model.
        
        Args:
            model_class: Model class to instantiate
            model_params: Parameters for model initialization
            X: Feature matrix
            y: Binary labels
            cv_method: Cross-validation method ('stratified', 'time_series', 'group')
            n_splits: Number of CV splits
            groups: Group labels for group-based CV
            
        Returns:
            Cross-validation results
        """
        # Initialize cross-validator
        if cv_method == 'stratified':
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        elif cv_method == 'time_series':
            cv = TimeSeriesSplit(n_splits=n_splits)
        elif cv_method == 'group':
            cv = TimeSeriesGroupKFold(n_splits=n_splits)
        else:
            raise ValueError(f"Unknown CV method: {cv_method}")
        
        fold_results = []
        all_scores = []
        all_labels = []
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
            try:
                # Split data
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train only on normal samples
                normal_idx = np.where(y_train == 0)[0]
                if len(normal_idx) == 0:
                    continue
                
                X_train_normal = X_train[normal_idx]
                
                # Initialize and train model
                model = model_class(**model_params)
                model.fit(X_train_normal)
                
                # Get scores
                if hasattr(model, 'reconstruction_error'):
                    scores = model.reconstruction_error(X_test)
                elif hasattr(model, 'anomaly_score'):
                    scores = model.anomaly_score(X_test)
                elif hasattr(model, 'score'):
                    scores = model.score(X_test)
                else:
                    raise ValueError("Model doesn't have compatible scoring method")
                
                # Evaluate
                metrics = self.compute_comprehensive_metrics(y_test, scores)
                metrics['fold'] = fold
                fold_results.append(metrics)
                
                all_scores.extend(scores)
                all_labels.extend(y_test)
                
            except Exception as e:
                print(f"Error in fold {fold}: {e}")
                continue
        
        if not fold_results:
            return {'error': 'All folds failed'}
        
        # Aggregate results
        df_results = pd.DataFrame(fold_results)
        
        # Overall metrics using all predictions
        overall_metrics = self.compute_comprehensive_metrics(
            np.array(all_labels), np.array(all_scores)
        )
        
        return {
            'fold_results': fold_results,
            'mean_metrics': df_results.select_dtypes(include=[np.number]).mean().to_dict(),
            'std_metrics': df_results.select_dtypes(include=[np.number]).std().to_dict(),
            'overall_metrics': overall_metrics,
            'n_folds': len(fold_results)
        }
    
    def subject_independent_evaluation(self, model_class: Any, model_params: Dict,
                                     X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> Dict[str, Any]:
        """
        Perform subject-independent evaluation (leave-one-subject-out).
        
        Args:
            model_class: Model class to instantiate
            model_params: Parameters for model initialization
            X: Feature matrix
            y: Binary labels
            subjects: Subject identifiers
            
        Returns:
            Subject-independent evaluation results
        """
        unique_subjects = np.unique(subjects)
        subject_results = []
        
        for test_subject in unique_subjects:
            try:
                # Split by subject
                test_mask = subjects == test_subject
                train_mask = ~test_mask
                
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]
                
                # Train only on normal samples
                normal_idx = np.where(y_train == 0)[0]
                if len(normal_idx) == 0:
                    continue
                
                X_train_normal = X_train[normal_idx]
                
                # Train model
                model = model_class(**model_params)
                model.fit(X_train_normal)
                
                # Get scores
                if hasattr(model, 'reconstruction_error'):
                    scores = model.reconstruction_error(X_test)
                elif hasattr(model, 'anomaly_score'):
                    scores = model.anomaly_score(X_test)
                elif hasattr(model, 'score'):
                    scores = model.score(X_test)
                else:
                    raise ValueError("Model doesn't have compatible scoring method")
                
                # Evaluate
                metrics = self.compute_comprehensive_metrics(y_test, scores)
                metrics['test_subject'] = test_subject
                subject_results.append(metrics)
                
            except Exception as e:
                print(f"Error evaluating subject {test_subject}: {e}")
                continue
        
        if not subject_results:
            return {'error': 'All subjects failed'}
        
        # Aggregate results
        df_results = pd.DataFrame(subject_results)
        
        return {
            'subject_results': subject_results,
            'mean_metrics': df_results.select_dtypes(include=[np.number]).mean().to_dict(),
            'std_metrics': df_results.select_dtypes(include=[np.number]).std().to_dict(),
            'n_subjects': len(subject_results)
        }
    
    def uncertainty_quantification(self, model: Any, X: np.ndarray, 
                                 n_bootstrap: int = 100, bootstrap_ratio: float = 0.8) -> Dict[str, np.ndarray]:
        """
        Quantify prediction uncertainty using bootstrap sampling.
        
        Args:
            model: Trained model
            X: Data to evaluate
            n_bootstrap: Number of bootstrap samples
            bootstrap_ratio: Ratio of data to use in each bootstrap
            
        Returns:
            Uncertainty statistics
        """
        n_samples = int(len(X) * bootstrap_ratio)
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(len(X), size=n_samples, replace=True)
            X_bootstrap = X[idx]
            
            try:
                # Get scores
                if hasattr(model, 'reconstruction_error'):
                    scores = model.reconstruction_error(X_bootstrap)
                elif hasattr(model, 'anomaly_score'):
                    scores = model.anomaly_score(X_bootstrap)
                elif hasattr(model, 'score'):
                    scores = model.score(X_bootstrap)
                else:
                    raise ValueError("Model doesn't have compatible scoring method")
                
                bootstrap_scores.append(scores)
                
            except Exception as e:
                print(f"Bootstrap iteration failed: {e}")
                continue
        
        if not bootstrap_scores:
            return {'error': 'All bootstrap iterations failed'}
        
        # Align scores (pad shorter arrays)
        max_len = max(len(scores) for scores in bootstrap_scores)
        aligned_scores = []
        
        for scores in bootstrap_scores:
            if len(scores) < max_len:
                # Pad with mean value
                padded = np.concatenate([scores, np.full(max_len - len(scores), np.mean(scores))])
            else:
                padded = scores[:max_len]
            aligned_scores.append(padded)
        
        bootstrap_matrix = np.array(aligned_scores)
        
        return {
            'mean_scores': np.mean(bootstrap_matrix, axis=0),
            'std_scores': np.std(bootstrap_matrix, axis=0),
            'lower_ci': np.percentile(bootstrap_matrix, 2.5, axis=0),
            'upper_ci': np.percentile(bootstrap_matrix, 97.5, axis=0),
            'median_scores': np.median(bootstrap_matrix, axis=0),
            'n_bootstrap': len(aligned_scores)
        }
    
    def generate_evaluation_report(self, results: Dict[str, Any], 
                                 output_path: Optional[Path] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: Evaluation results dictionary
            output_path: Path to save report (optional)
            
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("ANOMALY DETECTION EVALUATION REPORT")
        report_lines.append("=" * 60)
        
        # Cross-validation results
        if 'cross_validation' in results:
            cv_results = results['cross_validation']
            if 'error' not in cv_results:
                report_lines.append("\nCROSS-VALIDATION RESULTS:")
                report_lines.append("-" * 30)
                
                mean_metrics = cv_results['mean_metrics']
                std_metrics = cv_results['std_metrics']
                
                for metric in ['roc_auc', 'avg_precision', 'f1', 'precision', 'recall']:
                    if metric in mean_metrics:
                        mean_val = mean_metrics[metric]
                        std_val = std_metrics.get(metric, 0)
                        report_lines.append(f"{metric.upper()}: {mean_val:.4f} (±{std_val:.4f})")
                
                report_lines.append(f"Number of folds: {cv_results['n_folds']}")
        
        # Subject-independent results
        if 'subject_independent' in results:
            si_results = results['subject_independent']
            if 'error' not in si_results:
                report_lines.append("\nSUBJECT-INDEPENDENT RESULTS:")
                report_lines.append("-" * 30)
                
                mean_metrics = si_results['mean_metrics']
                std_metrics = si_results['std_metrics']
                
                for metric in ['roc_auc', 'avg_precision', 'f1', 'precision', 'recall']:
                    if metric in mean_metrics:
                        mean_val = mean_metrics[metric]
                        std_val = std_metrics.get(metric, 0)
                        report_lines.append(f"{metric.upper()}: {mean_val:.4f} (±{std_val:.4f})")
                
                report_lines.append(f"Number of subjects: {si_results['n_subjects']}")
        
        # Overall metrics
        if 'overall_metrics' in results:
            overall = results['overall_metrics']
            if 'error' not in overall:
                report_lines.append("\nOVERALL PERFORMANCE:")
                report_lines.append("-" * 30)
                
                for metric in ['roc_auc', 'avg_precision', 'f1', 'precision', 'recall', 'accuracy']:
                    if metric in overall:
                        report_lines.append(f"{metric.upper()}: {overall[metric]:.4f}")
        
        report_lines.append("\n" + "=" * 60)
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text


def plot_evaluation_curves(y_true: np.ndarray, y_scores: np.ndarray, 
                          output_path: Optional[Path] = None, title: str = "Evaluation Curves") -> None:
    """
    Plot ROC and Precision-Recall curves.
    
    Args:
        y_true: True binary labels
        y_scores: Anomaly scores
        output_path: Path to save plot
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    ax2.plot(recall, precision, color='darkorange', lw=2, 
             label=f'PR curve (AP = {avg_precision:.3f})')
    ax2.axhline(y=np.sum(y_true) / len(y_true), color='navy', linestyle='--', 
                label='Random')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         output_path: Optional[Path] = None, title: str = "Confusion Matrix") -> None:
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        output_path: Path to save plot
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'], 
                yticklabels=['Normal', 'Anomaly'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
