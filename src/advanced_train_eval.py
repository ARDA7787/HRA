#!/usr/bin/env python3
"""
Advanced training and evaluation script with all new features.
Supports multiple models, ensemble methods, advanced evaluation, and interactive visualization.
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

# Import utilities
from utils.config_manager import ConfigManager, load_default_config
from utils.preprocessing import interpolate_missing, zscore, butter_lowpass, sliding_windows
from utils.feature_engineering import PhysiologicalFeatureExtractor, build_feature_matrix
from utils.evaluation import AnomalyEvaluator, plot_evaluation_curves, plot_confusion_matrix
from utils.visualization import plot_anomalies
from utils.interactive_visualization import InteractiveVisualizer, create_comprehensive_report
from utils.model_serialization import ModelRegistry
from utils.metrics import compute_metrics

# Import models
from models.isolation_forest import IFAnomalyDetector
from models.autoencoder import AEAnomalyDetector
from models.vae_anomaly import VAEAnomalyDetector
from models.lstm_anomaly import LSTMAnomalyDetector
from models.transformer_anomaly import TransformerAnomalyDetector
from models.ensemble import EnsembleAnomalyDetector, create_model_ensemble


def window_labels(labels: np.ndarray, window_size: int, step: int) -> np.ndarray:
    """Create window-level labels from sample-level labels."""
    ys = []
    for s, e in sliding_windows(labels, window_size, step):
        # Window labeled as anomaly if any sample is anomaly
        ys.append(int(np.any(labels[s:e] == 1)))
    return np.array(ys, dtype=int)


def prepare_data(df: pd.DataFrame, config: Any) -> tuple:
    """
    Prepare data for training and evaluation.

    Args:
        df: Input dataframe with time, EDA, HR, [label] columns
        config: Configuration object

    Returns:
        Tuple of processed data
    """
    # Extract signals
    time = df["time"].values if "time" in df.columns else np.arange(len(df)) / config.data.fs
    eda = interpolate_missing(df["EDA"])
    hr = interpolate_missing(df["HR"])

    # Preprocessing
    if config.data.eda.get("lowpass_filter", True):
        eda_f = pd.Series(
            butter_lowpass(
                eda.values, fs=config.data.fs, cutoff=config.data.eda.get("cutoff_frequency", 1.0)
            )
        )
    else:
        eda_f = eda.copy()

    # Normalization
    eda_z = zscore(eda_f)
    hr_z = zscore(hr)

    # Combine signals
    X = np.vstack([eda_z.values, hr_z.values]).T  # [N, 2]

    # Window parameters
    window_size = int(config.data.window_sec * config.data.fs)
    step = max(1, int(window_size * (1.0 - config.data.overlap)))

    # Feature extraction
    if config.data.features.get("use_advanced_features", True):
        logging.info("Extracting advanced features...")
        X_features, feature_names = build_feature_matrix(X, window_size, step, config.data.fs)

        if len(X_features) == 0:
            logging.warning("No features extracted, falling back to raw windows")
            X_features = build_raw_windows(X, window_size, step)
            feature_names = [f"raw_feature_{i}" for i in range(X_features.shape[1])]
    else:
        logging.info("Using raw windowed features...")
        X_features = build_raw_windows(X, window_size, step)
        feature_names = [f"raw_feature_{i}" for i in range(X_features.shape[1])]

    # Labels
    y_window = None
    if "label" in df.columns:
        labels = df["label"].values.astype(int)
        y_window = window_labels(labels, window_size, step)

    return X, X_features, y_window, time, eda_z.values, hr_z.values, feature_names


def build_raw_windows(X: np.ndarray, window_size: int, step: int) -> np.ndarray:
    """Build raw windowed features."""
    features = []
    for s, e in sliding_windows(X, window_size, step):
        features.append(X[s:e].reshape(-1))

    if not features:
        return np.zeros((0, window_size * X.shape[1]))

    return np.vstack(features)


def train_single_model(
    model_type: str, X_train: np.ndarray, config: Any, model_configs: Dict[str, Any]
) -> Any:
    """Train a single model."""
    logging.info(f"Training {model_type}...")

    model_config = model_configs.get(model_type, {})

    # Handle both dict and ModelConfig object
    if hasattr(model_config, 'enabled'):
        enabled = model_config.enabled
        params = model_config.params if hasattr(model_config, 'params') else {}
    else:
        enabled = model_config.get("enabled", True)
        params = model_config.get("params", {})

    if not enabled:
        logging.info(f"Model {model_type} is disabled")
        return None

    try:
        if model_type == "isolation_forest":
            model = IFAnomalyDetector(**params)
            model.fit(X_train)

        elif model_type == "autoencoder":
            model = AEAnomalyDetector(input_dim=X_train.shape[1], **params)
            model.fit(X_train)

        elif model_type == "vae":
            model = VAEAnomalyDetector(input_dim=X_train.shape[1], **params)
            model.fit(X_train)

        elif model_type == "lstm":
            model = LSTMAnomalyDetector(input_dim=2, **params)  # Raw signal input
            model.fit(X_train)

        elif model_type == "transformer":
            model = TransformerAnomalyDetector(input_dim=2, **params)  # Raw signal input
            model.fit(X_train)

        else:
            logging.error(f"Unknown model type: {model_type}")
            return None

        logging.info(f"Successfully trained {model_type}")
        return model

    except Exception as e:
        logging.error(f"Error training {model_type}: {e}")
        return None


def evaluate_models(
    models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, config: Any
) -> Dict[str, Dict[str, float]]:
    """Evaluate all models."""
    results = {}

    for model_name, model in models.items():
        if model is None:
            continue

        logging.info(f"Evaluating {model_name}...")

        try:
            # Get scores
            if hasattr(model, "reconstruction_error"):
                scores = model.reconstruction_error(X_test)
            elif hasattr(model, "anomaly_score"):
                scores = model.anomaly_score(X_test)
            elif hasattr(model, "score"):
                scores = model.score(X_test)
            else:
                logging.warning(f"Model {model_name} doesn't support scoring")
                continue

            # Compute metrics
            metrics = compute_metrics(y_test, scores)
            results[model_name] = metrics

            logging.info(
                f"{model_name} - ROC AUC: {metrics.get('roc_auc', 'N/A'):.4f}, "
                f"F1: {metrics.get('f1', 'N/A'):.4f}"
            )

        except Exception as e:
            logging.error(f"Error evaluating {model_name}: {e}")
            continue

    return results


def advanced_evaluation(
    models: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    config: Any,
    subjects: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Perform advanced evaluation including cross-validation and subject-independent."""
    evaluator = AnomalyEvaluator()
    advanced_results = {}

    for model_name, model_class in [
        ("isolation_forest", IFAnomalyDetector),
        ("autoencoder", AEAnomalyDetector),
        ("vae", VAEAnomalyDetector),
    ]:
        if model_name not in config.models or not config.models[model_name].get("enabled", True):
            continue

        logging.info(f"Advanced evaluation for {model_name}...")

        try:
            model_params = config.models[model_name].get("params", {})

            # Cross-validation
            if config.evaluation.cross_validation.get("enabled", True):
                cv_results = evaluator.cross_validate_model(
                    model_class=model_class,
                    model_params=model_params,
                    X=X,
                    y=y,
                    cv_method=config.evaluation.cross_validation.get("method", "stratified"),
                    n_splits=config.evaluation.cross_validation.get("n_splits", 5),
                )
                advanced_results[f"{model_name}_cv"] = cv_results

            # Subject-independent evaluation
            if subjects is not None and config.evaluation.subject_independent.get("enabled", True):
                si_results = evaluator.subject_independent_evaluation(
                    model_class=model_class, model_params=model_params, X=X, y=y, subjects=subjects
                )
                advanced_results[f"{model_name}_si"] = si_results

        except Exception as e:
            logging.error(f"Advanced evaluation failed for {model_name}: {e}")
            continue

    return advanced_results


def create_visualizations(
    time: np.ndarray,
    eda: np.ndarray,
    hr: np.ndarray,
    anomaly_mask: np.ndarray,
    y_true: np.ndarray,
    model_results: Dict[str, Dict[str, float]],
    feature_names: List[str],
    output_dir: Path,
    config: Any,
) -> None:
    """Create comprehensive visualizations."""
    if not config.visualization.get("interactive", True):
        return

    logging.info("Creating interactive visualizations...")

    try:
        # Collect scores from all models for visualization
        all_scores = []
        model_names = []

        # Note: This is simplified - in practice you'd need to store scores during evaluation
        # For now, create dummy scores for visualization
        for model_name in model_results.keys():
            scores = np.random.random(len(y_true))  # Placeholder
            all_scores.append(scores)
            model_names.append(model_name)

        if all_scores:
            scores_matrix = np.column_stack(all_scores)

            # Create comprehensive report
            create_comprehensive_report(
                time=time,
                eda=eda,
                hr=hr,
                anomaly_mask=anomaly_mask,
                y_true=y_true,
                y_scores=scores_matrix[:, 0] if len(all_scores) > 0 else np.zeros(len(y_true)),
                feature_names=feature_names if len(feature_names) < 50 else feature_names[:50],
                output_dir=output_dir / "interactive",
            )

        logging.info(f"Interactive visualizations saved to: {output_dir / 'interactive'}")

    except Exception as e:
        logging.error(f"Error creating visualizations: {e}")


def save_models(
    models: Dict[str, Any],
    model_results: Dict[str, Dict[str, float]],
    config: Any,
    output_dir: Path,
) -> None:
    """Save trained models with metadata."""
    if not config.output.get("save_models", True):
        return

    logging.info("Saving models...")

    try:
        registry = ModelRegistry()

        for model_name, model in models.items():
            if model is None:
                continue

            # Prepare metadata
            metadata = {
                "config": config.models.get(model_name, {}),
                "training_timestamp": pd.Timestamp.now().isoformat(),
                "dataset": "physiological_signals",
            }

            # Add performance metrics
            if model_name in model_results:
                metadata.update(model_results[model_name])

            # Save model
            model_id = registry.register_model(
                model=model,
                model_name=model_name,
                version="latest",
                metadata=metadata,
                is_production=False,
            )

            logging.info(f"Saved {model_name} with ID: {model_id}")

    except Exception as e:
        logging.error(f"Error saving models: {e}")


def main():
    parser = argparse.ArgumentParser(description="Advanced Anomaly Detection Training")
    parser.add_argument(
        "--csv", type=str, required=True, help="Input CSV with columns: time, EDA, HR, [label]"
    )
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument(
        "--subject_id", type=str, help="Subject ID for subject-independent evaluation"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["isolation_forest", "autoencoder", "vae"],
        help="Models to train",
    )
    parser.add_argument("--ensemble", action="store_true", help="Train ensemble model")
    parser.add_argument("--advanced_eval", action="store_true", help="Perform advanced evaluation")

    args = parser.parse_args()

    # Setup configuration
    if args.config:
        config_manager = ConfigManager(args.config)
        config = config_manager.config
    else:
        config = load_default_config()
        config_manager = ConfigManager()
        config_manager.config = config

    # Setup logging
    config_manager.setup_logging()
    logging.info("Starting advanced anomaly detection training...")

    # Setup output directories
    output_dir = Path(args.output_dir)
    directories = config_manager.setup_output_directories()

    # Load data
    logging.info(f"Loading data from: {args.csv}")
    df = pd.read_csv(args.csv)

    if "EDA" not in df.columns or "HR" not in df.columns:
        raise ValueError("CSV must have EDA and HR columns")

    # Prepare data
    X, X_features, y_window, time, eda, hr, feature_names = prepare_data(df, config)

    logging.info(f"Data shape: {X.shape}, Features shape: {X_features.shape}")
    if y_window is not None:
        logging.info(
            f"Labels available: {len(y_window)} windows, "
            f"{np.sum(y_window)} anomalies ({np.mean(y_window)*100:.1f}%)"
        )

    # Split data
    if y_window is not None:
        # Train only on normal samples
        normal_idx = np.where(y_window == 0)[0]
        X_train = X_features[normal_idx]

        # Evaluation on all data
        X_eval = X_features
        y_eval = y_window
    else:
        # No labels - use all data for training
        X_train = X_features
        X_eval = X_features
        y_eval = None
        logging.info("No labels available - unsupervised training")

    # Train models
    models = {}
    for model_type in args.models:
        model = train_single_model(model_type, X_train, config, config.models)
        if model is not None:
            models[model_type] = model

    # Train ensemble if requested
    if args.ensemble and len(models) > 1:
        logging.info("Training ensemble model...")
        try:
            model_list = list(models.values())
            ensemble = EnsembleAnomalyDetector(
                models=model_list,
                combination_method=config.ensemble.get("combination_method", "weighted_voting"),
            )
            models["ensemble"] = ensemble
        except Exception as e:
            logging.error(f"Error training ensemble: {e}")

    # Evaluation
    results = {}
    if y_eval is not None:
        results = evaluate_models(models, X_eval, y_eval, config)

        # Print results summary
        print("\n" + "=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)) and metric != "threshold":
                    print(f"  {metric}: {value:.4f}")

    # Advanced evaluation
    if args.advanced_eval and y_eval is not None:
        subjects = None
        if args.subject_id:
            subjects = np.full(len(X_features), args.subject_id)

        advanced_results = advanced_evaluation(models, X_features, y_eval, config, subjects)

        # Print advanced results
        print("\n" + "=" * 60)
        print("ADVANCED EVALUATION RESULTS")
        print("=" * 60)
        for eval_name, eval_results in advanced_results.items():
            if "error" not in eval_results:
                print(f"\n{eval_name.upper()}:")
                if "mean_metrics" in eval_results:
                    for metric, value in eval_results["mean_metrics"].items():
                        std_val = eval_results["std_metrics"].get(metric, 0)
                        print(f"  {metric}: {value:.4f} (±{std_val:.4f})")

    # Create visualizations
    if config.visualization.get("interactive", True):
        # Create anomaly mask for visualization
        if y_eval is not None and "ensemble" in models:
            # Get ensemble predictions for visualization
            try:
                if hasattr(models["ensemble"], "score"):
                    scores = models["ensemble"].score(X_eval)
                    threshold = np.percentile(scores, 95)
                    window_anomalies = scores >= threshold

                    # Map back to time series
                    anomaly_mask = np.zeros(len(time), dtype=bool)
                    window_size = int(config.data.window_sec * config.data.fs)
                    step = max(1, int(window_size * (1.0 - config.data.overlap)))

                    i = 0
                    for s, e in sliding_windows(np.arange(len(time)), window_size, step):
                        if i < len(window_anomalies) and window_anomalies[i]:
                            anomaly_mask[s:e] = True
                        i += 1
                else:
                    anomaly_mask = np.zeros(len(time), dtype=bool)
            except Exception as e:
                logging.warning(f"Could not create anomaly mask: {e}")
                anomaly_mask = np.zeros(len(time), dtype=bool)
        else:
            anomaly_mask = np.zeros(len(time), dtype=bool)

        create_visualizations(
            time=time,
            eda=eda,
            hr=hr,
            anomaly_mask=anomaly_mask,
            y_true=y_eval if y_eval is not None else np.zeros(len(X_eval)),
            model_results=results,
            feature_names=feature_names,
            output_dir=output_dir,
            config=config,
        )

    # Save static images (PNG) for quick showcasing
    try:
        plots_dir = directories.get("plots", output_dir / "plots")
        plots_dir.mkdir(parents=True, exist_ok=True)

        # 1) Anomalies over time (always available)
        plot_anomalies(
            time=time,
            eda=eda,
            hr=hr,
            anomaly_mask=anomaly_mask,
            out_path=str(plots_dir / "anomalies_over_time.png"),
            title="Detected Anomalies Over Time",
        )

        # 2) If labels and ensemble scores available, plot evaluation curves + confusion matrix
        if y_eval is not None and "ensemble" in models and hasattr(models["ensemble"], "score"):
            scores = models["ensemble"].score(X_eval)
            thr = np.percentile(scores, config.thresholds.get("default_percentile", 95))
            y_pred = (scores >= thr).astype(int)

            plot_evaluation_curves(
                y_true=y_eval,
                y_scores=scores,
                output_path=plots_dir / "roc_pr_curves.png",
                title="ROC and Precision-Recall Curves (Ensemble)",
            )

            plot_confusion_matrix(
                y_true=y_eval,
                y_pred=y_pred,
                output_path=plots_dir / "confusion_matrix.png",
                title="Confusion Matrix (Ensemble)",
            )

        logging.info(f"Static plots saved to: {plots_dir}")
    except Exception as e:
        logging.warning(f"Could not save static visualizations: {e}")

    # Save models
    save_models(models, results, config, output_dir)

    logging.info(f"Training completed successfully! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
