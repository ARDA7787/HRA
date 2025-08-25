#!/usr/bin/env python3
"""
Interactive visualizations for anomaly detection using Plotly.
Provides rich, interactive plots for better analysis and presentation.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')


class InteractiveVisualizer:
    """Interactive visualizer for anomaly detection results."""
    
    def __init__(self, theme: str = 'plotly_white'):
        """
        Initialize visualizer.
        
        Args:
            theme: Plotly theme ('plotly', 'plotly_white', 'plotly_dark', etc.)
        """
        self.theme = theme
        
    def plot_time_series_with_anomalies(self, time: np.ndarray, eda: np.ndarray, 
                                      hr: np.ndarray, anomaly_mask: np.ndarray,
                                      uncertainty: Optional[np.ndarray] = None,
                                      title: str = "Physiological Signals with Anomaly Detection",
                                      output_path: Optional[Path] = None) -> go.Figure:
        """
        Plot interactive time series with anomaly detection results.
        
        Args:
            time: Time array
            eda: EDA signal
            hr: HR signal
            anomaly_mask: Boolean mask for anomalies
            uncertainty: Uncertainty scores (optional)
            title: Plot title
            output_path: Path to save plot
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=3 if uncertainty is not None else 2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=['EDA (Electrodermal Activity)', 'HR (Heart Rate)', 'Uncertainty'] if uncertainty is not None else ['EDA (Electrodermal Activity)', 'HR (Heart Rate)'],
            vertical_spacing=0.08
        )
        
        # EDA trace
        fig.add_trace(
            go.Scatter(
                x=time,
                y=eda,
                mode='lines',
                name='EDA',
                line=dict(color='blue', width=1),
                hovertemplate='Time: %{x:.1f}s<br>EDA: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # HR trace
        fig.add_trace(
            go.Scatter(
                x=time,
                y=hr,
                mode='lines',
                name='HR',
                line=dict(color='red', width=1),
                hovertemplate='Time: %{x:.1f}s<br>HR: %{y:.1f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Anomaly markers
        anomaly_indices = np.where(anomaly_mask)[0]
        if len(anomaly_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=time[anomaly_indices],
                    y=eda[anomaly_indices],
                    mode='markers',
                    name='Anomalies (EDA)',
                    marker=dict(color='orange', size=6, symbol='x'),
                    hovertemplate='Time: %{x:.1f}s<br>EDA: %{y:.3f}<br>ANOMALY<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=time[anomaly_indices],
                    y=hr[anomaly_indices],
                    mode='markers',
                    name='Anomalies (HR)',
                    marker=dict(color='orange', size=6, symbol='x'),
                    hovertemplate='Time: %{x:.1f}s<br>HR: %{y:.1f}<br>ANOMALY<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Uncertainty plot
        if uncertainty is not None:
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=uncertainty,
                    mode='lines',
                    name='Uncertainty',
                    line=dict(color='purple', width=1),
                    fill='tonexty',
                    hovertemplate='Time: %{x:.1f}s<br>Uncertainty: %{y:.3f}<extra></extra>'
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=600 if uncertainty is None else 800,
            showlegend=True,
            hovermode='closest',
            template=self.theme
        )
        
        fig.update_xaxes(title_text="Time (seconds)", row=3 if uncertainty is not None else 2, col=1)
        fig.update_yaxes(title_text="EDA (z-score)", row=1, col=1)
        fig.update_yaxes(title_text="HR (z-score)", row=2, col=1)
        if uncertainty is not None:
            fig.update_yaxes(title_text="Uncertainty", row=3, col=1)
        
        if output_path:
            fig.write_html(str(output_path))
        
        return fig
    
    def plot_roc_pr_curves(self, y_true: np.ndarray, y_scores: np.ndarray,
                          model_names: Optional[List[str]] = None,
                          title: str = "ROC and Precision-Recall Curves",
                          output_path: Optional[Path] = None) -> go.Figure:
        """
        Plot interactive ROC and Precision-Recall curves.
        
        Args:
            y_true: True binary labels
            y_scores: Anomaly scores (can be 2D for multiple models)
            model_names: Names of models (optional)
            title: Plot title
            output_path: Path to save plot
            
        Returns:
            Plotly figure
        """
        # Handle single model case
        if y_scores.ndim == 1:
            y_scores = y_scores.reshape(-1, 1)
        
        n_models = y_scores.shape[1]
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(n_models)]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['ROC Curve', 'Precision-Recall Curve'],
            horizontal_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set1[:n_models]
        
        for i in range(n_models):
            scores = y_scores[:, i]
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, scores)
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, scores)
            
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'{model_names[i]} (AUC = {auc:.3f})',
                    line=dict(color=colors[i], width=2),
                    hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # PR Curve
            precision, recall, _ = precision_recall_curve(y_true, scores)
            from sklearn.metrics import average_precision_score
            ap = average_precision_score(y_true, scores)
            
            fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode='lines',
                    name=f'{model_names[i]} (AP = {ap:.3f})',
                    line=dict(color=colors[i], width=2),
                    hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Add diagonal lines
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        baseline_precision = np.sum(y_true) / len(y_true)
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[baseline_precision, baseline_precision],
                mode='lines',
                name='Random',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=500,
            template=self.theme
        )
        
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)
        
        if output_path:
            fig.write_html(str(output_path))
        
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: List[str] = None,
                             title: str = "Confusion Matrix",
                             output_path: Optional[Path] = None) -> go.Figure:
        """
        Plot interactive confusion matrix heatmap.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            class_names: Class names
            title: Plot title
            output_path: Path to save plot
            
        Returns:
            Plotly figure
        """
        if class_names is None:
            class_names = ['Normal', 'Anomaly']
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations
        annotations = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                annotations.append(
                    dict(
                        x=j, y=i,
                        text=f"{cm[i, j]}<br>({cm_percent[i, j]:.1f}%)",
                        showarrow=False,
                        font=dict(color="white" if cm[i, j] > cm.max() / 2 else "black", size=14)
                    )
                )
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            annotations=annotations,
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=500,
            template=self.theme
        )
        
        if output_path:
            fig.write_html(str(output_path))
        
        return fig
    
    def plot_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray,
                              title: str = "Feature Importance",
                              top_k: int = 20,
                              output_path: Optional[Path] = None) -> go.Figure:
        """
        Plot interactive feature importance bar chart.
        
        Args:
            feature_names: Names of features
            importance_scores: Importance scores
            title: Plot title
            top_k: Number of top features to show
            output_path: Path to save plot
            
        Returns:
            Plotly figure
        """
        # Sort by importance
        sorted_idx = np.argsort(importance_scores)[::-1][:top_k]
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_scores = importance_scores[sorted_idx]
        
        fig = go.Figure(data=[
            go.Bar(
                x=sorted_scores,
                y=sorted_names,
                orientation='h',
                marker_color=px.colors.sequential.Viridis,
                hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(400, len(sorted_names) * 25),
            yaxis=dict(autorange="reversed"),
            template=self.theme
        )
        
        if output_path:
            fig.write_html(str(output_path))
        
        return fig
    
    def plot_latent_space(self, latent_vectors: np.ndarray, labels: np.ndarray,
                         method: str = 'tsne',
                         title: str = "Latent Space Visualization",
                         output_path: Optional[Path] = None) -> go.Figure:
        """
        Plot 2D visualization of latent space.
        
        Args:
            latent_vectors: Latent representations [N, D]
            labels: Binary labels
            method: Dimensionality reduction method ('tsne', 'pca')
            title: Plot title
            output_path: Path to save plot
            
        Returns:
            Plotly figure
        """
        if latent_vectors.shape[1] > 2:
            if method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42)
            elif method == 'pca':
                reducer = PCA(n_components=2, random_state=42)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            latent_2d = reducer.fit_transform(latent_vectors)
        else:
            latent_2d = latent_vectors
        
        df = pd.DataFrame({
            'x': latent_2d[:, 0],
            'y': latent_2d[:, 1],
            'label': ['Anomaly' if l == 1 else 'Normal' for l in labels]
        })
        
        fig = px.scatter(
            df, x='x', y='y', color='label',
            color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
            title=title,
            hover_data={'x': ':.3f', 'y': ':.3f'},
            template=self.theme
        )
        
        fig.update_layout(
            xaxis_title=f"{method.upper()} Component 1",
            yaxis_title=f"{method.upper()} Component 2",
            height=600
        )
        
        if output_path:
            fig.write_html(str(output_path))
        
        return fig
    
    def plot_score_distribution(self, scores: np.ndarray, labels: np.ndarray,
                              threshold: Optional[float] = None,
                              title: str = "Anomaly Score Distribution",
                              output_path: Optional[Path] = None) -> go.Figure:
        """
        Plot distribution of anomaly scores for normal and anomalous samples.
        
        Args:
            scores: Anomaly scores
            labels: Binary labels
            threshold: Decision threshold (optional)
            title: Plot title
            output_path: Path to save plot
            
        Returns:
            Plotly figure
        """
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]
        
        fig = go.Figure()
        
        # Normal distribution
        fig.add_trace(go.Histogram(
            x=normal_scores,
            name='Normal',
            opacity=0.7,
            nbinsx=50,
            marker_color='blue',
            hovertemplate='Score: %{x:.3f}<br>Count: %{y}<extra></extra>'
        ))
        
        # Anomaly distribution
        fig.add_trace(go.Histogram(
            x=anomaly_scores,
            name='Anomaly',
            opacity=0.7,
            nbinsx=50,
            marker_color='red',
            hovertemplate='Score: %{x:.3f}<br>Count: %{y}<extra></extra>'
        ))
        
        # Add threshold line
        if threshold is not None:
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="black",
                annotation_text=f"Threshold: {threshold:.3f}"
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Anomaly Score",
            yaxis_title="Count",
            barmode='overlay',
            height=500,
            template=self.theme
        )
        
        if output_path:
            fig.write_html(str(output_path))
        
        return fig
    
    def create_dashboard(self, results_dict: Dict[str, Any],
                        output_path: Optional[Path] = None) -> str:
        """
        Create an interactive dashboard combining multiple visualizations.
        
        Args:
            results_dict: Dictionary containing visualization data
            output_path: Path to save dashboard HTML
            
        Returns:
            HTML string of the dashboard
        """
        html_parts = []
        
        # HTML header
        html_parts.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Anomaly Detection Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .plot-container { margin: 20px 0; }
                .title { text-align: center; color: #2c3e50; }
                .section { margin: 40px 0; }
            </style>
        </head>
        <body>
            <h1 class="title">Anomaly Detection Dashboard</h1>
        """)
        
        # Add each plot
        plot_counter = 0
        for plot_name, plot_data in results_dict.items():
            if isinstance(plot_data, go.Figure):
                plot_id = f"plot_{plot_counter}"
                html_parts.append(f'<div class="section"><div id="{plot_id}" class="plot-container"></div></div>')
                
                # Add JavaScript to render the plot
                plot_json = plot_data.to_json()
                html_parts.append(f"""
                <script>
                    Plotly.newPlot('{plot_id}', {plot_json});
                </script>
                """)
                plot_counter += 1
        
        # Close HTML
        html_parts.append("</body></html>")
        
        html_content = "\n".join(html_parts)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html_content)
        
        return html_content


def create_comprehensive_report(time: np.ndarray, eda: np.ndarray, hr: np.ndarray,
                               anomaly_mask: np.ndarray, y_true: np.ndarray, 
                               y_scores: np.ndarray, feature_names: List[str] = None,
                               feature_importance: np.ndarray = None,
                               latent_vectors: np.ndarray = None,
                               output_dir: Path = Path("outputs/interactive")) -> None:
    """
    Create a comprehensive interactive report with all visualizations.
    
    Args:
        time: Time array
        eda: EDA signal
        hr: HR signal
        anomaly_mask: Boolean mask for anomalies
        y_true: True binary labels
        y_scores: Anomaly scores
        feature_names: Feature names (optional)
        feature_importance: Feature importance scores (optional)
        latent_vectors: Latent representations (optional)
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = InteractiveVisualizer()
    
    # Time series plot
    fig_ts = visualizer.plot_time_series_with_anomalies(
        time, eda, hr, anomaly_mask,
        title="Physiological Signals with Anomaly Detection",
        output_path=output_dir / "time_series.html"
    )
    
    # ROC and PR curves
    fig_curves = visualizer.plot_roc_pr_curves(
        y_true, y_scores,
        title="Model Performance Curves",
        output_path=output_dir / "performance_curves.html"
    )
    
    # Confusion matrix
    threshold = np.percentile(y_scores, 95)
    y_pred = (y_scores >= threshold).astype(int)
    fig_cm = visualizer.plot_confusion_matrix(
        y_true, y_pred,
        title="Confusion Matrix",
        output_path=output_dir / "confusion_matrix.html"
    )
    
    # Score distribution
    fig_dist = visualizer.plot_score_distribution(
        y_scores, y_true, threshold,
        title="Anomaly Score Distribution",
        output_path=output_dir / "score_distribution.html"
    )
    
    plots = {
        "time_series": fig_ts,
        "performance_curves": fig_curves,
        "confusion_matrix": fig_cm,
        "score_distribution": fig_dist
    }
    
    # Feature importance (if available)
    if feature_names is not None and feature_importance is not None:
        fig_fi = visualizer.plot_feature_importance(
            feature_names, feature_importance,
            title="Feature Importance",
            output_path=output_dir / "feature_importance.html"
        )
        plots["feature_importance"] = fig_fi
    
    # Latent space (if available)
    if latent_vectors is not None:
        fig_latent = visualizer.plot_latent_space(
            latent_vectors, y_true,
            title="Latent Space Visualization",
            output_path=output_dir / "latent_space.html"
        )
        plots["latent_space"] = fig_latent
    
    # Create dashboard
    dashboard_html = visualizer.create_dashboard(
        plots, output_path=output_dir / "dashboard.html"
    )
    
    print(f"Interactive report saved to: {output_dir}")
    print(f"Open {output_dir / 'dashboard.html'} in your browser to view the full report.")
