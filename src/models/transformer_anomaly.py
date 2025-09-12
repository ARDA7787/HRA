#!/usr/bin/env python3
"""
Transformer-based anomaly detection for physiological time series.
Uses a Transformer encoder to learn temporal patterns and detect anomalies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerEncoder(nn.Module):
    """Transformer encoder for anomaly detection."""
    
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 8, 
                 num_layers: int = 3, dim_feedforward: int = 256, 
                 dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.output_projection = nn.Linear(d_model, input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: [batch_size, seq_len, input_dim]
            src_mask: [seq_len, seq_len] attention mask
        Returns:
            reconstructed: [batch_size, seq_len, input_dim]
        """
        # Project input to model dimension
        src = self.input_projection(src) * math.sqrt(src.size(-1))
        src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1)
        src = self.dropout(src)
        
        # Apply transformer
        output = self.transformer_encoder(src, src_mask)
        
        # Project back to input dimension
        reconstructed = self.output_projection(output)
        
        return reconstructed


class TransformerAnomalyDetector:
    """Transformer-based anomaly detector for physiological signals."""
    
    def __init__(self, input_dim: int = 2, d_model: int = 64, nhead: int = 8,
                 num_layers: int = 3, dim_feedforward: int = 256, dropout: float = 0.1,
                 lr: float = 1e-3, epochs: int = 50, batch_size: int = 32,
                 device: Optional[str] = None):
        """
        Initialize Transformer anomaly detector.
        
        Args:
            input_dim: Input feature dimension (2 for EDA + HR)
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            lr: Learning rate
            epochs: Training epochs
            batch_size: Batch size
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        # Handle device configuration
        if device == "auto" or device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model.to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
    def _prepare_data(self, X: np.ndarray) -> torch.Tensor:
        """
        Prepare data for transformer training.
        X should be [num_windows, window_size * input_dim] - reshape to [num_windows, window_size, input_dim]
        """
        if X.ndim == 2:
            # Reshape from [num_windows, window_size * input_dim] to [num_windows, window_size, input_dim]
            num_windows = X.shape[0]
            window_size = X.shape[1] // 2  # Assuming input_dim = 2
            X_reshaped = X.reshape(num_windows, window_size, 2)
        else:
            X_reshaped = X
            
        return torch.tensor(X_reshaped, dtype=torch.float32)
    
    def fit(self, X: np.ndarray) -> 'TransformerAnomalyDetector':
        """
        Train the transformer on normal data.
        
        Args:
            X: Training data [num_windows, window_size * input_dim]
        """
        X_tensor = self._prepare_data(X)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        train_losses = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for (batch_data,) in dataloader:
                batch_data = batch_data.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                reconstructed = self.model(batch_data)
                loss = self.loss_fn(reconstructed, batch_data)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            train_losses.append(avg_loss)
            self.scheduler.step(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")
                
        self.model.eval()  # Set to evaluation mode after training            
        return self
    
    def reconstruction_error(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """
        Compute reconstruction error for anomaly detection.
        
        Args:
            X: Data to evaluate [num_windows, window_size * input_dim]
            batch_size: Batch size for evaluation
            
        Returns:
            Reconstruction errors [num_windows]
        """
        X_tensor = self._prepare_data(X)
        self.model.eval()
        
        errors = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch_data = X_tensor[i:i+batch_size].to(self.device)
                reconstructed = self.model(batch_data)
                
                # Compute MSE for each sample
                mse = torch.mean((reconstructed - batch_data) ** 2, dim=(1, 2))
                errors.append(mse.cpu().numpy())
        
        return np.concatenate(errors) if errors else np.array([])
    
    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """
        Predict anomalies based on reconstruction error threshold.
        
        Args:
            X: Data to predict [num_windows, window_size * input_dim]
            threshold: Anomaly threshold
            
        Returns:
            Binary predictions [num_windows] (1 = anomaly)
        """
        errors = self.reconstruction_error(X)
        return (errors >= threshold).astype(int)
    
    def get_attention_weights(self, X: np.ndarray, layer_idx: int = -1) -> np.ndarray:
        """
        Extract attention weights from a specific transformer layer.
        
        Args:
            X: Input data [num_windows, window_size * input_dim]
            layer_idx: Layer index to extract attention from (-1 for last layer)
            
        Returns:
            Attention weights [num_windows, num_heads, seq_len, seq_len]
        """
        X_tensor = self._prepare_data(X)
        self.model.eval()
        
        # Hook to capture attention weights
        attention_weights = []
        
        def attention_hook(module, input, output):
            # For TransformerEncoderLayer, we need to access the self-attention
            if hasattr(module, 'self_attn'):
                # This is a simplified approach - in practice, you'd need to modify
                # the transformer to return attention weights
                pass
        
        # Register hook
        target_layer = self.model.transformer_encoder.layers[layer_idx]
        hook = target_layer.register_forward_hook(attention_hook)
        
        try:
            with torch.no_grad():
                X_device = X_tensor.to(self.device)
                _ = self.model(X_device)
        finally:
            hook.remove()
        
        # Note: This is a placeholder - actual attention weight extraction
        # would require modifying the transformer architecture
        return np.zeros((len(X), 8, X_tensor.shape[1], X_tensor.shape[1]))
