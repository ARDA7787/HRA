#!/usr/bin/env python3
"""
LSTM-based Autoencoder for anomaly detection in physiological time series.
Captures temporal dependencies in sequential data.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple


class LSTMEncoder(nn.Module):
    """LSTM encoder for sequence encoding."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, 
                 latent_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sequence to latent representation.
        
        Args:
            x: Input sequence [batch_size, seq_len, input_dim]
            
        Returns:
            latent: Latent representation [batch_size, latent_dim]
            hidden_states: All hidden states [batch_size, seq_len, hidden_dim]
        """
        # LSTM forward pass
        hidden_states, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state for latent representation
        last_hidden = h_n[-1]  # [batch_size, hidden_dim]
        last_hidden = self.dropout(last_hidden)
        
        # Project to latent space
        latent = self.fc_latent(last_hidden)
        
        return latent, hidden_states


class LSTMDecoder(nn.Module):
    """LSTM decoder for sequence reconstruction."""
    
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, 
                 seq_len: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.output_dim = output_dim
        
        # Project latent to initial hidden state
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.fc_cell = nn.Linear(latent_dim, hidden_dim * num_layers)
        
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to sequence.
        
        Args:
            latent: Latent representation [batch_size, latent_dim]
            
        Returns:
            reconstructed: Reconstructed sequence [batch_size, seq_len, output_dim]
        """
        batch_size = latent.size(0)
        
        # Initialize hidden and cell states
        h_0 = self.fc_hidden(latent).view(batch_size, self.num_layers, self.hidden_dim)
        h_0 = h_0.transpose(0, 1).contiguous()  # [num_layers, batch_size, hidden_dim]
        
        c_0 = self.fc_cell(latent).view(batch_size, self.num_layers, self.hidden_dim)
        c_0 = c_0.transpose(0, 1).contiguous()
        
        # Prepare input for decoder (repeat latent for each time step)
        decoder_input = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # LSTM forward pass
        lstm_output, _ = self.lstm(decoder_input, (h_0, c_0))
        lstm_output = self.dropout(lstm_output)
        
        # Project to output dimension
        reconstructed = self.fc_output(lstm_output)
        
        return reconstructed


class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for sequence reconstruction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 32,
                 seq_len: int = 120, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers, latent_dim, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers, dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LSTM autoencoder.
        
        Args:
            x: Input sequence [batch_size, seq_len, input_dim]
            
        Returns:
            reconstructed: Reconstructed sequence [batch_size, seq_len, input_dim]
            latent: Latent representation [batch_size, latent_dim]
        """
        latent, _ = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


class LSTMAnomalyDetector:
    """LSTM-based anomaly detector for physiological signals."""
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, latent_dim: int = 32,
                 seq_len: int = 120, num_layers: int = 2, dropout: float = 0.2,
                 lr: float = 1e-3, epochs: int = 100, batch_size: int = 32,
                 device: Optional[str] = None):
        """
        Initialize LSTM anomaly detector.
        
        Args:
            input_dim: Input feature dimension (2 for EDA + HR)
            hidden_dim: LSTM hidden dimension
            latent_dim: Latent space dimension
            seq_len: Sequence length
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            lr: Learning rate
            epochs: Training epochs
            batch_size: Batch size
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = LSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.seq_len = seq_len
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=15, verbose=False
        )
        
        # Track training statistics
        self.training_losses = []
        
    def _prepare_data(self, X: np.ndarray) -> torch.Tensor:
        """
        Prepare data for LSTM training.
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
    
    def fit(self, X: np.ndarray) -> 'LSTMAnomalyDetector':
        """
        Train the LSTM autoencoder on normal data.
        
        Args:
            X: Training data [num_windows, window_size * input_dim]
        """
        X_tensor = self._prepare_data(X)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for (batch_data,) in dataloader:
                batch_data = batch_data.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                reconstructed, latent = self.model(batch_data)
                loss = self.loss_fn(reconstructed, batch_data)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            self.training_losses.append(avg_loss)
            self.scheduler.step(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
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
                reconstructed, _ = self.model(batch_data)
                
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
    
    def get_latent_representation(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """
        Get latent space representation of input data.
        
        Args:
            X: Input data [num_windows, window_size * input_dim]
            batch_size: Batch size for processing
            
        Returns:
            Latent representations [num_windows, latent_dim]
        """
        X_tensor = self._prepare_data(X)
        self.model.eval()
        
        latent_reps = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch_data = X_tensor[i:i+batch_size].to(self.device)
                _, latent = self.model(batch_data)
                latent_reps.append(latent.cpu().numpy())
        
        return np.concatenate(latent_reps) if latent_reps else np.array([])
    
    def get_sequence_importance(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """
        Compute importance scores for each time step in the sequence.
        
        Args:
            X: Input data [num_windows, window_size * input_dim]
            batch_size: Batch size for processing
            
        Returns:
            Importance scores [num_windows, seq_len]
        """
        X_tensor = self._prepare_data(X)
        self.model.eval()
        
        importance_scores = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch_data = X_tensor[i:i+batch_size].to(self.device)
                reconstructed, _ = self.model(batch_data)
                
                # Compute MSE for each time step
                timestep_errors = torch.mean((reconstructed - batch_data) ** 2, dim=2)
                importance_scores.append(timestep_errors.cpu().numpy())
        
        return np.concatenate(importance_scores) if importance_scores else np.array([])
