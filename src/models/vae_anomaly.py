#!/usr/bin/env python3
"""
Variational Autoencoder (VAE) for anomaly detection in physiological signals.
Uses probabilistic encoding to detect out-of-distribution patterns.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple


class VAE(nn.Module):
    """Variational Autoencoder for anomaly detection."""
    
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: Optional[list] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var


def vae_loss_function(x_recon: torch.Tensor, x: torch.Tensor, 
                      mu: torch.Tensor, log_var: torch.Tensor, 
                      beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss function combining reconstruction loss and KL divergence.
    
    Args:
        x_recon: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        log_var: Log variance of latent distribution
        beta: Weight for KL divergence term (beta-VAE)
    
    Returns:
        Total loss, reconstruction loss, KL divergence
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # KL divergence
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_div
    
    return total_loss, recon_loss, kl_div


class VAEAnomalyDetector:
    """VAE-based anomaly detector for physiological signals."""
    
    def __init__(self, input_dim: int, latent_dim: int = 32, 
                 hidden_dims: Optional[list] = None, beta: float = 1.0,
                 lr: float = 1e-3, epochs: int = 100, batch_size: int = 128,
                 device: Optional[str] = None):
        """
        Initialize VAE anomaly detector.
        
        Args:
            input_dim: Input feature dimension
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions
            beta: Beta parameter for beta-VAE (controls KL weight)
            lr: Learning rate
            epochs: Training epochs
            batch_size: Batch size
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = VAE(input_dim, latent_dim, hidden_dims)
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        # Handle device configuration
        if device == "auto" or device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=15
        )
        
        # Track training statistics
        self.training_losses = []
        self.recon_losses = []
        self.kl_losses = []
        
    def fit(self, X: np.ndarray) -> 'VAEAnomalyDetector':
        """
        Train the VAE on normal data.
        
        Args:
            X: Training data [num_samples, input_dim]
        """
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            num_batches = 0
            
            for (batch_data,) in dataloader:
                batch_data = batch_data.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                x_recon, mu, log_var = self.model(batch_data)
                
                # Compute loss
                total_loss, recon_loss, kl_loss = vae_loss_function(
                    x_recon, batch_data, mu, log_var, self.beta
                )
                
                # Normalize by batch size
                total_loss = total_loss / len(batch_data)
                recon_loss = recon_loss / len(batch_data)
                kl_loss = kl_loss / len(batch_data)
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            avg_recon = epoch_recon / num_batches if num_batches > 0 else 0.0
            avg_kl = epoch_kl / num_batches if num_batches > 0 else 0.0
            
            self.training_losses.append(avg_loss)
            self.recon_losses.append(avg_recon)
            self.kl_losses.append(avg_kl)
            
            self.scheduler.step(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, "
                      f"Loss: {avg_loss:.6f}, Recon: {avg_recon:.6f}, KL: {avg_kl:.6f}")
        
        self.model.eval()  # Set to evaluation mode after training
        return self
    
    def reconstruction_error(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """
        Compute reconstruction error for anomaly detection.
        
        Args:
            X: Data to evaluate [num_samples, input_dim]
            batch_size: Batch size for evaluation
            
        Returns:
            Reconstruction errors [num_samples]
        """
        self.model.eval()
        errors = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_data = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=self.device)
                x_recon, _, _ = self.model(batch_data)
                
                # Compute MSE for each sample
                mse = torch.mean((x_recon - batch_data) ** 2, dim=1)
                errors.append(mse.cpu().numpy())
        
        return np.concatenate(errors) if errors else np.array([])
    
    def anomaly_score(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """
        Compute anomaly score using both reconstruction error and latent space probability.
        
        Args:
            X: Data to evaluate [num_samples, input_dim]
            batch_size: Batch size for evaluation
            
        Returns:
            Anomaly scores [num_samples]
        """
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_data = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=self.device)
                x_recon, mu, log_var = self.model(batch_data)
                
                # Reconstruction error
                recon_error = torch.mean((x_recon - batch_data) ** 2, dim=1)
                
                # Latent space probability (negative log likelihood)
                # For standard normal prior
                latent_prob = 0.5 * torch.sum(mu ** 2 + torch.exp(log_var) - log_var - 1, dim=1)
                
                # Combined score
                combined_score = recon_error + self.beta * latent_prob
                scores.append(combined_score.cpu().numpy())
        
        return np.concatenate(scores) if scores else np.array([])
    
    def predict(self, X: np.ndarray, threshold: float, use_combined_score: bool = True) -> np.ndarray:
        """
        Predict anomalies based on threshold.
        
        Args:
            X: Data to predict [num_samples, input_dim]
            threshold: Anomaly threshold
            use_combined_score: Whether to use combined score or just reconstruction error
            
        Returns:
            Binary predictions [num_samples] (1 = anomaly)
        """
        if use_combined_score:
            scores = self.anomaly_score(X)
        else:
            scores = self.reconstruction_error(X)
        
        return (scores >= threshold).astype(int)
    
    def generate_samples(self, num_samples: int) -> np.ndarray:
        """
        Generate new samples from the learned distribution.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples [num_samples, input_dim]
        """
        self.model.eval()
        
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_samples, self.model.latent_dim, device=self.device)
            
            # Decode to data space
            generated = self.model.decode(z)
            
        return generated.cpu().numpy()
    
    def get_latent_representation(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """
        Get latent space representation of input data.
        
        Args:
            X: Input data [num_samples, input_dim]
            batch_size: Batch size for processing
            
        Returns:
            Latent representations [num_samples, latent_dim]
        """
        self.model.eval()
        latent_reps = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_data = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=self.device)
                mu, _ = self.model.encode(batch_data)
                latent_reps.append(mu.cpu().numpy())
        
        return np.concatenate(latent_reps) if latent_reps else np.array([])
