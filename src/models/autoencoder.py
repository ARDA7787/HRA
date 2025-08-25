from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DenseAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class AEAnomalyDetector:
    def __init__(self, input_dim: int, latent_dim: int = 32, lr: float = 1e-3, epochs: int = 30, batch_size: int = 128, device: str | None = None):
        self.model = DenseAE(input_dim, latent_dim)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.loss_fn = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X: np.ndarray):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for _ in range(self.epochs):
            for (xb,) in dl:
                xb = xb.to(self.device)
                self.opt.zero_grad()
                xh = self.model(xb)
                loss = self.loss_fn(xh, xb)
                loss.backward()
                self.opt.step()
        return self

    def reconstruction_error(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        self.model.eval()
        errs = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=self.device)
                xh = self.model(xb)
                e = torch.mean((xh - xb) ** 2, dim=1)
                errs.append(e.cpu().numpy())
        return np.concatenate(errs)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        scores = self.reconstruction_error(X)
        return (scores >= threshold).astype(int)  # 1 = anomaly