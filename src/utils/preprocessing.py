import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def interpolate_missing(series: pd.Series) -> pd.Series:
    # Linear interpolation, then forward/backward fill
    s = series.copy()
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.interpolate(method='linear').ffill().bfill()


def zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - mu) / sd


def butter_lowpass(data: np.ndarray, fs: float, cutoff: float = 1.0, order: int = 4) -> np.ndarray:
    # Low-pass filter for EDA (reduce motion artifacts)
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.99)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def sliding_windows(arr: np.ndarray, window_size: int, step: int):
    # Generate sliding windows indices (start,end)
    n = len(arr)
    i = 0
    while i + window_size <= n:
        yield i, i + window_size
        i += step


def build_window_matrix(X: np.ndarray, window_size: int, step: int) -> np.ndarray:
    # Shape: [num_windows, window_size * num_features]
    feats = []
    for s, e in sliding_windows(X, window_size, step):
        feats.append(X[s:e].reshape(-1))
    if not feats:
        return np.zeros((0, window_size * X.shape[1]))
    return np.vstack(feats)


def inject_synthetic_anomalies(X: np.ndarray, rate: float = 0.02, scale: float = 3.0, rng: np.random.RandomState | None = None) -> np.ndarray:
    # Add sparse spikes to improve robustness for unlabeled data
    if rng is None:
        rng = np.random.RandomState(42)
    X_aug = X.copy()
    n, d = X_aug.shape
    num_spikes = int(n * rate)
    if num_spikes <= 0:
        return X_aug
    idx = rng.choice(n, size=num_spikes, replace=False)
    ch = rng.choice(d, size=num_spikes, replace=True)
    X_aug[idx, ch] += scale * (rng.randn(num_spikes))
    return X_aug