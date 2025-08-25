#!/usr/bin/env python3
"""
Advanced feature engineering for physiological signals (EDA, HR).
Extracts statistical, frequency domain, and time-domain features from sliding windows.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, welch, periodogram
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class PhysiologicalFeatureExtractor:
    """
    Comprehensive feature extractor for physiological signals.
    Extracts multiple types of features from EDA and HR windows.
    """
    
    def __init__(self, fs: float = 4.0):
        """
        Initialize feature extractor.
        
        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
        
    def extract_statistical_features(self, signal: np.ndarray, prefix: str = "") -> Dict[str, float]:
        """Extract statistical features from a signal window."""
        if len(signal) == 0 or np.all(np.isnan(signal)):
            return {f"{prefix}mean": 0.0, f"{prefix}std": 0.0, f"{prefix}min": 0.0, 
                   f"{prefix}max": 0.0, f"{prefix}median": 0.0, f"{prefix}iqr": 0.0,
                   f"{prefix}skewness": 0.0, f"{prefix}kurtosis": 0.0, f"{prefix}rms": 0.0}
        
        # Remove NaN values
        clean_signal = signal[~np.isnan(signal)]
        if len(clean_signal) == 0:
            return {f"{prefix}mean": 0.0, f"{prefix}std": 0.0, f"{prefix}min": 0.0, 
                   f"{prefix}max": 0.0, f"{prefix}median": 0.0, f"{prefix}iqr": 0.0,
                   f"{prefix}skewness": 0.0, f"{prefix}kurtosis": 0.0, f"{prefix}rms": 0.0}
        
        try:
            features = {
                f"{prefix}mean": float(np.mean(clean_signal)),
                f"{prefix}std": float(np.std(clean_signal, ddof=0)),
                f"{prefix}min": float(np.min(clean_signal)),
                f"{prefix}max": float(np.max(clean_signal)),
                f"{prefix}median": float(np.median(clean_signal)),
                f"{prefix}iqr": float(np.percentile(clean_signal, 75) - np.percentile(clean_signal, 25)),
                f"{prefix}skewness": float(stats.skew(clean_signal)) if len(clean_signal) > 2 else 0.0,
                f"{prefix}kurtosis": float(stats.kurtosis(clean_signal)) if len(clean_signal) > 3 else 0.0,
                f"{prefix}rms": float(np.sqrt(np.mean(clean_signal**2))),
            }
        except:
            features = {f"{prefix}mean": 0.0, f"{prefix}std": 0.0, f"{prefix}min": 0.0, 
                       f"{prefix}max": 0.0, f"{prefix}median": 0.0, f"{prefix}iqr": 0.0,
                       f"{prefix}skewness": 0.0, f"{prefix}kurtosis": 0.0, f"{prefix}rms": 0.0}
        
        return features
    
    def extract_frequency_features(self, signal: np.ndarray, prefix: str = "") -> Dict[str, float]:
        """Extract frequency domain features using FFT and power spectral density."""
        if len(signal) < 4 or np.all(np.isnan(signal)):
            return {f"{prefix}dominant_freq": 0.0, f"{prefix}spectral_centroid": 0.0,
                   f"{prefix}spectral_spread": 0.0, f"{prefix}power_low": 0.0,
                   f"{prefix}power_high": 0.0, f"{prefix}spectral_entropy": 0.0}
        
        # Remove NaN values and ensure even length
        clean_signal = signal[~np.isnan(signal)]
        if len(clean_signal) < 4:
            return {f"{prefix}dominant_freq": 0.0, f"{prefix}spectral_centroid": 0.0,
                   f"{prefix}spectral_spread": 0.0, f"{prefix}power_low": 0.0,
                   f"{prefix}power_high": 0.0, f"{prefix}spectral_entropy": 0.0}
        
        try:
            # Power spectral density
            freqs, psd = welch(clean_signal, fs=self.fs, nperseg=min(len(clean_signal)//2, 32))
            
            # Avoid zero division
            psd = psd + 1e-12
            total_power = np.sum(psd)
            
            if total_power == 0 or len(freqs) == 0:
                return {f"{prefix}dominant_freq": 0.0, f"{prefix}spectral_centroid": 0.0,
                       f"{prefix}spectral_spread": 0.0, f"{prefix}power_low": 0.0,
                       f"{prefix}power_high": 0.0, f"{prefix}spectral_entropy": 0.0}
            
            # Dominant frequency
            dominant_freq = freqs[np.argmax(psd)]
            
            # Spectral centroid (weighted mean frequency)
            spectral_centroid = np.sum(freqs * psd) / total_power
            
            # Spectral spread (weighted standard deviation)
            spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / total_power)
            
            # Power in low (0-0.5 Hz) and high (0.5+ Hz) frequency bands
            low_band_mask = freqs <= 0.5
            high_band_mask = freqs > 0.5
            power_low = np.sum(psd[low_band_mask]) / total_power if np.any(low_band_mask) else 0.0
            power_high = np.sum(psd[high_band_mask]) / total_power if np.any(high_band_mask) else 0.0
            
            # Spectral entropy
            normalized_psd = psd / total_power
            spectral_entropy = -np.sum(normalized_psd * np.log2(normalized_psd + 1e-12))
            
            features = {
                f"{prefix}dominant_freq": float(dominant_freq),
                f"{prefix}spectral_centroid": float(spectral_centroid),
                f"{prefix}spectral_spread": float(spectral_spread),
                f"{prefix}power_low": float(power_low),
                f"{prefix}power_high": float(power_high),
                f"{prefix}spectral_entropy": float(spectral_entropy),
            }
        except:
            features = {f"{prefix}dominant_freq": 0.0, f"{prefix}spectral_centroid": 0.0,
                       f"{prefix}spectral_spread": 0.0, f"{prefix}power_low": 0.0,
                       f"{prefix}power_high": 0.0, f"{prefix}spectral_entropy": 0.0}
        
        return features
    
    def extract_time_domain_features(self, signal: np.ndarray, prefix: str = "") -> Dict[str, float]:
        """Extract time-domain features including variability and peak detection."""
        if len(signal) < 4 or np.all(np.isnan(signal)):
            return {f"{prefix}slope": 0.0, f"{prefix}num_peaks": 0.0, f"{prefix}peak_density": 0.0,
                   f"{prefix}zero_crossings": 0.0, f"{prefix}signal_energy": 0.0,
                   f"{prefix}autocorr_lag1": 0.0, f"{prefix}variability": 0.0}
        
        # Remove NaN values
        clean_signal = signal[~np.isnan(signal)]
        if len(clean_signal) < 4:
            return {f"{prefix}slope": 0.0, f"{prefix}num_peaks": 0.0, f"{prefix}peak_density": 0.0,
                   f"{prefix}zero_crossings": 0.0, f"{prefix}signal_energy": 0.0,
                   f"{prefix}autocorr_lag1": 0.0, f"{prefix}variability": 0.0}
        
        try:
            # Linear trend (slope)
            x = np.arange(len(clean_signal))
            slope, _, _, _, _ = stats.linregress(x, clean_signal)
            
            # Peak detection
            peaks, _ = find_peaks(clean_signal, distance=max(1, len(clean_signal)//10))
            num_peaks = len(peaks)
            peak_density = num_peaks / len(clean_signal)
            
            # Zero crossings (sign changes)
            mean_centered = clean_signal - np.mean(clean_signal)
            zero_crossings = np.sum(np.diff(np.sign(mean_centered)) != 0)
            
            # Signal energy
            signal_energy = np.sum(clean_signal ** 2)
            
            # Autocorrelation at lag 1
            if len(clean_signal) > 1:
                autocorr_lag1 = np.corrcoef(clean_signal[:-1], clean_signal[1:])[0, 1]
                if np.isnan(autocorr_lag1):
                    autocorr_lag1 = 0.0
            else:
                autocorr_lag1 = 0.0
            
            # Signal variability (successive differences)
            if len(clean_signal) > 1:
                variability = np.std(np.diff(clean_signal))
            else:
                variability = 0.0
            
            features = {
                f"{prefix}slope": float(slope),
                f"{prefix}num_peaks": float(num_peaks),
                f"{prefix}peak_density": float(peak_density),
                f"{prefix}zero_crossings": float(zero_crossings),
                f"{prefix}signal_energy": float(signal_energy),
                f"{prefix}autocorr_lag1": float(autocorr_lag1),
                f"{prefix}variability": float(variability),
            }
        except:
            features = {f"{prefix}slope": 0.0, f"{prefix}num_peaks": 0.0, f"{prefix}peak_density": 0.0,
                       f"{prefix}zero_crossings": 0.0, f"{prefix}signal_energy": 0.0,
                       f"{prefix}autocorr_lag1": 0.0, f"{prefix}variability": 0.0}
        
        return features
    
    def extract_cross_signal_features(self, eda: np.ndarray, hr: np.ndarray) -> Dict[str, float]:
        """Extract features that capture relationships between EDA and HR."""
        if len(eda) != len(hr) or len(eda) < 4:
            return {"eda_hr_correlation": 0.0, "eda_hr_lag_correlation": 0.0,
                   "eda_hr_coherence": 0.0, "eda_hr_phase_sync": 0.0}
        
        # Remove NaN values
        valid_idx = ~(np.isnan(eda) | np.isnan(hr))
        if np.sum(valid_idx) < 4:
            return {"eda_hr_correlation": 0.0, "eda_hr_lag_correlation": 0.0,
                   "eda_hr_coherence": 0.0, "eda_hr_phase_sync": 0.0}
        
        clean_eda = eda[valid_idx]
        clean_hr = hr[valid_idx]
        
        try:
            # Cross-correlation at zero lag
            correlation = np.corrcoef(clean_eda, clean_hr)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            # Lagged correlation (max correlation within ±2 samples)
            max_lag = min(2, len(clean_eda)//4)
            lag_correlations = []
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    continue
                if lag > 0:
                    if len(clean_eda[:-lag]) > 0 and len(clean_hr[lag:]) > 0:
                        lag_corr = np.corrcoef(clean_eda[:-lag], clean_hr[lag:])[0, 1]
                else:
                    if len(clean_eda[-lag:]) > 0 and len(clean_hr[:lag]) > 0:
                        lag_corr = np.corrcoef(clean_eda[-lag:], clean_hr[:lag])[0, 1]
                if not np.isnan(lag_corr):
                    lag_correlations.append(abs(lag_corr))
            
            lag_correlation = max(lag_correlations) if lag_correlations else 0.0
            
            # Simple coherence measure (correlation in frequency domain)
            if len(clean_eda) >= 8:
                fft_eda = np.abs(fft(clean_eda))
                fft_hr = np.abs(fft(clean_hr))
                coherence = np.corrcoef(fft_eda, fft_hr)[0, 1]
                if np.isnan(coherence):
                    coherence = 0.0
            else:
                coherence = 0.0
            
            # Phase synchronization (simplified)
            eda_norm = (clean_eda - np.mean(clean_eda)) / (np.std(clean_eda) + 1e-12)
            hr_norm = (clean_hr - np.mean(clean_hr)) / (np.std(clean_hr) + 1e-12)
            phase_sync = np.abs(np.mean(np.exp(1j * (eda_norm - hr_norm))))
            
            features = {
                "eda_hr_correlation": float(correlation),
                "eda_hr_lag_correlation": float(lag_correlation),
                "eda_hr_coherence": float(coherence),
                "eda_hr_phase_sync": float(phase_sync),
            }
        except:
            features = {"eda_hr_correlation": 0.0, "eda_hr_lag_correlation": 0.0,
                       "eda_hr_coherence": 0.0, "eda_hr_phase_sync": 0.0}
        
        return features
    
    def extract_all_features(self, eda_window: np.ndarray, hr_window: np.ndarray) -> Dict[str, float]:
        """
        Extract all features from EDA and HR windows.
        
        Args:
            eda_window: EDA signal window
            hr_window: HR signal window
            
        Returns:
            Dictionary with all extracted features
        """
        features = {}
        
        # EDA features
        features.update(self.extract_statistical_features(eda_window, "eda_"))
        features.update(self.extract_frequency_features(eda_window, "eda_"))
        features.update(self.extract_time_domain_features(eda_window, "eda_"))
        
        # HR features
        features.update(self.extract_statistical_features(hr_window, "hr_"))
        features.update(self.extract_frequency_features(hr_window, "hr_"))
        features.update(self.extract_time_domain_features(hr_window, "hr_"))
        
        # Cross-signal features
        features.update(self.extract_cross_signal_features(eda_window, hr_window))
        
        return features


def build_feature_matrix(X: np.ndarray, window_size: int, step: int, fs: float = 4.0) -> Tuple[np.ndarray, List[str]]:
    """
    Build a feature matrix from raw EDA/HR signals using sliding windows.
    
    Args:
        X: Input data [N, 2] where columns are [EDA, HR]
        window_size: Window size in samples
        step: Step size between windows
        fs: Sampling frequency
        
    Returns:
        Feature matrix [num_windows, num_features] and feature names
    """
    from .preprocessing import sliding_windows
    
    extractor = PhysiologicalFeatureExtractor(fs=fs)
    
    feature_list = []
    feature_names = None
    
    for s, e in sliding_windows(X, window_size, step):
        eda_window = X[s:e, 0]
        hr_window = X[s:e, 1]
        
        features = extractor.extract_all_features(eda_window, hr_window)
        
        if feature_names is None:
            feature_names = list(features.keys())
        
        feature_vector = [features.get(name, 0.0) for name in feature_names]
        feature_list.append(feature_vector)
    
    if not feature_list:
        return np.zeros((0, 0)), []
    
    feature_matrix = np.array(feature_list)
    
    # Replace any remaining NaN or inf values
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feature_matrix, feature_names


def get_feature_importance_ranking(feature_matrix: np.ndarray, labels: np.ndarray, 
                                 feature_names: List[str]) -> List[Tuple[str, float]]:
    """
    Rank features by importance using mutual information.
    
    Args:
        feature_matrix: Feature matrix [N, F]
        labels: Binary labels [N]
        feature_names: List of feature names
        
    Returns:
        List of (feature_name, importance_score) tuples sorted by importance
    """
    from sklearn.feature_selection import mutual_info_classif
    
    if len(np.unique(labels)) < 2:
        return [(name, 0.0) for name in feature_names]
    
    try:
        mi_scores = mutual_info_classif(feature_matrix, labels, random_state=42)
        importance_ranking = sorted(zip(feature_names, mi_scores), 
                                  key=lambda x: x[1], reverse=True)
        return importance_ranking
    except:
        return [(name, 0.0) for name in feature_names]
