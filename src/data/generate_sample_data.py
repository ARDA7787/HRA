#!/usr/bin/env python3
"""
Generate synthetic physiological data for testing and demonstration.
Creates realistic EDA and HR signals with injected anomalies.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
from typing import Tuple


def generate_realistic_eda(duration_sec: float, fs: float = 4.0, 
                          baseline: float = 2.0, noise_level: float = 0.1) -> np.ndarray:
    """
    Generate realistic EDA signal with physiological characteristics.
    
    Args:
        duration_sec: Duration in seconds
        fs: Sampling frequency
        baseline: Baseline EDA level (microsiemens)
        noise_level: Noise level
        
    Returns:
        EDA signal array
    """
    n_samples = int(duration_sec * fs)
    t = np.linspace(0, duration_sec, n_samples)
    
    # Base signal with slow drift
    eda = baseline + 0.3 * np.sin(2 * np.pi * 0.01 * t)  # Very slow oscillation
    
    # Add physiological responses (SCRs)
    scr_times = np.random.exponential(30, size=int(duration_sec / 20))  # SCRs every ~20-30 seconds
    scr_times = scr_times.cumsum()
    scr_times = scr_times[scr_times < duration_sec]
    
    for scr_time in scr_times:
        # SCR shape: exponential rise and decay
        scr_start = int(scr_time * fs)
        scr_duration = int(5 * fs)  # 5 second SCR
        if scr_start + scr_duration < n_samples:
            scr_indices = np.arange(scr_start, min(scr_start + scr_duration, n_samples))
            scr_t = (scr_indices - scr_start) / fs
            # Exponential rise and decay
            scr_amplitude = np.random.uniform(0.1, 0.5)
            scr_response = scr_amplitude * np.exp(-scr_t / 1.5) * (1 - np.exp(-scr_t / 0.3))
            eda[scr_indices] += scr_response
    
    # Add noise
    eda += np.random.normal(0, noise_level, n_samples)
    
    # Ensure positive values
    eda = np.maximum(eda, 0.1)
    
    return eda


def generate_realistic_hr(duration_sec: float, fs: float = 4.0,
                         baseline: float = 70.0, variability: float = 10.0) -> np.ndarray:
    """
    Generate realistic HR signal with physiological variability.
    
    Args:
        duration_sec: Duration in seconds
        fs: Sampling frequency
        baseline: Baseline heart rate (BPM)
        variability: HR variability
        
    Returns:
        HR signal array
    """
    n_samples = int(duration_sec * fs)
    t = np.linspace(0, duration_sec, n_samples)
    
    # Base HR with respiratory sinus arrhythmia
    hr = baseline + variability * 0.3 * np.sin(2 * np.pi * 0.25 * t)  # Respiratory frequency
    
    # Add slower variations (autonomic regulation)
    hr += variability * 0.2 * np.sin(2 * np.pi * 0.05 * t)  # Slower oscillation
    
    # Add random walk component
    random_walk = np.cumsum(np.random.normal(0, 0.5, n_samples))
    random_walk = random_walk - np.mean(random_walk)  # Center around zero
    hr += variability * 0.1 * random_walk / np.std(random_walk)
    
    # Add measurement noise
    hr += np.random.normal(0, 1.0, n_samples)
    
    # Physiological limits
    hr = np.clip(hr, 50, 120)
    
    return hr


def inject_anomalies(eda: np.ndarray, hr: np.ndarray, fs: float = 4.0,
                    anomaly_rate: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inject realistic anomalies into physiological signals.
    
    Args:
        eda: EDA signal
        hr: HR signal
        fs: Sampling frequency
        anomaly_rate: Fraction of data to be anomalous
        
    Returns:
        Modified EDA, modified HR, anomaly labels
    """
    n_samples = len(eda)
    labels = np.zeros(n_samples, dtype=int)
    
    # Calculate number of anomalous segments
    segment_duration = int(10 * fs)  # 10-second anomalous segments
    n_anomalies = int(anomaly_rate * n_samples / segment_duration)
    
    eda_modified = eda.copy()
    hr_modified = hr.copy()
    
    for _ in range(n_anomalies):
        # Random start position
        start_idx = np.random.randint(0, max(1, n_samples - segment_duration))
        end_idx = min(start_idx + segment_duration, n_samples)
        
        # Mark as anomalous
        labels[start_idx:end_idx] = 1
        
        # Type of anomaly
        anomaly_type = np.random.choice(['stress_response', 'artifact', 'extreme_values'])
        
        if anomaly_type == 'stress_response':
            # Simulate stress response: increased EDA and HR
            eda_modified[start_idx:end_idx] *= np.random.uniform(1.5, 3.0)
            hr_modified[start_idx:end_idx] += np.random.uniform(10, 30)
            
        elif anomaly_type == 'artifact':
            # Simulate motion artifacts: spikes and noise
            spike_amplitude = np.random.uniform(2, 5)
            eda_modified[start_idx:end_idx] += spike_amplitude * np.random.randn(end_idx - start_idx)
            hr_modified[start_idx:end_idx] += 20 * np.random.randn(end_idx - start_idx)
            
        elif anomaly_type == 'extreme_values':
            # Simulate sensor malfunction: extreme values
            if np.random.random() > 0.5:
                eda_modified[start_idx:end_idx] = np.random.uniform(10, 20)  # Very high EDA
            else:
                hr_modified[start_idx:end_idx] = np.random.uniform(120, 150)  # Very high HR
    
    # Ensure physiological limits
    eda_modified = np.maximum(eda_modified, 0.1)
    hr_modified = np.clip(hr_modified, 40, 180)
    
    return eda_modified, hr_modified, labels


def generate_sample_dataset(duration_minutes: float = 60, fs: float = 4.0,
                           anomaly_rate: float = 0.05, output_path: str = None) -> pd.DataFrame:
    """
    Generate a complete sample dataset with realistic physiological signals and anomalies.
    
    Args:
        duration_minutes: Duration in minutes
        fs: Sampling frequency
        anomaly_rate: Fraction of anomalous data
        output_path: Path to save CSV file
        
    Returns:
        DataFrame with columns: time, EDA, HR, label
    """
    duration_sec = duration_minutes * 60
    
    # Generate base signals
    eda = generate_realistic_eda(duration_sec, fs)
    hr = generate_realistic_hr(duration_sec, fs)
    
    # Inject anomalies
    eda_final, hr_final, labels = inject_anomalies(eda, hr, fs, anomaly_rate)
    
    # Create time array
    n_samples = len(eda_final)
    time = np.linspace(0, duration_sec, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': time,
        'EDA': eda_final,
        'HR': hr_final,
        'label': labels
    })
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Sample dataset saved to: {output_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Anomaly rate: {labels.mean():.3f}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic physiological data")
    parser.add_argument("--duration", type=float, default=60, 
                       help="Duration in minutes (default: 60)")
    parser.add_argument("--fs", type=float, default=4.0,
                       help="Sampling frequency in Hz (default: 4.0)")
    parser.add_argument("--anomaly_rate", type=float, default=0.05,
                       help="Fraction of anomalous data (default: 0.05)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output CSV file path")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Generate dataset
    df = generate_sample_dataset(
        duration_minutes=args.duration,
        fs=args.fs,
        anomaly_rate=args.anomaly_rate,
        output_path=args.output
    )
    
    print(f"\nDataset statistics:")
    print(f"Duration: {args.duration} minutes")
    print(f"Samples: {len(df)}")
    print(f"EDA range: {df['EDA'].min():.3f} - {df['EDA'].max():.3f}")
    print(f"HR range: {df['HR'].min():.1f} - {df['HR'].max():.1f}")
    print(f"Anomalies: {df['label'].sum()} / {len(df)} ({df['label'].mean():.3f})")


if __name__ == "__main__":
    main()
