#!/usr/bin/env python3
"""
Convert WESAD E4 wrist signals into per-subject CSV with columns: time, EDA, HR, label.
- EDA: 4 Hz
- BVP: 64 Hz (we compute HR via peak detection)
- Labels: map WESAD label stream to {0: baseline/non-stress, 1: stress}; amusement is treated as non-stress.

Usage:
python src/data/prepare_wesad_csv.py --wesad_root data_raw/WESAD --subjects 2 3 5 --out_dir data_csv

This script requires the standard WESAD folder structure with MATLAB .pkl/.p files.
We rely on publicly mirrored preprocessed pickles commonly used in community repos. If your extracted data only has .mat files, adapt the loader accordingly.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

try:
    import pickle
except Exception:
    pickle = None

# Fallback: try reading from .pkl dataset files (common community mirrors)

def load_subject_dict(subject_dir: Path):
    """Load subject dict with keys similar to community WESAD pickle format.
    Expected keys (subset):
      - 'signal': {'wrist': {'EDA': array(4Hz), 'BVP': array(64Hz)}}
      - 'label': array per sample at a synchronized rate
      - 'resampled': optional synced arrays
    This loader tries several common file patterns.
    """
    # Common file names: S2.pkl or S2.p
    for name in [subject_dir.name + '.pkl', subject_dir.name + '.p']:
        p = subject_dir / name
        if p.exists():
            with open(p, 'rb') as f:
                return pickle.load(f, encoding='latin1')
    # Some zips contain data in .p under parent folder
    alt = subject_dir.parent / (subject_dir.name + '.p')
    if alt.exists():
        with open(alt, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    raise FileNotFoundError(f"Could not find pickle for {subject_dir}")


def bvp_to_hr(bvp: np.ndarray, fs_bvp: float = 64.0) -> np.ndarray:
    """Compute heart rate (bpm) from BVP via peak detection and interpolate back to BVP timeline.
    Ensures input is a 1-D float array.
    """
    bvp = np.asarray(bvp)
    if bvp.ndim > 1:
        bvp = bvp.reshape(-1)
    bvp = bvp.astype(float)
    # Detect peaks in BVP
    distance = int(0.4 * fs_bvp)  # min 150 bpm cap avoidance
    peaks, _ = find_peaks(bvp, distance=distance)
    if len(peaks) < 2:
        return np.full_like(bvp, fill_value=np.nan, dtype=float)
    # Compute instantaneous HR from peak-to-peak intervals
    ibi = np.diff(peaks) / fs_bvp  # seconds
    hr_inst = 60.0 / ibi  # bpm
    # Map HR back to peak positions and interpolate
    hr_series = np.full_like(bvp, np.nan, dtype=float)
    hr_series[peaks[1:]] = hr_inst
    # Forward-fill then back-fill
    s = pd.Series(hr_series)
    s = s.ffill().bfill()
    return s.values


def map_labels_to_binary(labels: np.ndarray) -> np.ndarray:
    """Map WESAD labels to binary {0: non-stress, 1: stress}.
    Common mapping (from paper README): 1=baseline, 2=stress, 3=amusement, 0=undefined.
    We treat amusement as non-stress (0).
    """
    out = np.zeros_like(labels, dtype=int)
    out[labels == 2] = 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wesad_root', type=str, required=True)
    ap.add_argument('--subjects', type=int, nargs='+', required=True)
    ap.add_argument('--out_dir', type=str, required=True)
    args = ap.parse_args()

    wesad_root = Path(args.wesad_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sid in args.subjects:
        subj_name = f'S{sid}'
        subj_dir = wesad_root / subj_name
        if not subj_dir.exists():
            raise FileNotFoundError(f"Subject folder missing: {subj_dir}")

        data = load_subject_dict(subj_dir)
        # Prefer wrist signals
        wrist = data.get('signal', {}).get('wrist', {})
        if not wrist:
            raise RuntimeError("Wrist signals missing in pickle; please adjust loader to your structure.")
        eda = np.asarray(wrist.get('EDA'))
        bvp = np.asarray(wrist.get('BVP'))
        if eda is None or bvp is None:
            raise RuntimeError("EDA or BVP missing in wrist signals.")

        # Resample or align
        fs_eda = 4.0
        fs_bvp = 64.0
        # Create time bases
        t_eda = np.arange(len(eda)) / fs_eda
        t_bvp = np.arange(len(bvp)) / fs_bvp

        # Compute HR from BVP then resample HR to EDA time base via nearest index
        hr_bvp = bvp_to_hr(bvp, fs_bvp=fs_bvp)
        # Resample HR to 4Hz timeline
        # Map each EDA time to nearest BVP index
        idx = np.minimum(np.searchsorted(t_bvp, t_eda), len(t_bvp)-1)
        hr_4hz = hr_bvp[idx]

        # Labels: attempt to get synchronized label stream; many pickles have 'label'
        labels = np.asarray(data.get('label'))
        if labels is not None and labels.size > 0:
            # Labels are often at 700 Hz (chest) or aligned differently; fall back: resample by proportion to EDA length
            lab_resampled = pd.Series(labels).iloc[::max(1, int(len(labels)/len(eda)))]
            lab_resampled = lab_resampled.reindex(range(len(eda)), method='ffill')
            y_bin = map_labels_to_binary(lab_resampled.values[:len(eda)])
        else:
            y_bin = np.zeros(len(eda), dtype=int)

        df = pd.DataFrame({
            'time': t_eda,
            'EDA': eda.astype(float),
            'HR': hr_4hz.astype(float),
            'label': y_bin.astype(int)
        })
        out_path = out_dir / f'subject_{subj_name}.csv'
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} with shape {df.shape}")


if __name__ == '__main__':
    main()