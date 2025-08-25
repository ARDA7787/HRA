#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.preprocessing import (
    interpolate_missing,
    zscore,
    butter_lowpass,
    build_window_matrix,
    sliding_windows,
    inject_synthetic_anomalies,
)
from utils.metrics import compute_metrics
from utils.visualization import plot_anomalies
from models.isolation_forest import IFAnomalyDetector
from models.autoencoder import AEAnomalyDetector


def window_labels(labels: np.ndarray, window_size: int, step: int) -> np.ndarray:
    ys = []
    for s, e in sliding_windows(labels, window_size, step):
        # window labeled as anomaly if any sample is anomaly
        ys.append(int(np.any(labels[s:e] == 1)))
    return np.array(ys, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv", type=str, required=True, help="Input CSV with columns: time, EDA, HR, [label]"
    )
    ap.add_argument("--model", type=str, choices=["if", "ae"], default="if")
    ap.add_argument("--window_sec", type=float, default=30.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument(
        "--fs",
        type=float,
        default=4.0,
        help="Sampling rate (Hz) for time-series; default 4 for EDA",
    )
    ap.add_argument("--use_labels", type=str, default="true", help="true/false")
    ap.add_argument(
        "--inject_anomalies",
        type=str,
        default="false",
        help="true/false; only when labels not used",
    )
    ap.add_argument("--plot_out", type=str, default="")
    args = ap.parse_args()

    use_labels = args.use_labels.lower() == "true"
    inject = args.inject_anomalies.lower() == "true"

    df = pd.read_csv(args.csv)
    if "EDA" not in df.columns or "HR" not in df.columns:
        raise ValueError("CSV must have EDA and HR columns")

    time = df["time"].values if "time" in df.columns else np.arange(len(df)) / args.fs
    eda = interpolate_missing(df["EDA"])
    hr = interpolate_missing(df["HR"])

    # Filter EDA (low-pass) and z-score both
    eda_f = pd.Series(butter_lowpass(eda.values, fs=args.fs, cutoff=1.0))
    eda_z = zscore(eda_f)
    hr_z = zscore(hr)

    X = np.vstack([eda_z.values, hr_z.values]).T  # [N, 2]

    window_size = int(args.window_sec * args.fs)
    step = max(1, int(window_size * (1.0 - args.overlap)))

    Xw = build_window_matrix(X, window_size, step)

    # Build labels if available
    yw = None
    if use_labels and "label" in df.columns:
        labels = df["label"].values.astype(int)
        yw = window_labels(labels, window_size, step)

    # Split: use only normal windows for training
    if yw is not None and yw.size == len(Xw):
        idx_normal = np.where(yw == 0)[0]
        idx_anom = np.where(yw == 1)[0]
        Xw_train, Xw_test = Xw[idx_normal], Xw[idx_normal]
        # Keep a test set including anomalies for evaluation
        Xw_eval = np.concatenate([Xw[idx_normal], Xw[idx_anom]], axis=0)
        yw_eval = np.concatenate(
            [np.zeros(len(idx_normal), dtype=int), np.ones(len(idx_anom), dtype=int)]
        )
    else:
        # No labels: train on all windows, optionally inject anomalies for robustness
        Xw_train = Xw.copy()
        if inject:
            Xw_train = inject_synthetic_anomalies(Xw_train, rate=0.02, scale=3.0)
        Xw_eval = Xw.copy()
        yw_eval = None

    if args.model == "if":
        model = IFAnomalyDetector(contamination=None)
        model.fit(Xw_train)
        scores = model.score(Xw_eval)
    else:
        model = AEAnomalyDetector(input_dim=Xw.shape[1], latent_dim=32, epochs=40)
        model.fit(Xw_train)
        scores = model.reconstruction_error(Xw_eval)

    # Choose threshold (95th percentile of scores on normal data if labels available)
    if yw is not None:
        thr = np.percentile(scores[yw_eval == 0], 95)
    else:
        thr = np.percentile(scores, 95)

    preds = (scores >= thr).astype(int)

    if yw_eval is not None:
        report = compute_metrics(yw_eval, scores, preds)
        print("Metrics:", report)
    else:
        print("No labels provided; showing threshold and anomaly rate.")
        print({"threshold": float(thr), "anomaly_rate": float(preds.mean())})

    # Map window anomalies back to timeline for plotting
    anom_mask = np.zeros(len(X), dtype=bool)
    i = 0
    for s, e in sliding_windows(np.arange(len(X)), window_size, step):
        if preds[i] == 1:
            anom_mask[s:e] = True
        i += 1

    out_path = args.plot_out if args.plot_out else None
    plot_anomalies(
        time,
        eda_z.values,
        hr_z.values,
        anom_mask,
        out_path=out_path,
        title=f"{args.model.upper()} anomalies",
    )


if __name__ == "__main__":
    main()
