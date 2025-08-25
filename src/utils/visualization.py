import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_anomalies(time: np.ndarray, eda: np.ndarray, hr: np.ndarray, anomaly_mask: np.ndarray, out_path: str | None = None, title: str = "Anomalies"):
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(time, eda, label='EDA (z)', color='tab:blue', alpha=0.8)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('EDA (z)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(time, hr, label='HR (z)', color='tab:red', alpha=0.6)
    ax2.set_ylabel('HR (z)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Overlay anomalies
    mask = anomaly_mask.astype(bool)
    if mask.any():
        ax1.scatter(time[mask], eda[mask], color='black', s=8, label='Anomaly')

    fig.suptitle(title)
    fig.tight_layout()
    ax1.legend(loc='upper left')

    if out_path:
        plt.savefig(out_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)