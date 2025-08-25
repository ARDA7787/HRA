import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray | None = None):
    # y_score: higher => more anomalous
    if y_pred is None:
        # pick threshold at 95th percentile for report if not supplied
        thr = np.percentile(y_score, 95)
        y_pred = (y_score >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    auc = None
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = None
    # Check for invalid case (only one class)
    if len(np.unique(y_true)) < 2:
        return {
            'error': 'Need both normal and anomalous samples for evaluation',
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'roc_auc': None
        }
    
    return {
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'roc_auc': None if auc is None or np.isnan(auc) else float(auc),
    }