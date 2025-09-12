from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest


class IFAnomalyDetector:
    def __init__(self, contamination: float | None = 'auto', random_state: int = 42,
                 n_estimators: int = 200, max_samples: str = 'auto', n_jobs: int = -1, **kwargs):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,  # if None, model infers threshold via decision_function
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray):
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # Higher is more normal. Convert to anomaly score as negative of decision_function
        df = self.model.decision_function(X)  # normality score
        return -df

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        scores = self.score(X)
        return (scores >= threshold).astype(int)  # 1 = anomaly