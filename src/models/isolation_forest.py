from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest


class IFAnomalyDetector:
    def __init__(self, contamination: float | None = None, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=200,
            max_samples='auto',
            contamination=contamination,  # if None, model infers threshold via decision_function
            random_state=random_state,
            n_jobs=-1,
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