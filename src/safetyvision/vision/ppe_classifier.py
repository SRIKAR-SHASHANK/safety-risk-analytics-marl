"""PPE (hard-hat / hi-vis vest) classifier trained on detected person crops."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class PPEClassifier:
    """Two independent binary heads (helmet, vest) on the shared colour descriptor."""

    def __init__(self, seed: int = 0):
        def head():
            return make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=600, early_stopping=True,
                              random_state=seed),
            )
        self.helmet_clf = head()
        self.vest_clf = head()
        self.fitted = False

    def fit(self, X: np.ndarray, y_helmet: np.ndarray, y_vest: np.ndarray) -> "PPEClassifier":
        self.helmet_clf.fit(X, y_helmet)
        self.vest_clf.fit(X, y_vest)
        self.fitted = True
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(X) == 0:
            return np.zeros(0, bool), np.zeros(0, bool)
        return self.helmet_clf.predict(X).astype(bool), self.vest_clf.predict(X).astype(bool)

    def evaluate(self, X, y_helmet, y_vest) -> dict:
        ph, pv = self.predict(X)
        out = {}
        for name, y, p in (("helmet", y_helmet, ph), ("vest", y_vest, pv)):
            # positive class for safety purposes = *violation* (missing PPE)
            out[name] = dict(
                accuracy=float(accuracy_score(y, p)),
                violation_f1=float(f1_score(~y.astype(bool), ~p, zero_division=0)),
                confusion=confusion_matrix(y, p, labels=[False, True]).tolist(),
            )
        return out
