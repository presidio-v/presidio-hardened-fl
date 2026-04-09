"""Logistic-regression wrapper for federated learning rounds."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier


class FLModel:
    """Thin wrapper around SGDClassifier for federated weight exchange.

    Uses ``loss='log_loss'`` (logistic regression) with ``warm_start=True``
    so that each call to :meth:`fit` continues from the current weights
    rather than reinitialising them.
    """

    def __init__(self, n_features: int, seed: int = 42) -> None:
        self.n_features = n_features
        self._clf = SGDClassifier(
            loss="log_loss",
            max_iter=100,
            warm_start=True,
            random_state=seed,
            tol=1e-4,
        )
        self._initialised = False

    # ------------------------------------------------------------------
    # Weight accessors
    # ------------------------------------------------------------------

    def _ensure_initialised(self) -> None:
        """Initialise coef_ / intercept_ to zeros if not yet fit."""
        if not self._initialised:
            self._clf.classes_ = np.array([0, 1])
            self._clf.coef_ = np.zeros((1, self.n_features))
            self._clf.intercept_ = np.zeros(1)
            self._initialised = True

    def get_weights(self) -> dict[str, np.ndarray]:
        """Return a *copy* of the current model parameters."""
        self._ensure_initialised()
        return {
            "coef": self._clf.coef_.copy(),
            "intercept": self._clf.intercept_.copy(),
        }

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Overwrite model parameters with the supplied weights dict."""
        self._ensure_initialised()
        self._clf.coef_ = weights["coef"].copy()
        self._clf.intercept_ = weights["intercept"].copy()

    # ------------------------------------------------------------------
    # Training / inference
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> FLModel:
        """Run one round of local SGD training."""
        self._ensure_initialised()
        self._clf.fit(X, y)
        self._initialised = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self._ensure_initialised()
        return self._clf.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy on (X, y)."""
        self._ensure_initialised()
        return float(self._clf.score(X, y))
