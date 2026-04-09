"""Federated learning client: local training with gradient norm clipping."""

from __future__ import annotations

import numpy as np

from presidio_fl.model import FLModel


class FLClient:
    """Simulate a single federated learning node.

    Parameters
    ----------
    node_id:
        Integer identifier for this node.
    X:
        Local training features.
    y:
        Local training labels.
    clip_norm:
        Maximum L2 norm of the coefficient vector before returning weights.
        Set to ``None`` to disable clipping.
    seed:
        Random seed forwarded to the underlying :class:`FLModel`.
    """

    def __init__(
        self,
        node_id: int,
        X: np.ndarray,
        y: np.ndarray,
        clip_norm: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.node_id = node_id
        self.X = X
        self.y = y
        self.clip_norm = clip_norm
        self._model = FLModel(n_features=X.shape[1], seed=seed)

    @property
    def n_samples(self) -> int:
        return len(self.y)

    def train(self, global_weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run one round of local training and return (clipped) weights.

        Steps
        -----
        1. Set the model weights to *global_weights*.
        2. Fit on the local dataset for ``max_iter`` SGD steps.
        3. Clip the coefficient L2 norm to at most ``clip_norm``.
        4. Return the updated weights dict.
        """
        self._model.set_weights(global_weights)
        self._model.fit(self.X, self.y)
        weights = self._model.get_weights()

        # Gradient norm clipping on the coefficient vector
        if self.clip_norm is not None:
            coef = weights["coef"]
            norm = np.linalg.norm(coef) + 1e-8
            scale = min(1.0, self.clip_norm / norm)
            weights["coef"] = coef * scale

        return weights
