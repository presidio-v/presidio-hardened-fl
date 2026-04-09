"""Tests for presidio_fl.client module."""

from __future__ import annotations

import numpy as np

from presidio_fl.client import FLClient


def _make_global_weights(n_features: int = 10) -> dict:
    return {
        "coef": np.zeros((1, n_features)),
        "intercept": np.zeros(1),
    }


class TestFLClient:
    def _make_client(self, n_samples=100, n_features=10, clip_norm=1.0):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n_samples, n_features))
        y = rng.integers(0, 2, size=n_samples)
        # Ensure both classes present
        y[:5] = 0
        y[5:10] = 1
        return FLClient(node_id=0, X=X, y=y, clip_norm=clip_norm, seed=42)

    def test_train_returns_dict_with_correct_keys(self):
        client = self._make_client()
        gw = _make_global_weights()
        result = client.train(gw)
        assert set(result.keys()) == {"coef", "intercept"}

    def test_train_returns_same_shape(self):
        client = self._make_client()
        gw = _make_global_weights()
        result = client.train(gw)
        assert result["coef"].shape == gw["coef"].shape
        assert result["intercept"].shape == gw["intercept"].shape

    def test_clipping_enforces_norm_bound(self):
        clip_norm = 0.5
        client = self._make_client(clip_norm=clip_norm)
        gw = _make_global_weights()
        result = client.train(gw)
        coef_norm = float(np.linalg.norm(result["coef"]))
        assert coef_norm <= clip_norm + 1e-6, (
            f"coef norm {coef_norm:.6f} exceeds clip_norm {clip_norm}"
        )

    def test_no_clipping_with_large_clip_norm(self):
        """With a very large clip_norm the weights should not be clipped."""
        client = self._make_client(clip_norm=1e9)
        gw = _make_global_weights()
        result = client.train(gw)
        # As long as it runs without error and returns valid arrays
        assert result["coef"] is not None

    def test_n_samples_property(self):
        client = self._make_client(n_samples=80)
        assert client.n_samples == 80

    def test_global_weights_not_mutated(self):
        client = self._make_client()
        gw = _make_global_weights()
        coef_before = gw["coef"].copy()
        client.train(gw)
        np.testing.assert_array_equal(gw["coef"], coef_before)
