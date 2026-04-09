"""Tests for presidio_fl.aggregator module."""

from __future__ import annotations

import numpy as np
import pytest

from presidio_fl.aggregator import FedAvg


class TestFedAvg:
    def _w(self, coef_val: float, intercept_val: float = 0.0):
        return {
            "coef": np.array([[coef_val, coef_val]]),
            "intercept": np.array([intercept_val]),
        }

    def test_equal_weights_returns_mean(self):
        updates = [
            (100, self._w(1.0)),
            (100, self._w(3.0)),
        ]
        result = FedAvg.aggregate(updates)
        np.testing.assert_allclose(result["coef"], np.array([[2.0, 2.0]]))

    def test_unequal_weights_returns_weighted_mean(self):
        # 1/4 weight for node A, 3/4 weight for node B
        updates = [
            (1, self._w(0.0)),
            (3, self._w(4.0)),
        ]
        result = FedAvg.aggregate(updates)
        # expected = (1*0 + 3*4) / 4 = 3.0
        np.testing.assert_allclose(result["coef"], np.array([[3.0, 3.0]]), atol=1e-12)

    def test_single_node_identity(self):
        w = self._w(5.0, 1.5)
        result = FedAvg.aggregate([(50, w)])
        np.testing.assert_allclose(result["coef"], w["coef"])
        np.testing.assert_allclose(result["intercept"], w["intercept"])

    def test_three_equal_nodes(self):
        updates = [(10, self._w(c)) for c in [1.0, 2.0, 3.0]]
        result = FedAvg.aggregate(updates)
        np.testing.assert_allclose(result["coef"], np.array([[2.0, 2.0]]), atol=1e-12)

    def test_preserves_all_keys(self):
        updates = [
            (10, {"coef": np.array([[1.0]]), "intercept": np.array([0.0])}),
            (10, {"coef": np.array([[2.0]]), "intercept": np.array([1.0])}),
        ]
        result = FedAvg.aggregate(updates)
        assert "coef" in result
        assert "intercept" in result

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            FedAvg.aggregate([])

    def test_zero_samples_raises(self):
        with pytest.raises(ValueError):
            FedAvg.aggregate([(0, self._w(1.0))])
