"""Tests for presidio_fl.dp module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from presidio_fl.dp import GaussianMechanism, PrivacyAccountant, PrivacyBudgetExhausted

# ---------------------------------------------------------------------------
# GaussianMechanism
# ---------------------------------------------------------------------------


class TestGaussianMechanism:
    def _weights(self):
        return {
            "coef": np.array([[1.0, 2.0, 3.0]]),
            "intercept": np.array([0.5]),
        }

    def test_sigma_formula(self):
        eps, delta, s = 1.0, 1e-5, 1.0
        mech = GaussianMechanism(epsilon=eps, delta=delta, sensitivity=s)
        expected = s * math.sqrt(2.0 * math.log(1.25 / delta)) / eps
        assert abs(mech.sigma - expected) < 1e-12

    def test_sigma_larger_for_small_epsilon(self):
        mech_weak = GaussianMechanism(epsilon=10.0, delta=1e-5)
        mech_strong = GaussianMechanism(epsilon=0.1, delta=1e-5)
        assert mech_strong.sigma > mech_weak.sigma

    def test_add_noise_changes_weights(self):
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5)
        w = self._weights()
        noised = mech.add_noise(w, rng=np.random.default_rng(0))
        # Noised values must differ from original
        assert not np.allclose(noised["coef"], w["coef"])

    def test_add_noise_preserves_keys(self):
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5)
        w = self._weights()
        noised = mech.add_noise(w, rng=np.random.default_rng(1))
        assert set(noised.keys()) == set(w.keys())

    def test_add_noise_preserves_shape(self):
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5)
        w = self._weights()
        noised = mech.add_noise(w, rng=np.random.default_rng(2))
        for k in w:
            assert noised[k].shape == w[k].shape

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError):
            GaussianMechanism(epsilon=0.0, delta=1e-5)

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError):
            GaussianMechanism(epsilon=1.0, delta=0.0)
        with pytest.raises(ValueError):
            GaussianMechanism(epsilon=1.0, delta=1.0)

    def test_reproducibility_with_same_rng_seed(self):
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5)
        w = self._weights()
        n1 = mech.add_noise(w, rng=np.random.default_rng(99))
        n2 = mech.add_noise(w, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(n1["coef"], n2["coef"])


# ---------------------------------------------------------------------------
# PrivacyAccountant
# ---------------------------------------------------------------------------


class TestPrivacyAccountant:
    def test_initial_state(self):
        acc = PrivacyAccountant(epsilon_per_round=1.0, delta=1e-5, max_rounds=5)
        assert acc.spent_rounds == 0
        assert not acc.is_exhausted()

    def test_spend_round_increments(self):
        acc = PrivacyAccountant(epsilon_per_round=1.0, delta=1e-5, max_rounds=5)
        acc.spend_round()
        assert acc.spent_rounds == 1

    def test_remaining_budget_decreases(self):
        acc = PrivacyAccountant(epsilon_per_round=2.0, delta=1e-5, max_rounds=3)
        assert acc.remaining_budget() == pytest.approx(6.0)
        acc.spend_round()
        assert acc.remaining_budget() == pytest.approx(4.0)

    def test_total_budget(self):
        acc = PrivacyAccountant(epsilon_per_round=1.5, delta=1e-5, max_rounds=4)
        assert acc.total_budget() == pytest.approx(6.0)

    def test_exhausted_after_max_rounds(self):
        acc = PrivacyAccountant(epsilon_per_round=1.0, delta=1e-5, max_rounds=3)
        for _ in range(3):
            acc.spend_round()
        assert acc.is_exhausted()

    def test_raises_when_exhausted(self):
        acc = PrivacyAccountant(epsilon_per_round=1.0, delta=1e-5, max_rounds=2)
        acc.spend_round()
        acc.spend_round()
        with pytest.raises(PrivacyBudgetExhausted):
            acc.spend_round()
