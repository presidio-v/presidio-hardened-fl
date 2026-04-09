"""Gaussian differential privacy mechanism and privacy budget accounting."""

from __future__ import annotations

import math

import numpy as np


class PrivacyBudgetExhausted(Exception):
    """Raised when the total privacy budget has been consumed."""


class GaussianMechanism:
    """Add calibrated Gaussian noise to model weights.

    The noise standard deviation follows the standard analytic formula::

        σ = sensitivity * sqrt(2 * ln(1.25 / δ)) / ε

    Parameters
    ----------
    epsilon:
        Privacy parameter ε (privacy budget per application).
    delta:
        Privacy parameter δ (failure probability).
    sensitivity:
        L2 sensitivity of the query (equals ``clip_norm`` by convention).
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        sensitivity: float = 1.0,
    ) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity

    @property
    def sigma(self) -> float:
        """Noise standard deviation calibrated to (ε, δ)-DP."""
        return self.sensitivity * math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon

    def add_noise(
        self,
        weights: dict[str, np.ndarray],
        rng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray]:
        """Return a copy of *weights* with i.i.d. Gaussian noise added.

        Parameters
        ----------
        weights:
            Dict mapping parameter names to numpy arrays.
        rng:
            Optional :class:`numpy.random.Generator` for reproducibility.
            If ``None``, a fresh default generator is used.
        """
        if rng is None:
            rng = np.random.default_rng()
        noised: dict[str, np.ndarray] = {}
        for k, v in weights.items():
            noised[k] = v + rng.normal(loc=0.0, scale=self.sigma, size=v.shape)
        return noised


class PrivacyAccountant:
    """Track cumulative privacy budget under basic (sequential) composition.

    Under basic composition *n* mechanisms each with budget ε_r consume a
    total of *n · ε_r*.

    Parameters
    ----------
    epsilon_per_round:
        Privacy budget consumed per training round.
    delta:
        δ parameter (informational, not composed here).
    max_rounds:
        Maximum number of rounds before the budget is exhausted.
    """

    def __init__(
        self,
        epsilon_per_round: float,
        delta: float,
        max_rounds: int,
    ) -> None:
        self.epsilon_per_round = epsilon_per_round
        self.delta = delta
        self.max_rounds = max_rounds
        self.spent_rounds: int = 0

    def spend_round(self) -> None:
        """Consume one round of privacy budget.

        Raises :class:`PrivacyBudgetExhausted` if the budget is already
        exhausted before this call.
        """
        if self.spent_rounds >= self.max_rounds:
            raise PrivacyBudgetExhausted(
                f"Privacy budget exhausted after {self.max_rounds} rounds "
                f"(total ε = {self.total_budget():.4f})"
            )
        self.spent_rounds += 1

    def remaining_budget(self) -> float:
        """Remaining ε budget under basic composition."""
        return (self.max_rounds - self.spent_rounds) * self.epsilon_per_round

    def total_budget(self) -> float:
        """Total ε budget over all rounds."""
        return self.max_rounds * self.epsilon_per_round

    def is_exhausted(self) -> bool:
        """Return ``True`` when all rounds have been spent."""
        return self.spent_rounds >= self.max_rounds
