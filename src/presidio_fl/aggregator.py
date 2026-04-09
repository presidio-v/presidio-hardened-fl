"""FedAvg aggregator: sample-count weighted averaging of model parameters."""

from __future__ import annotations

import numpy as np


class FedAvg:
    """Federated Averaging aggregator.

    Combines weight updates from multiple nodes using a weighted average
    proportional to each node's sample count.
    """

    @staticmethod
    def aggregate(
        weighted_updates: list[tuple[int, dict[str, np.ndarray]]],
    ) -> dict[str, np.ndarray]:
        """Compute a weighted average of the supplied weight dicts.

        Parameters
        ----------
        weighted_updates:
            List of ``(n_samples, weights_dict)`` tuples, one per node.

        Returns
        -------
        dict
            Aggregated weights with the same keys as the input dicts.
        """
        if not weighted_updates:
            raise ValueError("weighted_updates must not be empty")

        total_samples = sum(n for n, _ in weighted_updates)
        if total_samples == 0:
            raise ValueError("Total sample count must be > 0")

        # Initialise accumulators
        keys = list(weighted_updates[0][1].keys())
        result: dict[str, np.ndarray] = {
            k: np.zeros_like(weighted_updates[0][1][k], dtype=float) for k in keys
        }

        for n_samples, weights in weighted_updates:
            weight_fraction = n_samples / total_samples
            for k in keys:
                result[k] += weight_fraction * weights[k].astype(float)

        return result
