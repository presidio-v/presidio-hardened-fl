"""FederationRound: orchestrate multi-round federated learning with optional DP."""

from __future__ import annotations

import numpy as np

from presidio_fl.aggregator import FedAvg
from presidio_fl.client import FLClient
from presidio_fl.dp import GaussianMechanism, PrivacyAccountant
from presidio_fl.model import FLModel
from presidio_fl.security import (
    log_privacy_budget_spent,
    log_training_complete,
    log_training_start,
)


class FederationRound:
    """Orchestrate multiple federated learning rounds across a set of nodes.

    Parameters
    ----------
    nodes:
        List of ``(X_train, y_train)`` tuples, one per node.
    epsilon:
        Privacy budget per round.  When ``None`` no differential privacy is
        applied.
    delta:
        δ parameter for the Gaussian mechanism.
    clip_norm:
        Gradient norm clipping threshold applied at each client.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        nodes: list[tuple[np.ndarray, np.ndarray]],
        epsilon: float | None = None,
        delta: float = 1e-5,
        clip_norm: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.seed = seed

        n_features = nodes[0][0].shape[1]

        # Clients return raw weights; clipping is applied in the federation
        # loop on the *update* vector (new_weights - old_weights), which is
        # the standard DP-FL approach (DP-FedAvg).
        self._clients: list[FLClient] = [
            FLClient(
                node_id=i,
                X=X,
                y=y,
                clip_norm=None,
                seed=seed + i,
            )
            for i, (X, y) in enumerate(nodes)
        ]
        self._global_model = FLModel(n_features=n_features, seed=seed)
        self._aggregator = FedAvg()

    def run(
        self,
        n_rounds: int,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """Execute *n_rounds* of federated learning.

        Returns
        -------
        dict with keys:
            - ``round_accuracies``: list of float (accuracy after each round)
            - ``final_accuracy``: float
            - ``privacy_budget_used``: float or ``None``
            - ``delta``: float or ``None``
            - ``n_rounds``: int
        """
        use_dp = self.epsilon is not None
        dp_mechanism: GaussianMechanism | None = None
        accountant: PrivacyAccountant | None = None
        rng: np.random.Generator | None = None

        if use_dp:
            dp_mechanism = GaussianMechanism(
                epsilon=self.epsilon,  # type: ignore[arg-type]
                delta=self.delta,
                sensitivity=self.clip_norm,
            )
            accountant = PrivacyAccountant(
                epsilon_per_round=self.epsilon,  # type: ignore[arg-type]
                delta=self.delta,
                max_rounds=n_rounds,
            )
            rng = np.random.default_rng(self.seed)

        log_training_start(
            epsilon=self.epsilon,
            delta=self.delta if use_dp else None,
            n_rounds=n_rounds,
        )

        round_accuracies: list[float] = []
        global_weights = self._global_model.get_weights()

        for round_num in range(1, n_rounds + 1):
            old_weights = {k: v.copy() for k, v in global_weights.items()}

            # --- Client local training ---
            weighted_updates: list[tuple[int, dict[str, np.ndarray]]] = []
            for client in self._clients:
                local_weights = client.train(global_weights)
                weighted_updates.append((client.n_samples, local_weights))

            # --- FedAvg aggregation ---
            aggregated = self._aggregator.aggregate(weighted_updates)

            # --- Optional DP: clip + noise the round *update*, not the weights ---
            # Standard DP-FedAvg: noise is calibrated to the update sensitivity
            # (= clip_norm), so the model accumulates signal across rounds while
            # each round's noise stays bounded.
            if use_dp and dp_mechanism is not None and accountant is not None:
                update = {k: aggregated[k] - old_weights[k] for k in aggregated}
                # Clip the update vector to clip_norm
                total_norm = (
                    float(np.sqrt(sum(np.linalg.norm(v) ** 2 for v in update.values()))) + 1e-8
                )
                scale = min(1.0, self.clip_norm / total_norm)
                update = {k: v * scale for k, v in update.items()}
                # Add calibrated Gaussian noise to the clipped update
                noised_update = dp_mechanism.add_noise(update, rng=rng)
                global_weights = {k: old_weights[k] + noised_update[k] for k in old_weights}
                accountant.spend_round()
                log_privacy_budget_spent(
                    round_num=round_num,
                    budget_remaining=accountant.remaining_budget(),
                )
            else:
                global_weights = aggregated
            self._global_model.set_weights(global_weights)

            # --- Evaluation ---
            acc = self._global_model.score(X_test, y_test)
            round_accuracies.append(acc)

        final_accuracy = round_accuracies[-1] if round_accuracies else 0.0
        budget_used = (n_rounds * self.epsilon) if use_dp else None

        log_training_complete(
            final_accuracy=final_accuracy,
            budget_used=budget_used,
        )

        return {
            "round_accuracies": round_accuracies,
            "final_accuracy": final_accuracy,
            "privacy_budget_used": budget_used,
            "delta": self.delta if use_dp else None,
            "n_rounds": n_rounds,
        }
