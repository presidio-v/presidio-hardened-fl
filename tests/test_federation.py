"""Tests for presidio_fl.federation module."""

from __future__ import annotations

import pytest

from presidio_fl.data import generate_dataset, split_iid
from presidio_fl.federation import FederationRound


def _make_nodes_and_test(n_nodes=3, n_per_node=150, seed=0):
    X, y = generate_dataset(
        n_samples=n_nodes * n_per_node + 100, n_features=10, n_informative=5, seed=seed
    )
    X_train, y_train = X[: n_nodes * n_per_node], y[: n_nodes * n_per_node]
    X_test, y_test = X[n_nodes * n_per_node :], y[n_nodes * n_per_node :]
    nodes = split_iid(X_train, y_train, n_nodes=n_nodes, seed=seed)
    return nodes, X_test, y_test


class TestFederationRound:
    def test_run_returns_expected_keys(self):
        nodes, X_test, y_test = _make_nodes_and_test()
        fed = FederationRound(nodes=nodes, epsilon=None, seed=1)
        result = fed.run(n_rounds=3, X_test=X_test, y_test=y_test)
        assert "round_accuracies" in result
        assert "final_accuracy" in result
        assert "privacy_budget_used" in result
        assert "delta" in result
        assert "n_rounds" in result

    def test_no_dp_privacy_budget_none(self):
        nodes, X_test, y_test = _make_nodes_and_test()
        fed = FederationRound(nodes=nodes, epsilon=None)
        result = fed.run(n_rounds=2, X_test=X_test, y_test=y_test)
        assert result["privacy_budget_used"] is None
        assert result["delta"] is None

    def test_final_accuracy_in_unit_interval(self):
        nodes, X_test, y_test = _make_nodes_and_test()
        fed = FederationRound(nodes=nodes, epsilon=None, seed=0)
        result = fed.run(n_rounds=3, X_test=X_test, y_test=y_test)
        assert 0.0 < result["final_accuracy"] < 1.0

    def test_round_accuracies_count_matches_n_rounds(self):
        nodes, X_test, y_test = _make_nodes_and_test()
        fed = FederationRound(nodes=nodes, epsilon=None)
        n_rounds = 4
        result = fed.run(n_rounds=n_rounds, X_test=X_test, y_test=y_test)
        assert len(result["round_accuracies"]) == n_rounds

    def test_no_dp_improves_or_maintains_accuracy(self):
        """Accuracy after 5 rounds should be higher than random (>55%)."""
        nodes, X_test, y_test = _make_nodes_and_test(seed=7)
        fed = FederationRound(nodes=nodes, epsilon=None, seed=7)
        result = fed.run(n_rounds=5, X_test=X_test, y_test=y_test)
        assert result["final_accuracy"] > 0.55

    def test_dp_runs_without_error(self):
        nodes, X_test, y_test = _make_nodes_and_test(seed=2)
        fed = FederationRound(nodes=nodes, epsilon=1.0, delta=1e-5, seed=2)
        result = fed.run(n_rounds=3, X_test=X_test, y_test=y_test)
        assert result["final_accuracy"] > 0.0

    def test_dp_privacy_budget_used_correct(self):
        nodes, X_test, y_test = _make_nodes_and_test()
        eps = 2.0
        n_rounds = 4
        fed = FederationRound(nodes=nodes, epsilon=eps, delta=1e-5, seed=3)
        result = fed.run(n_rounds=n_rounds, X_test=X_test, y_test=y_test)
        assert result["privacy_budget_used"] == pytest.approx(n_rounds * eps)

    def test_dp_delta_preserved_in_result(self):
        nodes, X_test, y_test = _make_nodes_and_test()
        fed = FederationRound(nodes=nodes, epsilon=1.0, delta=1e-6, seed=4)
        result = fed.run(n_rounds=2, X_test=X_test, y_test=y_test)
        assert result["delta"] == pytest.approx(1e-6)

    def test_weak_dp_accuracy_better_than_strong(self):
        """ε=10 should achieve higher accuracy than ε=0.1 over 5 rounds."""
        nodes, X_test, y_test = _make_nodes_and_test(n_per_node=200, seed=5)
        fed_weak = FederationRound(nodes=nodes, epsilon=10.0, delta=1e-5, seed=5)
        fed_strong = FederationRound(nodes=nodes, epsilon=0.1, delta=1e-5, seed=5)
        res_weak = fed_weak.run(n_rounds=5, X_test=X_test, y_test=y_test)
        res_strong = fed_strong.run(n_rounds=5, X_test=X_test, y_test=y_test)
        assert res_weak["final_accuracy"] >= res_strong["final_accuracy"]
