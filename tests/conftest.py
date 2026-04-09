"""Shared pytest fixtures for presidio-hardened-fl tests."""

from __future__ import annotations

import pytest

from presidio_fl.data import generate_dataset, split_iid


@pytest.fixture(scope="session")
def small_dataset():
    """A tiny binary classification dataset for fast tests."""
    X, y = generate_dataset(n_samples=300, n_features=10, n_informative=5, seed=0)
    return X, y


@pytest.fixture(scope="session")
def three_node_splits(small_dataset):
    X, y = small_dataset
    return split_iid(X, y, n_nodes=3, seed=0)


@pytest.fixture(scope="session")
def train_test_split_data():
    X, y = generate_dataset(n_samples=500, n_features=10, n_informative=5, seed=1)
    X_train, y_train = X[:400], y[:400]
    X_test, y_test = X[400:], y[400:]
    return X_train, y_train, X_test, y_test
