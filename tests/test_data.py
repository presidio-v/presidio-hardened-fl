"""Tests for presidio_fl.data module."""

from __future__ import annotations

import tempfile

import numpy as np

from presidio_fl.data import (
    generate_dataset,
    load_node_data,
    load_test_data,
    save_node_data,
    save_test_data,
    split_iid,
)


def test_generate_dataset_shape():
    X, y = generate_dataset(n_samples=200, n_features=10, n_informative=5, seed=42)
    assert X.shape == (200, 10)
    assert y.shape == (200,)


def test_generate_dataset_binary():
    _, y = generate_dataset(n_samples=200, seed=42)
    assert set(y.tolist()).issubset({0, 1})


def test_split_iid_sizes():
    X, y = generate_dataset(n_samples=300, n_features=10, n_informative=5, seed=0)
    splits = split_iid(X, y, n_nodes=3, seed=0)
    assert len(splits) == 3
    total = sum(len(s[1]) for s in splits)
    assert total == 300
    # Each split should be within 1 sample of equal
    sizes = [len(s[1]) for s in splits]
    assert max(sizes) - min(sizes) <= 1


def test_split_iid_no_overlap():
    X, y = generate_dataset(n_samples=90, n_features=10, n_informative=5, seed=0)
    splits = split_iid(X, y, n_nodes=3, seed=0)
    # No two nodes share the same sample (check first feature values)
    sets = [set(map(tuple, s[0].tolist())) for s in splits]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert sets[i].isdisjoint(sets[j]), "Nodes share samples — not IID"


def test_csv_round_trip():
    X, y = generate_dataset(n_samples=50, n_features=20, n_informative=10, seed=7)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_node_data(tmpdir, node_id=0, X_train=X, y_train=y)
        X_loaded, y_loaded = load_node_data(tmpdir, node_id=0)
    np.testing.assert_allclose(X, X_loaded, rtol=1e-5)
    np.testing.assert_array_equal(y, y_loaded)


def test_test_csv_round_trip():
    X, y = generate_dataset(n_samples=40, n_features=20, n_informative=10, seed=8)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_test_data(tmpdir, X, y)
        X_loaded, y_loaded = load_test_data(tmpdir)
    np.testing.assert_allclose(X, X_loaded, rtol=1e-5)
    np.testing.assert_array_equal(y, y_loaded)


def test_no_label_leakage_between_splits():
    """Train labels from one node must not appear in the test set by index."""
    X, y = generate_dataset(n_samples=100, n_features=20, n_informative=10, seed=99)
    n_train = 80
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    with tempfile.TemporaryDirectory() as tmpdir:
        save_node_data(tmpdir, node_id=0, X_train=X_train, y_train=y_train)
        save_test_data(tmpdir, X_test, y_test)

        X_node, y_node = load_node_data(tmpdir, node_id=0)
        X_t, y_t = load_test_data(tmpdir)

    # Verify sizes are correct
    assert len(y_node) == n_train
    assert len(y_t) == 20

    # Verify feature vectors do not overlap
    train_rows = {tuple(row.tolist()) for row in X_node}
    test_rows = {tuple(row.tolist()) for row in X_t}
    assert train_rows.isdisjoint(test_rows)
