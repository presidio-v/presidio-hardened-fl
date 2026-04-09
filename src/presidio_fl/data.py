"""Synthetic dataset generation and IID node splitting for federated learning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def generate_dataset(
    n_samples: int = 3000,
    n_features: int = 20,
    n_informative: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic binary classification dataset.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        class_sep=2.0,
        flip_y=0.01,
        random_state=seed,
    )
    return X, y


def split_iid(
    X: np.ndarray,
    y: np.ndarray,
    n_nodes: int,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Shuffle data and split equally into *n_nodes* IID partitions.

    Returns a list of (X_node, y_node) tuples.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    splits = np.array_split(np.arange(len(y)), n_nodes)
    return [(X_shuffled[idx], y_shuffled[idx]) for idx in splits]


def save_node_data(
    output_dir: str | Path,
    node_id: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Path:
    """Persist node training data as a CSV file.

    Columns: feature_0 … feature_N-1, label
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_features = X_train.shape[1]
    columns = [f"feature_{i}" for i in range(n_features)] + ["label"]
    df = pd.DataFrame(
        np.column_stack([X_train, y_train.reshape(-1, 1)]),
        columns=columns,
    )
    # Keep label as integer
    df["label"] = df["label"].astype(int)
    path = output_dir / f"node_{node_id}_train.csv"
    df.to_csv(path, index=False)
    return path


def save_test_data(
    output_dir: str | Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Path:
    """Persist test data as test.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_features = X_test.shape[1]
    columns = [f"feature_{i}" for i in range(n_features)] + ["label"]
    df = pd.DataFrame(
        np.column_stack([X_test, y_test.reshape(-1, 1)]),
        columns=columns,
    )
    df["label"] = df["label"].astype(int)
    path = output_dir / "test.csv"
    df.to_csv(path, index=False)
    return path


def load_node_data(
    input_dir: str | Path,
    node_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a node's training data from CSV.

    Returns (X, y) as numpy arrays.
    """
    path = Path(input_dir) / f"node_{node_id}_train.csv"
    df = pd.read_csv(path)
    y = df["label"].to_numpy(dtype=int)
    X = df.drop(columns=["label"]).to_numpy(dtype=float)
    return X, y


def load_test_data(input_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the held-out test set from test.csv.

    Returns (X_test, y_test) as numpy arrays.
    """
    path = Path(input_dir) / "test.csv"
    df = pd.read_csv(path)
    y = df["label"].to_numpy(dtype=int)
    X = df.drop(columns=["label"]).to_numpy(dtype=float)
    return X, y
