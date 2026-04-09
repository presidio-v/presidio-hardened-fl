"""Generate synthetic federated learning data and split across nodes.

Usage::

    python generate_data.py --nodes 3 --samples-per-node 1000 --seed 42 --output data/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from presidio_fl.data import (
    generate_dataset,
    save_node_data,
    save_test_data,
    split_iid,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic FL training data split across nodes."
    )
    parser.add_argument("--nodes", type=int, default=3, help="Number of FL nodes (default: 3)")
    parser.add_argument(
        "--samples-per-node",
        type=int,
        default=1000,
        help="Training samples per node (default: 1000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output", type=str, default="data/", help="Output directory (default: data/)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.nodes < 1:
        print("ERROR: --nodes must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.samples_per_node < 1:
        print("ERROR: --samples-per-node must be >= 1", file=sys.stderr)
        sys.exit(1)

    total_train = args.nodes * args.samples_per_node
    n_test = max(int(total_train * 0.25), 200)
    n_total = total_train + n_test

    print(f"Generating dataset: {n_total} total samples ({total_train} train, {n_test} test)")
    X, y = generate_dataset(n_samples=n_total, n_features=20, n_informative=10, seed=args.seed)

    X_train, y_train = X[:total_train], y[:total_train]
    X_test, y_test = X[total_train:], y[total_train:]

    node_splits = split_iid(X_train, y_train, n_nodes=args.nodes, seed=args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for node_id, (X_node, y_node) in enumerate(node_splits):
        path = save_node_data(output_dir, node_id, X_node, y_node)
        class_counts = {int(c): int((y_node == c).sum()) for c in sorted(set(y_node.tolist()))}
        print(f"  Node {node_id}: {len(y_node)} samples → {path}  classes={class_counts}")

    test_path = save_test_data(output_dir, X_test, y_test)
    test_class_counts = {int(c): int((y_test == c).sum()) for c in sorted(set(y_test.tolist()))}
    print(f"  Test set: {len(y_test)} samples → {test_path}  classes={test_class_counts}")
    print("Done.")


if __name__ == "__main__":
    main()
