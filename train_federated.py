"""Train a federated model with optional Gaussian differential privacy.

Usage::

    python train_federated.py --nodes 3 --rounds 10 --epsilon 1.0 --delta 1e-5 \\
        --model-out models/fl_model_strong.pkl
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

from presidio_fl.data import load_node_data, load_test_data
from presidio_fl.federation import FederationRound


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a federated logistic regression classifier."
    )
    parser.add_argument("--nodes", type=int, default=3, help="Number of nodes (default: 3)")
    parser.add_argument("--rounds", type=int, default=10, help="Number of FL rounds (default: 10)")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="DP privacy budget ε per round (omit for no DP)",
    )
    parser.add_argument("--delta", type=float, default=1e-5, help="DP δ parameter (default: 1e-5)")
    parser.add_argument(
        "--data-dir", type=str, default="data/", help="Data directory (default: data/)"
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="models/fl_model.pkl",
        help="Output model path (default: models/fl_model.pkl)",
    )
    parser.add_argument(
        "--clip-norm", type=float, default=1.0, help="Gradient clip norm (default: 1.0)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(
            f"ERROR: data directory '{data_dir}' not found. Run generate_data.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load data
    nodes: list = []
    for node_id in range(args.nodes):
        try:
            X, y = load_node_data(data_dir, node_id)
        except FileNotFoundError:
            print(f"ERROR: node_{node_id}_train.csv not found in '{data_dir}'.", file=sys.stderr)
            sys.exit(1)
        nodes.append((X, y))

    X_test, y_test = load_test_data(data_dir)

    dp_str = f"ε={args.epsilon}, δ={args.delta}" if args.epsilon is not None else "disabled"
    print(f"Federated training: {args.nodes} nodes, {args.rounds} rounds, DP={dp_str}")
    print(f"Clip norm: {args.clip_norm}")
    print()

    federation = FederationRound(
        nodes=nodes,
        epsilon=args.epsilon,
        delta=args.delta,
        clip_norm=args.clip_norm,
        seed=args.seed,
    )
    result = federation.run(n_rounds=args.rounds, X_test=X_test, y_test=y_test)

    for r, acc in enumerate(result["round_accuracies"], start=1):
        print(f"  Round {r:3d}: accuracy = {acc:.4f}")

    print(
        f"\nFinal accuracy: {result['final_accuracy']:.4f} ({result['final_accuracy'] * 100:.1f}%)"
    )
    if result["privacy_budget_used"] is not None:
        print(f"Total privacy budget used: ε = {result['privacy_budget_used']:.4f}")
    else:
        print("Differential privacy: not applied")

    # Save model + metadata
    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "final_accuracy": result["final_accuracy"],
        "round_accuracies": result["round_accuracies"],
        "epsilon": args.epsilon,
        "delta": args.delta if args.epsilon is not None else None,
        "n_rounds": args.rounds,
        "n_nodes": args.nodes,
        "clip_norm": args.clip_norm,
        "seed": args.seed,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    joblib.dump(
        {"model": federation._global_model, "metadata": metadata},
        out_path,
    )
    print(f"\nModel saved to {out_path}")


if __name__ == "__main__":
    main()
