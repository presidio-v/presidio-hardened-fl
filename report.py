"""Report generator for federated learning experiment results.

Usage::

    python report.py --experiment 4
    python report.py --compare fl_model_weak fl_model_strong fl_model_vstrong
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report on federated learning experiment results."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--experiment",
        type=int,
        help="Load latest model from models/ and print full summary (use --experiment 4).",
    )
    group.add_argument(
        "--compare",
        nargs="+",
        metavar="MODEL",
        help="Compare named models (stems without path/extension) from models/.",
    )
    return parser.parse_args(argv)


def _load_model(path: Path) -> dict:
    data = joblib.load(path)
    if isinstance(data, dict) and "metadata" in data:
        return data["metadata"]
    raise ValueError(f"Unexpected model file format at {path}")


def _latest_model_path(models_dir: Path) -> Path:
    candidates = list(models_dir.glob("*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No .pkl files found in {models_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _print_summary(meta: dict, name: str = "") -> None:
    title = f"Model: {name}" if name else "Model summary"
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    eps = meta.get("epsilon")
    print(f"  epsilon (ε):       {eps if eps is not None else 'N/A (no DP)'}")
    print(f"  delta (δ):         {meta.get('delta', 'N/A')}")
    print(f"  Rounds:            {meta.get('n_rounds', '?')}")
    print(f"  Final accuracy:    {meta['final_accuracy'] * 100:.2f}%")
    budget = meta.get("privacy_budget_used") or (
        (meta["n_rounds"] * meta["epsilon"]) if meta.get("epsilon") else None
    )
    print(f"  Budget used (ε):   {budget if budget is not None else 'N/A'}")
    print(f"  Nodes:             {meta.get('n_nodes', '?')}")
    print(f"  Timestamp:         {meta.get('timestamp', '?')}")
    accuracies = meta.get("round_accuracies", [])
    if accuracies:
        print("\n  Per-round accuracies:")
        for i, acc in enumerate(accuracies, start=1):
            print(f"    Round {i:3d}: {acc:.4f}")


def _print_comparison(records: list[tuple[str, dict]]) -> None:
    header = f"{'Model':<25} {'ε':>8} {'δ':>10} {'Rounds':>7} {'Accuracy':>10} {'Budget used':>12}"
    print(f"\n{header}")
    print("-" * len(header))
    for name, meta in records:
        eps = meta.get("epsilon")
        delta = meta.get("delta")
        rounds = meta.get("n_rounds", "?")
        acc = meta.get("final_accuracy", 0.0)
        budget = (rounds * eps) if (eps is not None and isinstance(rounds, int)) else None
        eps_str = f"{eps:.2f}" if eps is not None else "none"
        delta_str = f"{delta:.0e}" if delta is not None else "N/A"
        budget_str = f"{budget:.2f}" if budget is not None else "N/A"
        print(
            f"{name:<25} {eps_str:>8} {delta_str:>10} {rounds:>7} "
            f"{acc * 100:>9.1f}% {budget_str:>12}"
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    models_dir = Path("models")

    if args.experiment is not None:
        if args.experiment != 4:
            print(f"WARNING: only experiment 4 is supported (got {args.experiment})")
        try:
            path = _latest_model_path(models_dir)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        meta = _load_model(path)
        _print_summary(meta, name=path.stem)

    elif args.compare:
        records: list[tuple[str, dict]] = []
        for stem in args.compare:
            path = models_dir / f"{stem}.pkl"
            if not path.exists():
                print(f"ERROR: model file not found: {path}", file=sys.stderr)
                sys.exit(1)
            meta = _load_model(path)
            records.append((stem, meta))
        _print_comparison(records)


if __name__ == "__main__":
    main()
