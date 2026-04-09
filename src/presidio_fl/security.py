"""Security event logger for presidio-hardened-fl.

Events are written as JSON lines to ``logs/security.jsonl``.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_LOG_PATH = Path("logs/security.jsonl")

logger = logging.getLogger("presidio_fl")


def _setup_logger() -> None:
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _write_event(event: str, **kwargs: object) -> None:
    """Append a JSON event record to the security log file."""
    _setup_logger()
    record: dict[str, object] = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "event": event,
        **kwargs,
    }
    logger.info(
        "SECURITY_EVENT event=%s %s", event, " ".join(f"{k}={v}" for k, v in kwargs.items())
    )
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        logger.warning("Could not write to security log at %s", _LOG_PATH)


def log_training_start(
    epsilon: float | None,
    delta: float | None,
    n_rounds: int,
) -> None:
    """Log the start of a federated training run."""
    _write_event(
        "training_start",
        epsilon=epsilon,
        delta=delta,
        n_rounds=n_rounds,
        dp_enabled=epsilon is not None,
    )


def log_privacy_budget_spent(round_num: int, budget_remaining: float) -> None:
    """Log each round of privacy budget consumption."""
    _write_event(
        "privacy_budget_spent",
        round_num=round_num,
        budget_remaining=round(budget_remaining, 6),
    )


def log_training_complete(final_accuracy: float, budget_used: float | None) -> None:
    """Log the end of a federated training run."""
    _write_event(
        "training_complete",
        final_accuracy=round(final_accuracy, 6),
        budget_used=budget_used,
    )
