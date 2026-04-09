# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

Do not open a public GitHub issue for security vulnerabilities.

Email **security@presidio-group.eu** with description, reproduction steps, and impact.
Acknowledgement within 48 hours; resolution within 7 days.

## Security Features

| Feature | Description |
|---|---|
| **Input validation** | All CLI parameters are type-checked and bounds-validated |
| **No shell execution** | All computation is pure Python/NumPy/sklearn; no subprocess calls |
| **Security event logging** | Structured JSON events for training start, per-round budget consumption, and completion |
| **No data leakage** | Nodes communicate only via model weight vectors; raw training data never leaves the node |
| **Gradient norm clipping** | Limits the influence of any single training sample on the aggregated model |
| **Privacy budget enforcement** | `PrivacyBudgetExhausted` exception halts training when the ε budget is consumed |
| **Deterministic generation** | All random operations accept a seed for reproducibility |
| **Dependency audit** | Runs `pip-audit` non-blocking on import when available |

## Threat Model

| Threat | Mitigation |
|---|---|
| Gradient inversion / data reconstruction | Gradient norm clipping + Gaussian DP noise before aggregation |
| Privacy budget over-spend | `PrivacyAccountant` enforces hard stop; exception raised on violation |
| Malicious node injecting arbitrary weights | FedAvg weighted by sample count; clip_norm bounds per-node influence |
| Unvalidated CLI inputs | argparse type annotations + manual range checks in each script |
