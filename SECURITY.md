# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

Please report security vulnerabilities by opening a private GitHub Security Advisory
(via the "Security" tab → "Report a vulnerability") rather than a public issue.

Include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You will receive an acknowledgement within 5 business days. We aim to release a patch
within 30 days of a confirmed vulnerability.

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

## Software Development Lifecycle

This repository is developed under the Presidio hardened-family SDLC. The public report
— scope, standards mapping, threat-model gates, and supply-chain controls — is at
<https://github.com/presidio-v/presidio-hardened-docs/blob/main/sdlc/sdlc-report.md>.
