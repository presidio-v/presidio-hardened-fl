# Presidio-Hardened FL — Requirements

## Overview

`presidio-hardened-fl` is a federated learning simulation library for
Experiment 4 of PRES-EDU-CS-101 (Cloud Solutions with Cybersecurity & ML).
It demonstrates:

1. IID data partitioning across N configurable nodes
2. Federated Averaging (FedAvg) with sample-count weighting
3. Gaussian differential privacy with configurable ε and δ
4. Privacy budget tracking under basic composition
5. Gradient norm clipping before aggregation

## Mandatory Presidio Security Extensions

- Input validation on all CLI parameters (type bounds, positive-valued ε/δ)
- Security event logging to `logs/security.jsonl` for training start, per-round
  budget consumption, and training completion
- No shell execution; all computation is pure Python/NumPy/sklearn
- Full GitHub security files: SECURITY.md, .github/dependabot.yml, .github/workflows/codeql.yml
- No raw training data ever leaves individual node (nodes return weights only)

## Technical Requirements

- Python 3.10+
- `scikit-learn>=1.4`, `pandas>=2.1`, `numpy>=1.26`, `joblib>=1.3`
- `src/presidio_fl/` layout
- pytest ≥80% coverage
- ruff lint + format enforced
- MIT License, version 0.1.0

## Functional Requirements

| ID | Requirement |
|----|-------------|
| REQ-FL-01 | IID data split across N nodes (configurable, default 3) |
| REQ-FL-02 | FedAvg aggregation with sample-count weighting |
| REQ-FL-03 | Gaussian DP mechanism with configurable ε and δ |
| REQ-FL-04 | Privacy budget tracking (basic composition) |
| REQ-FL-05 | Privacy budget exhaustion must raise `PrivacyBudgetExhausted` and halt training |
| REQ-FL-06 | Gradient norm clipping before aggregation (clip_norm configurable, default 1.0) |
| REQ-FL-07 | All training metadata persisted to disk for reproducibility |
| REQ-FL-08 | No raw data ever leaves individual node (enforced by design — nodes only return weights) |

## Version Deliberation Log

### v0.1.0 — Initial release

**Scope decision:** `SGDClassifier(loss='log_loss', warm_start=True)` is used
as the base learner because it supports incremental fitting (compatible with
FL round semantics), is deterministic given a seed, and provides coef_/
intercept_ attributes that can be directly exchanged as weight vectors.

**Scope decision:** Basic (sequential) composition is used for privacy accounting
rather than Rényi or moments accountant. Basic composition is conservative but
analytically transparent and appropriate for a teaching setting where the
trade-off between ε and accuracy is the primary pedagogical objective.

**Scope decision:** The Gaussian mechanism is calibrated to (ε, δ)-DP using the
standard analytic formula σ = s · √(2 ln(1.25/δ)) / ε rather than the
numerically tighter formula from Balle & Wang (2018). This matches the
formulation presented in the lecture slides.

**Scope decision:** Gradient norm clipping is applied to the coefficient vector
(L2 norm) before returning from the client. The intercept is not clipped
separately because its magnitude is typically small relative to the coefficient
vector and clipping it independently would require separate sensitivity analysis.

<!-- Deliver the complete working project ready for GitHub publish. -->

## SDLC

These requirements are delivered under the family-wide Presidio SDLC:
<https://github.com/presidio-v/presidio-hardened-docs/blob/main/sdlc/sdlc-report.md>.
