# presidio-hardened-fl

Privacy-preserving federated learning simulation for PRES-EDU-CS-101
(Cloud Solutions with Cybersecurity & ML), Experiment 4.

Simulates 3 nodes training a logistic regression classifier using
FedAvg aggregation with optional Gaussian differential privacy.

## Quick start

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 1. Generate synthetic data
python generate_data.py --nodes 3 --samples-per-node 1000 --seed 42 --output data/

# 2. Inspect the data
python eda.py --input data/

# 3. Train with different privacy budgets
python train_federated.py --nodes 3 --rounds 10 --epsilon 10.0 --delta 1e-5 \
    --model-out models/fl_model_weak.pkl

python train_federated.py --nodes 3 --rounds 10 --epsilon 1.0 --delta 1e-5 \
    --model-out models/fl_model_strong.pkl

python train_federated.py --nodes 3 --rounds 10 --epsilon 0.1 --delta 1e-5 \
    --model-out models/fl_model_vstrong.pkl

# 4. Compare results
python report.py --compare fl_model_weak fl_model_strong fl_model_vstrong
```

## Architecture

```
src/presidio_fl/
  data.py        — synthetic data generation + IID node split
  model.py       — SGDClassifier wrapper (FL-compatible weights API)
  client.py      — FLClient: local training + gradient norm clipping
  aggregator.py  — FedAvg: weighted average of coef_ + intercept_
  dp.py          — GaussianMechanism + PrivacyAccountant
  federation.py  — FederationRound: orchestrate rounds, track accuracy + budget
  security.py    — structured security event logging to logs/security.jsonl
```

## License

MIT

---

## SDLC

This repository is developed under the Presidio hardened-family SDLC:
<https://github.com/presidio-v/presidio-hardened-docs/blob/main/sdlc/sdlc-report.md>.
