---
type: Method
title: "FedNova"
description: "Normalized averaging: rescales each client's accumulated update by its number of local steps, removing objective inconsistency from heterogeneous local work."
tags: [aggregation, heterogeneity, baseline]
introduced_by: /papers/wang-2020-fednova.md
timestamp: 2026-07-09T12:00:00Z
---

# FedNova

A normalized-averaging correction that fixes the *objective inconsistency* naive
FedAvg suffers when clients perform unequal numbers of local updates.

## 1. Mechanism

When client \(i\) runs \(\tau_i\) local steps, its cumulative update is larger and
differently scaled, so weighted averaging converges to a biased surrogate objective
\(\tilde F \ne F\). FedNova normalizes each client's accumulated update by its
effective step count before aggregation,
\( x^{(t+1)} = x^{(t)} - \big(\sum_i p_i \tau_i\big)\sum_i p_i \tfrac{\Delta_i}{\tau_i} \),
so no client's local progress dominates the global direction. The framework
subsumes FedAvg, FedProx, and momentum/adaptive local solvers as special cases.

## 2. Key hyperparameters

- Per-client normalization vector / effective \(\tau_i\) (reported by clients).
- \(\tau_{eff}\) — effective global step size.

## 3. Communication & computation profile

Re-weights aggregation only; transmits full-precision models. The server must know
each client's local-step count (and solver parameters) to normalize. Corrects update
*scale*, not representation/gradient drift — typically combined with a drift
corrector.

## 4. Papers

- Introduces: [FedNova](/papers/wang-2020-fednova.md).
- Generalizes [FedAvg](/papers/mcmahan-2017-fedavg.md) and
  [FedProx](/papers/li-2020-fedprox.md).

## 5. FedMAQ relevance

FedMAQ's "multi-adaptive" premise means clients contribute updates of differing
fidelity — a precision-space analogue of heterogeneous \(\tau_i\). FedNova's
principle argues the FedMAQ aggregator should be derived to provably target the true
objective when contributions are heterogeneous in compute *and* bit-width, e.g. by
reweighting for per-client quantization level, not just dataset size.

# Related

- [Non-IID data and client drift](/concepts/non-iid-heterogeneity.md)
- [Adaptive bit-width](/concepts/adaptive-bit-width.md)
- [FedAvg](/methods/fedavg.md), [SCAFFOLD](/methods/scaffold.md),
  [FedDyn](/methods/feddyn.md)
- [FedNova paper](/papers/wang-2020-fednova.md)
