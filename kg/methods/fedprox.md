---
type: Method
title: "FedProx"
description: "FedAvg plus a proximal term that penalizes local drift from the global model, and tolerance of variable (partial) local work."
tags: [regularization, fl-core, heterogeneity, baseline]
introduced_by: /papers/li-2020-fedprox.md
timestamp: 2026-07-09T12:00:00Z
---

# FedProx

A drift-limiting generalization of [FedAvg](/methods/fedavg.md) for statistically
and systems-heterogeneous networks.

## 1. Mechanism

Each client minimizes a proximally regularized local objective
\( h_k(w; w^t) = F_k(w) + \tfrac{\mu}{2}\lVert w - w^t\rVert^2 \), which restrains
local updates from straying far from the current global model \(w^t\). Clients need
only return a \(\gamma_k^t\)-inexact minimizer, so devices that cannot finish a full
local schedule (stragglers) contribute partial work instead of being dropped.
FedAvg is the special case \(\mu = 0\).

## 2. Key hyperparameters

- \(\mu\) — proximal strength; too large stalls local progress, too small fails to
  curb drift. The paper gives a loss-based heuristic to adapt \(\mu\).
- \(\gamma_k^t\) — per-device local-solve inexactness (amount of local work).

## 3. Communication & computation profile

Identical payload to FedAvg — full-precision models, one message each way, no
compression. The proximal term adds a cheap quadratic penalty to local gradients;
robustness gains come from optimization, not fewer bits.

## 4. Papers

- Introduces: [FedProx](/papers/li-2020-fedprox.md).
- Baseline in: [SCAFFOLD](/papers/karimireddy-2020-scaffold.md),
  [FedDyn](/papers/acar-2021-feddyn.md), [MOON](/papers/li-2021-moon.md),
  [FedNova](/papers/wang-2020-fednova.md).

## 5. FedMAQ relevance

The proximal term is a candidate stabilizer for FedMAQ: quantization noise perturbs
local updates much as heterogeneity does, and \(\tfrac{\mu}{2}\lVert w-w^t\rVert^2\)
could dampen that drift. FedProx also motivates tolerating variable client budgets —
useful when clients differ in bandwidth or quantization level.

# Related

- [Non-IID data and client drift](/concepts/non-iid-heterogeneity.md)
- [FedAvg](/methods/fedavg.md), [FedDyn](/methods/feddyn.md),
  [SCAFFOLD](/methods/scaffold.md)
- [FedProx paper](/papers/li-2020-fedprox.md)
