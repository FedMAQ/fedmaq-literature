---
type: Method
title: "SCAFFOLD"
description: "Variance-reduced FL: server and per-client control variates estimate and subtract client-drift from each local gradient step."
tags: [drift-correction, heterogeneity, baseline]
introduced_by: /papers/karimireddy-2020-scaffold.md
timestamp: 2026-07-09T12:00:00Z
---

# SCAFFOLD

Stochastic Controlled Averaging: the canonical variance-reduction method for
non-IID federated learning.

## 1. Mechanism

A server control variate \(c\) estimates the global update direction and a
per-client control variate \(c_i\) estimates client \(i\)'s direction; their
difference \((c - c_i)\) approximates the client-drift and is subtracted from each
local step, \( y_i \leftarrow y_i - \eta_l\big(g_i(y_i) - c_i + c\big) \). This is
SAGA/SVRG-style variance reduction applied across clients rather than data points,
which also makes it robust to client sampling. The convergence rate is provably
independent of the heterogeneity constant \(G\).

## 2. Key hyperparameters

- \(\eta_l\), \(\eta_g\) — local and global (server) step sizes.
- Control-variate update option (I: recomputed gradient; II: the cheaper
  \(c_i^+ = c_i - c + \tfrac{1}{K\eta_l}(x - y_i)\)).
- \(K\) — local steps per round.

## 3. Communication & computation profile

Doubles uplink and client state: each client transmits and persists a control
variate the size of the model. Reduces rounds but raises bits per round, and
control variates go stale under low participation — a poor fit for
transmit-once cross-device settings.

## 4. Papers

- Introduces: [SCAFFOLD](/papers/karimireddy-2020-scaffold.md).
- Related baselines: [FedAvg](/papers/mcmahan-2017-fedavg.md),
  [FedProx](/papers/li-2020-fedprox.md), [FedDyn](/papers/acar-2021-feddyn.md).

## 5. FedMAQ relevance

SCAFFOLD directly opposes FedMAQ's goal — it *adds* a full-model-sized message.
That tension is a concrete research question: can control variates be transmitted
at low bit-width without destroying drift correction? SCAFFOLD also formalizes that
heterogeneity, not just communication, governs convergence, arguing for
heterogeneity-aware quantization schedules.

# Related

- [Non-IID data and client drift](/concepts/non-iid-heterogeneity.md)
- [Adaptive bit-width](/concepts/adaptive-bit-width.md)
- [FedProx](/methods/fedprox.md), [FedDyn](/methods/feddyn.md),
  [FedNova](/methods/fednova.md)
- [SCAFFOLD paper](/papers/karimireddy-2020-scaffold.md)
