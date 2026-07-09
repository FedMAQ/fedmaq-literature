---
type: Method
title: "FedAvg"
description: "Iterative model averaging: clients run multiple local SGD epochs, the server aggregates their models weighted by local sample count."
tags: [drift-correction, fl-core, baseline]
introduced_by: /papers/mcmahan-2017-fedavg.md
timestamp: 2026-07-09T12:00:00Z
---

# FedAvg

The foundational federated optimization algorithm and the direct baseline against
which every other method here is measured.

## 1. Mechanism

Each round \(t\): the server broadcasts the global model \(w_t\) to a random
fraction \(C\) of clients; each selected client \(k\) runs \(E\) local SGD epochs
over minibatches of size \(B\) and returns its updated model; the server averages
the returned models weighted by local sample count,
\( w_{t+1} = \sum_k \tfrac{n_k}{n}\, w_{t+1}^k \). Substituting local training for
per-step gradient exchange cuts communication rounds 10–100x versus synchronous
SGD. Setting \(E=1, B=\infty\) recovers FedSGD (full-gradient averaging).

## 2. Key hyperparameters

- \(C\) — client fraction per round (parallelism).
- \(E\) — local epochs; larger \(E\) saves rounds but, under non-IID data, induces
  client drift and can diverge or plateau.
- \(B\) — local minibatch size.
- \(\eta\) — local learning rate.

## 3. Communication & computation profile

Full-precision model transmitted each round (no payload compression). Uplink and
downlink are both one model-sized message; no persistent client state. Local
compute per client per round is \(E\, n_k / B\) updates.

## 4. Papers

- Introduces: [FedAvg](/papers/mcmahan-2017-fedavg.md).
- Baseline in: virtually all successors, e.g.
  [FedProx](/papers/li-2020-fedprox.md),
  [SCAFFOLD](/papers/karimireddy-2020-scaffold.md),
  [MOON](/papers/li-2021-moon.md), [FedPAQ](/papers/reisizadeh-2020-fedpaq.md).

## 5. FedMAQ relevance

FedAvg is FedMAQ's primary baseline. It reduces rounds but leaves the per-round
payload uncompressed — precisely the axis FedMAQ attacks with multi-adaptive
quantization and knowledge distillation. Its non-IID sensitivity sets the accuracy
envelope aggressive compression must not breach.

# Related

- [Quantization](/concepts/quantization.md)
- [Communication efficiency](/concepts/communication-efficiency.md)
- [Non-IID data and client drift](/concepts/non-iid-heterogeneity.md)
- [FedProx](/methods/fedprox.md), [SCAFFOLD](/methods/scaffold.md)
- [FedAvg paper](/papers/mcmahan-2017-fedavg.md)
