---
type: Concept
title: "Non-IID heterogeneity"
description: "Statistical and system heterogeneity across federated clients — divergent local data distributions and unequal compute/bandwidth — that biases the global model and slows convergence."
tags: [heterogeneity, non-iid, client-drift, convergence]
timestamp: 2026-07-09T12:00:00Z
---

# Non-IID heterogeneity

## 1. Definition

In federated learning clients hold data drawn from distributions \(\mathcal D_i\) that
differ across clients (statistical heterogeneity) while also differing in compute,
bandwidth, and dataset size (system heterogeneity). When each client minimizes its
own \(F_i(\theta)\), the local optima \(\theta_i^\star\) diverge from the global
\(\theta^\star\); averaging these drifted updates yields *client drift*, an
inconsistency between the average of local solutions and the true global objective.
The effect worsens with more local epochs and higher distribution skew.

## 2. Why it matters for FedMAQ

Heterogeneity is the second axis FedMAQ must survive: a quantization schedule tuned
for IID data can amplify drift, because quantization noise and drift both push the
average away from \(\theta^\star\). FedMAQ therefore needs precision allocation that
is *heterogeneity-aware*, and distillation as a second channel that transfers
knowledge without forcing weight agreement across dissimilar clients.

## 3. Variants & dimensions

- **Statistical** — label skew, feature skew, quantity skew.
- **System** — straggler delay, bandwidth, and device capacity heterogeneity.
- **Model** — clients running different architectures (model-heterogeneous FL),
  which rules out weight averaging and motivates prototype/logit exchange.
- **Remedy family** — regularization (FedProx, FedDyn, MOON), variance reduction
  (SCAFFOLD), objective correction (FedNova), representation sharing (FedProto).

## 4. Methods & papers

- Methods: [FedProx](/methods/fedprox.md), [SCAFFOLD](/methods/scaffold.md),
  [FedDyn](/methods/feddyn.md), [MOON](/methods/moon.md),
  [FedNova](/methods/fednova.md), [FedProto](/methods/fedproto.md),
  [FedDistill (De-Biasing)](/methods/feddistill-debias.md).
- Papers: [FedAvg](/papers/mcmahan-2017-fedavg.md),
  [FedProx](/papers/li-2020-fedprox.md),
  [SCAFFOLD](/papers/karimireddy-2020-scaffold.md).
- Survey: [Jimenez 2024 non-IID survey](/papers/jimenez-2024-non-iid-survey.md).

# Related

- [Adaptive bit-width](/concepts/adaptive-bit-width.md)
- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Communication efficiency](/concepts/communication-efficiency.md)
