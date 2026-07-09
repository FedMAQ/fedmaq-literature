---
type: Method
title: "FedProto"
description: "Prototype-based FL: clients exchange per-class mean feature representations instead of gradients, supporting model-heterogeneous clients at low cost."
tags: [distillation, heterogeneity, representation, baseline]
introduced_by: /papers/tan-2022-fedproto.md
timestamp: 2026-07-09T12:00:00Z
---

# FedProto

Federated Prototype Learning: transfers compact per-class representation statistics
rather than model parameters, tolerating data *and* model heterogeneity.

## 1. Mechanism

A prototype for class \(j\) on client \(i\) is the mean embedding of that client's
class-\(j\) samples, \( C_i^{(j)} = \tfrac{1}{|D_{i,j}|}\sum f_{\phi_i}(x) \). The
server aggregates local prototypes into global prototypes
\( \bar C^{(j)} \) (weighted by class support) and broadcasts them; clients add a
prototype-alignment regularizer to local training,
\( \ell = \ell_{sup} + \lambda\sum_j \lVert C_i^{(j)} - \bar C^{(j)}\rVert \). Because
the exchanged object is embedding-sized and architecture-agnostic, clients may run
different network architectures.

## 2. Key hyperparameters

- \(\lambda\) — prototype-alignment weight.
- Embedding dimension — sets the (model-independent) communication size.

## 3. Communication & computation profile

Extreme compression: only class-mean vectors are transmitted, independent of model
size. Yields personalized per-client models rather than one deployable global
network; class-coverage skew weakens the signal for unobserved classes.

## 4. Papers

- Introduces: [FedProto](/papers/tan-2022-fedproto.md).
- Contrasts with public-dataset KD such as [FedDF](/papers/lin-2020-feddf.md).

## 5. FedMAQ relevance

A reference point for "how little can we send," reached from the prototype direction
rather than quantization. Like FedMAQ's distillation component it transfers
representations, not parameters; prototype-style semantic anchors could stabilize
accuracy when FedMAQ pushes bit-widths low.

# Related

- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Communication efficiency](/concepts/communication-efficiency.md)
- [MOON](/methods/moon.md), [FedDF](/methods/feddf.md)
- [FedProto paper](/papers/tan-2022-fedproto.md)
