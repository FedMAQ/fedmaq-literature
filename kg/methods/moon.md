---
type: Method
title: "MOON"
description: "Model-contrastive FL: a local regularizer pulls each client's representation toward the global model's and away from its previous local model's."
tags: [drift-correction, heterogeneity, representation, baseline]
introduced_by: /papers/li-2021-moon.md
timestamp: 2026-07-09T12:00:00Z
---

# MOON

Model-Contrastive Federated Learning: corrects non-IID drift in representation
space rather than gradient/weight space.

## 1. Mechanism

The global model, having aggregated all parties, learns a better representation than
any locally skewed model. MOON adds a model-level contrastive term to the local loss
that pulls the current local model's representation \(z\) of an input toward the
global model's \(z_{glob}\) (positive) and away from the previous-round local model's
\(z_{prev}\) (negative):
\( \ell = \ell_{sup} + \mu\,\ell_{con}(z, z_{glob}, z_{prev}) \), with
\(\ell_{con}\) an NT-Xent loss over cosine similarities at temperature \(\tau\). It
is a drop-in modification of the [FedAvg](/methods/fedavg.md) local step; aggregation
and communication are unchanged.

## 2. Key hyperparameters

- \(\mu\) — contrastive weight (representation-alignment strength).
- \(\tau\) — softmax temperature in the contrastive loss.

## 3. Communication & computation profile

FedAvg-sized full-precision payload (no compression). Adds client cost: three
forward passes per batch (local, global, previous-local) and storage of the previous
local model. Improves accuracy per round, not bits per round.

## 4. Papers

- Introduces: [MOON](/papers/li-2021-moon.md).
- Reports that [FedProx](/papers/li-2020-fedprox.md) and
  [SCAFFOLD](/papers/karimireddy-2020-scaffold.md) can stall on deep vision models.

## 5. FedMAQ relevance

MOON's representation-alignment premise is close to knowledge distillation, giving a
non-KD comparison point on the representation axis. Its warning: quantizing updates
perturbs the very representations MOON aligns, motivating representation-aware (not
purely magnitude-aware) quantization criteria in FedMAQ.

# Related

- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Non-IID data and client drift](/concepts/non-iid-heterogeneity.md)
- [FedProto](/methods/fedproto.md), [SCAFFOLD](/methods/scaffold.md)
- [MOON paper](/papers/li-2021-moon.md)
