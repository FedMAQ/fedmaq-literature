---
type: Concept
title: "Proxy-dataset distillation"
description: "Federated distillation that transfers knowledge through model outputs on a shared public/unlabeled proxy dataset, decoupling communication from model size."
tags: [distillation, kd, proxy-data, communication-efficiency]
timestamp: 2026-07-09T12:00:00Z
---

# Proxy-dataset distillation

## 1. Definition

Proxy-dataset distillation is the branch of federated KD in which every party
evaluates its model on a common *public* or *unlabeled* reference set and exchanges
only the resulting outputs — per-sample logits or class-averaged soft labels. The
server (or the clients in consensus) then distills against those outputs. Because the
message is a function of the proxy set rather than the parameters, payload is
independent of model size and architecture.

## 2. Why it matters for FedMAQ

It is the most communication-efficient distillation family and the natural KD partner
for quantization in FedMAQ: logit exchange over a proxy set already cuts payload by
orders of magnitude, and the residual model traffic can be quantized. The cost is the
dependence on a suitable shared dataset — a requirement FedMAQ must either satisfy or
relax (see [Data-free distillation](/concepts/data-free-distillation.md)).

## 3. Variants & dimensions

- **Transfer site** — client-side consensus (FedMD) vs. server-side ensemble
  distillation (FedDF, CFD).
- **Proxy source** — a labeled public set, an unlabeled public set, or a subset of
  client data designated as shared.
- **Signal granularity** — per-sample logits (FedMD, FedDF) vs. per-label averaged
  logits (FD + FAug, FedDistill).
- **Dependence** — the defining limitation: a distributionally relevant proxy set
  must exist.

## 4. Methods & papers

- Methods: [FedMD](/methods/fedmd.md), [FedDF](/methods/feddf.md),
  [FedKD](/methods/fedkd.md), [CFD](/methods/cfd.md),
  [AQFedAvg + FKD](/methods/quantized-kd.md).
- Papers: [FedMD](/papers/li-2019-fedmd.md), [FedDF](/papers/lin-2020-feddf.md).
- Surveys: [Qin 2025 KD survey](/papers/qin-2025-kd-survey.md),
  [Salman 2025 KD survey](/papers/salman-2025-kd-survey.md).

# Related

- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Data-free distillation](/concepts/data-free-distillation.md)
- [Communication efficiency](/concepts/communication-efficiency.md)
