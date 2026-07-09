---
type: Finding
title: "Federated distillation decouples payload from model size but classically depends on a shared proxy dataset"
description: "Exchanging logits over a reference set makes communication independent of model architecture and size, at the cost of requiring a distributionally relevant public/unlabeled dataset."
tags: [distillation, kd, proxy-data, communication-efficiency]
timestamp: 2026-07-09T12:00:00Z
---

# Federated distillation decouples payload from model size but needs a proxy set

## Scope

Covers proxy-dataset federated KD methods — client-side consensus and server-side
ensemble distillation — versus full-model averaging, on communication payload and on
tolerance of architecture heterogeneity.

## Claim

When parties exchange outputs on a shared reference set instead of parameters, the
message size is a function of the proxy set and label space, not of model size, so
uplink cost is decoupled from architecture and heterogeneous models can interoperate.
The recurring precondition is a *distributionally relevant proxy dataset*: performance
degrades when the public/unlabeled set does not cover the clients' data manifold.

## Evidence

| Paper | Key result | Link |
| --- | --- | --- |
| FedMD | Public-set logit consensus enables architecture-heterogeneous clients to co-train | [/papers/li-2019-fedmd.md](/papers/li-2019-fedmd.md) |
| FedDF | Server ensemble distillation over proxy data tolerates size/precision/architecture heterogeneity | [/papers/lin-2020-feddf.md](/papers/lin-2020-feddf.md) |
| FedKD | Mutual distillation plus SVD compression cuts communication far below weight averaging | [/papers/wu-2022-fedkd.md](/papers/wu-2022-fedkd.md) |
| CFD | Soft-label quantization over a reference set drives dual server distillation | [/papers/sattler-2022-cfd.md](/papers/sattler-2022-cfd.md) |
| Qin 2025 (survey) | Surveys proxy-data dependence as a defining constraint of federated KD | [/papers/qin-2025-kd-survey.md](/papers/qin-2025-kd-survey.md) |

## Open gaps

- Removing or relaxing the proxy-set requirement without the accuracy cost of
  data-free transfer: [/gaps/proxy-data-dependence.md](/gaps/proxy-data-dependence.md).

# Related

- [Proxy-dataset distillation](/concepts/proxy-dataset-distillation.md)
- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [FedMD](/methods/fedmd.md), [FedDF](/methods/feddf.md), [FedKD](/methods/fedkd.md)
- [Data-free KD removes the proxy requirement at an accuracy cost](/findings/data-free-kd-removes-proxy-cost.md)
