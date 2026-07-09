---
type: Method
title: "FedMD"
description: "Heterogeneous FL via model distillation: clients share only logits on a public dataset and distill the averaged consensus back into their own architectures."
tags: [distillation, heterogeneity, baseline]
introduced_by: /papers/li-2019-fedmd.md
timestamp: 2026-07-09T12:00:00Z
---

# FedMD

The founding public-dataset distillation method for FL, enabling fully
architecture-heterogeneous clients with no gradient or weight sharing.

## 1. Mechanism

Each client first transfer-learns on a shared public dataset \(D_0\), then
fine-tunes on private data. Per round: the server samples a public subset \(d_j\);
each client uploads its logits \(f_k(x)\) on \(d_j\); the server averages them into a
consensus \(\tilde f(x)=\tfrac1m\sum_k f_k(x)\) and broadcasts it; each client
*digests* (distills toward \(\tilde f\), e.g. MSE or KL on logits) then *revisits*
(fine-tunes on private data). Only logits — never parameters — cross the network.

## 2. Key hyperparameters

- \(S\) — public-subset size per round (sets communication cost).
- Digest loss (MSE vs. KL, temperature) and epochs.
- Optional consensus weights \(c_k\) to down-weight weaker models.

## 3. Communication & computation profile

Payload is logits over a public subset, independent of model architecture, but sent
as raw float32 (no compression). Requires a shared, representative public dataset and
a common label space; no convergence proof; extra digest-phase compute per client.

## 4. Papers

- Introduces: [FedMD](/papers/li-2019-fedmd.md).
- Proxy-data dependence later removed by [FedGen](/papers/zhu-2021-fedgen.md);
  ensemble variant [FedDF](/papers/lin-2020-feddf.md).

## 5. FedMAQ relevance

FedMD is a non-quantized baseline for FedMAQ's distillation component: it already
reduces communication by sending logits instead of models. FedMAQ's contribution is
to quantize those logits (and the consensus) — the paper transmits raw float32, an
obvious target for adaptive compression.

# Related

- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Proxy-dataset distillation](/concepts/proxy-dataset-distillation.md)
- [FedDF](/methods/feddf.md), [FedGen](/methods/fedgen.md)
- [FedMD paper](/papers/li-2019-fedmd.md)
