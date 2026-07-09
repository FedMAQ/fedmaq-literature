---
type: Method
title: "FedGen"
description: "Data-free KD: the server learns a lightweight label-conditioned generator from clients' prediction rules and broadcasts it to regularize local training."
tags: [distillation, data-free, heterogeneity, baseline]
introduced_by: /papers/zhu-2021-fedgen.md
timestamp: 2026-07-09T12:00:00Z
---

# FedGen

Data-Free Knowledge Distillation: removes the proxy-dataset requirement of FedDF/FedMD
by distilling the client ensemble into a generator instead of onto real data.

## 1. Mechanism

The model splits into a feature extractor \(f: X\to Z\) and predictor \(h: Z\to\triangle_Y\).
The server trains a generator \(G(\cdot\mid y)\) so that, for a target label \(y\), its
sampled latent features are confidently classified as \(y\) by the *ensemble* of client
heads, \( \max_G \mathbb{E}_{y,\,z\sim G(\cdot\mid y)}\log\sum_k w_k\,h_{\theta_p^k}(z)_y \) —
distilling "what clients collectively know" with no real data. \(G\) is broadcast and
each client augments local training with \(G\)-sampled latent features, injecting peer
knowledge as an inductive bias against local-data drift.

## 2. Key hyperparameters

- Latent dimension \(d\) of \(Z\) (keeps \(G\) lightweight).
- Distillation/regularization weight on the generator term in the local loss.
- Ensemble weights \(w_k\).

## 3. Communication & computation profile

Broadcasts a small latent-space generator (needs only client predictor heads, not full
models); adds per-round server training of \(G\). Assumes a shared classifier
head / latent space; generator quality inherits client-head bias under scarce data.

## 4. Papers

- Introduces: [FedGen](/papers/zhu-2021-fedgen.md).
- Removes proxy dependence of [FedDF](/papers/lin-2020-feddf.md) and
  [FedMD](/papers/li-2019-fedmd.md).

## 5. FedMAQ relevance

FedGen is the data-free KD baseline for FedMAQ where the server lacks data. The
lightweight generator is itself a compression of ensemble knowledge, aligning with
FedMAQ's "maximal knowledge, minimal bits" aim, and — because it regularizes *local*
training — it is a hook to stabilize clients under aggressive quantization by
compensating for information lost to low bit-widths.

# Related

- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Data-free distillation](/concepts/data-free-distillation.md)
- [FedDF](/methods/feddf.md), [FedMD](/methods/fedmd.md),
  [FedProto](/methods/fedproto.md)
- [FedGen paper](/papers/zhu-2021-fedgen.md)
