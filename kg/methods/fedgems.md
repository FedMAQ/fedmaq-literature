---
type: Method
title: "FedGEMS"
description: "Larger server-model FL: server self-trains/self-distills from a cached logit pool and, only when needed, distills from an entropy-weighted, misprediction-filtered ensemble of client logits."
tags: [distillation, sota, baseline]
introduced_by: /papers/cheng-2021-fedgems.md
timestamp: 2026-07-24T14:38:00Z
---

# FedGEMS

Extends the FedMD-style public-dataset distillation family by letting the server
train a strictly larger model than any client ("GEM"), and by making client
knowledge fusion *selective* rather than a flat average.

## 1. Mechanism

Per round, clients train locally then upload logits on a shared public dataset.
For each public sample, the server takes one of three paths: (1) if it already
predicts correctly, plain cross-entropy self-training, caching the logit into a
global pool; (2) if wrong but a cached logit exists, self-distill from the cache
(no client query needed); (3) otherwise, distill from the ensemble of clients that
predicted the sample correctly, weighted by \(\alpha_{C_j}=\mathrm{softmax}(1/H(p_{C_j}))\)
(inverse prediction entropy), with mispredicting clients weighted 0. The server
then broadcasts its logits and clients distill from them before local training.

## 2. Key hyperparameters

- \(\epsilon\) — cross-entropy vs. KL-distillation trade-off in every server loss term (best found: 0.75).
- Public/private dataset split ratio (affects both accuracy and communication cost).
- Server model capacity relative to clients (ResNet-20 to ResNet-110 in the paper's ablations).

## 3. Communication & computation profile

Payload is client logits on public-dataset samples, sent uncompressed (float);
the self-distillation cache and misprediction filtering *reduce how often* logits
must be exchanged but do not compress the logits themselves. Requires a labeled
public dataset shared by server and clients. Server-side compute is higher than
client-parity methods since it trains a strictly larger model.

## 4. Papers

- Introduces: [FedGEMS](/papers/cheng-2021-fedgems.md) (and its base variant FedGEM, same paper).
- Compared against as a KD-with-larger-server-model baseline alongside [FedMD](/methods/fedmd.md), [FedDF](/methods/feddf.md), and FedGKT.

## 5. FedMAQ relevance

FedGEMS's confidence-weighted client selection is a reference point for FedMAQ's
own server-side weighting of client contributions, though FedMAQ's signals are
resource/training-state/data-richness driven rather than purely prediction-entropy
driven. FedGEMS never quantizes the logits it transmits — only the frequency of
transmission is reduced — leaving logit/soft-label quantization as an open
opportunity that FedMAQ's design targets.

# Related

- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Proxy-dataset distillation](/concepts/proxy-dataset-distillation.md)
- [FedMD](/methods/fedmd.md), [FedDF](/methods/feddf.md)
- [FedGEMS paper](/papers/cheng-2021-fedgems.md)
