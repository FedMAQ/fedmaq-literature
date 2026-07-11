---
type: Method
title: "CFD"
description: "Compressed Federated Distillation: soft-label quantization plus delta coding and dual server-side distillation cut FD communication by orders of magnitude."
tags: [joint-q-kd, distillation, quantization, baseline]
introduced_by: /papers/sattler-2022-cfd.md
timestamp: 2026-07-09T12:00:00Z
---

# CFD

Compressed Federated Distillation: pushes federated distillation's already-small
soft-label payload down further with quantization and entropy/delta coding.

## 1. Mechanism

On a fixed random public subset, client soft-labels are compressed by five techniques:
distill-data curation, *constrained uniform quantization* \(\mathcal Q_b\) of the
probability vectors (at \(b=1\) this is a max-vote), lossless entropy + *delta coding*
that transmits only predictions that changed since the previous round, *dual
distillation* on the server (distill an aggregate model, then send its quantized
soft-labels down to prevent desync under partial participation), and downstream
quantization. Per-round communication is \(b_{total}=|X^{pub}|(H(Y_i)+\eta)\).

## 2. Key hyperparameters

- \(b\) — soft-label quantization bit-width (static).
- Distill-subset size \(|X^{pub}|\).
- Entropy coder (e.g. CABAC); delta-coding on/off.

## 3. Communication & computation profile

>2 orders of magnitude below standard FD and >4 below FedAvg, both directions.
Requires a public unlabeled dataset roughly matching client data; dual distillation
adds server compute and clients bear a local distillation step.

## 4. Papers

- Introduces: [CFD](/papers/sattler-2022-cfd.md).
- Compresses the [FedDF](/papers/lin-2020-feddf.md) /
  [FD+FAug](/papers/jeong-2023-feddistill-aug.md) distillation channel.

## 5. FedMAQ relevance

CFD is the closest baseline for FedMAQ's *soft-label* compression: its constrained
quantization and delta coding are directly reusable, though FedMAQ compresses
gradients rather than logits. Its bit-width \(b\) is static — [FedMAQ](/methods/fedmaq.md)
makes it multi-adaptive, driven by resource, training-state, and data-richness
signals at a per-client, per-round scalar granularity (no layer axis).

# Related

- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Quantization](/concepts/quantization.md)
- [Communication efficiency](/concepts/communication-efficiency.md)
- [FedDF](/methods/feddf.md), [DynFed](/methods/dynfed.md)
- [CFD paper](/papers/sattler-2022-cfd.md)
