---
type: Method
title: "AQFedAvg + FKD (Qu 2020)"
description: "Bandwidth-adaptive unbiased gradient quantization (AQFedAvg) followed by federated knowledge distillation of a smaller student from unlabeled data."
tags: [joint-q-kd, quantization, distillation]
introduced_by: /papers/qu-2020-quantization-kd.md
timestamp: 2026-07-09T12:00:00Z
---

# AQFedAvg + FKD

An early two-phase framework pairing bandwidth-adaptive quantization with federated
knowledge distillation for edge FL.

## 1. Mechanism

- *AQFedAvg*: extends FedAvg with QSGD-style unbiased stochastic gradient quantization
  whose level is set per device by bandwidth,
  \( s_i = 2^{\lceil \log_2(\phi_{\max}/\phi_i)\rceil} \); the number of participating
  clients \(K\) is chosen greedily to respect a bandwidth budget \(T\)
  (\(\sum_{i\le K}\phi_i \le T\)).
- *FKD*: the large model trained by AQFedAvg becomes a fixed teacher; smaller student
  models are trained on *unlabeled* client data to mimic teacher logits, combined with
  supervised loss on a labeled subset,
  \( L = \alpha L_s + (1-\alpha)\lVert z^T - z^S\rVert_2^2 \), where \(\alpha\) is the
  labeled fraction.

## 2. Key hyperparameters

- Bandwidth ratio \(\phi_{\max}/\phi_i\) → bits; budget \(T\).
- \(\alpha\) — labeled-vs-distillation mix.
- Client fraction \(C\).

## 3. Communication & computation profile

4x–16x compression within ~2% accuracy; reaches target accuracy with ~30% labeled data.
Two-stage (teacher then student), single teacher, uplink-only compression (downlink not
detailed); non-IID gains smaller than IID.

## 4. Papers

- Introduces: [AQFedAvg + FKD](/papers/qu-2020-quantization-kd.md).
- Builds on [QSGD](/papers/alistarh-2017-qsgd.md) and
  [FedAvg](/papers/mcmahan-2017-fedavg.md).

## 5. FedMAQ relevance

The earliest paper in the corpus to fuse adaptive quantization with federated
distillation — essentially a prototype of the FedMAQ thesis, and a baseline. Its
bandwidth→bits formula and FKD loss are integrable; FedMAQ extends it with per-layer /
multi-level precision, joint teacher-student training, and multi-teacher distillation.

# Related

- [Quantization](/concepts/quantization.md)
- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Communication efficiency](/concepts/communication-efficiency.md)
- [QSGD](/methods/qsgd.md), [FedAvg](/methods/fedavg.md),
  [DynFed](/methods/dynfed.md)
- [Qu 2020 paper](/papers/qu-2020-quantization-kd.md)
