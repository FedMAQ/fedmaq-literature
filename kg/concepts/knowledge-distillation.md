---
type: Concept
title: "Knowledge distillation"
description: "Transferring the behavior of a teacher model into a student by matching soft outputs or intermediate features, used in FL to share knowledge without sharing weights."
tags: [distillation, kd, communication-efficiency, heterogeneity]
timestamp: 2026-07-09T12:00:00Z
---

# Knowledge distillation

## 1. Definition

Knowledge distillation (KD) trains a student model to reproduce a teacher's soft
outputs rather than only hard labels. Introduced by Hinton et al. as matching
temperature-softened logits, the loss combines a supervised term with a divergence
term \( \mathcal L = \alpha\,\mathcal L_{CE} + (1-\alpha)\,T^2\,\mathrm{KL}(p_T^{(\tau)}\,\|\,p_S^{(\tau)}) \),
where \(\tau\) is the softmax temperature. In federated learning the teacher and
student live on different parties, so KD becomes a way to communicate *behavior*
(logits or features) instead of parameters.

## 2. Why it matters for FedMAQ

KD gives FedMAQ two things quantization cannot: (i) a payload independent of model
size — clients exchange per-class logits, not gradients — and (ii) tolerance of
model heterogeneity, since teacher and student need not share architecture. The
thesis pairs distillation with quantization so the two compression mechanisms
compound, and uses distillation to recover the accuracy that aggressive quantization
sacrifices.

## 3. Variants & dimensions

- **Signal matched** — logit/response distillation vs. feature/hidden-state
  distillation (AdaDQ-KD, FedKD).
- **Where transfer happens** — client-side co-distillation (FedMD, mutual KD in
  FedKD) vs. server-side ensemble distillation (FedDF, CFD).
- **Data used** — a shared proxy/public set
  ([Proxy-dataset distillation](/concepts/proxy-dataset-distillation.md)) vs.
  [data-free](/concepts/data-free-distillation.md) generator/prototype transfer.
- **Teacher count** — single global teacher vs. multi-teacher / ensemble.

## 4. Methods & papers

- Methods: [FedMD](/methods/fedmd.md), [FedDF](/methods/feddf.md),
  [FedGen](/methods/fedgen.md), [FedDistill (De-Biasing)](/methods/feddistill-debias.md),
  [FedKD](/methods/fedkd.md), [FD + FAug](/methods/fd-faug.md),
  [CFD](/methods/cfd.md), [DynFed](/methods/dynfed.md), [FedDT](/methods/feddt.md),
  [AdaDQ-KD](/methods/adadq-kd.md), [AQFedAvg + FKD](/methods/quantized-kd.md),
  [FedProto](/methods/fedproto.md), [FedGEMS](/methods/fedgems.md).
- Foundational: [Hinton 2015 distillation](/papers/hinton-2015-distillation.md).
- Surveys: [Qin 2025 KD survey](/papers/qin-2025-kd-survey.md),
  [Salman 2025 KD survey](/papers/salman-2025-kd-survey.md).

# Related

- [Proxy-dataset distillation](/concepts/proxy-dataset-distillation.md)
- [Data-free distillation](/concepts/data-free-distillation.md)
- [Model compression](/concepts/model-compression.md)
- [Communication efficiency](/concepts/communication-efficiency.md)
