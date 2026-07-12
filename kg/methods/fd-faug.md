---
type: Method
title: "FD + FAug"
description: "Federated Distillation exchanges per-label average logits (payload independent of model size), paired with GAN-based Federated Augmentation to counter non-IID skew."
tags: [distillation, data-free, heterogeneity]
introduced_by: /papers/jeong-2023-feddistill-aug.md
timestamp: 2026-07-09T12:00:00Z
---

# FD + FAug

Federated Distillation (FD) with Federated Augmentation (FAug): an early on-device
scheme that decouples communication from model size and rectifies data skew.

## 1. Mechanism

- *FD*: each device periodically uploads its **per-label average output logits**; the
  server averages them into global per-label logit means \( \bar y^{(\ell)} \), broadcast
  as distillation targets. Local loss = supervised CE + \( \gamma\,\mathrm{KL}(\sigma(f_k(x)/T)\,\lVert\,\sigma(\bar y^{(y)}/T)) \).
  Payload \(\propto |\mathcal Y|\times\dim(\text{output})\), independent of network
  width/depth.
- *FAug*: devices upload a few seed samples of missing classes; the server trains a GAN;
  the generator is downloaded so each device synthesizes missing-class samples locally,
  pushing its data toward IID and stabilizing the per-label logit statistics FD relies on.

## 2. Key hyperparameters

- \(T\), \(\gamma\) — distillation temperature and weight.
- FAug seed-sample budget (privacy-vs-communication trade-off).

## 3. Communication & computation profile

~26x less communication than FL, reaching 95–98% of its accuracy — an explicit lossy
trade. Payload scales with output dimension, not parameters. FAug leaks some information
via seed samples; the logit-only channel is classification-centric.

## 4. Papers

- Introduces: [FD + FAug](/papers/jeong-2023-feddistill-aug.md).
- Precursor to [FedDF](/papers/lin-2020-feddf.md),
  [CFD](/papers/sattler-2022-cfd.md), and data-free
  [FedGen](/papers/zhu-2021-fedgen.md).

## 5. FedMAQ relevance

FD is a canonical demonstration that exchanging *outputs* instead of *parameters*
decouples communication from model size — a premise complementary to FedMAQ's parameter
quantization. FedMAQ can combine both: quantized weight updates plus a cheap per-label
logit channel that is naturally robust to aggressive weight quantization. FAug adds a
generative, data-side heterogeneity lever distinct from optimization-based correctors.

# Related

- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Data-free distillation](/concepts/data-free-distillation.md)
- [Communication efficiency](/concepts/communication-efficiency.md)
- [FedGen](/methods/fedgen.md), [CFD](/methods/cfd.md), [FedDF](/methods/feddf.md)
- [FedDistill](/methods/feddistill.md) — the FD component alone, as implemented as
  the FedMAQ codebase's "FedDistill" baseline (no FAug/GAN augmentation)
- [FD+FAug paper](/papers/jeong-2023-feddistill-aug.md)
