---
type: Concept
title: "Data-free distillation"
description: "Federated distillation that needs no shared dataset — knowledge is carried by a learned generator or by class prototypes instead of proxy-set outputs."
tags: [distillation, kd, data-free, generator, prototype]
timestamp: 2026-07-09T12:00:00Z
---

# Data-free distillation

## 1. Definition

Data-free distillation removes the proxy-dataset requirement of federated KD. Instead
of exchanging outputs on a shared set, the knowledge medium is either (i) a lightweight
*generator* that synthesizes feature/label-conditioned samples the server broadcasts to
regularize local training (FedGen), or (ii) *class prototypes* — averaged latent
representations per label that clients align to (FedProto), or (iii) locally generated
augmentations that make per-label logit exchange viable without shared data
(FD + FAug). No client raw data and no public corpus leaves the device.

## 2. Why it matters for FedMAQ

The proxy-set assumption is the sharpest limitation of the proxy-dataset family; where
no representative public data exists — common in energy, sensor, and IoT deployments —
data-free transfer is the only distillation option. FedMAQ must weigh the payload
savings of prototype/generator transfer against its accuracy cost relative to
proxy-based distillation, and decide which regime its target deployments occupy.

## 3. Variants & dimensions

- **Generator-based** — a conditional generator learns the aggregated data manifold
  and induces a distillation signal (FedGen).
- **Prototype-based** — per-class embedding centroids are averaged and shared;
  payload scales with classes, not model size (FedProto).
- **Augmentation-based** — clients synthesize local data (GAN augmentation) so
  per-label logits become shareable (FD + FAug).
- **Trade-off** — no data dependency, but typically lower accuracy and extra
  generator/prototype machinery.

## 4. Methods & papers

- Methods: [FedGen](/methods/fedgen.md), [FedProto](/methods/fedproto.md),
  [FD + FAug](/methods/fd-faug.md), [FedDistill](/methods/feddistill.md).
- Papers: [FedGen](/papers/zhu-2021-fedgen.md),
  [FedProto](/papers/tan-2022-fedproto.md),
  [FD + FAug](/papers/jeong-2023-feddistill-aug.md).

# Related

- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Proxy-dataset distillation](/concepts/proxy-dataset-distillation.md)
- [Non-IID heterogeneity](/concepts/non-iid-heterogeneity.md)
