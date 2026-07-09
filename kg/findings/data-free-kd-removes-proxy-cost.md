---
type: Finding
title: "Data-free and prototype distillation remove the proxy-dataset requirement at an accuracy cost"
description: "Generator-, prototype-, and augmentation-based transfer eliminate the shared-dataset dependency of federated KD, but generally trade some accuracy and add generator/prototype machinery."
tags: [distillation, data-free, prototype, heterogeneity]
timestamp: 2026-07-09T12:00:00Z
---

# Data-free and prototype distillation remove the proxy requirement at a cost

## Scope

Covers federated KD variants that carry knowledge without a shared dataset — learned
generators, class prototypes, and local augmentation — against proxy-dataset KD, on
data dependency and accuracy.

## Claim

Data-free transfer resolves the sharpest limitation of proxy-dataset KD: no public
corpus need exist. A broadcast conditional generator, averaged per-class prototypes,
or GAN-augmented local data each supply a distillation signal from on-device
information alone. The consistent cost is lower accuracy than a well-matched proxy set
plus the overhead of training and communicating generators or prototypes, and prototype
methods further restrict what can be transferred (representations, not full logits).

## Evidence

| Paper | Key result | Link |
| --- | --- | --- |
| FedGen | Data-free broadcast generator regularizes local training, no proxy set needed | [/papers/zhu-2021-fedgen.md](/papers/zhu-2021-fedgen.md) |
| FedProto | Per-class prototype exchange supports model-heterogeneous FL with payload scaling in classes, not model size | [/papers/tan-2022-fedproto.md](/papers/tan-2022-fedproto.md) |
| FD + FAug | GAN augmentation makes per-label logit exchange viable without shared data | [/papers/jeong-2023-feddistill-aug.md](/papers/jeong-2023-feddistill-aug.md) |
| FedDistill | Group distillation de-biases classifiers over local class imbalance | [/papers/song-2024-feddistill.md](/papers/song-2024-feddistill.md) |

## Open gaps

- Closing the accuracy gap to proxy-based KD while keeping data-free operation:
  [/gaps/proxy-data-dependence.md](/gaps/proxy-data-dependence.md).

# Related

- [Data-free distillation](/concepts/data-free-distillation.md)
- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [FedGen](/methods/fedgen.md), [FedProto](/methods/fedproto.md),
  [FD + FAug](/methods/fd-faug.md)
- [Distillation cuts uplink but needs a proxy set](/findings/distillation-cuts-uplink-needs-proxy.md)
