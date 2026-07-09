---
type: Concept
title: "Model compression"
description: "The umbrella of techniques — quantization, distillation, pruning, low-rank factorization — that shrink a model or its communicated updates."
tags: [compression, quantization, distillation, pruning]
timestamp: 2026-07-09T12:00:00Z
---

# Model compression

## 1. Definition

Model compression reduces the size of a neural network or the messages it exchanges
while preserving task accuracy. The principal families are quantization (fewer bits
per parameter), knowledge distillation (a smaller student replaces a larger teacher),
pruning/sparsification (fewer non-zero parameters), and low-rank factorization
(fewer effective parameters, e.g. SVD of gradients). In federated learning the target
of compression is usually the *communicated update* rather than only the stored model.

## 2. Why it matters for FedMAQ

FedMAQ sits at the intersection of the two compression families that dominate FL —
quantization and distillation — and treats them as composable rather than
alternatives. Framing them under one concept clarifies where FedMAQ's novelty lies:
not in a new single compressor, but in adaptively combining precision reduction with
knowledge transfer to push the accuracy-vs-size frontier further than either alone.

## 3. Variants & dimensions

- **Quantization** — precision reduction; see [Quantization](/concepts/quantization.md).
- **Distillation** — a compact student mimics a teacher; see
  [Knowledge distillation](/concepts/knowledge-distillation.md).
- **Sparsification / pruning** — drop small-magnitude entries (top-k gradients).
- **Low-rank factorization** — SVD/dynamic-rank compression of gradients (FedKD).
- **Composition** — stacking two families (quantization + KD in the joint-Q+KD
  methods; ternary + distillation in FedDT).

## 4. Methods & papers

- Methods: [FedDT](/methods/feddt.md), [FedKD](/methods/fedkd.md),
  [CFD](/methods/cfd.md), [DynFed](/methods/dynfed.md),
  [AQFedAvg + FKD](/methods/quantized-kd.md).
- Survey: [Le 2024 compression survey](/papers/le-2024-compression-survey.md).

# Related

- [Quantization](/concepts/quantization.md)
- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Communication efficiency](/concepts/communication-efficiency.md)
