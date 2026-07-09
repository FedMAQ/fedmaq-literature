---
type: Concept
title: "Communication efficiency"
description: "Reducing the total bits exchanged to reach a target accuracy in FL, framed as a bits-per-round versus rounds-to-converge trade-off."
tags: [communication-efficiency, compression, convergence]
timestamp: 2026-07-09T12:00:00Z
---

# Communication efficiency

## 1. Definition

Communication efficiency measures the total data transferred to reach a target
accuracy, not merely the payload of one round. It decomposes as
\( \text{total bits} \approx (\text{bits per round}) \times (\text{rounds to converge}) \).
Compression cuts the first factor but, by injecting error, can inflate the second, so
the meaningful objective is the product. Uplink (client→server) and downlink
(server→client) budgets are usually asymmetric, and partial participation adds a
third lever.

## 2. Why it matters for FedMAQ

This is the objective FedMAQ optimizes. The thesis title's "communication-efficient"
names the goal that quantization and distillation jointly serve. FedMAQ's argument is
that a single-axis compressor (fixed-bit quantization or proxy-data distillation
alone) leaves the product un-minimized, whereas multi-adaptive precision plus
distillation lowers bits-per-round while distillation limits the rounds penalty.

## 3. Variants & dimensions

- **Levers** — quantization/sparsification (fewer bits per update), distillation
  (payload independent of model size), periodic averaging and partial participation
  (fewer/lighter rounds).
- **Direction** — uplink vs. downlink compression; many methods compress only uplink.
- **Metric** — bits to target accuracy, compression ratio, or wall-clock round time
  under bandwidth heterogeneity.

## 4. Methods & papers

- Methods: [FedPAQ](/methods/fedpaq.md), [QSGD](/methods/qsgd.md),
  [signSGD](/methods/signsgd.md), [FedKD](/methods/fedkd.md),
  [FD + FAug](/methods/fd-faug.md), [CFD](/methods/cfd.md),
  [AdaGQ](/methods/adagq.md), [AQFedAvg + FKD](/methods/quantized-kd.md).
- Survey: [Le 2024 compression survey](/papers/le-2024-compression-survey.md),
  [Cajas-Ordonez 2025 edge survey](/papers/cajas-ordonez-2025-edge-computing-survey.md).

# Related

- [Quantization](/concepts/quantization.md)
- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Model compression](/concepts/model-compression.md)
- [Adaptive bit-width](/concepts/adaptive-bit-width.md)
- [Edge / IoT deployment](/concepts/edge-iot-deployment.md)
