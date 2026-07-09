---
type: Gap
title: "Quantization schedules rarely condition on statistical heterogeneity, only on bandwidth"
description: "Adaptive precision is typically allocated by bandwidth or straggler delay; conditioning the schedule on statistical (label/feature) heterogeneity to protect drifting clients is largely unexplored."
tags: [quantization, heterogeneity, adaptive, non-iid]
timestamp: 2026-07-09T12:00:00Z
---

# Quantization schedules rarely condition on statistical heterogeneity

## Problem

Adaptive-quantization methods mostly allocate bits by *system* signals — bandwidth,
device speed, straggler delay — treating all clients' data as interchangeable. Whether
precision should also condition on *statistical* heterogeneity (label/feature skew), so
that clients whose updates carry more drift are quantized differently, is largely
unexplored. Uniformly compressing a highly skewed client may discard exactly the signal
needed to counter drift.

## State of the art

AdaGQ and LAQ-HC allocate by round time, bandwidth, or a data-quality proxy; DAdaQuant
by client weight; DynFed and AdaDQ-KD by resource/delay. None sets precision from a
direct measure of a client's distributional divergence, and drift-correction methods
that do model heterogeneity assume full precision.

## FedMAQ's angle

FedMAQ's heterogeneity-aware precision means the schedule should read statistical skew,
not just bandwidth — spending bits where drift is worst and leaning on distillation
elsewhere. This gap defines a design axis of the multi-adaptive quantizer.

## Sources

- Motivating findings: [Adaptive bit-width beats uniform quantization](/findings/adaptive-quantization-beats-uniform.md),
  [No method jointly optimizes compression and heterogeneity robustness](/findings/no-unified-compression-heterogeneity-method.md).
- Papers: [AdaGQ](/papers/liu-2023-adagq.md), [LAQ-HC](/papers/cui-2026-laq-hc.md),
  [DynFed](/papers/he-2025-dynfed.md).

# Related

- [Adaptive bit-width](/concepts/adaptive-bit-width.md)
- [Non-IID heterogeneity](/concepts/non-iid-heterogeneity.md)
- [Joint precision scheduling is unsolved](/gaps/adaptive-precision-scheduling.md)
- [Quantization-drift interaction](/gaps/quantization-drift-interaction.md)
- [AdaGQ](/methods/adagq.md), [LAQ-HC](/methods/laq-hc.md)
