---
type: Gap
title: "Joint methods pair single-precision quantization with a single teacher; multi-adaptive Q+KD is scarce"
description: "The few methods combining quantization and distillation fix one bit-width and one global teacher; none unifies multi-axis adaptive precision with multi-teacher distillation — the direct FedMAQ niche."
tags: [joint-q-kd, quantization, distillation, gap]
timestamp: 2026-07-09T12:00:00Z
---

# Multi-adaptive quantization + knowledge distillation is scarce

## Problem

Quantization and distillation are complementary and a handful of methods combine them,
but the combinations are shallow: a single fixed or coarsely adaptive precision paired
with a single global teacher. No corpus method unifies *multi-adaptive* precision
(across round, client, and layer) with *multi-teacher* / ensemble distillation under
explicit heterogeneity robustness. This is the exact intersection FedMAQ claims.

## State of the art

AQFedAvg+FKD is an early two-stage single-teacher prototype; FedDT distills into a
single ternary-precision student; DynFed fuses adaptive bit-widths with multi-teacher
distillation but not across all precision axes; AdaDQ-KD uses a single global teacher
with straggler-driven precision. Each realizes part of the target; none realizes the
full multi-axis, multi-teacher design.

## FedMAQ's angle

FedMAQ is defined by this gap: multi-adaptive quantization coupled with (multi-)teacher
distillation, jointly trained and heterogeneity-aware. Closing it is the thesis
contribution rather than an incremental improvement on any single baseline.

## Sources

- Motivating findings: [Quantization and KD are complementary](/findings/quantization-kd-complementary.md),
  [No method jointly optimizes compression and heterogeneity robustness](/findings/no-unified-compression-heterogeneity-method.md).
- Papers: [AQFedAvg+FKD](/papers/qu-2020-quantization-kd.md),
  [FedDT](/papers/he-2025-feddt.md), [DynFed](/papers/he-2025-dynfed.md),
  [AdaDQ-KD](/papers/wang-2026-adadq-kd.md).

# Related

- [Quantization](/concepts/quantization.md),
  [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Adaptive bit-width](/concepts/adaptive-bit-width.md)
- [CFD](/methods/cfd.md), [DynFed](/methods/dynfed.md), [FedDT](/methods/feddt.md),
  [AdaDQ-KD](/methods/adadq-kd.md), [AQFedAvg + FKD](/methods/quantized-kd.md)
