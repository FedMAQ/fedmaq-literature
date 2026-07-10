---
type: Gap
title: "Joint Q+KD methods omit a data-richness signal and a study of how adaptive signals should combine"
description: "Methods combining quantization and distillation either pair a single-precision quantizer with one teacher, or (DynFed) fuse resource+state-adaptive bit-widths with multi-teacher KD but omit data richness and any study of the combination logic -- the direct FedMAQ niche."
tags: [joint-q-kd, quantization, distillation, gap]
timestamp: 2026-07-10T00:00:00Z
---

# Multi-signal adaptive quantization coupled to ensemble distillation is scarce

## Problem

Quantization and distillation are complementary and a handful of methods combine
them, but the combinations leave two things unaddressed. Most pair a single fixed or
coarsely adaptive precision with a single global teacher. The closest prior work,
DynFed, does fuse *resource*- and *training-state*-adaptive bit-widths with
multi-teacher server-side distillation -- but it conditions precision on only those
two signals, omits a *data-richness* signal, and fixes the way its signals combine
rather than studying it. What remains scarce is a quantizer driven by *multiple*
complementary signals (resource + training-state + data-richness) whose combination
logic is *studied* rather than assumed, coupled to ensemble distillation. This is the
exact intersection FedMAQ claims -- a combination-logic contribution over DynFed's
resource-and-state design, not a novel multi-axis precision mechanism.

## State of the art

AQFedAvg+FKD is an early two-stage single-teacher prototype; FedDT distills into a
single ternary-precision student; AdaDQ-KD uses a single global teacher with
straggler-driven precision; DynFed fuses resource+state-adaptive bit-widths with
multi-teacher distillation but without data-richness awareness or a studied
combination rule. Each realizes part of the target; none adds the data-richness
signal *and* systematically determines how the signals should combine.

## FedMAQ's angle

FedMAQ is defined by this gap: it adds a data-richness signal to DynFed's
resource-and-state design and evaluates specific combination logics (linear,
multiplicative, gradient-primary data-modulated, and threshold-based) to determine
the optimal interaction between gradient norm and dataset size under a hard memory
constraint, coupled to server-side ensemble distillation. The contribution is the
combination *principle* and an open, reproducible implementation -- not a claimed
novel round-by-client-by-layer precision mechanism.

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
