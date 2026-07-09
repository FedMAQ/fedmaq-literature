---
type: Gap
title: "Joint precision scheduling across round, client, and layer is unsolved"
description: "Existing adaptive-quantization methods adapt precision along a single axis; how to schedule bit-width jointly across time, client, and layer under a communication budget is open."
tags: [quantization, adaptive, scheduling, communication-efficiency]
timestamp: 2026-07-09T12:00:00Z
---

# Joint precision scheduling across round, client, and layer is unsolved

## Problem

Adaptive quantization has been shown to beat fixed precision, but each method adapts
along essentially one axis: precision rising over rounds (time), or allocated by
client weight/bandwidth, or set per layer. A policy that co-optimizes all three
simultaneously under a total communication budget — the "multi-adaptive" schedule the
FedMAQ name promises — has no established formulation or algorithm.

## State of the art

DAdaQuant is doubly adaptive (time + client) but not layer-wise; AdaGQ adapts
resolution over rounds and bit-width per client for round-time equalization;
LAQ-HC sets per-client precision from data quality and bandwidth; FedDT uses a
layer-wise ternary threshold but a single global schedule. None searches the joint
(round × client × layer) precision space, and none couples that search to a
distillation signal.

## FedMAQ's angle

FedMAQ proposes precision that varies across all three axes at once, driven by a
unified signal (progress, heterogeneity, bandwidth) and coupled to distillation, which
is exactly the scheduling problem left open here.

## Sources

- Motivating finding: [Adaptive bit-width beats uniform quantization](/findings/adaptive-quantization-beats-uniform.md).
- Papers: [DAdaQuant](/papers/honig-2022-dadaquant.md),
  [AdaGQ](/papers/liu-2023-adagq.md), [LAQ-HC](/papers/cui-2026-laq-hc.md).

# Related

- [Adaptive bit-width](/concepts/adaptive-bit-width.md)
- [Heterogeneity-aware quantization schedules](/gaps/heterogeneity-aware-quantization.md)
- [DAdaQuant](/methods/dadaquant.md), [AdaGQ](/methods/adagq.md)
