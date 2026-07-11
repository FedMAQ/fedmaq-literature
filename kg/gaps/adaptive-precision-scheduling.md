---
type: Gap
title: "How multiple adaptive signals should be combined into a client-level precision budget is under-explored"
description: "Adaptive-quantization methods key precision on a single signal; how resource, training-state, and data-richness signals should be combined into one client-level bit-width under a communication budget is open."
tags: [quantization, adaptive, signal-combination, communication-efficiency]
timestamp: 2026-07-10T00:00:00Z
---

# How multiple adaptive signals should be combined into a client-level precision budget is under-explored

## Problem

Adaptive quantization beats fixed precision, but each established method keys its
bit-width on essentially one signal: precision rising over rounds (time), or
allocated by client weight/bandwidth, or set per layer. How *multiple* complementary
signals -- resource (memory), training-state (gradient norm), and data-richness
(dataset size) -- should be *combined* into a single client-level bit-width under a
communication budget has no established formulation. The open question is the
combination *logic* (additive, multiplicative, or a modulated interaction), not a new
precision axis: the allocation targeted here is one scalar per client per round, with
round-to-round variation an implicit consequence of the evolving gradient-norm signal
rather than an explicitly scheduled time axis, and without per-layer resolution.

## State of the art

DAdaQuant is doubly adaptive (time + client) but fuses those signals through a fixed
doubling heuristic rather than a studied interaction; AdaGQ adapts resolution over
rounds and bit-width per client for round-time equalization; LAQ-HC sets per-client
precision from data quality and bandwidth; DynFed fuses a memory cap with a recursive
gradient-norm tracker but omits data richness. None studies how resource,
training-state, and data-richness signals should jointly determine a client's
precision, and none couples that combination to a distillation signal.

## FedMAQ's angle

[FedMAQ](/methods/fedmaq.md) targets exactly this combination question for the
client-level scalar: it evaluates a resource-only control plus linear, multiplicative,
gradient-primary data-modulated, and threshold-based rules for fusing the
gradient-norm and data-richness signals within a hard memory ceiling. The candidates'
internal weights are fixed by one shared convention rather than tuned per formula, so
the comparison isolates combination *structure*. The winner is selected by a
pre-registered rule — least cumulative communication to a per-dataset/skew target
accuracy (90% of uncompressed FedAvg), subject to an accuracy-floor disqualification —
on CIFAR-10 across both skew regimes, then generalization-tested on the remaining
datasets/skews via the main benchmark grid. It does *not* resolve the layer-wise or
explicit-round scheduling sub-problems, which remain open.

## Sources

- Motivating finding: [Adaptive bit-width beats uniform quantization](/findings/adaptive-quantization-beats-uniform.md).
- Papers: [DAdaQuant](/papers/honig-2022-dadaquant.md),
  [AdaGQ](/papers/liu-2023-adagq.md), [LAQ-HC](/papers/cui-2026-laq-hc.md),
  [DynFed](/papers/he-2025-dynfed.md).

# Related

- [Adaptive bit-width](/concepts/adaptive-bit-width.md)
- [Heterogeneity-aware quantization schedules](/gaps/heterogeneity-aware-quantization.md)
- [DAdaQuant](/methods/dadaquant.md), [AdaGQ](/methods/adagq.md)
