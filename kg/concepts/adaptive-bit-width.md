---
type: Concept
title: "Adaptive bit-width"
description: "Allocating quantization precision non-uniformly — across rounds, clients, or layers — in response to training progress, data quality, or bandwidth."
tags: [quantization, adaptive, communication-efficiency, heterogeneity]
timestamp: 2026-07-09T12:00:00Z
---

# Adaptive bit-width

## 1. Definition

Adaptive bit-width replaces a single fixed precision \(b\) with a schedule
\(b_{i,\ell,t}\) that can depend on the client \(i\), layer \(\ell\), and round \(t\).
The allocation is driven by a signal — loss-decrease rate, gradient variance,
bandwidth, or data quality — so that bits are spent where they most reduce
error-per-round or round time. It generalizes [Quantization](/concepts/quantization.md)
from a scalar knob to a policy.

## 2. Why it matters for FedMAQ

This is the core mechanism of the thesis: "multi-adaptive quantization" means
precision adapts along multiple axes simultaneously. A well-designed schedule spends
coarse bits early (when gradients are noisy anyway) and finer bits late, and gives
low-bandwidth stragglers coarser precision to equalize round time. FedMAQ's
contribution is to unify these axes and couple the schedule to a distillation signal.

## 3. Variants & dimensions

- **Temporal** — precision rises over rounds as training converges (DAdaQuant's
  "time-adaptive"; AdaGQ's loss-decrease trigger).
- **Client-wise** — bits allocated by client weight, bandwidth, or straggler delay
  (DAdaQuant "client-adaptive"; AdaGQ; LAQ-HC; AdaDQ-KD).
- **Layer-wise** — different precision per layer (ternary thresholds in FedDT).
- **Signal driving adaptation** — loss slope, gradient norm/variance, bandwidth,
  data quality, or expected local delay.

## 4. Methods & papers

- Methods: [DAdaQuant](/methods/dadaquant.md), [AdaGQ](/methods/adagq.md),
  [LAQ-HC](/methods/laq-hc.md), [DynFed](/methods/dynfed.md),
  [AdaDQ-KD](/methods/adadq-kd.md), [QSGD](/methods/qsgd.md) (tunable levels).
- Papers: [DAdaQuant](/papers/honig-2022-dadaquant.md),
  [AdaGQ](/papers/liu-2023-adagq.md), [LAQ-HC](/papers/cui-2026-laq-hc.md),
  [DynFed](/papers/he-2025-dynfed.md), [AdaDQ-KD](/papers/wang-2026-adadq-kd.md).

# Related

- [Quantization](/concepts/quantization.md)
- [Communication efficiency](/concepts/communication-efficiency.md)
- [Non-IID heterogeneity](/concepts/non-iid-heterogeneity.md)
