---
type: Finding
title: "Quantization and knowledge distillation are complementary; jointly they exceed either alone"
description: "Quantization compresses the update while distillation both shrinks payload independently of model size and recovers the accuracy aggressive quantization sacrifices, so combined schemes dominate single-axis compression."
tags: [joint-q-kd, quantization, distillation, communication-efficiency]
timestamp: 2026-07-09T12:00:00Z
---

# Quantization and distillation are complementary; jointly they exceed either alone

## Scope

Covers methods that combine precision reduction with knowledge transfer against
single-mechanism baselines, on the communication-vs-accuracy frontier.

## Claim

The two dominant compression families act on different terms and do not cannibalize
each other. Quantization reduces bits per parameter but injects error that lowers
accuracy; distillation makes payload independent of model size and, used as a
regularizer, recovers accuracy lost to quantization noise. Stacking them therefore
pushes the accuracy-vs-bits frontier past either alone — several corpus methods report
large compression ratios at near-baseline accuracy by combining ternary/adaptive
quantization with a distillation loss. This complementarity is the technical premise
of the FedMAQ thesis.

## Evidence

| Paper | Key result | Link |
| --- | --- | --- |
| AQFedAvg + FKD | 4x-16x compression within ~2% accuracy by pairing adaptive quantization with federated distillation | [/papers/qu-2020-quantization-kd.md](/papers/qu-2020-quantization-kd.md) |
| FedDT | ~78% communication reduction with accuracy gains via distillation into a ternary-quantized student | [/papers/he-2025-feddt.md](/papers/he-2025-feddt.md) |
| DynFed | Adaptive bit-widths fused by multi-teacher distillation beat fixed-precision or KD-only baselines | [/papers/he-2025-dynfed.md](/papers/he-2025-dynfed.md) |
| CFD | Soft-label quantization plus dual server distillation compresses beyond either component | [/papers/sattler-2022-cfd.md](/papers/sattler-2022-cfd.md) |
| AdaDQ-KD | Dithering quantization plus feature-level KD trades privacy, bits, and accuracy jointly | [/papers/wang-2026-adadq-kd.md](/papers/wang-2026-adadq-kd.md) |

## Open gaps

- Joint methods remain few and mostly pair a *single* precision with a *single*
  teacher: [/gaps/multi-adaptive-q-kd-scarcity.md](/gaps/multi-adaptive-q-kd-scarcity.md).

# Related

- [Quantization](/concepts/quantization.md)
- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Model compression](/concepts/model-compression.md)
- [CFD](/methods/cfd.md), [DynFed](/methods/dynfed.md), [FedDT](/methods/feddt.md),
  [AdaDQ-KD](/methods/adadq-kd.md), [AQFedAvg + FKD](/methods/quantized-kd.md)
