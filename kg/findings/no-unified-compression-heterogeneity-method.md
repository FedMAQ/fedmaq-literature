---
type: Finding
title: "No existing method jointly optimizes multi-axis compression and heterogeneity robustness"
description: "Surveys and the method corpus converge on a gap: quantization, distillation, and drift correction are each strong alone, but no single method unifies adaptive multi-axis compression with heterogeneity robustness — the FedMAQ niche."
tags: [survey, gap, communication-efficiency, heterogeneity, joint-q-kd]
timestamp: 2026-07-09T12:00:00Z
---

# No existing method jointly optimizes multi-axis compression and heterogeneity robustness

## Scope

Synthesizes the survey layer and the joint-Q+KD methods to characterize what is
unaddressed across the whole corpus, framing the thesis contribution.

## Claim

Each capability exists in isolation — adaptive quantization (DAdaQuant, AdaGQ),
distillation-based compression (FedDF, FedKD), and drift correction (SCAFFOLD,
FedDyn) — and a handful of methods pair two of them. But no corpus method unifies
*multi-adaptive* precision (across round, client, and layer simultaneously) with
*multi-teacher* distillation under explicit heterogeneity robustness; existing joint
methods fix a single precision or a single teacher. Surveys across compression, KD,
non-IID, and edge FL independently note this missing unification. That unaddressed
intersection is precisely the FedMAQ thesis niche.

## Evidence

| Paper | Key result | Link |
| --- | --- | --- |
| Le 2024 compression survey | No single compression scheme jointly handles heterogeneity robustness | [/papers/le-2024-compression-survey.md](/papers/le-2024-compression-survey.md) |
| Jimenez 2024 non-IID survey | Heterogeneity remedies surveyed separately from compression | [/papers/jimenez-2024-non-iid-survey.md](/papers/jimenez-2024-non-iid-survey.md) |
| Qin 2025 / Salman 2025 KD surveys | Federated KD reviewed without adaptive-quantization integration | [/papers/qin-2025-kd-survey.md](/papers/qin-2025-kd-survey.md) |
| Cajas-Ordonez 2025 edge survey | Edge FL compression surveyed without a unified adaptive-Q + KD method | [/papers/cajas-ordonez-2025-edge-computing-survey.md](/papers/cajas-ordonez-2025-edge-computing-survey.md) |
| AdaDQ-KD / FedDT / DynFed | State of the art pairs Q + KD but with single precision or single teacher | [/papers/wang-2026-adadq-kd.md](/papers/wang-2026-adadq-kd.md) |

## Open gaps

- The scarcity of multi-adaptive Q + KD methods:
  [/gaps/multi-adaptive-q-kd-scarcity.md](/gaps/multi-adaptive-q-kd-scarcity.md).
- Heterogeneity-aware quantization schedules:
  [/gaps/heterogeneity-aware-quantization.md](/gaps/heterogeneity-aware-quantization.md).

# Related

- [Communication efficiency](/concepts/communication-efficiency.md)
- [Quantization](/concepts/quantization.md),
  [Knowledge distillation](/concepts/knowledge-distillation.md),
  [Non-IID heterogeneity](/concepts/non-iid-heterogeneity.md)
- [Quantization and KD are complementary](/findings/quantization-kd-complementary.md)
