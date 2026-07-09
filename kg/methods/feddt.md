---
type: Method
title: "FedDT"
description: "Two-level compression: personalized teacher-to-shared-student distillation plus trained ternary quantization of the student model for aggregation."
tags: [joint-q-kd, quantization, distillation, sota, baseline]
introduced_by: /papers/he-2025-feddt.md
timestamp: 2026-07-09T12:00:00Z
---

# FedDT

Knowledge Distillation and Ternary compression: masks model heterogeneity via a shared
student and compresses communication to ternary weights.

## 1. Mechanism

Each client holds a personalized heterogeneous *teacher* \(T_i\) (pre-trained on local
data) and a copy of the shared global *student* \(S\). Local *adaptive distillation*
transfers teacher knowledge into the student with task, distillation (KL), and hidden
(MSE) losses each weighted inversely by the teacher/student task-loss sum — the shared
student masks architectural differences. The distilled student is then compressed by
*Trained Ternary Quantization* into \(\{-1,0,+1\}\) with a layer-wise adaptive threshold
\(\Delta=\tfrac{T_k}{d^2}\sum|\Theta_i^s|\) and learned positive/negative scales
\(w_p,w_n\). The server aggregates quantized students by weighted average and
re-quantizes (\(\Delta_s=0.05\max|\Theta_r|\)) before broadcasting.

## 2. Key hyperparameters

- Ternary threshold schedule \(T_k\); server threshold factor (0.05).
- Adaptive-distillation temperature and loss weighting.
- Student architecture (sets the residual communication size).

## 3. Communication & computation profile

~78% communication reduction with reported accuracy gains; \(O(1/NR)\) convergence under
strong-convexity assumptions (unbiasedness assumes weights uniform in \([-1,1]\)).
Clients must host and train both teacher and student — a cost for weak devices.

## 4. Papers

- Introduces: [FedDT](/papers/he-2025-feddt.md).
- Distillation lineage with [FedKD](/papers/wu-2022-fedkd.md); sibling SOTA to
  [DynFed](/papers/he-2025-dynfed.md).

## 5. FedMAQ relevance

FedDT is a SOTA joint KD+quantization baseline. Its adaptive distillation loss and
layer-wise ternary scheme are integrable, and its single (ternary) precision is exactly
what FedMAQ generalizes to *multi-adaptive* mixed precision per layer/client. Its
server re-quantization step informs FedMAQ's aggregator design.

# Related

- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Quantization](/concepts/quantization.md)
- [Model compression](/concepts/model-compression.md)
- [DynFed](/methods/dynfed.md), [FedKD](/methods/fedkd.md)
- [FedDT paper](/papers/he-2025-feddt.md)
