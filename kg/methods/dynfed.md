---
type: Method
title: "DynFed"
description: "Quantization-aware KD: resource- and gradient-adaptive per-client bit-widths, fused server-side by active multi-teacher distillation over heterogeneous precisions."
tags: [joint-q-kd, quantization, distillation, adaptive, sota]
introduced_by: /papers/he-2025-dynfed.md
timestamp: 2026-07-09T12:00:00Z
---

# DynFed

Adaptive FL via Quantization-Aware Knowledge Distillation: allocates bit-widths by
client capacity and training dynamics, then aggregates heterogeneously-quantized
models by distillation rather than averaging.

## 1. Mechanism

- *Dynamic bit-width allocation*: initial level \(q_k=\min(c_{\max},\lfloor c_k/c_p\rfloor)\)
  from memory capacity; then per-round adjustment by gradient norm,
  \( b_i^{(t)} = b_i^{(t-1)} + \eta(\tfrac{|\nabla F_i|}{\max|\nabla F_i|} - \tfrac{b_i^{(t-1)}}{B_{\max}}) \)
  (bigger gradients → more bits). Uses a QSGD-style stochastic quantizer.
- *Active multi-teacher distillation*: rather than average heterogeneous quantized
  models, the server scores client models per public sample by MCMC-entropy uncertainty
  and a bit-width/confidence *comprehensive score* with a diversity penalty, selects
  teachers, and trains the global model on \( L=\gamma L_{distill}+(1-\gamma)L_{CE} \).

## 2. Key hyperparameters

- \(\eta\) (bit-width step), \(B_{\max}\), \(c_p\).
- Teacher-selection weights \(\alpha,\beta\), diversity \(\lambda\), MCMC samples \(M\).
- \(\gamma\) — distillation/CE mix.

## 3. Communication & computation profile

Bits tied to capacity + gradient magnitude; distillation aggregation tolerates mixed
precision that averaging cannot. Needs a small public unlabeled set (~200 samples);
server bears MCMC + multi-model inference; clients still send full (quantized) gradient
vectors.

## 4. Papers

- Introduces: [DynFed](/papers/he-2025-dynfed.md).
- Reported to beat [FedPAQ](/papers/reisizadeh-2020-fedpaq.md),
  [DAdaQuant](/papers/honig-2022-dadaquant.md), and
  [FedKD](/papers/wu-2022-fedkd.md).

## 5. FedMAQ relevance

DynFed is excluded from the primary benchmark grid (no public codebase) but is the
nearest neighbor to [FedMAQ](/methods/fedmaq.md)'s thesis: adaptive quantization
*plus* KD aggregation under heterogeneity. FedMAQ reproduces its core
mechanism — memory-capped, gradient-norm-adaptive bit-width refined by
server-side distillation, minus data-richness — as a DynFed-style ablation
reference arm; FedMAQ's added data-richness signal and combination-logic study
isolate what it contributes over this design. FedMAQ does **not** adopt DynFed's
uncertainty-based active teacher selection — its ensemble averages all
participating clients' soft labels with equal weight.

# Related

- [Adaptive bit-width](/concepts/adaptive-bit-width.md)
- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Non-IID data and client drift](/concepts/non-iid-heterogeneity.md)
- [FedDT](/methods/feddt.md), [AdaDQ-KD](/methods/adadq-kd.md),
  [DAdaQuant](/methods/dadaquant.md)
- [DynFed paper](/papers/he-2025-dynfed.md)
