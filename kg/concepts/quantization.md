---
type: Concept
title: "Quantization"
description: "Reducing the numerical precision of communicated model updates or gradients to fewer bits, trading representation error for a smaller payload."
tags: [quantization, communication-efficiency, compression]
timestamp: 2026-07-09T12:00:00Z
---

# Quantization

## 1. Definition

Quantization maps a high-precision tensor (a gradient, weight, or model delta) to a
low-precision representation drawn from a small set of levels. For a vector \(v\), a
stochastic quantizer \(Q(v)\) rounds each coordinate to one of \(s\) levels such that
\(\mathbb E[Q(v)] = v\) (unbiased) while the encoding costs on the order of
\(\log_2 s\) bits per coordinate plus a shared scale. The central trade-off is
variance versus bits: fewer levels shrink the payload but inflate
\(\mathbb E\lVert Q(v)-v\rVert^2\), which slows convergence.

## 2. Why it matters for FedMAQ

Uplink communication is the dominant bottleneck in federated learning over edge and
IoT links. Quantization is the primary lever [FedMAQ](/methods/fedmaq.md) pulls to
cut per-round payload, and the "multi-adaptive" in the thesis name refers to
allocating one client-level, per-round scalar bit-width from *multiple adaptive
signals* (resource, training-state, data-richness) rather than a single fixed
bit-width — not to resolving precision per layer. The open question FedMAQ
inherits is how quantization noise interacts with non-IID drift and with the
distillation signal.

## 3. Variants & dimensions

- **Biased vs. unbiased** — sign compression (biased) vs. QSGD-style unbiased
  stochastic rounding.
- **Fixed vs. adaptive** — a static bit-width vs. precision that varies by round,
  client, or layer (see [Adaptive bit-width](/concepts/adaptive-bit-width.md)).
- **Uniform vs. non-uniform levels** — evenly spaced levels vs. learned or
  logarithmic spacing (e.g. ternary \(\{-1,0,+1\}\)).
- **Object quantized** — gradients (QSGD, signSGD), model deltas (FedPAQ), full
  weights (ternary), or soft labels (CFD).

## 4. Methods & papers

- Methods: [QSGD](/methods/qsgd.md), [signSGD](/methods/signsgd.md),
  [FedPAQ](/methods/fedpaq.md), [DAdaQuant](/methods/dadaquant.md),
  [AdaGQ](/methods/adagq.md), [LAQ-HC](/methods/laq-hc.md),
  [CFD](/methods/cfd.md), [DynFed](/methods/dynfed.md), [FedDT](/methods/feddt.md),
  [AdaDQ-KD](/methods/adadq-kd.md), [AQFedAvg + FKD](/methods/quantized-kd.md).
- Foundational: [QSGD](/papers/alistarh-2017-qsgd.md),
  [signSGD](/papers/bernstein-2018-signsgd.md),
  [FedPAQ](/papers/reisizadeh-2020-fedpaq.md).
- Survey: [Le 2024 compression survey](/papers/le-2024-compression-survey.md).

# Related

- [Adaptive bit-width](/concepts/adaptive-bit-width.md)
- [Communication efficiency](/concepts/communication-efficiency.md)
- [Model compression](/concepts/model-compression.md)
- [Knowledge distillation](/concepts/knowledge-distillation.md)
