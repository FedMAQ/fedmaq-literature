---
type: Method
title: "AdaDQ-KD"
description: "Adaptive dithering quantization whose error doubles as DP noise, plus feature-level KD, jointly trading privacy, communication, and accuracy under stragglers."
tags: [joint-q-kd, quantization, distillation, privacy, sota, baseline]
introduced_by: /papers/wang-2026-adadq-kd.md
timestamp: 2026-07-09T12:00:00Z
---

# AdaDQ-KD

Adaptive Dithering Quantization with Knowledge Distillation for privacy-preserving FL:
one quantizer delivers compression *and* differential privacy, with KD recovering
accuracy.

## 1. Mechanism

- *Dithering quantization*: dither \(u_i\sim U(-\Delta_i/2,\Delta_i/2)\) is added before
  quantization and subtracted on decode, so the residual error is provably equivalent to
  Gaussian noise \(\mathcal N(0,\sigma^2 n_{i,t}^2)\) — a DP guarantee with no separate
  noise injection. Step size \(\Delta_i=2n_i\sigma\sqrt{v_i}\) sets precision.
- *Adaptive precision*: clients are ranked by expected local delay; the top \(K=\lfloor kP\rfloor\)
  stragglers have their precision coefficient \(n_i\) reduced (coarser quantization) until
  their delay drops below the \((K{+}1)\)-th client's — an NP-hard scheduling heuristic.
- *Feature-level KD*: local loss \( \mathcal L=\mathcal L_{CE}+\lambda\mathcal L_{KD} \)
  against a pre-trained global teacher regularizes against DP-noise/quantization damage.

## 2. Key hyperparameters

- \(\sigma\) — target DP noise std; \(n_i\) — per-client precision coefficient.
- \(k\) — straggler fraction; search length \(h\).
- \(\lambda\) — KD weight.

## 3. Communication & computation profile

Straggler-aware precision minimizes round time; DQ folds privacy into compression, but
per-round privacy budget fluctuates, complicating accounting. KD adds teacher forward
passes; effectiveness hinges on teacher quality.

## 4. Papers

- Introduces: [AdaDQ-KD](/papers/wang-2026-adadq-kd.md).
- Single-teacher; sibling SOTA to [DynFed](/papers/he-2025-dynfed.md) and
  [FedDT](/papers/he-2025-feddt.md).

## 5. FedMAQ relevance

A direct SOTA baseline that already unites adaptive quantization + KD, plus a privacy
angle. Its delay-driven precision and dithering-as-DP mechanism are integrable into
FedMAQ's multi-adaptive quantizer; FedMAQ's stated improvement over it is multi-teacher /
ensemble distillation instead of a single global teacher.

# Related

- [Adaptive bit-width](/concepts/adaptive-bit-width.md)
- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Privacy in federated learning](/concepts/privacy-in-fl.md)
- [DynFed](/methods/dynfed.md), [FedDT](/methods/feddt.md)
- [AdaDQ-KD paper](/papers/wang-2026-adadq-kd.md)
