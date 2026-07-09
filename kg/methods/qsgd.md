---
type: Method
title: "QSGD"
description: "Unbiased stochastic gradient quantization with a tunable number of levels and lossless encoding, giving a provable bits-vs-variance trade-off."
tags: [quantization, baseline]
introduced_by: /papers/alistarh-2017-qsgd.md
timestamp: 2026-07-09T12:00:00Z
---

# QSGD

Quantized SGD: the canonical provably-convergent, unbiased, tunable gradient
quantizer and theoretical ancestor of federated quantization methods.

## 1. Mechanism

Each gradient coordinate is stochastically rounded to one of \(s\) levels between 0
and the vector's \(\ell_2\) norm, with probabilities set so the quantized value is an
*unbiased* estimator: \( Q_s(v_i) = \lVert v\rVert_2\,\mathrm{sgn}(v_i)\,\xi_i \),
\(\mathbb{E}[\xi_i]=|v_i|/\lVert v\rVert_2\). The message is the scalar norm plus a
sign and small-integer level per coordinate, then Elias-coded losslessly.
Unbiasedness with a bounded second moment keeps standard SGD convergence, with the
variance term inflated by an explicit quantization factor.

## 2. Key hyperparameters

- \(s\) — number of quantization levels; more levels lower variance at more bits
  (\(s=1\) approaches 1-bit-like compression).
- Variance bound: \( \mathbb{E}\lVert Q_s(v)-v\rVert_2^2 \le \min(n/s^2,\sqrt n/s)\lVert v\rVert_2^2 \).

## 3. Communication & computation profile

Compresses *gradients* in data-parallel SGD (maps onto FedSGD, not directly onto
multi-local-step FedAvg). Transmitting the full-precision norm and variable-length
codes add overhead; small \(s\) inflates variance and can slow wall-clock
convergence. Analysis is IID-leaning.

## 4. Papers

- Introduces: [QSGD](/papers/alistarh-2017-qsgd.md).
- Adapted for FL parameter deltas by [FedPAQ](/papers/reisizadeh-2020-fedpaq.md) and
  [DAdaQuant](/papers/honig-2022-dadaquant.md); used by
  [AdaGQ](/papers/liu-2023-adagq.md).

## 5. FedMAQ relevance

QSGD's explicit variance bound as a function of \(s\) is exactly the lever FedMAQ
makes *multi-adaptive* — varying \(s\) across clients, layers, and rounds. It also
supplies the essential lesson: unbiasedness is what preserves convergence, so an
adaptive quantizer must remain unbiased (or correct for bias) as it varies bit-widths.

# Related

- [Quantization](/concepts/quantization.md)
- [Adaptive bit-width](/concepts/adaptive-bit-width.md)
- [signSGD](/methods/signsgd.md), [FedPAQ](/methods/fedpaq.md),
  [DAdaQuant](/methods/dadaquant.md)
- [QSGD paper](/papers/alistarh-2017-qsgd.md)
