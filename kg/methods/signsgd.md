---
type: Method
title: "signSGD"
description: "Extreme 1-bit gradient compression transmitting only the sign of each coordinate, with majority-vote aggregation in the distributed setting."
tags: [quantization, baseline]
introduced_by: /papers/bernstein-2018-signsgd.md
timestamp: 2026-07-09T12:00:00Z
---

# signSGD

The 1-bit endpoint of the quantization spectrum: keep only the sign bit of each
gradient coordinate.

## 1. Mechanism

Update \( x_{k+1} = x_k - \delta\,\mathrm{sign}(\tilde g_k) \) discards exponent and
mantissa. The momentum variant SIGNUM signs a momentum buffer and empirically matches
ADAM. In distributed training, each worker sends
\(\mathrm{sign}(\tilde g_m)\) and the server returns
\(\mathrm{sign}[\sum_m \mathrm{sign}(\tilde g_m)]\) — a majority vote, giving 1-bit
communication in *both* directions. Via a unimodality argument, majority vote attains
the same variance reduction as full-precision distributed SGD.

## 2. Key hyperparameters

- \(\delta\) — step size.
- \(\beta\) — momentum coefficient (SIGNUM).
- Regime of advantage governed by the \(\ell_1/\ell_2\) geometry of gradient, noise,
  and curvature — dense gradients favor sign compression.

## 3. Communication & computation profile

Maximal compression (1 bit/coordinate) but a *biased* compressor: convergence is
conditional on gradient geometry and can degrade when gradients are sparse or
noise-dominated. Magnitude is discarded entirely. Gradient-level and IID-leaning.

## 4. Papers

- Introduces: [signSGD](/papers/bernstein-2018-signsgd.md).
- Multi-level unbiased counterpart: [QSGD](/papers/alistarh-2017-qsgd.md).

## 5. FedMAQ relevance

signSGD marks the aggressive extreme FedMAQ's schedule moves toward. Its cautionary
lesson: at low bit-widths compressors become biased and convergence becomes
conditional, so FedMAQ must detect when a client's gradients stop satisfying such
conditions and back off. Majority-vote aggregation also shows server-side design can
recover the value of many low-fidelity updates.

# Related

- [Quantization](/concepts/quantization.md)
- [Adaptive bit-width](/concepts/adaptive-bit-width.md)
- [QSGD](/methods/qsgd.md)
- [signSGD paper](/papers/bernstein-2018-signsgd.md)
