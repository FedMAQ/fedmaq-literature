---
type: Method
title: "AdaGQ"
description: "Adaptive gradient quantization that adjusts resolution per round by loss-decrease rate and assigns per-client bit-widths to equalize round time under heterogeneity."
tags: [quantization, adaptive]
introduced_by: /papers/liu-2023-adagq.md
timestamp: 2026-07-09T12:00:00Z
---

# AdaGQ

Adaptive Gradient Quantization for heterogeneous edge devices, optimizing *wall-clock*
training time rather than round count.

## 1. Mechanism

- *Adaptive average level*: the server updates the round's average quantization level
  \(\bar s_k\) by online gradient descent on a loss-decrease-rate objective
  \(R_k=(L_{k-1}-L_k)/T_{k-1,k}\), using the sign of the derivative estimated from two
  candidate levels, then calibrates by the change in gradient norm
  (\(+\lambda_g(\log_2\lVert g_k\rVert-\log_2\lVert g_{k-1}\rVert)\)).
- *Heterogeneous assignment*: per-client bits \(b_{i,k}\) are set so all clients'
  expected round times match — the slow client is aligned with the fast — subject to
  \(\tfrac1n\sum_i s_{i,k}=\bar s_k\). Uses the QSGD quantizer underneath.

## 2. Key hyperparameters

- \(\lambda_1,\lambda_2\) — bit decrease/increase steps in the online rule.
- \(\lambda_g\) — gradient-norm calibration weight.
- \(s_0\) — initial level.

## 3. Communication & computation profile

Clients do two forward passes per round (current and one-bit-lower level) to estimate
the derivative sign — light overhead. Synchronous; the server solves a per-round
linear system across \(n\) clients. Explicitly models bits-to-time to mitigate
stragglers, unlike DAdaQuant which keys on data size.

## 4. Papers

- Introduces: [AdaGQ](/papers/liu-2023-adagq.md).
- Uses the [QSGD](/papers/alistarh-2017-qsgd.md) quantizer; sibling adaptive method
  to [DAdaQuant](/papers/honig-2022-dadaquant.md) and
  [LAQ-HC](/papers/cui-2026-laq-hc.md).

## 5. FedMAQ relevance

AdaGQ is a strong adaptive-quantization baseline and a direct source of reusable
machinery: the loss-decrease-rate adaptation rule and the bits-to-time client
assignment are candidate modules for FedMAQ's multi-adaptive scheduler, which can
unify them with a distillation objective and additional compression axes.

# Related

- [Quantization](/concepts/quantization.md)
- [Adaptive bit-width](/concepts/adaptive-bit-width.md)
- [Communication efficiency](/concepts/communication-efficiency.md)
- [DAdaQuant](/methods/dadaquant.md), [LAQ-HC](/methods/laq-hc.md),
  [QSGD](/methods/qsgd.md)
- [AdaGQ paper](/papers/liu-2023-adagq.md)
