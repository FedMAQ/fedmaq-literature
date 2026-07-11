---
type: Method
title: "DAdaQuant"
description: "Doubly-adaptive quantization: increases precision over rounds (time-adaptive) and allocates bits across clients by aggregation weight (client-adaptive)."
tags: [quantization, adaptive, baseline]
introduced_by: /papers/honig-2022-dadaquant.md
timestamp: 2026-07-09T12:00:00Z
---

# DAdaQuant

A state-of-the-art adaptive quantizer that varies precision along *two* axes on top
of a Federated QSGD engine.

## 1. Mechanism

- *Time-adaptive*: the quantization level \(q_t\) is increased (doubled, capped at
  \(q_{\max}\)) when a moving average of the estimated global loss plateaus — coarse
  early, fine late.
- *Client-adaptive*: within a round, bits are allocated to minimize total
  communication \(\sum_k q_k\) while holding the aggregate variance equal to a static
  quantizer's; the Lagrangian optimum is \(q_i \propto w_i^{2/3}\) in the client
  aggregation weights \(w_i\).
- *Federated QSGD*: unbiased stochastic fixed-point quantization of the model delta,
  with 0-run-length + Elias \(\omega\) coding.

## 2. Key hyperparameters

- \(q_{\min}, q_{\max}\) — precision bounds.
- \(\phi\) (plateau lookback), \(\psi=0.9\) (loss smoothing).
- Client weights \(w_i = |D_i|/\sum_j|D_j|\) drive the \(w_i^{2/3}\) allocation.

## 3. Communication & computation profile

Uplink-only compression (downlink assumed cheaper); reported 1.2–2.8x over static
Federated QSGD. ~1% overhead for the extra loss forward pass. Client-adaptivity keys
on data size, not compute; the optimality theorem assumes uniform parameter
distribution.

## 4. Papers

- Introduces: [DAdaQuant](/papers/honig-2022-dadaquant.md).
- Builds on [QSGD](/papers/alistarh-2017-qsgd.md) and inherits convergence from
  [FedPAQ](/papers/reisizadeh-2020-fedpaq.md).

## 5. FedMAQ relevance

DAdaQuant is the closest single-technique precursor to [FedMAQ](/methods/fedmaq.md)'s
*multi-adaptive* quantization and a primary SOTA baseline. FedMAQ does not extend
DAdaQuant's two axes directly — it replaces time-adaptivity with a per-round
gradient-norm (training-state) signal and adds a data-richness signal, combining
both at the client level within a hard memory ceiling, then pairs this with
server-side ensemble distillation — the paper itself suggests driving the
time-adaptive schedule by a distillation loss. Its closed-form \(w_i^{2/3}\) rule
remains a candidate building block outside FedMAQ's current scope.

# Related

- [Quantization](/concepts/quantization.md)
- [Adaptive bit-width](/concepts/adaptive-bit-width.md)
- [QSGD](/methods/qsgd.md), [FedPAQ](/methods/fedpaq.md),
  [AdaGQ](/methods/adagq.md), [LAQ-HC](/methods/laq-hc.md)
- [DAdaQuant paper](/papers/honig-2022-dadaquant.md)
