---
type: Method
title: "LAQ-HC"
description: "Lightweight adaptive quantization for heterogeneous clients: fits a per-round tanh impact function to pick each client's bit-width by data quality and bandwidth."
tags: [quantization, adaptive]
introduced_by: /papers/cui-2026-laq-hc.md
timestamp: 2026-07-09T12:00:00Z
---

# LAQ-HC

Lightweight Adaptive Quantization for Heterogeneous Clients: chooses per-client
precision from a cheaply fitted impact model, giving weaker-link clients more
compression to mitigate stragglers.

## 1. Mechanism

The impact of a quantization level on loss reduction follows a hyperbolic-tangent
function \( IQ(t,\ell) = a\tfrac{e^{b2^\ell}-1}{e^{c2^\ell}+1}+d \) that is consistent
across clients and nearly stable between adjacent rounds, so it is fitted once per
round from the previous round's data (avoiding per-client per-level training). Each
client gets a quality score \(q_{i,\ell}^t = D_i F_i(w^t)\,IQ(t{-}1,\ell)\); a *flag*
function balances quality and bandwidth per unit level to select the level (or, under
a bandwidth budget, solves a grouped-knapsack selection). A cold-start round fits the
four tanh parameters on one client across four levels.

## 2. Key hyperparameters

- \(\alpha\) — quality-vs-bandwidth trade-off in the flag function.
- Candidate level set \(\ell(i)\); uniform affine quantization range \(\mathcal R\).
- Bandwidth budget (constrained mode).

## 3. Communication & computation profile

Uses uniform affine quantization of gradients; the tanh fit makes level selection
nearly free of extra training. Proven \(O(1/\sqrt T)\) non-convex rate under unbiased
quantization. Discards low-quality clients under a budget, which can slow early
convergence; dropout above ~20% degrades results.

## 4. Papers

- Introduces: [LAQ-HC](/papers/cui-2026-laq-hc.md).
- Sibling adaptive quantizers: [DAdaQuant](/papers/honig-2022-dadaquant.md),
  [AdaGQ](/papers/liu-2023-adagq.md).

## 5. FedMAQ relevance

LAQ-HC is a recent SOTA adaptive-quantization baseline. The paper's own suggested
future work of adding knowledge distillation is the part of its program
[FedMAQ](/methods/fedmaq.md) pursues; its further suggestion of combining
quantization with sparsification or low-rank compression is **not** — FedMAQ's
scope is quantization only, excluding architectural compression. LAQ-HC's
lightweight impact estimation remains a candidate alternative to FedMAQ's
gradient-norm signal, outside its current formulation set.

# Related

- [Quantization](/concepts/quantization.md)
- [Adaptive bit-width](/concepts/adaptive-bit-width.md)
- [Non-IID data and client drift](/concepts/non-iid-heterogeneity.md)
- [DAdaQuant](/methods/dadaquant.md), [AdaGQ](/methods/adagq.md)
- [LAQ-HC paper](/papers/cui-2026-laq-hc.md)
