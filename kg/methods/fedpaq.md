---
type: Method
title: "FedPAQ"
description: "Federated learning with periodic averaging, partial participation, and unbiased quantization of model deltas, with convergence guarantees."
tags: [quantization, fl-core, baseline]
introduced_by: /papers/reisizadeh-2020-fedpaq.md
timestamp: 2026-07-09T12:00:00Z
---

# FedPAQ

The first FL method to combine local-SGD periodic averaging, partial device
participation, and quantized uploads with a rigorous convergence analysis.

## 1. Mechanism

Three complementary modules: (1) *periodic averaging* — each device runs \(\tau\)
local SGD steps before syncing, cutting rounds from \(T\) to \(T/\tau\);
(2) *partial participation* — only a random subset of \(r\le n\) devices is active per
round; (3) *quantized upload* — each device quantizes its model delta
\(\Delta_k^{(i)} = x_{k,\tau}^{(i)} - x_k\) with an unbiased low-precision quantizer
\(Q_{\mathrm{LP}}\) (QSGD-style, level \(s\)) before sending. The server aggregates
\( x_{k+1} = x_k + \tfrac1r\sum_{i\in\mathcal S_k} Q(\Delta_k^{(i)}) \).

## 2. Key hyperparameters

- \(\tau\) — local steps per round; \(\tau = o(\sqrt T)\) retains the optimal rate.
- \(s\) — quantizer level (fixed, single-precision).
- \(r\) — participating devices per round.

## 3. Communication & computation profile

Quantizes the *model delta* (not raw gradients), so it applies to multi-local-step
FedAvg — the key advance over QSGD/signSGD. Uplink-focused. Guarantees:
\(O(1/T)\) strongly convex, \(O(1/\sqrt T)\) non-convex, but the analysis assumes IID
data and an unbiased, variance-bounded quantizer.

## 4. Papers

- Introduces: [FedPAQ](/papers/reisizadeh-2020-fedpaq.md).
- Uses the [QSGD](/papers/alistarh-2017-qsgd.md) quantizer; extended adaptively by
  [DAdaQuant](/papers/honig-2022-dadaquant.md).

## 5. FedMAQ relevance

FedPAQ is a key baseline that already covers two of FedMAQ's three pillars (periodic
averaging + quantization). Its quantization is *fixed* single-precision; FedMAQ makes
it multi-adaptive (per client / layer / round) and adds knowledge distillation.
DAdaQuant's convergence result explicitly inherits from FedPAQ, so FedPAQ anchors the
theoretical chain FedMAQ extends.

# Related

- [Quantization](/concepts/quantization.md)
- [Communication efficiency](/concepts/communication-efficiency.md)
- [QSGD](/methods/qsgd.md), [DAdaQuant](/methods/dadaquant.md),
  [FedAvg](/methods/fedavg.md)
- [FedPAQ paper](/papers/reisizadeh-2020-fedpaq.md)
