---
type: Method
title: "FedDyn"
description: "Dynamic regularization: a per-device linear+quadratic penalty whose optima track the global model, aligning local and global stationary points."
tags: [regularization, drift-correction, heterogeneity, baseline]
introduced_by: /papers/acar-2021-feddyn.md
timestamp: 2026-07-09T12:00:00Z
---

# FedDyn

Federated Dynamic regularization: makes device objectives *consistent* with the
global objective so extra local computation does not stall short of the global
optimum.

## 1. Mechanism

Each round, device \(k\) minimizes
\( L_k(\theta) - \langle \nabla L_k(\theta_k^{t-1}), \theta\rangle
+ \tfrac{\alpha}{2}\lVert\theta - \theta^{t-1}\rVert^2 \). The *linear* term,
accumulating the device's past gradients, is what makes the regularizer dynamic:
in the limit the device optima coincide with the global optimum. The server keeps a
running gradient-state term \(h\) and corrects the averaged model,
\( \theta^t = \bar\theta^t - \tfrac{1}{\alpha}h^t \). Unlike FedProx's static
proximal term, the anchor moves toward global consistency.

## 2. Key hyperparameters

- \(\alpha\) — regularization strength (searched over \([10^{-3},10^{-1}]\)); the
  single key knob, trading local progress against consistency.

## 3. Communication & computation profile

Full-precision model per round (no compression); reduces round count, not bits per
round. Requires persistent per-device gradient state, assuming devices are revisited
or state is server-mirrored.

## 4. Papers

- Introduces: [FedDyn](/papers/acar-2021-feddyn.md).
- Family baselines: [FedProx](/papers/li-2020-fedprox.md),
  [SCAFFOLD](/papers/karimireddy-2020-scaffold.md).

## 5. FedMAQ relevance

A strong non-IID baseline whose drift correction is orthogonal to payload
compression, so FedMAQ can quantize FedDyn updates and test whether dynamic
regularization tolerates aggressive bit-width reduction. Its lesson — trading
compute for rounds is only "free" if local-global consistency holds — warns that
quantization noise must not re-introduce the drift FedDyn removes.

# Related

- [Non-IID data and client drift](/concepts/non-iid-heterogeneity.md)
- [FedProx](/methods/fedprox.md), [SCAFFOLD](/methods/scaffold.md)
- [FedDyn paper](/papers/acar-2021-feddyn.md)
