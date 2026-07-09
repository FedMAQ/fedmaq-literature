---
type: Paper
title: "Federated Learning Based on Dynamic Regularization"
description: "FedDyn adds a per-device dynamic regularizer so that local minima align with the global optimum, decoupling communication rounds from statistical heterogeneity."
authors: "Acar et al."
year: 2021
bibkey: acar-2021-feddyn
baseline: FedDyn
tags: [fl-core, heterogeneity]
resource: markdown/acar-2021-feddyn/paper.md
timestamp: 2026-07-09T10:18:16Z
---

## 1. Overview & Objectives

- **Core Problem**: In heterogeneous federated learning, the minimizers of each device's local empirical loss are **inconsistent** with the minimizer of the global loss. Methods that push more computation onto devices (many local epochs) to save communication therefore drift toward local optima and stall short of the global solution — a "fundamental dilemma" between local computation and global consistency.
- **Main Objectives**:
  - Introduce **FedDyn** (Federated Dynamic regularization): a method where, at each round, each device optimizes its local loss plus a **dynamically updated regularization term** so that, in the limit, the stationary points of the device objectives coincide with the global stationary point.
  - Achieve communication efficiency that is **agnostic to device heterogeneity** and robust to partial participation, large device counts, and unbalanced data.
  - Provide convergence guarantees in both convex and non-convex settings.

## 2. Methodology & Key Innovations

- **Key Idea**: Replace the static proximal penalty of FedProx with a **dynamic linear+quadratic regularizer** whose linear term accumulates the device's own past gradients. This makes each device's regularized objective's optimum track the global model rather than the device's local optimum.
- **Per-device objective** at round \(t\), device \(k\) with local loss \(L_k\), server model \(\theta^{t-1}\):
  - Minimize \(L_k(\theta) - \langle \nabla L_k(\theta_k^{t-1}), \theta \rangle + \tfrac{\alpha}{2}\lVert \theta - \theta^{t-1} \rVert^2\).
  - The **linear term** (negative inner product with the accumulated local gradient) is what makes the regularizer *dynamic* — it is updated each round from the device's latest local gradient.
- **Server aggregation** additionally maintains a running average of the accumulated local gradients (an implicit global state term \(h\)), so the server model corrects for the aggregate drift:
  - \(h^t = h^{t-1} - \alpha \tfrac{1}{m}\sum_{k \in \mathcal{P}^t}(\theta_k^t - \theta^{t-1})\), \(\quad \theta^t = \bar{\theta}^t - \tfrac{1}{\alpha}h^t\).
- **Consequence**: at convergence the first-order condition of the device problems sums to the global first-order condition, so device and global solutions are **aligned** without requiring IID data or full participation.

## 3. Mathematical Formulation

- **Global objective**: \(\min_{\theta}\; \tfrac{1}{N}\sum_{k=1}^N L_k(\theta)\), with each device holding a possibly non-IID local risk \(L_k\).
- **Dynamic regularized local risk** at device \(k\), round \(t\):

\[
\theta_k^t = \arg\min_\theta \; L_k(\theta) - \langle \nabla L_k(\theta_k^{t-1}),\, \theta \rangle + \frac{\alpha}{2}\lVert \theta - \theta^{t-1}\rVert^2 .
\]

- **Server update** (average of received models corrected by the aggregated gradient state):

\[
\theta^t = \frac{1}{|\mathcal{P}^t|}\sum_{k \in \mathcal{P}^t}\theta_k^t \; - \; \frac{1}{\alpha} h^t .
\]

- The regularization strength \(\alpha\) is the single key hyperparameter (searched over \([10^{-3},10^{-1}]\) in the experiments).

## 4. Limitations & Constraints

- **Persistent device state**: each device must store its accumulated local gradient term across rounds, which assumes devices are revisited (or that state is server-mirrored) — a mismatch with massively cross-device settings where a client is seen once.
- **Regularization sensitivity**: performance depends on tuning \(\alpha\); too large slows local progress, too small fails to enforce consistency.
- **No update compression**: like FedAvg/FedProx, FedDyn communicates full-precision model parameters; it reduces the *number* of rounds but not the *bits per round*.
- **Assumes reliable local optimization**: the alignment argument relies on devices approximately solving their regularized subproblem each round.

## 5. FedMAQ Thesis Relevance

- **Heterogeneity baseline**: FedDyn is a strong non-IID baseline in the same family as [FedProx](/papers/li-2020-fedprox.md) and [SCAFFOLD](/papers/karimireddy-2020-scaffold.md), correcting client drift via regularization rather than control variates. It anchors the "statistical-heterogeneity" axis that FedMAQ must remain competitive on while adding compression.
- **Orthogonality to compression**: FedDyn's drift correction operates on the optimization objective, leaving the communicated payload uncompressed. FedMAQ's multi-adaptive quantization and knowledge distillation are **complementary** — one could quantize the FedDyn model updates, testing whether dynamic regularization tolerates aggressive bit-width reduction.
- **Key insight to integrate**: FedDyn demonstrates that trading extra local computation for fewer rounds is only "free" if local-global consistency is enforced. FedMAQ's adaptive schedule should likewise guard against quantization noise re-introducing the very drift FedDyn removes.

# Related

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](/papers/mcmahan-2017-fedavg.md)
- [Federated Optimization in Heterogeneous Networks](/papers/li-2020-fedprox.md)
- [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](/papers/karimireddy-2020-scaffold.md)

# Citations

[1] Full-text conversion: [markdown/acar-2021-feddyn/paper.md](markdown/acar-2021-feddyn/paper.md)
[2] Source PDF: `papers/01 FL, Heterogeneity/Acar et al. - 2021 - Federated Learning Based on Dynamic Regularization.pdf`
