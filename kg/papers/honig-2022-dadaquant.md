---
type: Paper
title: "DAdaQuant: Doubly-adaptive quantization for communication-efficient Federated Learning"
description: "Federated Learning (FL) suffers from high communication costs due to repeated transmission of model parameters between server and clients."
authors: "Hönig et al."
year: 2022
bibkey: honig-2022-dadaquant
baseline: DAdaQuant
tags: [quantization]
resource: markdown/honig-2022-dadaquant/paper.md
timestamp: 2026-06-21T04:46:15Z
---

## 1. Overview & Objectives

**Core Problem:**
Federated Learning (FL) suffers from high communication costs due to repeated transmission of model parameters between server and clients. Uplink (client → server) is particularly bottlenecked. Existing static quantization methods use a fixed quantization level, leaving room for dynamic adaptation to improve compression without sacrificing model accuracy.

**Main Objectives:**

- Introduce **time-adaptive quantization**: dynamically change the quantization level across training rounds (low precision early, high precision later).
- Introduce **client-adaptive quantization**: assign different quantization levels to clients according to their weight in the global aggregation (clients with larger dataset sizes get higher precision).
- Combine both into **DAdaQuant** to achieve state-of-the-art uplink compression while maintaining convergence and accuracy.
- Adapt QSGD (stochastic fixed-point quantizer) to work with FL parameter updates, creating **Federated QSGD** as a strong baseline.

## 2. Methodology & Key Innovations

**System Model:**

- FL with a server and \( N \) clients. Each client \( c_k \) has local dataset \( D_k \).
- Use FedAvg (with FedProx proximal term) for local training.
- In each round \( t \), server sends global model \( p_t \) to a random subset \( S_t \) of \( K \) clients. Clients compute updates \( p_t^{k+1} - p_t \), quantize them, and send to server. Server aggregates:
  \[
  p_{t+1} = p_t + \sum_{k \in S_t} \frac{|D_k|}{\sum_j |D_j|} Q(p_{t+1}^k - p_t)
  \]

**Key Innovations:**

1. **Federated QSGD**: Adaptation of QSGD gradient quantizer for parameter updates in FL. Uses difference coding (quantize \( p\_{t+1}^k - p_t \)), stochastic fixed-point quantization with \( q \) bins per sign, followed by 0-run-length encoding and Elias \( \omega \) coding.
2. **Time-Adaptive Quantization**: Monotonically increases quantization level \( q_t \) based on a moving average of the estimated global loss. \( q_t \) doubles when the loss plateaus, bounded by \( q_{\max} \).
3. **Client-Adaptive Quantization**: Assigns quantization levels \( q_k \) to clients in a round to minimize total communication \( \sum_k q_k \) while keeping the expected variance of the quantized aggregate equal to that of a static quantizer with level \( q \). Optimal solution derived via Lagrangian optimization: \( q_i \propto w_i^{2/3} \), where \( w_i \) are client aggregation weights.

## 3. Mathematical Formulation

### Federated QSGD Quantization

For a parameter vector \( p \), quantize elementwise:

- For each element \( x \in [ -t, t ] \), let \( s = t/q \), \( b = \text{rem}(x, s) \), \( u = s - b \).
- Stochastic rounding:
  \[
  Q_q(x) = \begin{cases}
  \lfloor x/s \rfloor s & \text{with prob } 1 - b/s \\
  \lceil x/s \rceil s & \text{with prob } b/s
  \end{cases}
  \]
- Unbiased: \( \mathbb{E}[Q_q(x)] = x \)
- Variance: \( \text{Var}(Q_q(x)) = u b \leq s^2/4 \)

### Time-Adaptive Quantization Rule

- Initialize \( q_1 = q_{\min} \).
- Estimate global loss: \( \hat{G}_t = \sum_{k \in S_t} \frac{|D_k|}{\sum_l |D_l|} F_k(p_t) \)
- Moving average: \( \bar{G}_t = \psi \bar{G}_{t-1} + (1-\psi) \hat{G}\_t \), where the smoothing weight \( \psi \) is set to 0.9.
- If \( \bar{G}_t \geq \bar{G}_{t-\phi} - \phi \) (plateau detected over the lookback window of \( \phi \) rounds), then double the quantization level: \( q_t = 2 q_{t-1} \) (capped at \( q\_{\max} \)).
- Once the quantization level is doubled, it is kept fixed for at least \( \phi \) rounds to allow loss reductions to manifest in the moving average before another doubling can occur.

### Client-Adaptive Quantization (Theorem 1)

Assume parameters \( p_1, \dots, p_K \sim \mathcal{U}[-t, t] \) and weights \( w_i = |D_i| / \sum_j |D_j| \).
Define the expected variance of the quantized aggregate:
\[
\mathbb{E}_{p_1\dots p_K}[\text{Var}(e^{q_1\dots q_K}_p)] = \frac{t^2}{6} \sum_{i=1}^K \frac{w_i^2}{q_i^2}
\]
Minimize total cost \( \sum_i q_i \) subject to fixing this variance equal to that of static quantizer with level \( q \).
Optimal solution:
\[
q_i = \sqrt{\frac{a}{b}} \cdot w_i^{2/3}, \quad a = \sum_{j=1}^K w_j^{2/3}, \quad b = \sum_{j=1}^K \frac{w_j^2}{q^2}
\]
In practice, round to integer: \( q_i = \max(1, \text{round}\big(\sqrt{a/b} \cdot w_i^{2/3}\big)) \).

### Overall DAdaQuant Update

At round \( t \):

1. Determine \( q_t \) via time-adaptive rule (using moving average loss).
2. For all sampled clients \( k \in S_t \), compute client quantization levels \( q_k \) using client-adaptive formula with \( q = q_t \).
3. Each client sends quantized update \( Q_{q_k}(p_{t+1}^k - p_t) \).
4. Server aggregates: \( p_{t+1} = p_t + \sum_k w_k Q_{q_k}(p_{t+1}^k - p_t) \).

**Convergence guarantee:** DAdaQuant inherits convergence from FedPAQ if the quantizer is unbiased and has bounded variance for minimum \( q=1 \). Federated QSGD satisfies this.

## 4. Limitations & Constraints

- **Communication bottleneck focus**: Only addresses uplink compression; downlink (server→client) assumed less constrained.
- **Assumption on parameter distribution**: Client-adaptive optimality theorem assumes parameters are uniformly distributed in \([-t, t]\). May not hold exactly for all models/layers.
- **Time-adaptation sensitivity**: Requires careful tuning of \( \phi, \psi, q_{\min}, q_{\max} \). For datasets that converge quickly or need high precision early (e.g., Shakespeare), initialization \( q_{\min} = q_{\max}/2 \) is used, reducing adaptation range.
- **Overhead**: Additional forward pass per client to compute local loss \( F_k(p_t) \) for moving average (≈1% overhead reported).
- **No knowledge distillation**: Paper relies solely on quantization; no student-teacher or KD component.
- **Heterogeneity handling**: Simulated system heterogeneity by reducing epochs for some clients, but client-adaptive quantization does not account for varying compute power (only data size).

## 5. FedMAQ Thesis Relevance

### Baseline Role

DAdaQuant serves as a **state-of-the-art baseline** for adaptive quantization in FL, achieving compression factors of 1.2×–2.8× over strong static baselines (Federated QSGD). FedMAQ can directly compare its performance against DAdaQuant to demonstrate improvements from multi-adaptive quantization and knowledge distillation.

### Integration Potential

- **Client-adaptive quantization**: FedMAQ could adopt DAdaQuant’s optimal quantization level allocation based on client weights. The formula \( q_i \propto w_i^{2/3} \) provides a principled way to assign bits across clients without requiring retraining.
- **Time-adaptive quantization**: The moving-average loss plateau detection can be integrated into FedMAQ to dynamically adjust overall quantization resolution over rounds.
- **Federated QSGD**: This adaptation of QSGD for parameter updates is a strong compression engine. FedMAQ could build upon it or replace it with its own quantizer, using DAdaQuant’s adaptation strategies (time and client) as additional dimensions.

### Complementarity with Knowledge Distillation

DAdaQuant does not use KD, but FedMAQ could combine its adaptive quantization with a KD loss to further improve convergence when using very low precision (e.g., early rounds). The time-adaptive schedule could be driven by distillation loss rather than training loss, potentially achieving even better compression-accuracy trade-offs.

### Summary for FedMAQ

DAdaQuant provides:

- A **design pattern** for two-level adaptive quantization (time + client).
- **Closed-form formulas** for client-level quantization that are computationally cheap.
- **Empirical baselines** across multiple datasets and models (Synthetic, FEMNIST, Sent140, Shakespeare, CelebA) with clear compression-accuracy curves.
- A **building block** for FedMAQ: use DAdaQuant for pure quantization, then extend with multi-adaptation (e.g., per-layer) and KD to achieve further gains.

# Related

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](/papers/mcmahan-2017-fedavg.md)
- [Federated Optimization in Heterogeneous Networks](/papers/li-2020-fedprox.md)
- [QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding](/papers/alistarh-2017-qsgd.md)
- [FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization](/papers/reisizadeh-2020-fedpaq.md)

# Citations

[1] Full-text conversion: [markdown/honig-2022-dadaquant/paper.md](markdown/honig-2022-dadaquant/paper.md)
[2] Source PDF: `papers/02 Quantization/Hönig et al. - 2022 - DAdaQuant Doubly-adaptive quantization for communication-efficient Federated Learning.pdf`
