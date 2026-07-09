---
type: Paper
title: "DynFed: Adaptive Federated Learning via Quantization-Aware Knowledge Distillation"
description: "Federated Learning (FL) suffers from high communication overhead, client resource heterogeneity (varying memory/compute), and data heterogeneity (non-IID distributions)."
authors: "He et al."
year: 2025
bibkey: he-2025-dynfed
tags: [quantization, kd, sota]
resource: markdown/he-2025-dynfed/paper.md
timestamp: 2026-06-21T07:25:10Z
---

### 1. Overview & Objectives

**Core Problem:** Federated Learning (FL) suffers from high communication overhead, client resource heterogeneity (varying memory/compute), and data heterogeneity (non-IID distributions). Existing methods either ignore resource disparities by using uniform quantization or fail to properly aggregate models with heterogeneous bit-widths, leading to accuracy degradation and straggler issues.

**Main Objectives:**
- Dynamically allocate quantization bit-widths to clients based on their resource capacity and training dynamics (loss/gradient behavior).
- Design an error-resilient aggregation mechanism that handles heterogeneous bit-widths without direct parameter averaging.
- Improve global model accuracy and convergence under both resource and data heterogeneity.

### 2. Methodology & Key Innovations

DynFed operates in two core phases per communication round:

1. **Dynamic Bit-Width Allocation (Client Side):**
   - **Initialization:** Each client’s quantization level \(q_k\) is set based on its memory capacity \(c_k\) using QSGD (Quantized Stochastic Gradient Descent).
   - **Adaptive Adjustment:** During training, the bit-width \(b_i^{(t)}\) is updated according to the gradient norm of the local loss. Clients with larger gradients receive higher bit-widths to capture more precise updates, while those with smaller gradients use lower bit-widths to save communication.

2. **Multi-Teacher Knowledge Distillation (Server Side):**
   - Instead of directly averaging heterogeneous quantized models, the server performs knowledge distillation.
   - **Active Teacher Selection:** For each unlabeled sample in a public dataset, the server evaluates client models using an uncertainty score (entropy of MCMC-sampled predictions) and a comprehensive score that combines bit-width and prediction confidence. A diversity penalty prevents over-selection of the same client.
   - **Distillation:** The selected teacher models produce soft labels, which are used to train the global model via a combination of distillation loss and cross-entropy loss.

**Key Innovations:**
- Resource-aware and loss-adaptive quantization that aligns precision with client capabilities and training progress.
- Server-side multi-teacher distillation that leverages client diversity and mitigates quantization-induced errors.
- Active learning-based teacher selection to choose the most informative and reliable teachers per sample.

### 3. Mathematical Formulation

#### 3.1 Quantization Initialization (Eq. 2 & 3)

Given client \(k\) with memory capacity \(c_k\) and capacity per bin \(c_p\):
\[
q_k = \min\left(c_{\max}, \left\lfloor \frac{c_k}{c_p} \right\rfloor\right)
\]

Quantization of normalized gradient \(p_{\text{norm}} = p / \|p\|_2\):
\[
Q_{q_k}(p_i) = \text{sig}(p_i) \cdot \left( \frac{\lfloor q_k |p_i| \rfloor + \mathbb{I}[U < (q_k |p_i| - \lfloor q_k |p_i| \rfloor)]}{q_k} \right)
\]
where \(\mathbb{I}[\cdot]\) is the indicator function introducing stochastic rounding.

#### 3.2 Dynamic Bit-Width Adjustment (Eq. 4)

\[
b_i^{(t)} = b_i^{(t-1)} + \eta \cdot \left( \frac{|\nabla F_i(w, t)|}{\max(|\nabla F_i(w, t)|)} - \frac{b_i^{(t-1)}}{B_{\max}} \right)
\]
where \(\eta\) is the learning rate for bit-width adjustment, \(|\nabla F_i(w, t)|\) is the gradient norm of client \(i\) over a unit time interval, and \(B_{\max}\) is the maximum allowed bit-width.

#### 3.3 Uncertainty Evaluation (Eq. 5 & 6)

For sample \(x_i\), the global model \(\mathcal{M}_G\) produces \(M\) Monte Carlo samples:
\[
p(y|x_i, \mathcal{M}_G) = \frac{1}{M} \sum_{m=1}^M p_m(y|x_i, \mathcal{M}_G)
\]
\[
U(x_i) = -\sum_{c=1}^C p(y=c|x_i, \mathcal{M}_G) \log p(y=c|x_i, \mathcal{M}_G)
\]

#### 3.4 Comprehensive Score for Teacher Selection (Eq. 7 & 8)

For client model \(\mathcal{M}_k\) on sample \(x_i\):
\[
S_i^k = \alpha \cdot b(x_i, \mathcal{M}_k) + \beta \cdot \min p(y|x_i, \mathcal{M}_k)
\]
where \(b(x_i, \mathcal{M}_k)\) is the quantization bit-width of \(\mathcal{M}_k\) on a validation set similar to \(x_i\), and \(\alpha, \beta\) are weighting factors.

With diversity penalty \(D_k\):
\[
S_i^{k'} = S_i^k - \lambda D_k
\]

#### 3.5 Knowledge Distillation Loss (Eq. 9, 10, 11)

Soft label from selected teacher set \(\mathcal{M}_T\):
\[
\tau_i = \frac{1}{|\mathcal{M}_T|} \sum_{\mathcal{M}_j \in \mathcal{M}_T} p(y|x_i, \mathcal{M}_j)
\]

Distillation loss:
\[
L_{\text{distill}} = -\sum_{i=1}^K \sum_{c=1}^C \tau_i(c) \log p(y=c|x_i, \mathcal{M}_G)
\]

Final loss for global model:
\[
L = \gamma L_{\text{distill}} + (1-\gamma) L_{CE}
\]

#### 3.6 Convergence Bound (Eq. 13)

Under assumptions of \(L\)-smoothness, bounded variance \(\sigma^2\), bounded quantization error \(\epsilon_q\), and bounded distillation error \(\epsilon_d\):
\[
\mathbb{E}[F(w_T) - F(w^*)] \leq \frac{L\|w_0 - w^*\|^2}{2T\eta} + \frac{\eta L \sigma^2}{2} + L\epsilon_q + L\epsilon_d
\]

### 4. Limitations & Constraints

- **Assumptions:** The convergence proof relies on standard assumptions (smoothness, bounded variance, bounded quantization/distillation errors). In practice, these may not hold for highly non-convex models or extreme heterogeneity.
- **Public Dataset Requirement:** The server-side distillation requires a small public unlabeled dataset (200 samples in experiments). This may not always be available or may introduce privacy concerns if the public data distribution differs from client data.
- **Server Computation Overhead:** The active teacher selection and distillation process (MCMC sampling, inference on all client models) adds computational burden on the server, which may be a bottleneck in large-scale deployments.
- **Communication Overhead of Gradients:** Although quantization reduces message size, clients still send full gradient vectors (not just model parameters) to the server, which may be costly for very deep models.
- **Static Resource Estimation:** Initial bit-width allocation depends on pre-measured memory capacity \(c_k\); dynamic changes in client resources during training are not explicitly handled.

### 5. FedMAQ Thesis Relevance

**DynFed as a Baseline:** DynFed can serve as a strong baseline for FedMAQ (Communication-Efficient FL via Multi-Adaptive Quantization and Knowledge Distillation). It demonstrates that adaptive quantization combined with KD-based aggregation outperforms fixed-quantization methods (FedPAQ, DAdaQuant) and vanilla KD methods (FedKD) under heterogeneous settings.

**Techniques Integrable into FedMAQ:**
- **Dynamic Bit-Width Adjustment (Eq. 4):** The gradient-norm-based adaptation rule can be directly adopted or extended in FedMAQ to handle varying client capacities.
- **Multi-Teacher Distillation with Active Selection:** The uncertainty-based teacher selection (Eq. 5-8) and diversity penalty can be integrated into FedMAQ’s aggregation module to improve knowledge transfer from heterogeneous quantized models.
- **Server-Side Distillation:** The idea of performing distillation entirely on the server (reducing client burden) aligns with FedMAQ’s goal of communication efficiency.
- **Convergence Analysis Framework:** The proof structure (local update error + aggregation error + distillation error) provides a template for analyzing FedMAQ’s convergence under similar assumptions.

**Potential Extensions for FedMAQ:**
- Replace the public dataset requirement with a synthetic data generator or a small subset of global model’s own predictions.
- Combine with gradient compression (e.g., sparsification) to further reduce uplink communication.
- Extend the teacher selection to consider both bit-width and data distribution similarity (e.g., using prototype-based metrics).

# Related

- [QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding](/papers/alistarh-2017-qsgd.md)
- [DAdaQuant: Doubly-adaptive quantization for communication-efficient Federated Learning](/papers/honig-2022-dadaquant.md)
- [FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization](/papers/reisizadeh-2020-fedpaq.md)
- [Communication-efficient federated learning via knowledge distillation](/papers/wu-2022-fedkd.md)

# Citations

[1] Full-text conversion: [markdown/he-2025-dynfed/paper.md](markdown/he-2025-dynfed/paper.md)
[2] Source PDF: `papers/04 Q+KD/He et al. - 2025 - DynFed Adaptive Federated Learning via Quantization-Aware Knowledge Distillation.pdf`
