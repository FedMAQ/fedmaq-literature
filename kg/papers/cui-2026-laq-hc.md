---
type: Paper
title: "Lightweight Adaptive Quantization Algorithms for Federated Learning With Heterogeneous Clients"
description: "In federated learning (FL) with heterogeneous edge clients, uniform quantization levels fail to account for variations in client data quality, communication bandwidth, and computational capabilities."
authors: "Cui et al."
year: 2026
bibkey: cui-2026-laq-hc
tags: [quantization, adaptive]
resource: markdown/cui-2026-laq-hc/paper.md
timestamp: 2026-06-21T06:51:10Z
---

## 1. Overview & Objectives

**Core Problem:** In federated learning (FL) with heterogeneous edge clients, uniform quantization levels fail to account for variations in client data quality, communication bandwidth, and computational capabilities. This mismatch leads to straggler effects, increased overall runtime, and suboptimal convergence.

**Main Objectives:**
- Enable each client to adaptively select its quantization level based on its data quality and communication capabilities without increasing computation costs.
- Model the relationship between quantization levels and their impact on training convergence in a lightweight manner.
- Achieve faster convergence and higher accuracy under limited bandwidth constraints while reducing computation time, overall runtime, and communication overhead compared to existing adaptive quantization algorithms.

## 2. Methodology & Key Innovations

**Core Idea:** Clients with lower communication capabilities should use higher quantization levels (more compression), while those with higher capabilities should use lower levels (less compression). This ensures all clients complete uploads in similar time, mitigating the straggler problem.

**Key Innovations:**
1. **Lightweight Impact Estimation:** The impact of different quantization levels on training loss reduction follows a hyperbolic tangent function that is consistent across clients and nearly identical between adjacent rounds. This allows fitting the impact function once per round using previous round data, avoiding costly per-client per-level training.
2. **Client Quality Metric:** Combines data quantity, data quality (loss value), and quantization impact into a single quality score \( q_{i,\ell(i)}^t \) for each client \( i \) at quantization level \( \ell(i) \) in round \( t \).
3. **Flag-Based Selection:** A flag function \( \text{flag}_{i,\ell(i)} \) balances client quality and bandwidth per unit quantization level. Two scenarios are considered:
   - **Without constraint:** Select the quantization level maximizing flag for each client.
   - **With constraint (e.g., limited total bandwidth):** Formulated as a grouped knapsack problem (NP-hard) to select clients and their quantization levels maximizing total weighted quality under bandwidth budget.
4. **Cold-Start Phase:** First round requires training on a single client across four quantization levels to fit the four-parameter hyperbolic tangent function; overhead is negligible relative to total training rounds.

## 3. Mathematical Formulation

### 3.1 FL Framework with Gradient Quantization

**Global objective:**
\[
F(w) = \sum_{i=1}^n \frac{D_i}{D} F_i(w), \quad F_i(w) = \frac{1}{D_i} \sum_{y \in \mathcal{D}_i} f(w, y)
\]

**Local gradient update (SGD):**
\[
w_i^t \leftarrow w^t - \eta g_i(w^t; \xi_i^t)
\]

**Quantization (uniform affine quantization):**
\[
x_{\text{int}} = \text{round}\left( \frac{g_i(w^t; \xi_i^t)}{\mathcal{R}} \right) + z, \quad \mathcal{R} = \frac{x_{\max} - x_{\min}}{2^{\ell(i)} - 1}, \quad z = \text{round}(-x_{\min}/\mathcal{R})
\]
\[
Q_{i,\ell(i)}^t = \text{clamp}(0, 2^{\ell(i)}-1, x_{\text{int}})
\]

**De-quantization:**
\[
\hat{g}_i(w^t) = (Q_{i,\ell(i)}^t - z) \mathcal{R}
\]

**Model aggregation:**
\[
w^{t+1} = w^t - \frac{\eta}{D_c} \sum_{i \in \mathcal{C}^t} D_i \hat{g}_i(w^t), \quad D_c = \sum_{i \in \mathcal{C}^t} D_i
\]

### 3.2 Client Quality Estimation

**Without quantization (approximation):**
\[
q_{i,\ell(i)}^t = D_i F_i(w^t)
\]

**With quantization (using impact function):**
\[
q_{i,\ell(i)}^t = D_i F_i(w^t) \times IQ(t-1, \ell(i))
\]

**Impact function (hyperbolic tangent fit):**
\[
IQ(t, \ell(i)) = a \times \frac{e^{b \times 2^{\ell(i)}} - 1}{e^{c \times 2^{\ell(i)}} + 1} + d
\]
Parameters \(a,b,c,d\) are fitted using previous round data from selected clients.

### 3.3 Flag Functions for Client/Level Selection

**Without constraint:**
\[
\text{flag}_{i,\ell(i)} = \frac{\alpha q_{i,\ell(i)}^t + (1-\alpha) B_i}{\ell(i)}
\]

**With constraint (limited total bandwidth):**
\[
\text{flag}_{i,\ell(i)} = \frac{\alpha q_{i,\ell(i)}^t + (1-\alpha) B_i}{\ell(i) \times s_i}
\]
where \(s_i\) is the bandwidth of client \(i\).

### 3.4 Convergence Analysis

**Assumptions:**
- Unbiased stochastic gradient with bounded variance \(\sigma^2\).
- Bounded gradient magnitude \(G\).
- Unbiased quantization with error bound \(\epsilon_i = \sqrt{\min\{d/\ell(i)^2, \sqrt{d}/\ell(i)\}}\).

**Theorem 2 (Strongly convex case):** With learning rate \(\eta \leq 1/(3L)\),
\[
\mathbb{E}[F(w^{t+1}) - F(w^*)] \leq (1 - \eta\mu)^t [F(w^1) - F(w^*) - A] + A
\]
where \(A = \frac{3L\eta(\epsilon G^2 + \sigma^2 N)}{2N^2\mu}\) and \(\epsilon = \sum_{i=1}^N \epsilon_i^2\).

**Theorem 3 (Non-convex case, bound on average gradient):**
\[
\frac{1}{T} \sum_{t=1}^T \mathbb{E}\|\nabla F(w^t)\|^2 \leq \frac{2|F(w^1)-F(w^*)|}{\eta T} + \frac{3L\eta(\epsilon G^2 + \sigma^2 N)}{N^2}
\]

**Theorem 4 (Non-convex convergence rate):** With \(\eta = \sqrt{\frac{2|F(w^1)-F(w^*)|N^2}{3L(\epsilon G^2 + \sigma^2 N)T}}\),
\[
\frac{1}{T} \sum_{t=1}^T \mathbb{E}\|\nabla F(w^t)\|^2 \leq 2\sqrt{\frac{6[F(w^1)-F(w^*)]L(\epsilon G^2 + \sigma^2 N)}{N^2}} \cdot \frac{1}{\sqrt{T}}
\]
Thus LAQ-HC achieves \(O(1/\sqrt{T})\) convergence rate, same order as non-compression SGD.

## 4. Limitations & Constraints

**Statistical/System Assumptions:**
- Unbiased stochastic gradients with bounded variance and bounded gradient magnitude.
- Unbiased quantization with error bound proportional to gradient norm.
- Impact function consistency across clients and adjacent rounds (validated empirically but may not hold for all datasets/models).
- Cold-start phase requires initial training on four quantization levels; overhead is negligible but assumes availability of a representative client.

**Limitations:**
- The hyperbolic tangent fitting may not generalize to all data distributions or model architectures; requires re-fitting per dataset.
- Constrained selection is NP-hard (grouped knapsack), necessitating approximate solutions for large-scale systems.
- Dropout rates above ~20% degrade accuracy and runtime due to loss of high-quality clients.
- Sensitivity to hyperparameter \(\alpha\): optimal value depends on bandwidth variance and dataset complexity; requires tuning.
- Does not incorporate knowledge distillation or sparsification, which could further improve communication efficiency.

**Communication Bottleneck Concerns:**
- While LAQ-HC reduces overall communication overhead, the per-round overhead includes uploading client parameters (loss, data size) for selection, which is lightweight but non-zero.
- Under extreme heterogeneity, the algorithm may discard many low-quality clients, potentially slowing initial convergence.

## 5. FedMAQ Thesis Relevance

**LAQ-HC as a Baseline:** LAQ-HC is a state-of-the-art adaptive quantization method for heterogeneous FL. It can serve as a direct baseline for FedMAQ (Communication-Efficient FL via Multi-Adaptive Quantization and Knowledge Distillation). FedMAQ should compare against LAQ-HC on metrics such as convergence accuracy, communication overhead, computation time, and overall runtime under both unconstrained and bandwidth-constrained settings.

**Integration Opportunities:**
- **Knowledge Distillation (KD):** LAQ-HC does not use KD. FedMAQ could integrate LAQ-HC's adaptive quantization with KD to further reduce communication while preserving model accuracy. For example, clients could use LAQ-HC's impact-aware quantization for gradient compression, while the server employs KD to distill knowledge from multiple quantized updates.
- **Multi-Adaptive Quantization:** LAQ-HC adapts quantization levels per client per round based on data quality and bandwidth. FedMAQ could extend this to multi-adaptive strategies, e.g., combining quantization with sparsification or low-rank approximation, as suggested in the paper's future work.
- **Lightweight Impact Estimation:** The hyperbolic tangent fitting method is computationally cheap and could be adopted by FedMAQ to estimate the effect of different compression levels on training, enabling more informed client selection and resource allocation.

**Key Differences to Address:**
- LAQ-HC focuses solely on quantization; FedMAQ aims to combine multiple compression techniques (quantization, sparsification, KD).
- LAQ-HC does not consider privacy beyond standard FL; FedMAQ may incorporate differential privacy or blockchain-based mechanisms.
- LAQ-HC's convergence analysis assumes unbiased quantization; FedMAQ may need to handle biased compression methods.

**Conclusion:** LAQ-HC provides a strong foundation for adaptive quantization in heterogeneous FL. FedMAQ can build upon its lightweight impact estimation and flag-based selection, while adding knowledge distillation and multi-adaptive compression to achieve even greater communication efficiency.

# Citations

[1] Full-text conversion: [markdown/cui-2026-laq-hc/paper.md](markdown/cui-2026-laq-hc/paper.md)
[2] Source PDF: `papers/02 Quantization/Cui et al. - 2026 - Lightweight Adaptive Quantization Algorithms for Federated Learning With Heterogeneous Clients.pdf`
