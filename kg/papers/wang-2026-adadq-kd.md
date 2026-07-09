---
type: Paper
title: "AdaDQ-KD: An Adaptive Dithering Quantization with Knowledge Distillation in Privacy-Preserving Federated Learning"
description: "Federated Learning (FL) with Differential Privacy (DP) faces a critical trade-off between privacy, communication efficiency, and model accuracy."
authors: "Wang et al."
year: 2026
bibkey: wang-2026-adadq-kd
tags: [quantization, kd, sota]
resource: markdown/wang-2026-adadq-kd/paper.md
timestamp: 2026-06-21T07:55:20Z
---

### 1. Overview & Objectives

- **Core Problem:** Federated Learning (FL) with Differential Privacy (DP) faces a critical trade-off between privacy, communication efficiency, and model accuracy. Standard DP noise injection degrades model performance, while communication bottlenecks are exacerbated by client heterogeneity (stragglers) and non-IID data.
- **Main Objectives:**
  1.  To improve training efficiency in privacy-preserving FL by reducing communication overhead.
  2.  To mitigate the accuracy loss caused by DP noise injection and gradient quantization.
  3.  To handle system heterogeneity (stragglers) and statistical heterogeneity (non-IID data) simultaneously.
- **Proposed Solution:** The paper introduces **AdaDQ-KD**, an algorithm that integrates an adaptive dithering quantization (DQ) scheme with a knowledge distillation (KD) module to achieve a better trade-off among privacy, efficiency, and accuracy.

### 2. Methodology & Key Innovations

The AdaDQ-KD algorithm operates within a standard synchronous FL framework with a semi-honest parameter server (PS). Its core innovations are:

1.  **Adaptive Dithering Quantization (AdaDQ):**
    - **DQ for DP & Compression:** Dithering quantization is used to compress client gradients before transmission. The key insight is that the quantization error from DQ is mathematically equivalent to Gaussian noise, thus providing a DP guarantee without separate noise injection.
    - **Adaptive Precision:** The quantization precision (step size $\Delta$) is dynamically adjusted per client per round based on their estimated local delay. Stragglers (clients with high expected delay) are assigned a lower precision (coarser quantization) to reduce their communication load, thereby minimizing the overall round time.

2.  **Knowledge Distillation (KD) for Robustness:**
    - A pre-trained global teacher model is distributed to clients.
    - During local training, each client minimizes a combined loss function: a standard cross-entropy loss ($\mathcal{L}_{CE}$) and a feature-level distillation loss ($\mathcal{L}_{KD}$).
    - This KD process acts as an implicit regularizer, guiding local training and mitigating the adverse effects of data heterogeneity and quantization/noise injection on model accuracy.

### 3. Mathematical Formulation

- **Global Objective:**
  $$ \min_{w \in \mathbb{R}^m} F(w) := \frac{1}{P} \sum_{i=1}^{P} F_i(w) $$
    where $F_i(w)$ is the local objective for client $i$.

- **Local Training Objective (with KD):**
  $$ \mathcal{L}_{\text{local}} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{KD}} $$
  The local gradient is then $g_i^t = \nabla_w \mathcal{L}_{\text{local}}(w)$.

- **Dithering Quantization (DQ) Process:**
  - **Step Size:** $\Delta_i = 2 n_i \sigma \sqrt{v_i}$, where $v_i \sim \Gamma(3/2, 1/2)$, $\sigma$ is the target DP noise std, and $n_i$ is the **quantization precision coefficient**.
  - **Quantization Function:** $q_i^t = Q(g_i^t + u_i)$, where $u_i \sim U(-\Delta_i/2, \Delta_i/2)$ and $Q(x) = \lceil x/\Delta_i - 1/2 \rfloor \Delta_i + \Delta_i/2$.
  - **Noisy Estimation (Decoding):** $\hat{g}_i^t = q_i^t - u_i$.
  - **Gaussian Noise Equivalence (Theorem 1):** The error from DQ is equivalent to Gaussian noise: $\hat{g}_i^t \sim g_i^t + \mathcal{N}(0, \mathbb{I}_m \sigma^2 n_{i,t}^2)$.

- **Adaptive Precision Adjustment (Algorithm 1):**
  - **Expected Local Delay:**
    $$ \mathbb{E}(T_{i,t+1}^{loc}) = \mathbb{E}(T_{i,t+1}^{comp}) + \frac{\log_2(2\lceil \frac{C}{\Delta_{i,t+1}} + 1 \rceil)}{\log_2(2\lceil \frac{C}{\Delta_{i,t}} + 1 \rceil)} T\_{i,t}^{comm} $$
  - **Straggler Identification:** Clients are sorted by $\mathbb{E}(T_{i,t}^{loc})$. The top $K = \lfloor kP \rfloor$ clients are identified as stragglers.
  - **Precision Reduction:** For each straggler, the precision coefficient $n_i$ is reduced by a search length $h$ until its expected delay is below a threshold $S$ (the delay of the $(K+1)$-th client).

- **Global Aggregation:**
  $$ w^{t+1} = w^t - \alpha_t \left( \frac{1}{P} \sum_{i=1}^{P} \hat{g}\_i^t \right) $$

- **Convergence Bound (Theorem 4):**
  Under standard assumptions (L-smoothness, bounded gradients), the convergence is bounded by:
  $$ \sum_{t=0}^{T-1} \mathbb{E}[ \| \nabla F(w^t) \|^2 ] \leq \frac{1}{P} (M^2 + \sigma^2) + \frac{2[F(w^0) - F_{inf}]}{\alpha_0 T} $$
    where $M^2$ bounds the gradient variance and $\sigma^2$ is the DP noise variance.

### 4. Limitations & Constraints

- **Dependence on Teacher Model Quality:** The effectiveness of the KD module is contingent on the quality of the pre-trained global teacher model. A poor teacher can provide misleading guidance.
- **Increased Computational Overhead:** The KD process introduces additional forward passes through the teacher model and computation of feature distillation loss, increasing local computation time (though this is argued to be acceptable compared to communication savings).
- **Assumption of Stable Communication:** The adaptive precision adjustment relies on the assumption that the transmission rate changes smoothly between rounds, which may not hold in highly dynamic wireless environments.
- **Privacy Budget Fluctuation:** The adaptive DQ scheme causes the per-round privacy budget consumption to fluctuate, making the total privacy accounting more complex than in standard DP-FL.
- **Complexity of Optimization:** The problem of minimizing the longest local delay is NP-hard, and the proposed algorithm is a heuristic solution that finds a "good feasible solution" rather than an optimal one.

### 5. FedMAQ Thesis Relevance

This paper is **highly relevant** to the FedMAQ thesis (Communication-Efficient FL via Multi-Adaptive Quantization and Knowledge Distillation).

- **Direct Baseline:** The proposed **AdaDQ-KD** algorithm is a direct and strong baseline for FedMAQ. It explicitly combines adaptive quantization and knowledge distillation to address the same core challenges: communication efficiency, privacy, and heterogeneity.
- **Technique Integration:**
  - **Adaptive Quantization:** The method of dynamically adjusting quantization precision based on client delay (straggler mitigation) is a key technique that can be integrated into FedMAQ's "Multi-Adaptive Quantization" component.
  - **Knowledge Distillation:** The use of feature-level KD loss to mitigate accuracy loss from quantization and noise is a powerful technique that can be integrated into FedMAQ's KD module.
  - **DQ for DP:** The concept of using dithering quantization to simultaneously achieve compression and DP is a novel and efficient approach that FedMAQ could adopt or compare against.
- **Gap Addressed by FedMAQ:** While AdaDQ-KD is a strong baseline, it uses a single, global teacher model. FedMAQ could potentially improve upon this by exploring **multi-teacher KD** or **ensemble distillation** to provide more robust and personalized guidance across heterogeneous clients, further improving accuracy under high heterogeneity.

# Related

- [DynFed: Adaptive Federated Learning via Quantization-Aware Knowledge Distillation](/papers/he-2025-dynfed.md)
- [FedDT: A Communication-Efficient Federated Learning via Knowledge Distillation and Ternary Compression](/papers/he-2025-feddt.md)
- [Quantization and Knowledge Distillation for Efficient Federated Learning on Edge Devices](/papers/qu-2020-quantization-kd.md)
- [CFD: Communication-Efficient Federated Distillation via Soft-Label Quantization and Delta Coding](/papers/sattler-2022-cfd.md)

# Citations

[1] Full-text conversion: [markdown/wang-2026-adadq-kd/paper.md](markdown/wang-2026-adadq-kd/paper.md)
[2] Source PDF: `papers/04 Q+KD/Wang et al. - 2026 - AdaDQ-KD An Adaptive Dithering Quantization with Knowledge Distillation in Privacy-Preserving Feder.pdf`
