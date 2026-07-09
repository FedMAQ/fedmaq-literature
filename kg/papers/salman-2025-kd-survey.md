---
type: Paper
title: "Knowledge distillation in federated learning a comprehensive survey"
description: "This paper is a comprehensive survey on the integration of Knowledge Distillation (KD) within Federated Learning (FL) systems."
authors: "Salman et al."
year: 2025
bibkey: salman-2025-kd-survey
tags: [survey, kd]
resource: markdown/salman-2025-kd-survey/paper.md
timestamp: 2026-06-21T08:57:52Z
---

### 1. Overview & Objectives

This paper is a comprehensive survey on the integration of Knowledge Distillation (KD) within Federated Learning (FL) systems. The core problem addressed is the inherent inefficiency and performance degradation in FL caused by **data heterogeneity** (non-IID data) and **model heterogeneity** (differing client architectures). Traditional FL algorithms like FedAvg suffer from client drift, slow convergence, and high communication costs when these heterogeneities are present.

The main objectives of the survey are to:
- Systematically categorize and analyze existing KD methods designed to mitigate the challenges of data and model heterogeneity in FL.
- Provide a structured overview of how KD can improve communication efficiency, enable personalization, and enhance model performance in decentralized settings.
- Synthesize empirical findings from the literature to highlight the benefits and limitations of KD in FL.
- Identify open research questions and real-world deployment challenges.

### 2. Methodology & Key Innovations

The paper does not propose a single novel method but instead provides a taxonomy of existing KD-in-FL techniques. The key innovation is the structured organization of these methods based on the specific FL challenge they address and the architectural role of KD.

The core system model revolves around a **teacher-student paradigm** adapted for FL. The "teacher" can be a global model, a server-side ensemble, or a previous model snapshot, while the "student" is typically a local client model. The survey categorizes methods into two main lines of research:

1.  **Handling Model Heterogeneity:** These methods use KD to allow clients with different model architectures to collaborate without sharing model parameters. Instead, they share **model outputs (soft labels/logits)** or **intermediate features**.
    - **Broadcasting Model Outputs:** Clients receive global logits/soft targets from the server, use them for local distillation, and send their own local logits back for aggregation. This avoids the need for a common model architecture.
    - **Using Intermediate Features:** Methods like FedAD and FedGKT use model-agnostic intermediate representations (e.g., attention maps) to transfer knowledge, enabling model diversity.

2.  **Handling Data Heterogeneity:** These methods use KD to stabilize training and prevent client drift caused by non-IID data.
    - **Server-Side Refinement:** The server uses a proxy dataset or a data-free generator to fine-tune the aggregated global model via ensemble distillation (e.g., FedDF, FedFTG).
    - **Client-Side Regularization:** A KD-based regularization term is added to the client's local loss function. This term penalizes the divergence between the client's predictions and the global model's predictions, effectively "distilling" global knowledge to control local drift (e.g., FedGKD, FedCAD).

### 3. Mathematical Formulation

The paper extracts and presents the core mathematical formulas for the key components of KD in FL.

**A. Global Objective in FL:**
The goal is to minimize the weighted sum of local loss functions across all clients.
$$
\min_{w} f(w) = \sum_{u=1}^{N} \frac{n_u}{N} F_u(w)
$$
where $w$ are the model parameters, $N$ is the number of users, $n_u$ is the number of data points for user $u$, and $F_u(w)$ is the local loss function.

**B. Knowledge Distillation Loss Types:**

- **Response-Based Loss:** Measures the distance between the teacher's logits ($\mathcal{Z}_t$) and the student's logits ($\mathcal{Z}_s$).
  $$
  L_{ResD}(\mathcal{Z}_t, \mathcal{Z}_s) = \mathcal{L}_{\mathcal{R}}(\mathcal{Z}_t, \mathcal{Z}_s)
  $$

- **Feature-Based Loss:** Measures the distance between intermediate feature maps ($f_t(x)$ and $f_s(x)$) from the teacher and student, often after a transformation ($\Phi$) to match dimensions.
  $$
  L_{FeaD}(f_t(x), f_s(x)) = \mathcal{L}_F(\Phi_t(f_t(x)), \Phi_s(f_s(x)))
  $$

- **Relation-Based Loss:** Measures the difference in the relationships between pairs of feature maps (e.g., using a similarity function $\psi$).
  $$
  L_{RelD}(f_t, f_s) = \mathcal{L}_{\mathcal{R}^1}(\psi_t(\hat{f}_t, \check{f}_t), \psi_s(\hat{f}_s, \check{f}_s))
  $$

**C. Client-Side Regularization Loss (for Data Heterogeneity):**
The local training objective combines a classification loss (e.g., cross-entropy) with a KD-based regularization loss.
$$
L_{client} = \sum_{i=1}^{N} \alpha l_k(C_i, G_i) + (1 - \alpha) l_c(t_i, y_i)
$$
where $C_i$ is the client model's output, $G_i$ is the global model's output, $l_k$ is the KD loss, $l_c$ is the classification loss, $t_i$ is the truth label, $y_i$ is the predicted value, and $\alpha$ is a learning rate/weighting parameter.

**D. Classification Loss (Cross-Entropy):**
$$
L = -\frac{1}{N} \left[ \sum_{j=1}^{N} [t_j \log(p_j) + (1 - t_j) \log(1 - p_j)] \right]
$$

**E. KD Loss (Kullback-Leibler Divergence):**
$$
D_{KL}(P || Q) = \sum_{x \in X} P(x) \log\left(\frac{P(x)}{Q(x)}\right)
$$
where $P$ and $Q$ are the probability distributions from the teacher and student models, respectively.

### 4. Limitations & Constraints

The survey explicitly discusses several limitations and constraints of current KD-in-FL methods:

- **Statistical Assumptions:** Many server-side methods (e.g., FedDF) rely on a **proxy dataset** that may not accurately represent the statistical properties of the private client data, leading to performance degradation in domain shift scenarios.
- **System Constraints:**
    - **Communication Bottleneck:** While KD can reduce communication, some methods (e.g., FedGKD with M>1) can double communication costs. Sharing soft labels or intermediate features still incurs overhead.
    - **Computational Overhead:** Client-side KD introduces significant computational and memory overhead (30-70% increase in training time, 1.5-2x memory usage), which is a major constraint for resource-constrained devices (IoT, mobile phones).
    - **Battery Consumption:** The increased computational load can raise battery usage by 25-40% on mobile devices.
- **Privacy Risks:** Sharing model outputs (logits/soft labels) or intermediate features can expose clients to **model inversion attacks**, where adversaries can reconstruct training data features.
- **Data Dependency:** Methods requiring labeled public datasets (e.g., MHAT, FedGEMS) are limited by the availability and quality of such data, and can introduce bias if the public data distribution differs from the private one.
- **Scalability:** Server-side ensemble distillation methods (e.g., FedDF) face scaling challenges as the computational complexity grows with the number of clients.

### 5. FedMAQ Thesis Relevance

This survey is highly relevant to the FedMAQ thesis (Communication-Efficient FL via Multi-Adaptive Quantization and Knowledge Distillation) and can serve as a foundational reference.

- **As a Baseline:** The paper provides a comprehensive overview of existing KD-in-FL methods that can serve as **strong baselines** for FedMAQ. Specifically, methods like **FedDistill** (for communication efficiency and model heterogeneity) and **FedGKD** (for handling data heterogeneity via regularization) are directly comparable. The survey's empirical synthesis provides performance benchmarks (e.g., accuracy improvements, communication cost reductions) against which FedMAQ can be evaluated.

- **Techniques for Integration:** The survey details several techniques that can be **integrated into the FedMAQ framework**:
    1.  **Client-Side Regularization:** The mathematical formulation for the local KD loss ($L_{client}$) can be directly adopted as the "Knowledge Distillation" component of FedMAQ. The paper's discussion of adaptive weighting (e.g., FedCAD's class-specific weights) provides a clear path for the "Multi-Adaptive" aspect.
    2.  **Server-Side Refinement:** The concept of server-side ensemble distillation (e.g., FedDF) could be combined with FedMAQ's quantization to further refine the global model after aggregation.
    3.  **Communication Efficiency:** The survey explicitly identifies communication cost as a key challenge and discusses methods like CFD (Compressed Federated Distillation) that use quantization and delta coding. This directly aligns with FedMAQ's goal of "Communication-Efficient FL" and provides a direct point of comparison for the quantization component.
    4.  **Model Heterogeneity:** The survey's detailed analysis of methods that share logits instead of parameters (e.g., FedMD, Cronus) provides a clear alternative aggregation strategy that FedMAQ could explore or adapt.

In summary, this survey is an essential resource for the FedMAQ thesis, providing a taxonomy of the field, a library of mathematical formulations, a set of strong baselines, and a clear map of techniques that can be directly integrated or adapted to achieve the thesis's goals.

# Related

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](/papers/mcmahan-2017-fedavg.md)
- [FedMD: Heterogenous Federated Learning via Model Distillation](/papers/li-2019-fedmd.md)
- [FedDistill: Global Model Distillation for Local Model De-Biasing in Non-IID Federated Learning](/papers/song-2024-feddistill.md)
- [CFD: Communication-Efficient Federated Distillation via Soft-Label Quantization and Delta Coding](/papers/sattler-2022-cfd.md)
- [Ensemble Distillation for Robust Model Fusion in Federated Learning](/papers/lin-2020-feddf.md)

# Citations

[1] Full-text conversion: [markdown/salman-2025-kd-survey/paper.md](markdown/salman-2025-kd-survey/paper.md)
[2] Source PDF: `papers/00 Surveys/Salman et al. - 2025 - Knowledge distillation in federated learning a comprehensive survey.pdf`
