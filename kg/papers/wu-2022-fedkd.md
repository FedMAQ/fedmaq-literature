---
type: Paper
title: "Communication-efficient federated learning via knowledge distillation"
description: "Standard Federated Learning (FL) requires communicating large model updates (gradients/weights) between clients and server, leading to prohibitive communication costs, especially for modern deep models with billions..."
authors: "Wu et al."
year: 2022
bibkey: wu-2022-fedkd
baseline: FedKD
tags: [quantization, kd]
resource: markdown/wu-2022-fedkd/paper.md
timestamp: 2026-06-21T09:54:02Z
---

Wu et al., _Nature Communications_, 2022

### 1. Overview & Objectives

- **Core Problem:** Standard Federated Learning (FL) requires communicating large model updates (gradients/weights) between clients and server, leading to prohibitive communication costs, especially for modern deep models with billions of parameters. This limits the practical deployment of FL in bandwidth-constrained or environmentally-conscious settings.
- **Main Objectives:**
  1.  Drastically reduce the communication cost of FL without significantly sacrificing model accuracy.
  2.  Maintain strong model performance, ideally comparable to centralized learning, even under non-IID (heterogeneous) data distributions across clients.
  3.  Provide a practical, privacy-preserving framework suitable for cross-silo FL scenarios (e.g., hospitals, organizations).

### 2. Methodology & Key Innovations

The proposed **FedKD** framework combines two core techniques to achieve communication efficiency:

- **Federated Knowledge Distillation (KD) via Mentee-Mentor Architecture:**
  - Each client maintains a **large local mentor model** ($\Theta^t_i$) and a copy of a smaller **global mentee model** ($\Theta^s$).
  - **Communication reduction:** Only the small mentee model updates are exchanged between clients and the server, not the large mentor. The mentor is trained locally and never leaves the client.
  - **Adaptive Mutual Distillation:** The mentor and mentee models teach each other using three loss components:
    1.  **Task loss:** Standard supervised loss for the local task.
    2.  **Adaptive distillation loss (output logits):** KL divergence between the soft predictions of mentor and mentee, weighted inversely by the sum of their task losses.
    3.  **Adaptive hidden loss (representations):** MSE between hidden states and attention maps (if applicable) of the mentor and mentee, similarly weighted.
  - The adaptive weighting ensures reliable knowledge transfer: if either model makes poor predictions (high task loss), the distillation signal is weakened, preventing catastrophic forgetting or negative transfer.

- **Dynamic Gradient Compression via SVD:**
  - To further compress the communicated mentee gradients ($g_i$), FedKD applies **Singular Value Decomposition (SVD)** to factorize each gradient matrix: $g_i \approx U_i \Sigma_i V_i$.
  - **Dynamic Precision:** Rather than using a fixed rank $K$, FedKD uses a **dynamic energy threshold** $T(t)$ that increases linearly with the training progress $t$ (from $T_{start}$ to $T_{end}$). This retains fewer singular values (higher compression) in early rounds when gradients are low-rank and focused on coarse convergence, and retains more (lower compression but higher accuracy) in later rounds when gradients contain finer, high-frequency adjustments.
  - The server aggregates the reconstructed (uncompressed) gradients and re-factorizes the global gradient before sending it back to clients.

### 3. Mathematical Formulation

Let client $i$ have data $D_i$, mentor output distributions $y_i^t$, mentee output $y_i^s$, hidden states $H_i^t, H^s$, and attention maps $A_i^t, A^s$. Gold label: $y_i$.

**Local Mentor Loss on client $i$:**

$$
\mathcal{L}_{t,i} = \mathcal{L}_{t,i}^t + \mathcal{L}_{t,i}^d + \mathcal{L}_{t,i}^h
$$

**Local Mentee Loss on client $i$:**

$$
\mathcal{L}_{s,i} = \mathcal{L}_{s,i}^t + \mathcal{L}_{s,i}^d + \mathcal{L}_{s,i}^h
$$

where

**Task Loss (CE = Cross-Entropy):**

$$
\mathcal{L}_{t,i}^t = \text{CE}(y_i, y_i^t), \quad \mathcal{L}_{s,i}^t = \text{CE}(y_i, y_i^s) \quad (1), (2)
$$

**Adaptive Distillation Loss (KL = Kullback-Leibler divergence):**

$$
\mathcal{L}_{t,i}^d = \frac{\text{KL}(y_i^s, y_i^t)}{\mathcal{L}_{t,i}^t + \mathcal{L}_{s,i}^t}, \quad
\mathcal{L}_{s,i}^d = \frac{\text{KL}(y_i^t, y_i^s)}{\mathcal{L}_{t,i}^t + \mathcal{L}_{s,i}^t} \quad (3), (4)
$$

**Adaptive Hidden Loss (MSE = Mean Squared Error):**

$$
\mathcal{L}_{t,i}^h = \mathcal{L}_{s,i}^h = \frac{\text{MSE}(H_i^t, W_i^h H^s) + \text{MSE}(A_i^t, A^s)}{\mathcal{L}_{t,i}^t + \mathcal{L}_{s,i}^t} \quad (5)
$$

**SVD Gradient Compression – Energy Threshold:**

$$
\min_K \frac{\sum_{j=1}^K \sigma_j^2}{\sum_{j=1}^Q \sigma_j^2} > T \quad (8)
$$

where $\sigma_j$ are the singular values of gradient matrix $g_i \in \mathbb{R}^{P \times Q}$.

**Dynamic Threshold Scheduling:**

$$
T(t) = T_{start} + (T_{end} - T_{start}) \cdot t, \quad t \in [0,1] \quad (9)
$$

**Aggregation (FedAvg-style):**
Server receives $g_i$ from clients, reconstructs full matrices via SVD factors, then computes the sum:

$$
g = \sum_{i=1}^N g_i
$$

The global gradient sum $g$ is then factorized again using SVD (with the same $T(t)$) and distributed to clients. Each client updates its local mentee model by dividing by $N$ to obtain the average gradient:

$$
\Theta^s \leftarrow \Theta^s - \eta_s \cdot \frac{g}{N}
$$

_(Note: effectively this performs a standard average update, but the division by $N$ occurs on the client side during local updates rather than on the server)._

### 4. Limitations & Constraints

- **Statistical Assumptions:**
  - Requires sufficient local data on each client to train a reasonable mentor model; if local data is too scarce, mentor accuracy may be poor, hindering distillation.
  - Assumes labeled data is available on each client for the task loss.
- **System Assumptions:**
  - **Trusted Server & Secure Channels:** The method assumes the server is honest and communication channels are secure. If the server is malicious or there are eavesdroppers, model updates may leak private information.
  - **Computation Increase:** Clients must maintain and train both a large mentor and a small mentee. The paper states that this will only "slightly increase the computational cost" (rather than doubling it) because the mentee model is much smaller than the mentor model. However, it may still present a constraint for extremely low-resource edge devices.
  - **Cross-Silo Focus:** Designed for clients with moderate computational resources (e.g., hospitals, businesses), not for massive cross-device settings with thousands of weak mobile devices.
- **Communication Bottleneck Concerns:**
  - While mentee size and SVD reduce upload/download volume, the aggregation step still requires reconstructing full gradients on the server, which could become a memory bottleneck for extremely large models.

### 5. FedMAQ Thesis Relevance

- **Direct Baseline for FedMAQ:** The paper explicitly names its method **FedKD**, which serves as a strong baseline for the **FedMAQ** thesis (**Communication-Efficient FL via Multi-Adaptive Quantization and Knowledge Distillation**). FedKD demonstrates the effectiveness of combining knowledge distillation with gradient compression.
- **Integrable Techniques:**
  - **Adaptive Mutual Distillation:** The core idea of using small mentee models for communication and adaptively weighting distillation losses based on prediction correctness can be directly integrated into FedMAQ to improve model quality under high compression ratios.
  - **Dynamic SVD Compression:** The SVD-based compression with a dynamic energy threshold (Eq. 9) is a natural complement to quantization methods. FedMAQ could adopt a similar dynamic scheduling for its quantization bit-widths or combine quantization with SVD for multi-level compression.
  - **Mentee-Mentor Paradigm:** This architecture provides a principled way to separate model size (used locally for personalization) from communication size (small shared model), which FedMAQ can adopt as its base framework.

- **Comparison Point:** FedKD achieves up to **94.89% communication reduction** while matching centralized performance. This sets a high bar for FedMAQ, which can aim for similar or better compression ratios by introducing **multiple adaptive quantization strategies** (e.g., per-layer, per-round) that may outperform the single SVD-based approach of FedKD, especially on hardware that is quantization-friendly rather than SVD-friendly.

# Related

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](/papers/mcmahan-2017-fedavg.md)

# Citations

[1] Full-text conversion: [markdown/wu-2022-fedkd/paper.md](markdown/wu-2022-fedkd/paper.md)
[2] Source PDF: `papers/04 Q+KD/Wu et al. - 2022 - Communication-efficient federated learning via knowledge distillation.pdf`
