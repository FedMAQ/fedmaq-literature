---
type: Paper
title: "FedDT: A Communication-Efficient Federated Learning via Knowledge Distillation and Ternary Compression"
description: "Federated learning (FL) suffers from three major challenges: (i) data heterogeneity (non-IID distributions) causing model drift and performance degradation; (ii) model heterogeneity (different client architectures)..."
authors: "He et al."
year: 2025
bibkey: he-2025-feddt
tags: [quantization, kd, sota]
resource: markdown/he-2025-feddt/paper.md
timestamp: 2026-06-21T08:20:32Z
---

### 1. Overview & Objectives

**Core Problem:** Federated learning (FL) suffers from three major challenges: (i) **data heterogeneity** (non-IID distributions) causing model drift and performance degradation; (ii) **model heterogeneity** (different client architectures) creating knowledge transfer barriers; and (iii) **excessive communication overhead** due to frequent exchange of large model parameters.

**Main Objectives:**  
- Mitigate the negative impact of data heterogeneity through personalized teacher models.  
- Resolve model heterogeneity by using a shared student model (the global model) as a common intermediary.  
- Reduce communication costs via a two-level compression strategy combining knowledge distillation (KD) and ternary quantization.

### 2. Methodology & Key Innovations

**System Model:**  
- **Clients:** Each client owns a private dataset \(D_i\), a personalized heterogeneous teacher model \(T_i\) (parameters \(\Theta_i^t\)), and a local copy of the global student model \(S\) (parameters \(\Theta^s\)).  
- **Server:** Aggregates quantized student models and broadcasts the updated global model.

**Key Innovations:**  
1. **Personalized Federated Distillation:** Each client pre-trains a teacher model on its local data. Knowledge is transferred from the teacher to the shared student model via an adaptive distillation loss that dynamically weights task, distillation, and hidden-state losses.  
2. **Homogeneous Student Model:** The server-initialized global model serves as a unified student across all clients, masking local architectural variations and enabling consistent aggregation.  
3. **Two-Level Compression:**  
   - **Level 1 (KD):** The student model is trained via distillation, reducing the parameter size compared to the teacher.  
   - **Level 2 (Ternary Quantization):** The distilled student model is quantized layer-by-layer into ternary values \(\{-1, 0, +1\}\) using an adaptive threshold, drastically reducing communication volume.

**Training Process (per round):**  
- Clients download the global student model.  
- Local distillation updates both teacher and student models using the composite loss.  
- Student model is quantized via TTQ (Trained Ternary Quantization).  
- Quantized models are uploaded to the server.  
- Server aggregates (weighted average) and re-quantizes the global model before broadcasting.

### 3. Mathematical Formulation

**Softmax with Temperature (Definition 3):**  
\[
q_j(T, w) = \frac{\exp(z_j(w)/T)}{\sum_j \exp(z_j(w)/T)}
\]

**Cross-Entropy Loss (Definition 4):**  
\[
CE(a, b) = -\sum_i a_i \log(b_i)
\]

**KL Divergence (Definition 5):**  
\[
D_{KL}(P \| Q) = \sum_i P(i) \log\frac{P(i)}{Q(i)}
\]

**Adaptive Distillation Loss (Equations 5–10):**  
- Task loss:  
  \[
  F_{(t,i)}^t = CE(y_i, y_i^t), \quad F_{(s,i)}^t = CE(y_i, y_i^s)
  \]
- Distillation loss (adaptive weight):  
  \[
  F_{(t,i)}^d = \frac{KL(y_i^s, y_i^t)}{F_{(t,i)}^t + F_{(s,i)}^t}, \quad
  F_{(s,i)}^d = \frac{KL(y_i^t, y_i^s)}{F_{(t,i)}^t + F_{(s,i)}^t}
  \]
- Adaptive hidden loss (MSE on hidden states and attention maps):  
  \[
  F_{(t,i)}^h = F_{(s,i)}^h = \frac{MSE(H_i^t, W_i^h H^s) + MSE(A_i^t, A^s)}{F_{(t,i)}^t + F_{(s,i)}^t}
  \]
- Total loss for teacher and student:  
  \[
  F_{(t,i)} = F_{(t,i)}^d + F_{(t,i)}^h + F_{(t,i)}^t, \quad
  F_{(s,i)} = F_{(s,i)}^d + F_{(s,i)}^h + F_{(s,i)}^t
  \]

**Ternary Quantization (TTQ) – Equations 13–22:**  
- Normalization: \(\Theta^s = g(\Theta)\)  
- Adaptive threshold: \(\Delta = \frac{T_k}{d^2} \sum_{i=1}^{d^2} |\Theta_i^s|\) with \(T_k\) from a random schedule.  
- Mask: \(\text{mask}(\Theta^s) = \varepsilon(|\Theta^s| - \Delta)\)  
- Ternary weights: \(I^t = \text{sign}(\text{mask} \odot \Theta^s)\)  
- Quantized output: \(\Theta^s = \omega^q \times I^t\)  
- Positive/negative scaling factors:  
  \[
  w_p = \frac{1}{|I_p|}\sum_{i \in I_p} \Theta_i, \quad w_n = -\frac{1}{|I_n|}\sum_{j \in I_n} \Theta_j
  \]

**Server Aggregation (Equation 23):**  
\[
\Theta_r \leftarrow \sum_{k=1}^{\lambda N} \frac{|D_k|}{\sum_{k=1}^{\lambda N} |D_k|} \Theta_{k,r}^s
\]

**Server Re-quantization (Equations 24–25):**  
- Threshold: \(\Delta_s = 0.05 \times \max(|\Theta_r|)\)  
- Quantized global model: \(\Theta_r \leftarrow \omega_p \times I_p - \omega_n \times I_n\)

**Convergence Rate (Proposition 3):**  
Under L-smooth and μ-strongly convex assumptions, FedDT achieves \(O(1/NR)\) convergence, matching FedAvg.

### 4. Limitations & Constraints

- **Statistical Assumptions:** The unbiasedness proof (Proposition 2) assumes weights are uniformly distributed in \([-1,1]\) after normalization. This may not hold for all deep networks.  
- **Convergence Proof:** Assumes L-smooth and μ-strongly convex loss functions, which is rarely true for neural networks. The proof relies on results from Qu et al. [49] and may not generalize to non-convex settings.  
- **System Constraints:**  
  - Each client must pre-train a personalized teacher model, adding local computation overhead.  
  - The student model size still determines communication cost; ternary quantization reduces it but introduces lossy compression.  
  - The method assumes all clients can host both a teacher and a student model, which may be challenging for extremely resource-constrained devices.  
- **Communication Bottleneck:** While reduced, the server still broadcasts a quantized global model; the compression ratio depends on the student model architecture and quantization granularity.

### 5. FedMAQ Thesis Relevance

FedDT is directly relevant to the FedMAQ thesis (**Communication-Efficient FL via Multi-Adaptive Quantization and Knowledge Distillation**) as it combines both KD and quantization for communication efficiency.

- **Baseline Comparison:** FedDT can serve as a strong baseline for FedMAQ. It demonstrates that a two-level compression (KD + ternary quantization) achieves ~78% communication reduction with 7.85% accuracy improvement over baselines.  
- **Techniques for Integration:**  
  - The **adaptive distillation loss** (dynamic weighting of task, distillation, and hidden losses) could be incorporated into FedMAQ’s KD module to handle non-IID data more robustly.  
  - The **ternary quantization scheme** (TTQ) with layer-wise adaptive thresholds is FedDT's own strategy; [FedMAQ](/methods/fedmaq.md) instead varies bit-width per client per round from resource, training-state, and data-richness signals, with no layer axis.
  - The **server-side re-quantization** step ensures consistency between aggregated and broadcast models, a design choice relevant for FedMAQ's aggregation.  
- **Gap:** FedDT uses a single quantization level (ternary) and a fixed student model. FedMAQ instead uses multiple adaptive quantization levels per client per round (not per layer), selected via its formulation study.

# Related

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](/papers/mcmahan-2017-fedavg.md)

# Citations

[1] Full-text conversion: [markdown/he-2025-feddt/paper.md](markdown/he-2025-feddt/paper.md)
[2] Source PDF: `papers/04 Q+KD/He et al. - 2025 - FedDT A Communication-Efficient Federated Learning via Knowledge Distillation and Ternary Compressi.pdf`
