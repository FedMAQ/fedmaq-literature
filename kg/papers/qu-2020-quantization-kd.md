---
type: Paper
title: "Quantization and Knowledge Distillation for Efficient Federated Learning on Edge Devices"
description: "Communication is a critical bottleneck in federated learning (FL) due to the large size of deep neural network weights and gradients that must be transmitted between clients and server, especially over heterogeneous..."
authors: "Qu et al."
year: 2020
bibkey: qu-2020-quantization-kd
tags: [quantization, kd]
resource: markdown/qu-2020-quantization-kd/paper.md
timestamp: 2026-06-21T08:52:21Z
---

## 1. Overview & Objectives

**Core Problem:** Communication is a critical bottleneck in federated learning (FL) due to the large size of deep neural network weights and gradients that must be transmitted between clients and server, especially over heterogeneous edge devices with limited bandwidth.

**Main Objectives:**
- Reduce communication costs in FL by dynamically quantizing neural network gradients based on client bandwidth heterogeneity.
- Achieve high-quality small models with limited labeled data by combining federated learning with knowledge distillation using unlabeled data.
- Provide a compression ratio of 4×–16× while retaining accuracy within 2% of the full-precision model.
- Enable federated knowledge distillation to reach target accuracy (e.g., 83.35%) using only 30% labeled data.

**Key Contribution:** Integration of adaptive quantization and federated knowledge distillation into a single efficient FL framework.

---

## 2. Methodology & Key Innovations

The paper proposes a two-phase approach:

1. **Adaptive Quantized Federated Average (AQFedAvg) Algorithm**  
   - Extends QSGD (Alistarh et al., 2017) and FedAvg (McMahan et al., 2016).
   - Stochastically quantizes client-to-server gradients in an unbiased manner.
   - The quantization level (number of bits) is automatically tuned per device based on its communication bandwidth \(\phi_i\) and the maximum bandwidth \(\phi_{\text{max}}\).
   - The number of participating clients \(K\) is also adaptively determined to respect a bandwidth threshold \(T\).

2. **Federated Knowledge Distillation (FKD)**  
   - Uses the large model trained via AQFedAvg as a teacher.
   - Student models (smaller networks) are trained on unlabeled data by mimicking the teacher's logits.
   - The distillation loss is combined with supervised cross-entropy loss on a small labeled subset.
   - Teacher model is distributed to unlabeled clients once per round (no teacher update during distillation).

**System Model:**
- Central server + \(K\) clients.
- Each client has local data (IID or Non-IID).
- Teacher model trained with AQFedAvg → then student models trained via FKD.
- Only a fraction of clients participate per round (controlled by \(C\)).

---

## 3. Mathematical Formulation

### Federated Learning Objective (Equation 1)

\[
\min_{w \in \mathbb{R}^d} f(w) \quad \text{where} \quad f(w) \stackrel{\text{def}}{=} \sum_{k=1}^K \frac{n_k}{n} \, F_k(w)
\]

with local objective:
\[
F_k(w) = \frac{1}{n_k} \sum_{i=1}^{n_k} f_i(w)
\]

### Stochastic Quantization Function (Equations 4 & 5)

For any non-zero vector \(v \in \mathbb{R}^n\):

\[
Q_s(v_i) = \|v\|_2 \cdot \operatorname{sgn}(v_i) \cdot \xi_i(v, s)
\]

where \(\xi_i(v, s)\) is an independent stochastic rounding function:

\[
\xi_i(v, s) = \begin{cases} 
l / s & \text{with probability } 1 - p\!\left(\frac{|v_i|}{\|v\|_2}, s\right) \\
(l+1)/s & \text{otherwise}
\end{cases}
\]

Here \(p(a, s) = a s - l\) for \(a \in [0,1]\), and \(l = \lfloor a s \rfloor\). This is unbiased.

### Adaptive Quantization Level (Equation 6)

Quantization bits for client \(i\):

\[
s_i = 2^{\lceil \log_2(\phi_{\text{max}} / \phi_i) \rceil}
\]

where \(\phi_i\) is the bandwidth of the \(i\)-th device and \(\phi_{\text{max}}\) is the maximum bandwidth.

### Adaptive Client Selection (Equation 7)

The number of participating clients \(K\) satisfies:

\[
\sum_{i=0}^K \phi_i \leq T \leq \sum_{i=0}^{K+1} \phi_i
\]

where \(T\) is a bandwidth threshold. Implemented greedily.

### Federated Knowledge Distillation Loss (Equation 8)

\[
\begin{aligned}
L &= \alpha L_s + (1-\alpha) L_{\text{KD}} \\
  &= -\alpha \, t_i \log y_i^S - (1-\alpha) \|z^T - z^S\|_2^2
\end{aligned}
\]

- \(L_s\): cross-entropy on labeled data (with logit vector \(t_i\) as target, \(y_i^S\) as student prediction).
- \(L_{\text{KD}}\): L2 loss between teacher logits \(z^T\) and student logits \(z^S\).
- \(\alpha\): ratio of labeled data (controls the trade-off).

### Federated Knowledge Distillation Algorithm (Algorithm 1)

Key steps:
- Server distributes teacher model to unlabeled clients (line 3).
- At each round \(t\):
  - Randomly select \(\alpha C\) labeled clients and \((1-\alpha)C\) unlabeled clients.
  - Labeled clients perform local SGD on supervised loss.
  - Unlabeled clients perform local SGD using distillation loss \(L_{\text{distill}} = \|z^T - z^S\|_2^2\).
- Server aggregates all client weights: \(w^{t+1} = \sum_{k=1}^K \frac{n_k}{n} w_k^{t+1}\).

---

## 4. Limitations & Constraints

**Statistical Assumptions:**
- The paper assumes unlabeled data is readily available and can be easily fetched, but does not discuss how unlabeled data is distributed or its quality.
- IID and Non-IID settings are tested, but Non-IID performance gains from FKD are smaller than IID gains (e.g., LeNet: 15.5% gain on Non-IID vs 9.7% on IID – still significant but not analyzed in depth).

**System Assumptions:**
- A centralized server is assumed; no peer-to-peer or decentralized aggregation.
- The teacher model must be trained first (two-stage process) – no joint training of teacher and student.
- The adaptive quantization only applies to client-to-server traffic; server-to-client traffic uses low-precision but is not detailed.

**Communication Bottleneck Concerns:**
- Compression ratio is achieved only for upstream gradients; downstream model distribution still costs full precision unless separately compressed.
- The threshold \(T\) for client selection is not adaptive; may need tuning.

**Other Limitations:**
- No privacy guarantees beyond standard FL; no differential privacy or secure aggregation discussed.
- The distillation relies on a single teacher; ensemble or multi-teacher scenarios are not considered.

---

## 5. FedMAQ Thesis Relevance

This paper is highly relevant to **FedMAQ (Communication-Efficient FL via Multi-Adaptive Quantization and Knowledge Distillation)** as both focus on exactly the same problem: reducing communication cost in FL via quantization and knowledge distillation.

### Can it serve as a baseline?
- **Yes.** The adaptive quantization scheme (AQFedAvg) and the federated knowledge distillation (FKD) provide a direct baseline for FedMAQ. The performance numbers (compression ratio 4×–16×, accuracy within 2%, 30% labeled data reaching full-supervision accuracy) are benchmarks that FedMAQ can be compared against.

### Integration potential:
- The adaptive quantization formula (Equation 6) mapping bandwidth to quantization bits can be integrated into FedMAQ's multi-adaptive component.
- The distillation loss (Equation 8) and the FKD algorithm (Algorithm 1) can be adopted with modifications (e.g., multi-teacher, asynchronous distillation).
- The idea of using a teacher trained with quantized FL to then distill smaller models is directly usable in FedMAQ's pipeline.

### Gaps addressed by FedMAQ:
- FedMAQ could extend this work by supporting multiple quantization levels per layer, dynamic switching between quantization and sparsification, or joint training of teacher and student.
- FedMAQ could also address security and Non-IID robustness more thoroughly.

**Conclusion:** Qu et al. (2020) provides a foundational approach that FedMAQ builds upon and improves, making it both a strong baseline and a source of integrable techniques.

# Related

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](/papers/mcmahan-2017-fedavg.md)
- [QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding](/papers/alistarh-2017-qsgd.md)

# Citations

[1] Full-text conversion: [markdown/qu-2020-quantization-kd/paper.md](markdown/qu-2020-quantization-kd/paper.md)
[2] Source PDF: `papers/04 Q+KD/Qu et al. - 2020 - Quantization and Knowledge Distillation for Efficient Federated Learning on Edge Devices.pdf`
