---
type: Paper
title: "Communication-Efficient Federated Learning for Power Load Forecasting in Electric IoTs"
description: "Frequent communication of high-dimensional model updates between heterogeneous clients and a central server in Federated Learning (FL) leads to prohibitively high communication costs and privacy risks, especially in..."
authors: "Mao et al."
year: 2023
bibkey: mao-2023-power-load
tags: [application]
resource: markdown/mao-2023-power-load/paper.md
timestamp: 2026-06-21T08:45:59Z
---

### 1. Overview & Objectives

**Core Problem:**  
Frequent communication of high-dimensional model updates between heterogeneous clients and a central server in Federated Learning (FL) leads to prohibitively high communication costs and privacy risks, especially in power load forecasting for Electric Internet of Things (IoTs). Existing methods reduce communication only from one dimension (either compression or communication frequency) and ignore the downloading stage, while compression and lazy upload introduce biases that degrade model utility.

**Main Objectives:**  
- Propose a **systematic** communication-efficient FL algorithm that reduces communication cost from **both** per-epoch transmission bits **and** transmission frequency.  
- Achieve bidirectional compression (uplink and downlink) and lazy upload.  
- Integrate an **error compensation** strategy to mitigate the bias from compression and lazy upload, preserving model utility.  
- Validate the method on a real-world power load dataset, achieving at least 60% reduction in communication cost with controlled accuracy loss.

---

### 2. Methodology & Key Innovations

**Proposed Algorithm: CMULA-FL (Compressed Model Updates and Lazy Upload – FL)**  
The algorithm consists of three core components working on both client and server sides:

1. **Bidirectional Model Quantization (γ-compression operator):**  
   Both client-to-server (upload) and server-to-client (download) model updates are compressed using a multi-bits quantization operator that satisfies the γ-compression property. This reduces per-epoch communication cost on both directions.

2. **Lazy Upload (Communication Frequency Reduction):**  
   Each client computes a local model update after \(E\) local iterations. The client uploads only if the norm of the quantized update exceeds a threshold \(\epsilon\) or if a maximum staleness \(t_{\text{max}}\) is reached. Otherwise, the update is skipped and accumulated for the next epoch, reducing uplink frequency.

3. **Error Compensation:**  
   Both clients and server accumulate quantization errors from the previous epoch and add them to the current update before compression. This compensates for the bias introduced by compression and lazy upload, improving model convergence and accuracy.

**System Workflow:**  
- **Server:** Initializes model; broadcasts quantized model update; collects quantized updates from clients; aggregates with previous server error; quantizes the aggregated update and broadcasts it.  
- **Client:** Receives model update; trains locally for \(E\) iterations; computes update and adds accumulated error; quantizes the corrected update; applies lazy upload condition; if uploaded, computes new quantization error; otherwise, accumulates lazy upload error.

---

### 3. Mathematical Formulation

**Global objective (standard FL):**  
\[
\min_w F(w) = \frac{1}{N} \sum_{\xi_i \in \mathcal{D}} f(w, \xi_i)
\]

**γ-Compression Operator (Definition 1):**  
If a compression operator \(Q\) satisfies  
\[
\|x - Q(x)\|^2 \leq (1 - \gamma) \|x\|^2,
\]  
then \(Q\) is a γ-compression operator.

**Quantization of updates:**  
- Client upload: \(\Delta \tilde{w}_{i,j} = Q(\Delta \bar{w}_{i,j})\)  
- Server download: \(\Delta \tilde{w}_j = Q(\Delta w_j)\)

**Error compensation on client side:**  
\[
\begin{aligned}
\text{err}_{i,j-1} &= \Delta \tilde{w}_{i,j-1} - Q(\Delta \bar{w}_{i,j-1}), \\
\Delta \bar{w}_{i,j} &= \Delta w_{i,j} + \text{err}_{i,j-1}.
\end{aligned}
\]

**Error compensation on server side:**  
\[
\begin{aligned}
\text{err}_{j-1} &= \Delta \tilde{w}_{j-1} - Q(\Delta w_{j-1}), \\
\Delta w_j &= \text{Agg}(\Delta \tilde{w}_{i,j}) + \text{err}_{j-1}.
\end{aligned}
\]

**Lazy upload condition (client side):**  
\[
\text{if } \| \Delta \tilde{w}_{i,j} \| \geq \epsilon \quad \text{or} \quad t_i \geq t_{\text{max}}, \text{ then upload,}
\]  
otherwise accumulate lazy upload error:  
\[
\text{err}_{i,j} = w_{i,j}^E - w_{i,j}^0 + \text{err}_{i,j-1}.
\]

**Model aggregation rule (server):**  
\[
\Delta w_j = \frac{1}{S_j} \sum_{i \in S_j} \Delta \tilde{w}_{i,j} + \text{err}_{j-1},
\]  
where \(S_j\) is the set of clients that uploaded in epoch \(j\).

**Communication cost metric:**  
\[
\mathcal{C}_{\text{tot}} = \sum_{j=1}^J \sum_{i \in S_j} \sum_{k=1}^{M_{\text{num}}} \mathcal{C}(\Delta \widetilde{w}_{i,j,k}),
\]  
where \(\mathcal{C}(\cdot)\) is the number of bits for the \(k\)-th parameter.

---

### 4. Limitations & Constraints

- **Hyperparameter sensitivity:** The method introduces several hyperparameters (\(\epsilon\), \(t_{\text{max}}\), quantization bits) that are set empirically; tuning is difficult and may require domain knowledge.
- **No theoretical convergence analysis:** The authors did not provide formal convergence guarantees or bounds on the error accumulation, relying only on empirical validation.
- **Dataset and model specificity:** Experiments are conducted on a single power load dataset using LSTM; generalizability to other models (e.g., CNNs, transformers) or non-IID data distributions is not tested.
- **Neglect of client heterogeneity:** The algorithm assumes symmetric communication bandwidth but does not explicitly address heterogeneous client compute or network resources aside from a binary upload decision.
- **Download cost reduction only via quantization:** While server broadcasts are quantized, the download frequency is not reduced (e.g., no selective broadcasting).

---

### 5. FedMAQ Thesis Relevance

**Connection to FedMAQ (Communication-Efficient FL via Multi-Adaptive Quantization and Knowledge Distillation):**  
- **Baseline potential:** CMULA-FL can serve as a **strong baseline** for techniques using **bidirectional quantization** and **error compensation**. However, it **does not utilize knowledge distillation** (KD), which is a core component of FedMAQ.  
- **Integration possibilities:**  
  - The **error compensation scheme** for both client and server directly applies to FedMAQ to mitigate bias from aggressive compression.  
  - The **lazy upload mechanism** (norm threshold + staleness) is a candidate for reducing communication rounds in FedMAQ, possibly combined with KD-based soft label exchange.  
  - FedMAQ’s **multi-adaptive quantization** could be built upon CMULA-FL by dynamically adjusting compression ratios based on update importance, extending the fixed γ‑compression approach.  
- **Comparison gap:** CMULA-FL lacks KD, so it cannot leverage model‑level knowledge transfer; hence FedMAQ would likely achieve better utility under severe communication budgets by using a KD loss. CMULA-FL is a complementary work focusing solely on compression and upload frequency.

# Related

- [Advancing Electric Load Forecasting: Leveraging Federated Learning for Distributed, Non-Stationary, and Discontinuous Time Series](/papers/richter-2024-electric-load.md)
- [Air Quality Prediction Using Communication-Efficient Federated Learning with Compressed Deep Learning Models](/papers/joseph-2026-air-quality.md)
- [FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization](/papers/reisizadeh-2020-fedpaq.md)

# Citations

[1] Full-text conversion: [markdown/mao-2023-power-load/paper.md](markdown/mao-2023-power-load/paper.md)
[2] Source PDF: `papers/05 Applications/Mao et al. - 2023 - Communication-Efficient Federated Learning for Power Load Forecasting in Electric IoTs.pdf`
