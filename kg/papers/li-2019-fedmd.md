---
type: Paper
title: "FedMD: Heterogenous Federated Learning via Model Distillation"
description: "Standard federated learning (e.g., FedAvg) requires all participants to share a single global model architecture."
authors: "Li and Wang"
year: 2019
bibkey: li-2019-fedmd
baseline: FedMD
tags: [kd]
resource: markdown/li-2019-fedmd/paper.md
timestamp: 2026-06-21T08:36:22Z
---

## 1. Overview & Objectives

**Core Problem:**
Standard federated learning (e.g., FedAvg) requires all participants to share a single global model architecture. This assumption fails in many real-world scenarios where each participant independently designs its own model due to intellectual property concerns, heterogeneous tasks, or varying computational resources. The paper addresses the challenge of enabling federated learning when each client has a _unique, black-box model_ and only private data.

**Main Objectives:**

- Propose a framework (FedMD) that allows participants with _different model architectures_ to collaboratively improve their local models without sharing data or model details.
- Leverage transfer learning and knowledge distillation to enable communication between heterogeneous models.
- Demonstrate that collaboration via FedMD yields significant accuracy gains over isolated training, approaching the performance of pooled-data training.

## 2. Methodology & Key Innovations

**Core Idea:**
FedMD replaces the traditional gradient/weight aggregation with a _knowledge distillation_ step on a public dataset. Each participant shares only the _output logits_ (class scores) on a public dataset, not gradients or model parameters. The server computes a consensus (average logits), and each client distills this consensus back into its own model.

**Key Components:**

1. **Transfer Learning Phase:** Each model is first trained on a large public dataset \(D_0\), then fine-tuned on its private dataset \(D_k\). This establishes a strong baseline.
2. **Communication Phase:** At each round, a random subset \(d_j \subset D_0\) is selected. Each client computes logits \(f_k(x)\) for \(x \in d_j\) and sends them to the server.
3. **Aggregation:** Server computes the average logits \(\tilde{f}(x) = \frac{1}{m} \sum\_{k=1}^m f_k(x)\).
4. **Digest Phase:** Each client trains its model to minimize the divergence between its own logits and the consensus \(\tilde{f}(x)\) on the public subset.
5. **Revisit Phase:** Each client fine-tunes on its private data for a few epochs to retain task-specific knowledge.

**Innovation:**

- Enables _full model heterogeneity_ – no shared architecture, no gradient exchange.
- Uses public dataset as a communication bridge, avoiding privacy leakage.
- Communication cost is controlled by using a small random subset of the public data each round.

## 3. Mathematical Formulation

**Notation:**

- \(m\) participants, each with private dataset \(D*k = \{(x_i^k, y_i)\}*{i=1}^{N*k}\) and public dataset \(D_0 = \{(x_i^0, y_i^0)\}*{i=1}^{N_0}\).
- Model \(f_k\) outputs logits (pre-softmax) for input \(x\).
- At round \(j\), a random subset \(d_j \subset D_0\) of size \(S\) is selected.

**Aggregation (Consensus):**
\[
\tilde{f}(x) = \frac{1}{m} \sum\_{k=1}^m f_k(x), \quad \forall x \in d_j
\]

**Digest Phase (Knowledge Distillation):**
Each client \(k\) updates its model by training it to "approach the consensus" on the public subset. The paper is agnostic to the specific loss function used for this knowledge transfer step. According to the paper's supplementary notes, either mean squared error (MSE) on raw logits or Kullback-Leibler (KL) divergence (with or without temperature scaling) can be used. As a representative formulation using MSE between logits:
\[
\mathcal{L}_{\text{distill}}^{(k)} = \frac{1}{|d_j|} \sum_{x \in d_j} \| f_k(x) - \tilde{f}(x) \|^2
\]

**Revisit Phase:**
Each client fine-tunes on its private data using standard cross-entropy loss:
\[
\mathcal{L}_{\text{private}}^{(k)} = -\frac{1}{|D_k|} \sum_{(x,y) \in D_k} \log \text{softmax}(f_k(x))\_y
\]

**Overall Algorithm (FedMD):**

```
1. Transfer learning: train f_k on D_0, then on D_k.
2. For round j = 1 to P:
   a. Server selects subset d_j ⊂ D_0.
   b. Each client computes logits f_k(x) for x ∈ d_j and sends to server.
   c. Server computes consensus: \tilde{f}(x) = (1/m) Σ_k f_k(x).
   d. Server broadcasts \tilde{f}(x) to all clients.
   e. Each client updates f_k by minimizing L_distill on d_j (Digest).
   f. Each client fine-tunes f_k on D_k for a few epochs (Revisit).
```

**Weighted Consensus (optional):**
\[
\tilde{f}(x) = \sum\_{k=1}^m c_k f_k(x), \quad \text{with } \sum_k c_k = 1
\]
The paper uses equal weights by default but notes that weaker models can be down-weighted.

## 4. Limitations & Constraints

**Assumptions:**

- A large public dataset \(D_0\) is available to all participants. This may not always be feasible (e.g., in highly sensitive domains).
- All models solve the same classification task (same label space). Extending to different tasks is not addressed.
- The public dataset must be representative enough to serve as a communication bridge; otherwise, knowledge transfer may be poor.

**Statistical Constraints:**

- Private datasets can be very small (e.g., 3 samples per class) – the method relies heavily on transfer learning from the public dataset.
- Non-i.i.d. data across clients is handled implicitly via the consensus on public data, but extreme heterogeneity (e.g., completely disjoint label sets) is not tested.

**System Constraints:**

- Communication cost: sending logits for a subset of public data each round. The paper uses 5000 samples per round, which is manageable but may still be large for low-bandwidth settings.
- No explicit compression or quantization of logits is considered – raw float32 logits are transmitted.
- The server must store and average logits from all clients; this could become a bottleneck with many participants.

**Limitations:**

- The framework does not guarantee convergence proofs; empirical results are shown only for small-scale experiments (10 participants).
- The digest phase requires each client to train on the public subset, which adds computational overhead.
- The method may suffer if some models are significantly weaker – the paper suggests down-weighting their contributions.

## 5. FedMAQ Thesis Relevance

**Connection to FedMAQ (Communication-Efficient FL via Multi-Adaptive Quantization and Knowledge Distillation):**
FedMD is a **direct baseline** for the knowledge distillation component of FedMAQ. It demonstrates that KD alone can enable heterogeneous federated learning without any gradient or weight sharing. Key points for integration:

- **Baseline Role:** FedMD can serve as a non-quantized, non-compressed baseline for FedMAQ. Any communication savings from quantization/compression in FedMAQ should be measured against the performance of FedMD (which already reduces communication by sending only logits instead of full model updates).
- **Techniques to Integrate:**
  - FedMD’s consensus aggregation (average logits) can be combined with **adaptive quantization** of logits before transmission to further reduce communication cost.
  - The digest phase (distillation loss) can be augmented with **multi-adaptive quantization** of the consensus before broadcasting, as proposed in FedMAQ.
  - The revisit phase (private fine-tuning) is orthogonal and can be retained.
- **Gap Addressed by FedMAQ:** FedMD does not address the communication burden of sending raw logits (e.g., 5000 samples × number of classes × 4 bytes). FedMAQ’s quantization and compression techniques can be applied to the logits transmitted in the communication phase, making the framework more communication-efficient while preserving the benefits of heterogeneous model support.

**Summary:** FedMD provides the foundational idea of using KD for heterogeneous FL. FedMAQ can build upon it by introducing multi-adaptive quantization of the communicated logits and the consensus, thereby reducing the communication overhead without sacrificing model heterogeneity or accuracy.

# Related

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](/papers/mcmahan-2017-fedavg.md)

# Citations

[1] Full-text conversion: [markdown/li-2019-fedmd/paper.md](markdown/li-2019-fedmd/paper.md)
[2] Source PDF: `papers/03 KD/Li and Wang - 2019 - FedMD Heterogenous Federated Learning via Model Distillation.pdf`
