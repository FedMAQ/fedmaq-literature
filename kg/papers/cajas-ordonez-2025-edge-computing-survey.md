---
type: Paper
title: "Intelligent Edge Computing and Machine Learning: A Survey of Optimization and Applications"
description: "This survey addresses the critical challenges of deploying machine learning (ML) models on resource-constrained edge devices within next-generation network infrastructures."
authors: "Cajas Ordóñez et al."
year: 2025
bibkey: cajas-ordonez-2025-edge-computing-survey
tags: [survey]
resource: markdown/cajas-ordonez-2025-edge-computing-survey/paper.md
timestamp: 2026-06-21T08:33:55Z
---

## 1. Overview & Objectives

This survey addresses the critical challenges of deploying machine learning (ML) models on resource-constrained edge devices within next-generation network infrastructures. The core problems include computational limitations, memory constraints, energy-efficiency requirements, and the need for real-time intelligent inference. The main objectives are to:

- Systematically review soft computing optimization strategies (pruning, quantization, knowledge distillation, low-rank decomposition) for edge deployment.
- Explore intelligent MLOps frameworks tailored for edge environments, including continuous model adaptation, monitoring under data drift, and federated learning (FL) for privacy-preserving distributed intelligence.
- Identify research gaps in multimodal model deployment, streaming learning under concept drift, and integration of soft computing with edge orchestration.

## 2. Methodology & Key Innovations

The paper is a comprehensive survey, not a novel method. It synthesizes existing techniques and categorizes them into:

- **Model Compression**: Pruning (structured/unstructured), quantization (post-training, quantization-aware training), knowledge distillation (response-based, feature-based, relational), low-rank factorization (LoRA, QLoRA, QA-LoRA, DoRA).
- **Federated Learning**: FedAvg, FedProx, FedPer, FedOpt, SplitFL, FedTL, FedEL.
- **MLOps Pillars**: Intelligent model deployment, monitoring, production deployment, and release preparation.
- **Edge-Specific Hardware**: CPUs, GPUs, NPUs, TPUs, FPGAs.

Key innovations highlighted include:

- **Hybrid quantization** (e.g., BitNet, BitNet b1.58) that replaces matrix multiplication with integer addition.
- **Parameter-Efficient Fine-Tuning** (PEFT) methods like LoRA and its variants for reducing trainable parameters.
- **Knowledge distillation** frameworks with teacher-student architectures for edge deployment.

## 3. Mathematical Formulation

The survey extracts formulas from the literature. Key mathematical expressions include:

### 3.1 Quantization

The survey outlines quantization as a key soft computing optimization technique that converts tensors from high-precision floating-point numbers (e.g., FP32) to lower-precision representations (e.g., INT8, INT4, INT2, or INT1/binary) to reduce the memory footprint and latency of edge inference.

- **Post-Training Quantization (PTQ):** Applies quantization directly to weights and activations after training, which is simpler but can result in accuracy degradation. Extreme low-bit implementations (such as BitNet variants) replace floating-point matrix multiplications with integer addition.
- **Quantization-Aware Training (QAT):** Simulates quantization noise during training, allowing the model to adapt and preserve accuracy more effectively.

### 3.2 Performance Evaluation Metrics

To evaluate edge AI systems, the survey details performance metrics across computational, resource, quality, and system-level dimensions:

- **Computational Metrics:**
  - **Latency ($L$):** Time elapsed from input arrival to output generation:
    $$L = t_{\text{end}} - t_{\text{start}}$$
  - **Throughput ($T$):** The number of operations processed per unit time:
    $$T = \frac{N_{\text{operations}}}{t_{\text{elapsed}}}$$

- **Resource Utilization Metrics:**
  - **Energy per Inference ($E_{\text{inference}}$):**
    $$E_{\text{inference}} = \frac{P_{\text{avg}} \times t_{\text{inference}}}{N_{\text{inferences}}}$$
    where $P_{\text{avg}}$ is the average power consumption.
  - **Memory Utilization ($M_{\text{util}}$):**
    $$M_{\text{util}} = \frac{M_{\text{used}}}{M_{\text{total}}} \times 100\%$$

- **Model Quality Metrics:**
  - **Classification Metrics:**
    $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
    $$\text{Precision} = \frac{TP}{TP + FP}$$
    $$\text{Recall} = \frac{TP}{TP + FN}$$
    $$\text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **System-Level Metrics:**
  - **Availability:**
    $$\text{Availability} = \frac{\text{MTBF}}{\text{MTBF} + \text{MTTR}} \times 100\%$$
    where $\text{MTBF}$ is mean time between failures and $\text{MTTR}$ is mean time to repair.

### 3.3 Federated Learning Formulations

Federated Learning (FL) is presented as a paradigm to train models collaboratively on decentralized edge nodes while preserving data privacy. The survey highlights several aggregation and personalization formulations:

- **FedAvg:** Aggregates local client updates to update the global server model.
- **FedProx:** Introduces a regularization term to restrict local updates close to the global model, addressing heterogeneous environments.
- **FedPer / personalized FL:** Trains personalized local layers alongside shared base layers to handle client-specific data.
- **Other variants:** FedOpt, SplitFL, FedTL, and FedEL.

## 4. Limitations & Constraints

- **Statistical Assumptions:** Edge deployments assume data is highly heterogeneous (non-IID) and susceptible to temporal fluctuations (concept drift/data drift). Statistical drift detection requires continuous window-based monitoring.
- **System Constraints:** Real-time edge inference is constrained by battery life (energy consumption), limited local memory, and communication bandwidth. Edge hardware accelerators (NPUs, TPUs, FPGAs) must support low-precision arithmetic (e.g., INT4/INT8) to achieve target latencies.
- **Security & Privacy:** Privacy protection mechanisms (such as hashing encryption) must balance communication overhead with security guarantees under spatial-temporal exposure risks.

## 5. FedMAQ Thesis Relevance

- **Baseline Context:** This survey serves as a comprehensive reference mapping the landscape of edge AI optimization. It systematically defines the broader family of compression strategies (pruning, quantization, knowledge distillation), of which [FedMAQ](/methods/fedmaq.md) combines only quantization and knowledge distillation — pruning/sparsification is explicitly out of scope.
- **Integration Potential:** The evaluation metrics (latency, throughput, energy per inference, memory utilization) defined in the paper represent the exact target optimization parameters for FedMAQ. Furthermore, the survey's discussion of MLOps pipelines under drift and federated learning variants validates the multi-adaptive approach of FedMAQ in heterogeneous and dynamic environments.

# Related

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](/papers/mcmahan-2017-fedavg.md)
- [Federated Optimization in Heterogeneous Networks](/papers/li-2020-fedprox.md)

# Citations

[1] Full-text conversion: [markdown/cajas-ordonez-2025-edge-computing-survey/paper.md](markdown/cajas-ordonez-2025-edge-computing-survey/paper.md)
[2] Source PDF: `papers/00 Surveys/Cajas Ordóñez et al. - 2025 - Intelligent Edge Computing and Machine Learning A Survey of Optimization and Applications.pdf`
