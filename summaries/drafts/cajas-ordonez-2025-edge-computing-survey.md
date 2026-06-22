# Research Summary: Cajas Ordóñez et al. (2025) – Intelligent Edge Computing and Machine Learning: A Survey

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

- **Post-training quantization** (INT8):  
  $$Q(x) = \text{round}\left(\frac{x}{\Delta}\right) \cdot \Delta, \quad \Delta = \frac{\max(x) - \