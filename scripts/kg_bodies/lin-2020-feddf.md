description: FedDF replaces parameter averaging with ensemble distillation on unlabeled/generated data, fusing heterogeneous client models (differing size, precision, or architecture) in far fewer rounds.

## 1. Overview & Objectives

- **Core Problem**: FedAvg fuses clients by **averaging parameters**, which (a) requires all clients to share identical model size/structure, and (b) produces "blurred" decision boundaries when client models are averaged in weight space under non-IID data.
- **Main Objectives**:
  - Propose **FedDF** (Federated Distillation Fusion): aggregate client knowledge via **ensemble distillation** — train the server model on the **averaged output logits** of client models over an **unlabeled or GAN-generated** dataset — instead of averaging weights.
  - Support **heterogeneous clients** that differ in model size, numerical precision, or architecture, which parameter averaging cannot handle.
  - Show faster convergence (fewer communication rounds) than existing FL methods across CV/NLP benchmarks (CIFAR-10/100, ImageNet, AG News, SST2).

## 2. Methodology & Key Innovations

- **Key Idea**: After receiving client models each round, the server treats them as an **ensemble of teachers** and distills their consensus into the (student) server model using **unlabeled proxy data**: forward each proxy sample through all client models, average their logits, and train the server model to match that averaged soft target.
- **Heterogeneity tolerance**: because fusion happens in **function/output space** (logits) rather than weight space, clients may have arbitrary architectures — only their outputs on the proxy set need to be comparable.
- **Proxy data**: uses either real unlabeled data or samples from a GAN generator, keeping the privacy/communication cost comparable to baseline FL.
- **BN robustness**: the distillation step also repairs the Batch-Normalization quality loss that afflicts weight-averaging even in homogeneous settings.

## 3. Mathematical Formulation

- **Ensemble soft target** on proxy input \(x\) from \(N\) received client models \(\{\theta_i\}\):

\[
\bar{p}(x) = \frac{1}{N}\sum_{i=1}^N \sigma\!\big(f_{\theta_i}(x)/T\big),
\]

with temperature \(T\).
- **Server distillation objective** (student parameters \(\theta_s\), initialized from the averaged model):

\[
\min_{\theta_s}\; \mathbb{E}_{x\sim \mathcal{D}_{proxy}}\; \mathrm{KL}\big(\bar{p}(x)\,\lVert\,\sigma(f_{\theta_s}(x)/T)\big).
\]

- Fusion is thus a few SGD steps of distillation on unlabeled data, run **after** (and seeded by) standard aggregation.

## 4. Limitations & Constraints

- **Requires proxy data**: FedDF needs an unlabeled dataset or a trained generator on the server; performance depends on the proxy distribution matching the task — a prerequisite not always satisfiable (the gap [FedGen](/papers/zhu-2021-fedgen.md) later removes).
- **Server-side compute**: distillation over an ensemble of client models adds server computation each round.
- **Refines global model only**: knowledge flows client→server; it does not directly regularize subsequent local training.
- **Soft-label transfer limits**: logits over proxy data convey less than full parameters when proxy coverage is poor.

## 5. FedMAQ Thesis Relevance

- **Foundational KD-fusion baseline**: FedDF is the canonical ensemble-distillation aggregator underpinning FedMAQ's knowledge-distillation component, and the reference point contrasted by data-free ([FedGen](/papers/zhu-2021-fedgen.md)) and quantized-distillation ([CFD](/papers/sattler-2022-cfd.md)) successors.
- **Enables precision/architecture heterogeneity**: crucially for FedMAQ, FedDF fuses clients that differ in **numerical precision** — exactly the situation multi-adaptive quantization creates when clients send updates at different bit-widths. Distillation-based fusion is a natural aggregator for heterogeneously-quantized clients where weight averaging breaks down.
- **Key insight to integrate**: fusing in output space sidesteps the weight-space blurring that quantization noise would worsen. FedMAQ can pair low-bit parameter updates with an FedDF-style distillation correction on the server to recover accuracy lost to aggressive quantization.
