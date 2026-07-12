---
type: Paper
title: "Communication-Efficient On-Device Machine Learning: Federated Distillation and Augmentation under Non-IID Private Data"
description: "Federated Distillation exchanges per-label averaged logits (payload independent of model size) and pairs it with GAN-based Federated Augmentation to counter non-IID skew, cutting communication ~26x versus FL."
authors: "Jeong et al."
year: 2023
bibkey: jeong-2023-feddistill-aug
tags: [kd]
resource: markdown/jeong-2023-feddistill-aug/paper.md
timestamp: 2026-07-09T10:35:40Z
---

## 1. Overview & Objectives

- **Core Problem**: On-device FL communication scales with **model size**, forbidding large models; and user data is **non-IID**, degrading accuracy (up to 11% MNIST, 51% CIFAR-10 vs. IID). Exchanging raw samples to fix skew is communication-heavy and privacy-leaking.
- **Main Objectives**:
  - Propose **Federated Distillation (FD)**: an online co-distillation scheme whose communication payload depends on the **model output dimension**, not the model size — drastically smaller for large models.
  - Propose **Federated Augmentation (FAug)**: each device helps train a shared **GAN** so it can locally augment its data toward an IID distribution, under a privacy-vs-communication trade-off.
  - Show FD+FAug achieves ~**26× less communication** while reaching **95–98%** of FL's test accuracy.

## 2. Methodology & Key Innovations

- **Federated Distillation (FD)**: instead of sharing model parameters, each device periodically uploads its **per-label average output logits** (its "local knowledge"). The server averages these into **global per-label logit means**, broadcast back as distillation targets. Each device trains with its supervised loss plus a distillation loss matching the global per-label logits.
  - Payload size ∝ (#labels × output dimension), **independent of network width/depth** — the key communication win for large models.
- **Federated Augmentation (FAug)**: devices identify labels they lack ("target" labels), upload a few seed samples to the server, which trains a **GAN generator**; the trained generator is downloaded so each device can synthesize the missing-class samples locally, **rectifying non-IID skew** before/during FD.
- **Synergy**: FAug makes each device's data closer to IID, which stabilizes the per-label logit statistics FD relies on.

## 3. Mathematical Formulation

- **Local training loss** (schematic), supervised term + distillation to global logits \(\bar{y}^{(\ell)}\) for label \(\ell\):

\[
\ell_{local} = \ell_{CE}(f_k(x), y) + \gamma\, \mathrm{KL}\big(\sigma(f_k(x)/T)\,\lVert\,\sigma(\bar{y}^{(y)}/T)\big),
\]

with temperature \(T\) and distillation weight \(\gamma\).
- **Global logit aggregation**: \(\bar{y}^{(\ell)} = \tfrac{1}{|\mathcal{K}_\ell|}\sum_{k \in \mathcal{K}_\ell} \bar{y}_k^{(\ell)}\), the mean over devices holding label \(\ell\).
- **Communication payload** per round ∝ \(|\mathcal{Y}| \times \dim(\text{output})\), vs. FL's ∝ \(|\theta|\).

## 4. Limitations & Constraints

- **Accuracy gap**: FD trades a modest accuracy drop (reaching 95–98% of FL) for large communication savings; it is not strictly lossless.
- **FAug privacy/communication trade-off**: uploading seed samples to train the GAN leaks some information and adds overhead; the balance must be tuned.
- **Logit-only knowledge**: sharing per-label output statistics conveys less than full parameters, which can limit performance on hard, high-capacity tasks.
- **Classification-centric**: the output-dimension payload argument is cleanest for classification; extension to other output structures is less direct.

## 5. FedMAQ Thesis Relevance

- **Distillation-based communication reduction**: FD is an early, canonical demonstration that exchanging **outputs instead of parameters** decouples communication from model size — the core premise of FedMAQ's knowledge-distillation component, later refined by [FedDF](/papers/lin-2020-feddf.md) and [CFD](/papers/sattler-2022-cfd.md). Note: a different, unrelated method also named "FedDistill" ([Song et al. 2024](/papers/song-2024-feddistill.md)) does not reduce communication and is not part of this lineage — see that paper's note for disambiguation.
- **Data-side heterogeneity handling**: FAug tackles non-IID by generative augmentation rather than optimization correction — a complementary lever to the [FedProx](/papers/li-2020-fedprox.md)/[SCAFFOLD](/papers/karimireddy-2020-scaffold.md) family, and a precursor to data-free generators like [FedGen](/papers/zhu-2021-fedgen.md).
- **Key insight to integrate**: FD's payload scales with output dimension while FedMAQ's quantization scales the *parameter* payload. FedMAQ can combine both — quantized parameter updates plus logit-level distillation — and use FD's per-label logit exchange as a cheap knowledge channel that is naturally robust to aggressive weight quantization.

# Related

- [Federated Optimization in Heterogeneous Networks](/papers/li-2020-fedprox.md)
- [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](/papers/karimireddy-2020-scaffold.md)
- [FedDistill: Global Model Distillation for Local Model De-Biasing in Non-IID Federated Learning](/papers/song-2024-feddistill.md)
- [CFD: Communication-Efficient Federated Distillation via Soft-Label Quantization and Delta Coding](/papers/sattler-2022-cfd.md)
- [Ensemble Distillation for Robust Model Fusion in Federated Learning](/papers/lin-2020-feddf.md)
- [Data-Free Knowledge Distillation for Heterogeneous Federated Learning](/papers/zhu-2021-fedgen.md)

# Citations

[1] Full-text conversion: [markdown/jeong-2023-feddistill-aug/paper.md](markdown/jeong-2023-feddistill-aug/paper.md)
[2] Source PDF: `papers/03 KD/Jeong et al. - 2023 - Communication-Efficient On-Device Machine Learning Federated Distillation and Augmentation under No.pdf`
