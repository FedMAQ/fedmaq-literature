---
type: Paper
title: "Model-Contrastive Federated Learning"
description: "MOON corrects local drift in non-IID FL by model-level contrastive learning, maximizing agreement between local and global model representations."
authors: "Li et al."
year: 2021
bibkey: li-2021-moon
baseline: MOON
tags: [fl-core, heterogeneity]
resource: markdown/li-2021-moon/paper.md
timestamp: 2026-07-09T10:25:47Z
---

## 1. Overview & Objectives

- **Core Problem**: Under non-IID data, local training drifts each party's model away from the global objective, so the averaged model is sub-optimal. The authors observe that prior optimization-based fixes ([FedProx](/papers/li-2020-fedprox.md), [SCAFFOLD](/papers/karimireddy-2020-scaffold.md)) **fail on image datasets with deep networks**, sometimes performing no better than FedAvg.
- **Main Objectives**:
  - Address non-IID from a **representation-learning** perspective rather than a gradient/weight-space perspective.
  - Propose **MOON** (Model-Contrastive Federated Learning): use the similarity between the *global* model's and the *local* model's feature representations to regularize local training.
  - Deliver large accuracy gains on CIFAR-10/100 and Tiny-ImageNet with only lightweight modifications to FedAvg.

## 2. Methodology & Key Innovations

- **Key Idea**: The global model, having aggregated all parties, learns a **better representation** than any local model trained on skewed data. MOON therefore performs **contrastive learning at the model level**: pull the local model's representation of an input toward the global model's representation (positive pair), and push it away from the previous-round local model's representation (negative pair).
- **Three encoders per input** \(x\): current local model \(w_i^t\) → \(z\); global model \(w^t\) → \(z_{glob}\) (positive); previous local model \(w_i^{t-1}\) → \(z_{prev}\) (negative).
- **Local objective**: standard supervised loss + a **model-contrastive** term \(\ell_{con}\), weighted by \(\mu\):
  - \(\ell = \ell_{sup}(w_i; x, y) + \mu\, \ell_{con}(z, z_{glob}, z_{prev})\).
- **Practicality**: MOON is a drop-in modification to the FedAvg local step; it requires each client to keep its previous-round local model but changes nothing in aggregation or communication.

## 3. Mathematical Formulation

- **Model-contrastive loss** (NT-Xent form with temperature \(\tau\), cosine similarity \(\mathrm{sim}\)):

\[
\ell_{con} = -\log \frac{\exp(\mathrm{sim}(z, z_{glob})/\tau)}{\exp(\mathrm{sim}(z, z_{glob})/\tau) + \exp(\mathrm{sim}(z, z_{prev})/\tau)} .
\]

- **Total local loss**: \(\ell = \ell_{sup} + \mu\,\ell_{con}\), with \(\mu\) controlling the strength of representation alignment.
- **Reported gain**: on CIFAR-100 with 100 parties MOON reaches **61.8%** top-1 vs. ~55% for the best prior method — at least a 2% margin in most settings.

## 4. Limitations & Constraints

- **Extra compute/memory**: three forward passes per batch (local, global, previous-local) and storage of the previous local model increase client-side cost.
- **Vision-centric**: the representation-agreement premise is validated on image classification; benefit on other modalities/tasks is less established.
- **No payload compression**: MOON communicates full FedAvg-sized models; it improves accuracy per round but not bits per round.
- **Hyperparameter sensitivity**: the contrastive weight \(\mu\) and temperature \(\tau\) require tuning per dataset/heterogeneity level.

## 5. FedMAQ Thesis Relevance

- **Representation-space heterogeneity baseline**: MOON complements the optimization-space correctors [FedProx](/papers/li-2020-fedprox.md), [SCAFFOLD](/papers/karimireddy-2020-scaffold.md), and [FedDyn](/papers/acar-2021-feddyn.md), and its finding that those methods stall on deep vision models is important context for evaluating FedMAQ on image tasks.
- **KD adjacency**: model-level contrastive alignment is conceptually close to knowledge distillation (aligning student/teacher representations). FedMAQ's KD component can be viewed through the same lens, and MOON is a useful non-KD point of comparison for representation-based drift correction.
- **Key insight to integrate**: quantizing model updates perturbs the very representations MOON aligns. FedMAQ should verify that adaptive bit-width reduction does not erode representation agreement — suggesting representation-aware (rather than purely magnitude-aware) quantization criteria.

# Related

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](/papers/mcmahan-2017-fedavg.md)
- [Federated Optimization in Heterogeneous Networks](/papers/li-2020-fedprox.md)
- [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](/papers/karimireddy-2020-scaffold.md)
- [Federated Learning Based on Dynamic Regularization](/papers/acar-2021-feddyn.md)

# Citations

[1] Full-text conversion: [markdown/li-2021-moon/paper.md](markdown/li-2021-moon/paper.md)
[2] Source PDF: `papers/01 FL, Heterogeneity/Li et al. - 2021 - Model-Contrastive Federated Learning.pdf`
