---
type: Paper
title: "FedGEMS: Federated Learning of Larger Server Models via Selective Knowledge Fusion"
description: "FedGEMS trains a larger-capacity server model in FL via selective, entropy-weighted ensemble distillation from clients plus server self-distillation, improving both server and client accuracy, robustness, and communication cost."
authors: "Cheng et al."
year: 2021
bibkey: cheng-2021-fedgems
baseline: FedGEMS
tags: [kd, sota]
resource: markdown/cheng-2021-fedgems/paper.md
timestamp: 2026-07-24T14:37:00Z
---

## 1. Overview & Objectives

- **Core Problem**: Standard FL (FedAvg) caps model capacity at what resource-constrained edge clients can train, and cannot exploit the server's own comparatively abundant compute — the server merely aggregates client parameters instead of learning a strictly larger model (GEM, "larger server model").
- **Main Objectives**:
  - Propose **FedGEM**, a paradigm where the server trains a model larger than any client's, learning consensus knowledge fused from multiple client teachers via logit-based knowledge distillation on a shared public dataset, and transfers knowledge back to clients.
  - Propose **FedGEMS**, which adds a **selective, confidence-weighted knowledge-fusion criterion** so the server ignores unreliable/malicious clients and weights reliable ones by prediction confidence, preventing negative knowledge transfer.
  - Show the combined framework simultaneously improves server accuracy, client accuracy, robustness to poisoning attacks, and communication cost relative to FedAvg, FedMD, Cronus, DS-FL, FedDF, and FedGKT baselines.

## 2. Methodology & Key Innovations

- **Setup**: \(K\) clients each hold a private labeled dataset \(X_k\) and train a local model \(f_c^k\) (architecture may be heterogeneous across clients); a public dataset \(X_0\) is accessible to server and clients; the server trains a strictly larger model \(f_s\).
- **FedGEM (base algorithm)**: each round, clients train locally then send predicted logits on the public dataset to the server; the server aggregates and trains \(f_s\) guided by the fused client logits, then broadcasts its own logits back so clients distill from the server and continue local training — a bidirectional, alternating mutual-distillation loop.
- **Selective knowledge fusion (FedGEMS, Sec. 3.3)** has three regimes per public sample \(x^i\):
  - **Self-training** (\(\mathcal{L}_{S_1}\)): if the server already predicts \(x^i\) correctly, train with plain cross-entropy and cache the correct logit into a global logit pool \(l^i_{Global}\).
  - **Self-distillation** (\(\mathcal{L}_{S_2}\)): if the server predicts \(x^i\) wrongly but a cached \(l^i_{Global}\) exists, distill from its own stored logit rather than re-querying clients — this is what keeps communication low over training.
  - **Selective ensemble distillation** (\(\mathcal{L}_{S_3}\)): only for samples with no cached logit, discard clients that mispredict \(x^i\) (weight 0) and weight the remaining "reliable" clients by \(\alpha_{C_j} = \mathrm{softmax}(1/H(p_{C_j}))\), i.e. inverse prediction entropy (confidence).
- **Client-side digestion**: each client distills from the server's broadcast logits on the public data, then continues cross-entropy training on its own private data.

## 3. Mathematical Formulation

- Self-training loss: \(\mathcal{L}_{S_1} = \mathcal{L}_{CE} = -y^i\log(f_s(x^i))\).
- Self-distillation loss: \(\mathcal{L}_{S_2} = \epsilon\,\mathcal{L}_{CE} + (1-\epsilon)\,D_{KL}\big(l^i_{Global}\,\|\,l^i_s\big)\).
- Selective ensemble distillation loss: \(\mathcal{L}_{S_3} = \epsilon\,\mathcal{L}_{CE} + (1-\epsilon)\,D_{KL}\Big(\sum_{j=1}^K \alpha_{C_j} l_{C_j}\,\Big\|\,l_s\Big)\), with entropy \(H(p_{C_j}) = -\sum_i p(x_i)\log p(x_i)\) and \(\alpha_{C_j}=0\) for clients that mispredict the ground truth.
- \(\epsilon\) (best value found empirically: 0.75) trades off cross-entropy against distillation across all three losses; this is the single global hyperparameter controlling how much the server trusts hard labels versus soft consensus knowledge.

## 4. Limitations & Constraints

- **Requires a labeled public dataset** shared by server and all clients — a stronger assumption than FedDF's unlabeled/generated proxy data or FedGen's data-free approach, limiting applicability where no suitable public corpus exists.
- **Clients must expose logits on the public dataset every round they are queried**, which is a (bounded) information-leakage channel even though model parameters are never shared.
- **Evaluated only on CIFAR-10/CIFAR-100 image classification with ResNet-family clients/server** (up to 64 clients, ResNet-11 to ResNet-17 clients, ResNet-20 to ResNet-110 server); no results on non-vision tasks or larger real-world server/client capacity gaps.
- **All logit exchange is uncompressed**: the paper's own communication-cost analysis (Table 4) reports FedGEMS payload in raw megabytes with no quantization of logits, only fewer transmissions (via self-distillation caching and client filtering).
- **No formal convergence guarantee**: gains are empirical (400 rounds to convergence in experiments); the selective-fusion mechanism is a heuristic confidence filter, not a provably robust aggregator.

## 5. FedMAQ Thesis Relevance

- **Direct comparison point for FedMAQ's server-side proxy ensemble distillation**: FedGEMS's selective, entropy-weighted client fusion is a concrete precedent for *how* to weight client teachers when consolidating knowledge at the server, distinct from FedMAQ's own weighting signals (adaptive quantization/resource/training-state driven rather than purely prediction-confidence driven). Comparing the two weighting philosophies is a useful ablation angle.
- **Names quantization as an unclaimed opportunity**: FedGEMS reduces communication only by *transmitting fewer logits* (self-distillation caching, client filtering), never by *quantizing* the logits it does send — every payload in Table 4 is full-precision. This leaves the exact gap FedMAQ's quantization of the distillation signal (logits/soft-labels) is positioned to fill, analogous to the gap already noted for [FedMD](/papers/li-2019-fedmd.md) and [FedDF](/papers/lin-2020-feddf.md).
- **Reinforces the "larger server model" pattern is complementary to, not competing with, FedMAQ's scope**: FedMAQ does not adopt a GEM-style asymmetric server architecture (per the FedMAQ specification's scope boundaries), but FedGEMS's self-distillation cache is a useful design reference if a future extension considers reducing FedMAQ's server-side re-query rate.

# Related

- [FedMD: Heterogenous Federated Learning via Model Distillation](/papers/li-2019-fedmd.md)
- [Ensemble Distillation for Robust Model Fusion in Federated Learning](/papers/lin-2020-feddf.md)
- [Distilling the Knowledge in a Neural Network](/papers/hinton-2015-distillation.md)
- [Data-Free Knowledge Distillation for Heterogeneous Federated Learning](/papers/zhu-2021-fedgen.md)

# Citations

[1] Full-text conversion: [markdown/cheng-2021-fedgems/paper.md](markdown/cheng-2021-fedgems/paper.md)
[2] Source PDF: `papers/03 KD/Cheng et al. - 2021 - FedGEMS Federated Learning of Larger Server Models via Selective Knowledge Fusion.pdf`
