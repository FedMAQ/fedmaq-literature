---
type: Method
title: "FedDistill"
description: "Federated Distillation (FD): clients exchange per-label averaged output logits instead of parameters; the server aggregates them into global per-label logit means broadcast as distillation targets, decoupling communication from model size."
tags: [distillation, communication-efficiency, baseline]
introduced_by: /papers/jeong-2023-feddistill-aug.md
timestamp: 2026-07-12T00:00:00Z
---

# FedDistill

Federated Distillation (FD): an online co-distillation scheme whose communication
payload depends on the model's **output dimension**, not its parameter count — the
mechanism implemented as the "FedDistill" baseline in the FedMAQ codebase (the pure-KD
group alongside FedMD).

## 1. Mechanism

Each client periodically computes its **per-label average output logits** over its
local data and uploads them in place of model parameters. The server averages the
per-label logits across clients holding each label into **global per-label logit
means** \( \bar y^{(\ell)} \), and broadcasts these back as distillation targets. Each
client then trains on its supervised loss plus a distillation term pulling its own
per-label logits toward the corresponding global mean, at temperature \(T\):

\[
\ell_{local} = \ell_{CE}(f_k(x), y) + \gamma\, \mathrm{KL}\big(\sigma(f_k(x)/T)\,\lVert\,\sigma(\bar y^{(y)}/T)\big)
\]

\[
\bar y^{(\ell)} = \frac{1}{|\mathcal K_\ell|}\sum_{k \in \mathcal K_\ell} \bar y_k^{(\ell)}
\]

No model weights are ever exchanged — only a tensor of shape (#labels x output
dimension) per round in each direction.

## 2. Key hyperparameters

- Distillation temperature \(T\).
- Distillation loss weight \(\gamma\).

## 3. Communication & computation profile

Payload per round scales with \(|\mathcal Y| \times \dim(\text{output})\), independent
of network width/depth — the key win for large models, and a strict communication
reduction versus FedAvg (not merely a round-count reduction). Trades a modest accuracy
gap (reported 95-98% of full-parameter FL) for that savings; logit-only knowledge
conveys less than full parameters, which can limit performance on hard, high-capacity
tasks.

## 4. Papers

- Introduces: [FD + FAug](/papers/jeong-2023-feddistill-aug.md) (this method node
  covers the FD component only; see [FD + FAug](/methods/fd-faug.md) for the full
  paper method including the GAN-based Federated Augmentation component).

## 5. FedMAQ relevance

A canonical demonstration that exchanging **outputs instead of parameters** decouples
communication from model size — complementary to FedMAQ's parameter-side quantization.
FedMAQ can combine both: quantized weight updates plus a cheap per-label logit channel
that is naturally robust to aggressive weight quantization. As the FedMAQ codebase's
"pure KD" baseline, it isolates the distillation-only communication-efficiency
contribution against FedMAQ's joint quantization + KD design.

# Related

- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Communication efficiency](/concepts/communication-efficiency.md)
- [FD + FAug](/methods/fd-faug.md) — full paper method (adds GAN augmentation)
- [FedDistill (De-Biasing / Song 2024)](/methods/feddistill-debias.md) — unrelated
  mechanism that shares the "FedDistill" name
- [FedKD](/methods/fedkd.md)
- [FD + FAug paper](/papers/jeong-2023-feddistill-aug.md)
