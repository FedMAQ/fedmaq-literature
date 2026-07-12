---
type: Method
title: "FedDistill (De-Biasing / Song 2024)"
description: "Global-to-local group distillation that de-biases local classifiers under non-IID data, splitting the KD loss over true / rich-sample / few-sample classes."
tags: [distillation, heterogeneity]
introduced_by: /papers/song-2024-feddistill.md
timestamp: 2026-07-12T00:00:00Z
---

# FedDistill (De-Biasing / Song 2024)

Global Model Distillation for Local De-Biasing: attacks non-IID *forgetting* of
under-represented classes by restructuring the distillation loss, with no extra
communication. Shares the name "FedDistill" with [Jeong et al.'s per-label logit
exchange method](/methods/feddistill.md) but describes an unrelated mechanism — see
disambiguation note in that node's Related section.

## 1. Mechanism

Cross-entropy gives positive gradient to the true class and negative to others, so
few-sample classes are forgotten locally. FedDistill decomposes the global-to-local KD
loss into three group terms — true class (TC-KD), rich-sample classes (RC-KD), and
few-sample classes (FC-KD), split at threshold \(\gamma=1/|\mathcal C|\) — combined as
\( GD = \alpha_t TC + \alpha_r RC + \alpha_f FC \) (reducing to plain KL when all
\(\alpha=1\)). It further decomposes each model into extractor + classifier and uses
four global/local prediction paths, giving losses
\( \mathcal L = CE + \beta_L\mathcal L_L + \beta_E\mathcal L_E + \beta_{FC}\mathcal L_{FC} \)
to de-bias the local classifier while generalizing the extractor.

## 2. Key hyperparameters

- Group weights \(\alpha_t,\alpha_r,\alpha_f\); path weights \(\beta_L,\beta_E,\beta_{FC}\).
- Few-sample threshold \(\gamma\).

## 3. Communication & computation profile

No communication reduction — it modifies only local training (per-round cost equals
FedAvg) but reaches target accuracy in fewer rounds. Many hyperparameters; per-class
computation scales with class count; assumes the global model is less biased than
locals.

## 4. Papers

- Introduces: [FedDistill (De-Biasing)](/papers/song-2024-feddistill.md).
- Reported to outperform [MOON](/papers/li-2021-moon.md) and FedNTD on non-IID.

## 5. FedMAQ relevance

A strong self-distillation baseline for FedMAQ's KD axis that is orthogonal to
compression: its group-distillation loss and extractor/classifier split compose with
adaptive quantization, e.g. prioritizing precision (or KD weight) for the few-sample
classes most vulnerable to both non-IID forgetting and quantization noise. Not the
mechanism implemented as the codebase's "FedDistill" baseline (see
[FedDistill](/methods/feddistill.md) for that).

# Related

- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Non-IID data and client drift](/concepts/non-iid-heterogeneity.md)
- [MOON](/methods/moon.md), [FedKD](/methods/fedkd.md)
- [FedDistill](/methods/feddistill.md) — unrelated method sharing this name (Jeong
  et al.'s per-label logit exchange)
- [FedDistill (De-Biasing) paper](/papers/song-2024-feddistill.md)
