---
type: Method
title: "FedDF"
description: "Ensemble distillation fusion: the server distills the averaged logits of client models over unlabeled/generated proxy data into the global model, tolerating heterogeneity."
tags: [distillation, aggregation, heterogeneity, baseline]
introduced_by: /papers/lin-2020-feddf.md
timestamp: 2026-07-09T12:00:00Z
---

# FedDF

Federated Distillation Fusion: replaces weight averaging with server-side ensemble
distillation, fusing clients that differ in size, precision, or architecture.

## 1. Mechanism

After receiving client models, the server treats them as an ensemble of teachers:
for each proxy sample \(x\) it averages their tempered logits into a soft target
\( \bar p(x)=\tfrac1N\sum_i \sigma(f_{\theta_i}(x)/T) \) and trains the student
(server) model, seeded from the averaged model, to match it,
\( \min_{\theta_s} \mathbb{E}_{x\sim D_{proxy}} \mathrm{KL}(\bar p(x)\,\lVert\,\sigma(f_{\theta_s}(x)/T)) \).
Because fusion happens in output space, client architectures need not match; the step
also repairs BatchNorm damage from weight averaging.

## 2. Key hyperparameters

- \(T\) — distillation temperature.
- Proxy dataset (real unlabeled or GAN-generated) and its coverage.
- Server distillation steps per round.

## 3. Communication & computation profile

Communication like baseline FL, but adds server-side compute (forward the ensemble
over proxy data + distill). Requires a proxy dataset whose distribution matches the
task — its central limitation, later removed by FedGen.

## 4. Papers

- Introduces: [FedDF](/papers/lin-2020-feddf.md).
- Data-free successor [FedGen](/papers/zhu-2021-fedgen.md); quantized-distillation
  successor [CFD](/papers/sattler-2022-cfd.md).

## 5. FedMAQ relevance

FedDF is the canonical distillation *aggregator* underpinning FedMAQ's KD component,
and uniquely relevant because it fuses clients of differing *numerical precision* —
exactly what multi-adaptive quantization produces. An FedDF-style server distillation
can recover accuracy lost to aggressive per-client quantization where weight averaging
would blur.

# Related

- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Proxy-dataset distillation](/concepts/proxy-dataset-distillation.md)
- [FedGen](/methods/fedgen.md), [CFD](/methods/cfd.md),
  [FedProto](/methods/fedproto.md)
- [FedDF paper](/papers/lin-2020-feddf.md)
