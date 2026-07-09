---
type: Method
title: "FedKD"
description: "Mentee-mentor mutual distillation plus dynamic SVD gradient compression: only a small shared mentee model is communicated, cutting cost up to ~95%."
tags: [distillation, compression, joint-q-kd, baseline]
introduced_by: /papers/wu-2022-fedkd.md
timestamp: 2026-07-09T12:00:00Z
---

# FedKD

Communication-efficient FL that combines knowledge distillation with low-rank gradient
compression so only a compact shared model crosses the network.

## 1. Mechanism

Each client keeps a large local *mentor* and a copy of a small shared *mentee*; only
the mentee is communicated. Mentor and mentee train by *adaptive mutual distillation* —
task loss plus KL over logits and MSE over hidden states/attention, each weighted
inversely by the pair's task losses so unreliable predictions weaken the signal. The
mentee gradient is further compressed by SVD, keeping the top singular values under a
*dynamic energy threshold* \(T(t)=T_{start}+(T_{end}-T_{start})t\) — aggressive early,
finer later. The server reconstructs, sums, re-factorizes, and returns the global
gradient.

## 2. Key hyperparameters

- \(T_{start}, T_{end}\) — SVD energy-threshold schedule (compression vs. accuracy).
- Mentor/mentee capacities.
- Distillation temperature; adaptive-weight formulation.

## 3. Communication & computation profile

Up to ~94.9% communication reduction while matching centralized accuracy. Clients bear
extra compute (train both models); server reconstructs full gradients (a memory cost at
scale). Cross-silo focus; assumes an honest server and secure channels.

## 4. Papers

- Introduces: [FedKD](/papers/wu-2022-fedkd.md).
- Combines distillation with compression, adjacent to
  [CFD](/papers/sattler-2022-cfd.md) and the quantized-KD line.

## 5. FedMAQ relevance

FedKD is the clearest precursor to FedMAQ's *joint* KD + compression premise, and a
high-bar baseline (~95% reduction). Its dynamic-threshold schedule is a design template
FedMAQ can mirror for adaptive *bit-widths*; FedMAQ argues quantization may beat SVD on
quantization-friendly hardware while keeping the mentee-mentor communication/personalization
split.

# Related

- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Model compression](/concepts/model-compression.md)
- [Communication efficiency](/concepts/communication-efficiency.md)
- [CFD](/methods/cfd.md), [FedDistill](/methods/feddistill.md)
- [FedKD paper](/papers/wu-2022-fedkd.md)
