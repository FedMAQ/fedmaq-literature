---
type: Finding
title: "Communication efficiency is a bits-per-round versus rounds-to-converge trade-off, not raw compression"
description: "Compression that ignores its effect on convergence can raise total bits to target accuracy; the meaningful objective is the product of per-round payload and rounds required."
tags: [communication-efficiency, convergence, compression]
timestamp: 2026-07-09T12:00:00Z
---

# Communication efficiency is a bits-per-round versus rounds-to-converge trade-off

## Scope

Covers the objective that compression methods optimize: total data to reach a target
accuracy, decomposed across per-round payload, number of rounds, participation, and
uplink/downlink asymmetry.

## Claim

Maximizing single-round compression is the wrong objective. Compression injects error
that can raise the rounds needed to converge, so total cost is the *product*
\((\text{bits/round})\times(\text{rounds})\); an aggressive compressor that doubles
rounds can lose to a milder one. Methods that reason about this product — bounding
quantization variance, adding periodic averaging and partial participation, or letting
distillation limit the rounds penalty — achieve lower total communication than those
optimizing payload alone. Bandwidth heterogeneity adds wall-clock time as a second
metric distinct from raw bits.

## Evidence

| Paper | Key result | Link |
| --- | --- | --- |
| QSGD | Formalizes the bits-vs-variance trade-off underlying the product objective | [/papers/alistarh-2017-qsgd.md](/papers/alistarh-2017-qsgd.md) |
| FedPAQ | Periodic averaging + partial participation + quantized deltas with convergence guarantees | [/papers/reisizadeh-2020-fedpaq.md](/papers/reisizadeh-2020-fedpaq.md) |
| AdaGQ | Optimizes round *time* under bandwidth heterogeneity, not just bits | [/papers/liu-2023-adagq.md](/papers/liu-2023-adagq.md) |
| FedKD | Cuts total communication via mutual distillation plus dynamic SVD compression | [/papers/wu-2022-fedkd.md](/papers/wu-2022-fedkd.md) |
| Le 2024 (survey) | Frames FL compression around communication-to-accuracy rather than raw ratio | [/papers/le-2024-compression-survey.md](/papers/le-2024-compression-survey.md) |

## Open gaps

- Evaluation in application deployments often reports raw compression, not
  bits-to-accuracy: [/gaps/application-eval-gaps.md](/gaps/application-eval-gaps.md).

# Related

- [Communication efficiency](/concepts/communication-efficiency.md)
- [Quantization](/concepts/quantization.md)
- [FedPAQ](/methods/fedpaq.md), [AdaGQ](/methods/adagq.md), [FedKD](/methods/fedkd.md)
