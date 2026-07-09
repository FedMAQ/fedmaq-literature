---
type: Finding
title: "Drift correction improves convergence under heterogeneity but adds client state or communication"
description: "Regularization, control variates, and objective normalization curb client drift under non-IID data, each paying with extra local state, per-round payload, or tuning."
tags: [heterogeneity, client-drift, convergence, regularization]
timestamp: 2026-07-09T12:00:00Z
---

# Drift correction aids convergence under heterogeneity but adds state or communication

## Scope

Covers non-IID remedies layered on FedAvg — proximal/dynamic regularization,
contrastive alignment, control variates, and objective normalization — on convergence
under heterogeneity versus their added cost.

## Claim

Client drift, where local optima diverge from the global optimum and their average is
biased, is the central obstacle non-IID FL. Each remedy family reduces it but pays a
distinct price: regularizers (FedProx, FedDyn, MOON) add tuning and compute but no
extra payload; control variates (SCAFFOLD) achieve variance reduction at the cost of
doubled uplink and persistent client state; objective normalization (FedNova) removes
inconsistency from unequal local work with minimal overhead but does not address label
skew. There is no free correction — the choice is which cost to bear.

## Evidence

| Paper | Key result | Link |
| --- | --- | --- |
| FedProx | Proximal term tolerates partial work and stabilizes non-IID training, needs \(\mu\) tuning | [/papers/li-2020-fedprox.md](/papers/li-2020-fedprox.md) |
| SCAFFOLD | Control variates give variance reduction but double uplink and require client state | [/papers/karimireddy-2020-scaffold.md](/papers/karimireddy-2020-scaffold.md) |
| FedDyn | Dynamic regularization aligns local and global optima with no extra communication | [/papers/acar-2021-feddyn.md](/papers/acar-2021-feddyn.md) |
| MOON | Model-contrastive alignment corrects representation drift | [/papers/li-2021-moon.md](/papers/li-2021-moon.md) |
| FedNova | Normalized averaging removes objective inconsistency from unequal local epochs | [/papers/wang-2020-fednova.md](/papers/wang-2020-fednova.md) |

## Open gaps

- How drift correction interacts with quantization noise when both perturb the update:
  [/gaps/quantization-drift-interaction.md](/gaps/quantization-drift-interaction.md).

# Related

- [Non-IID heterogeneity](/concepts/non-iid-heterogeneity.md)
- [FedProx](/methods/fedprox.md), [SCAFFOLD](/methods/scaffold.md),
  [FedDyn](/methods/feddyn.md), [MOON](/methods/moon.md), [FedNova](/methods/fednova.md)
