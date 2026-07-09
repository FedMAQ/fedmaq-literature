---
type: Gap
title: "The interaction between quantization noise and client drift is poorly characterized"
description: "Quantization and non-IID drift both perturb the aggregated update, yet how compression error compounds with or is corrected by drift-correction mechanisms is not systematically studied."
tags: [quantization, client-drift, heterogeneity, convergence]
timestamp: 2026-07-09T12:00:00Z
---

# The interaction between quantization noise and client drift is poorly characterized

## Problem

Quantization injects (often unbiased) noise into updates, and non-IID heterogeneity
injects a (biased) drift; both move the aggregate away from the global optimum. Whether
these effects add, cancel, or amplify — and whether variance-reduction machinery such
as control variates still works once its inputs are quantized — is not systematically
established. A quantization schedule tuned on IID data may destabilize under drift.

## State of the art

Drift-correction methods (SCAFFOLD, FedDyn, FedProx, FedNova) assume full-precision
updates; quantization methods (QSGD, FedPAQ) prove convergence under IID or bounded-
heterogeneity assumptions without a drift-correction term. Joint-Q+KD methods sidestep
the analysis by leaning on distillation to absorb the damage rather than modeling the
interaction. No corpus method both quantizes *and* applies control-variate drift
correction with a joint convergence guarantee.

## FedMAQ's angle

FedMAQ must set precision under real heterogeneity, so it needs a heterogeneity-aware
schedule and, ideally, a distillation channel that compensates for the compounded
noise — turning this uncharacterized interaction into a design constraint.

## Sources

- Motivating finding: [Drift correction improves convergence but adds cost](/findings/drift-correction-tradeoff.md).
- Papers: [SCAFFOLD](/papers/karimireddy-2020-scaffold.md),
  [FedPAQ](/papers/reisizadeh-2020-fedpaq.md), [FedDyn](/papers/acar-2021-feddyn.md).

# Related

- [Quantization](/concepts/quantization.md)
- [Non-IID heterogeneity](/concepts/non-iid-heterogeneity.md)
- [Heterogeneity-aware quantization schedules](/gaps/heterogeneity-aware-quantization.md)
- [SCAFFOLD](/methods/scaffold.md), [FedPAQ](/methods/fedpaq.md)
