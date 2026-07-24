---
type: Paper
title: "Towards Federated Learning at Scale: System Design"
description: "Google's production FL system design for Android devices: synchronous round protocol, actor-model server, and Secure Aggregation, deployed to tens of millions of devices."
authors: "Bonawitz et al."
year: 2019
bibkey: bonawitz-2019-fl-scale
tags: [system, fl-core]
resource: markdown/bonawitz-2019-fl-scale/paper.md
timestamp: 2026-07-24T14:35:00Z
---

## 1. Overview & Objectives

- **Core Problem**: Federated Averaging (McMahan et al., 2017) was proven algorithmically, but no published system showed how to run synchronous FL rounds in production across a device population that is intermittently connected, resource-constrained, and drops out unpredictably.
- **Main Objectives**:
  - Describe a scalable, TensorFlow-based FL system design for Android that runs synchronous rounds of Federated Averaging over populations ranging from tens to hundreds of millions of devices.
  - Solve the systems problems this creates: device eligibility and selection, connection pacing, in-memory ephemeral aggregation, and integration with Secure Aggregation.
  - Report an operational profile from a system that has run production workloads for over a year, and enumerate open problems (bias, convergence speed, bandwidth) for future systems research.

## 2. Methodology & Key Innovations

- **Round protocol**: each round has three phases — **Selection** (server picks a subset, typically hundreds, of eligible connected devices via reservoir sampling), **Configuration** (server sends the FL plan and current model checkpoint), and **Reporting** (server aggregates updates via Federated Averaging as they arrive; devices that do not report in time are dropped, and the round is abandoned if too few report).
- **Pace steering**: a stateless, probabilistic flow-control mechanism that tells devices when to reconnect, avoiding "thundering herd" behavior at scale and synchronizing enough devices for small populations (also a requirement of Secure Aggregation).
- **Device eligibility**: devices only participate when idle, charging, and on an unmetered network — a deliberate policy to avoid harming user experience, battery, or data usage, which also introduces participation bias.
- **Server actor model**: Coordinators (per-population, globally registered), Selectors (accept/forward device connections, globally distributed), and ephemeral Master Aggregators/Aggregators (spawned per FL task, keep all state in memory, never persist per-device updates) — chosen for scalability and so that no logs of individual updates ever exist.
- **Secure Aggregation** (Sec. 6): a four-round interactive Secure Multi-Party Computation protocol layered into the Reporting phase so the server only ever observes the sum of masked updates, never individual device updates; because its server-side cost grows quadratically with participants, it is run per-Aggregator over groups of size \(\ge k\) and the resulting partial sums are combined without further Secure Aggregation at the Master Aggregator.
- **Versioned FL plans and tooling**: FL tasks are defined once and automatically compiled into device- and version-specific FL plans, tested in simulation against proxy data before deployment, addressing the versioning mismatch between the server's TensorFlow runtime and the (often much older) runtime installed on deployed devices.

## 3. Mathematical Formulation

- This is a systems paper, not an algorithmic one: it runs Federated Averaging (McMahan et al., 2017) as-is (pseudocode given in its Appendix B) rather than proposing a new update rule.
- The only formal quantities are the aggregation sums needed by Secure Aggregation and by FedAvg/SGD — the weighted parameter sum \(\bar w_t\) and the weight-count sum \(\bar n_t\) — since Secure Aggregation only needs to reveal *sums*, never per-device terms.
- No convergence analysis, no communication-cost bound, and no formal heterogeneity model are given; all quantitative results (Sec. 9, e.g., the 1.4M-parameter next-word-prediction model converging in 3000 rounds over 5 days) are empirical operational measurements, not derived from a stated objective.

## 4. Limitations & Constraints

- **Selection bias**: eligibility (idle, charging, unmetered network) correlates with time zone, region, and device tier, biasing which devices — and hence which data — participate; the authors detect this only indirectly via live A/B evaluation, not via a corrected estimator.
- **No compression or quantization**: all communication (models, updates, control messages) is full-precision; the paper explicitly lists quantized-representation training (Jacob et al., 2017) and gradient/update compression (Konečný et al., 2016b; Caldas et al., 2018) as unimplemented future work needed to reduce bandwidth.
- **Secure Aggregation cost scales quadratically** with per-group participant count, capping practical group sizes at hundreds of devices and forcing a two-tier (per-Aggregator, then Master Aggregator) aggregation split.
- **Convergence is slow relative to datacenter training** (~7x slower in the reported case study), attributed to FedAvg's limited ability to exploit more than hundreds of parallel devices per round; the authors identify this as an open algorithmic problem, not one they solve.
- **No formal non-IID treatment**: the system assumes whatever heterogeneity FedAvg already tolerates; it does not measure or adapt to per-device data or resource heterogeneity beyond the coarse eligibility gate.

## 5. FedMAQ Thesis Relevance

- **Establishes the deployment reality FedMAQ targets**: this is the reference system-level description of the cross-device constraints — intermittent connectivity, battery/charging gating, bandwidth variance by region, drop-out tolerance — that motivate *why* per-client adaptive compression (rather than a single fixed bit-width) is needed in practice. FedMAQ's resource-heterogeneity signal is effectively a formalization of this paper's device eligibility/selection criteria.
- **Names the exact gap FedMAQ (and its quantization baselines) fill**: the paper explicitly flags quantized training and update compression as necessary, unimplemented future work. Algorithmic quantization baselines such as [DAdaQuant](/papers/honig-2022-dadaquant.md) and [FedPAQ](/papers/reisizadeh-2020-fedpaq.md) are the direct algorithmic answers to the bandwidth problem raised here.
- **Grounds FedMAQ's non-adoption of Secure Aggregation and formal DP** in scope: this system shows both are deployable in principle, but FedMAQ's thesis scope (per the FedMAQ specification) is quantization-and-KD focused and does not attempt to jointly optimize with Secure Aggregation's quadratic-cost constraints — a boundary worth citing when scoping FedMAQ's contribution against full production systems.

# Related

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](/papers/mcmahan-2017-fedavg.md)
- [DAdaQuant: Doubly-adaptive quantization for communication-efficient Federated Learning](/papers/honig-2022-dadaquant.md)
- [FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization](/papers/reisizadeh-2020-fedpaq.md)

# Citations

[1] Full-text conversion: [markdown/bonawitz-2019-fl-scale/paper.md](markdown/bonawitz-2019-fl-scale/paper.md)
[2] Source PDF: `papers/01 FL, Heterogeneity/Bonawitz et al. - 2019 - Towards Federated Learning at Scale System Design.pdf`
