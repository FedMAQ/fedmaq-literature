---
type: Method
title: "FedMAQ"
description: "Multi-Adaptive Quantization + server-side proxy ensemble distillation: a per-client, per-round scalar bit-width driven by resource, training-state, and data-richness signals under a hard memory ceiling, refined by knowledge distillation over a server-only proxy set."
tags: [joint-q-kd, quantization, distillation, adaptive, thesis]
timestamp: 2026-07-11T00:00:00Z
---

# FedMAQ

This thesis's own algorithm. Unlike every other node in `methods/`, FedMAQ is not
introduced by a corpus paper — it has no `introduced_by` paper node; its
specification lives in the manuscript (`../../fedmaq-manuscript/chapter_3.tex`
Sections 3.3, 3.5 and `chapter_4.tex` Sections 4.2, 4.4). This node exists so the
rest of the bundle's "FedMAQ relevance" sections have one accurate node to point
to, rather than each paraphrasing the design independently.

## 1. Mechanism

FedMAQ combines two mechanisms across the round: client-side multi-adaptive
quantization and server-side proxy ensemble distillation. Clients perform no
local distillation — the local objective is strict cross-entropy task-loss
minimization; all error-mitigation compute is offloaded to the server.

**"Multi-adaptive" means multiple adaptive *signals*, not multiple precision
*axes*.** Each client emits a single scalar bit-width per round — precision is
not resolved layer-wise, and round-to-round variation is an implicit consequence
of the evolving gradient-norm signal, not an explicitly scheduled time axis. This
is a deliberate scope boundary: architectural compression (pruning,
sparsification, low-rank factorization) is excluded, and this is quantization
only.

**Two-tier precision scaling:**

- *Tier 1 (hard feasibility cap)*: client \(k\)'s bit-width is capped by its
  memory capacity \(c_k \sim \mathcal U(2048, 16384)\) MB,
  \(Q_k^{max} = \max\{q \in \mathcal Q \mid q \le \lfloor c_k / c_{unit}\rfloor\}\),
  \(c_{unit} = 512\) MB.
- *Tier 2 (soft quality allocation)*: within that cap, a soft target
  \(\hat q_k^{(t)}\) is computed from two normalized signals — training-state
  (\(\tilde g_k^{(t)}\), the per-round gradient norm, replacing DynFed's
  recursive inertial tracker with a direct per-round measurement) and
  data-richness (\(\tilde n_k\), dataset size, static per client) — then floored
  into \(\mathcal Q = \{1,2,3,4,5,6,7,8,16,32\}\) and capped by \(Q_k^{max}\):
  \(q_k^{(t)} = \max\{q \in \mathcal Q \mid q \le \min(Q_k^{max}, \hat q_k^{(t)})\}\).

Five candidate formulations for \(\hat q_k^{(t)}\) are evaluated in a dedicated
[formulation study](#3-formulation-study-and-winner-rule) (Formulation 0,
resource-only hard cap, is the non-adaptive control; Formulations 1-4 combine
\(\tilde g_k^{(t)}\) and \(\tilde n_k\) linearly, multiplicatively,
gradient-primary-data-modulated, or via a threshold rule).

**Server-side two-stage aggregation:** de-quantized client updates are first
combined by standard FedAvg-style \(n_k/n\) parameter averaging into a warm-start
global model — this stage, not distillation, attenuates the zero-mean
quantization noise (unbiased quantizer → averaging over \(K_{active}\) clients
shrinks variance \(\propto 1/K_{active}\)). The warm-start model is then refined
by proxy-based ensemble knowledge distillation: the server evaluates all
participating client models on a tiny, unlabeled, server-side-only proxy set
\(D_{proxy}\) (1600 samples, held out before Dirichlet partitioning,
label-stratified then unlabeled, never touched by clients), extracts softmax
temperature-scaled soft labels (\(T=1.0\)), averages them into an ensemble target
\(\tilde q(x)\) with **equal weight per teacher — no active/uncertainty-based
teacher selection** (unlike DynFed), and trains the global model to minimize
\(D_{KL}(\tilde q \parallel q_{global})\). Distillation targets the *functional*
non-IID drift parameter averaging leaves behind, not the quantization noise
itself; its predicted marginal benefit is larger under severe skew (\(\alpha=0.1\))
than moderate skew (\(\alpha=1.0\)).

Clients train only a compact student network (~1/5 the parameters of the
ResNet-18/LeNet-5 teacher architecture at the same input resolution); FedMAQ does
**not** retain FedKD's client-side mentor-mentee joint training — the teacher
role is deferred entirely to the server-side ensemble.

## 2. Key hyperparameters

- \(\mathcal Q = \{1,2,3,4,5,6,7,8,16,32\}\) — permissible bit-width set.
- \(c_{unit} = 512\) MB — memory consumed per quantization bit (Tier 1 calibration).
- \(D_{proxy} = 1600\) samples — server-only unlabeled proxy set size.
- \(T = 1.0\) — server-side KD softmax temperature.
- \(R = 100\) — fixed communication-round budget (not a convergence guarantee).
- Formulation weight convention (fixed, not tuned, identical across candidates):
  \(\omega_1 = \omega_2\); modulation/threshold constants \(\kappa = 1.0\),
  \(\tau_g = \tau_n = 0.5\) each held at one documented default.

## 3. Formulation study and winner rule

Formula selection is a first-class experiment, not a hidden pilot: determining
*how* the adaptive signals combine is the thesis's primary methodological
contribution. Run on CIFAR-10 across both skew regimes (\(\alpha \in \{0.1,
1.0\}\)) under the full pipeline with server-side KD active (only post-processing
coding is withheld). **Winner rule**: among the five candidates, select the one
reaching the target accuracy — 90% of the uncompressed FedAvg reference's
top-1 accuracy under the identical dataset/skew/seed — using the least cumulative
communication (MB) summed over rounds, subject to an accuracy-floor guard: any
formulation that fails to reach target within the \(R=100\) budget is disqualified
regardless of payload savings. This operationalizes the Objective-4
bits-to-accuracy product ([Communication efficiency](/concepts/communication-efficiency.md))
and blocks a degenerate low-bit formula from winning on compression alone. The
main benchmark grid across the remaining datasets/skews then serves as an
independent generalization test of the CIFAR-10-selected winner.

## 4. Communication & computation profile

Uplink: quantized student gradients at the per-client, per-round scalar
bit-width above. Downlink: full-precision global model broadcast. Server bears
proxy-inference-and-distillation compute in addition to aggregation; this and
FedKD's client-side dual-model training penalty are both modeled in the
manuscript's analytical cost model rather than measured wall-clock time.

## 5. Papers

FedMAQ has no `introduced_by` paper node. It is positioned directly against
[DynFed](/methods/dynfed.md) — the nearest published design (memory-capped,
gradient-adaptive bit-width + server-side multi-teacher distillation) — via a
DynFed-style ablation reference arm (Configuration 4: state-awareness-only, with
distillation retained) that isolates the value added by FedMAQ's data-richness
signal and its combination-logic contribution, since DynFed itself is excluded
from the primary benchmark grid for lacking a public codebase. It is benchmarked
against [FedAvg](/methods/fedavg.md), [FedProx](/methods/fedprox.md) (seminal),
[FedPAQ](/methods/fedpaq.md), [DAdaQuant](/methods/dadaquant.md) (pure
quantization), [FedMD](/methods/fedmd.md), [FedDistill](/methods/feddistill.md)
(pure KD), and [FedKD](/methods/fedkd.md), [CFD](/methods/cfd.md) (hybrid Q+KD).

# Related

- [Adaptive bit-width](/concepts/adaptive-bit-width.md)
- [Quantization](/concepts/quantization.md)
- [Knowledge distillation](/concepts/knowledge-distillation.md)
- [Communication efficiency](/concepts/communication-efficiency.md)
- [Non-IID heterogeneity](/concepts/non-iid-heterogeneity.md)
- [DynFed](/methods/dynfed.md), [FedKD](/methods/fedkd.md), [CFD](/methods/cfd.md)
- [Gap: signal-combination logic](/gaps/adaptive-precision-scheduling.md)
- [Gap: multi-adaptive Q+KD scarcity](/gaps/multi-adaptive-q-kd-scarcity.md)
