# FedMAQ — Finalized Thesis Structure, Narrative & Novelty Blueprint

## Context

FedMAQ ("Communication-Efficient Federated Learning via Multi-Adaptive
Quantization and Knowledge Distillation") is a proposal-stage BS/MS thesis
(DLSU; candidate C. J. Bunyi, advisor F. Flores). Chapters 1–4 are drafted as
outlines, Ch 5 drafted, Ch 6 pending. Before committing to prose, we ran a
relentless grilling pass to make the thesis **highly defensible yet feasible**.

The grill — cross-checked against the actual manuscript (`fedmaq-manuscript/`),
the experiments code (`fedmaq-experiments/`), and the knowledge graph
(`fedmaq-literature/kg/`) — surfaced one existential risk and several coherence
gaps:

- **The novelty overlaps heavily with DynFed** (He et al. 2025), which already
  fuses memory-capacity bit-width capping + gradient-norm-adaptive quantization
  - server-side proxy KD — and DynFed is currently _excluded_ from baselines.
- **The KG over-claims** the novelty as a "joint round × client × layer precision
  space," but the implementation emits **one scalar bit-width per client** (no
  layer-wise, round only implicit).
- **Three constraints wear one coat**: the cap is _memory_, the metric is
  _bandwidth (MB)_, the motivation is _OOM crashes_.
- **The central causal claim** ("KD filters quantization noise") is asserted, and
  its validity secretly depends on quantization being **stochastic/unbiased**.
- **The objective mapping** (client=system het, server=statistical het) does not
  survive the math.

This document records the **resolved decisions** from that grilling and the
concrete, per-artifact changes they imply. It is a planning/decision blueprint —
outlines only, no prose yet. Execution spans three repos and, per workspace
convention, each repo's edits are made in a session scoped to that repo.

---

## Locked decisions (from the grill)

1. **Novelty framing (honest, defensible).** FedMAQ's contribution is _not_ a
   novel mechanism. It is: (a) a **systematic empirical characterization of how
   multiple adaptive signals — resource (memory), training-state (gradient norm),
   and data-richness (dataset size) — should be _combined_ to allocate
   quantization precision** — the study supports "the combination _principle_
   (how to weight training-state vs. data-richness) generalizes across skew and
   datasets," **not** "transfers to arbitrary new signal sets" (the grid tests 5
   _fixed_ formulations, not new signals); and (b) a
   **simple, rigorous, fully open-source, reproducible DynFed-inspired
   implementation** others can branch from. We own the DynFed lineage openly.

2. **DynFed is not fully reproduced.** Faithful reproduction of its MCMC
   active-teacher-selection is infeasible to verify; omitting it is the honest
   call.

3. **DynFed-core reference via an elevated ablation arm.** The existing
   **"state-awareness-only + KD"** ablation arm _is_ DynFed's core mechanism
   (gradient-norm-adaptive quantization + multi-teacher server KD, minus active
   selection). Narrate it explicitly as a **DynFed-_style_ reference point** —
   framed as "what our data signal + combination logic add over a
   gradient-norm-only reference," **not** an implied win over DynFed itself.
   FedMAQ's gain over it quantifies the value of the added data-richness signal +
   learned combination logic. **Zero new code — VERIFIED**:
   `conf/algorithm/fedmaq_state_only.yaml` has `formulation: 1, gamma1=1.0,
gamma2=0.0, kd_epochs: 1` (gradient-only quantization + KD on). _Optional (now cheap on datacenter): add a
   6th arm using DynFed's recursive/inertial bit-tracker (Eq. 4) to empirically
   justify FedMAQ's direct-per-round-norm choice over recursive tracking._

4. **Memory vs. bandwidth — communication-primary.** Communication efficiency is
   the optimized objective and headline metric (MB transmitted). **Memory is a
   per-client _hard feasibility ceiling_** on affordable precision (high-bit
   quant/dequant + activation buffers have a footprint a constrained client can't
   sustain). Rewrite the OOM motivation honestly; DynFed is cited as precedent for
   the memory→precision cap, but `c_unit` must be defended _physically_ (now
   matched to a real ~45 MB ResNet-18 footprint), not by precedent alone.

5. **KD mechanism — attribute noise vs. drift to the RIGHT stage + commit to
   stochastic quantization.** _(Corrected attribution — this is the thesis's
   central causal claim, so precision matters.)_
   - **Noise attenuation happens at PARAMETER AGGREGATION, not at KD.** Because the
     client quantizer uses **unbiased stochastic rounding**, per-client error is
     zero-mean; FedAvg-style averaging of the de-quantized updates shrinks its
     variance ~1/K_active (the standard FedPAQ argument). This occurs _whether or
     not_ we distill. Do **not** credit the KD soft-label ensemble with the noise
     cancellation — the weight→logit map is nonlinear (softmax), so "unbiased
     weights → unbiased soft labels" does not survive Jensen. A quantization-
     literate panel will pull exactly this thread.
   - **KD's job is NON-IID DRIFT RECONCILIATION** in function space, where
     parameter-averaging of divergent clients degrades (FedDF, Lin 2020). This is
     KD's _primary_ and correct contribution.
   - **Falsifiable prediction this yields (a strength):** KD's marginal benefit
     should be **larger at α=0.1 than α=1.0**. The ablation already tests it —
     arm 5 (quant, no KD) vs arm 7 (full FedMAQ) across both skews.
   - **Switch the client quantizer to unbiased stochastic quantization** (a
     Bernoulli tie-break on the QSGD operator; DynFed already does this). Note:
     deterministic error doesn't cancel under averaging either, so this commitment
     stands regardless of the attribution fix. Do **not** reinstate the
     commented-out convergence proof (a shaky bound is worse than none).

6. **Formulation study elevated to a first-class result.** The 5-formulation
   comparison is the thesis's primary methodological contribution, not a hidden
   pilot. Run all five across **α ∈ {0.1, 1.0} on CIFAR-10** (was α=0.1 only;
   +15 runs). The main benchmark then serves as an **independent generalization
   test** of the selected formulation. If the winner shifts with skew, that is a
   headline finding.

7. **Compute — datacenter available.** The lab datacenter removes the compute
   constraint; the local RTX 5060 (8 GB) is a test rig only. Heavier
   architectures and the extra runs above are affordable.

8. **Architecture — ResNet-18 + GroupNorm.** CIFAR-10/100 use **ResNet-18 with
   BatchNorm replaced by GroupNorm** (standard FL-quant benchmark; non-trivial
   payload makes the communication + memory story real; GroupNorm is correct under
   α=0.1 non-IID and makes gradient quantization well-defined by avoiding
   BN-running-stat quantization). FEMNIST uses the **LEAF-standard small 2-conv
   CNN**.

9. **Objectives restructured around the RQ's two axes** (replaces the
   client=system / server=statistical 2×2 that the math contradicts):
   - **Obj 1** — Simulate FL under statistical _and_ system heterogeneity (the
     operating _condition_).
   - **Obj 2** — **Minimize communication overhead** via multi-signal adaptive
     quantization within each client's memory feasibility ceiling.
   - **Obj 3** — **Preserve predictive utility** via server-side KD, recovering
     loss from _both_ quantization noise _and_ non-IID drift.
   - **Obj 4** — Benchmark the overhead↔utility frontier against reproducible
     baselines.

10. **Proxy dataset — in-distribution, unlabeled, server-only.** 1,600-sample
    unlabeled holdout (FedDF/DynFed precedent). Defense: unlabeled + tiny +
    server-only (relaxes vs FedMD's labeled public / CFD's client-distributed
    proxy); every KD-family baseline gets equivalent proxy access (fair within
    family); the no-KD ablation isolates its contribution. No provenance/size
    sweep (unnecessary runtime).

---

## Resulting changes by artifact

### A. Manuscript (`fedmaq-manuscript/`, own session)

- **Ch 1 — Introduction**
  - Problem statement / RQ: keep the RQ (overhead vs utility), but rewrite the
    OOM-crash framing (ch1:36–53) so the constraint chain is coherent
    (communication-primary; memory = feasibility ceiling). [Decision 4]
  - Objectives (ch1:130–140): replace with the two-axis structure. [Decision 9]
  - Scope & Limitations (ch1:163–173): reframe Obj-2/Obj-3 paragraphs to the new
    mapping; upgrade the DynFed-exclusion paragraph to name the **DynFed-core
    reference arm** as the nearest-neighbor anchor (not a bare "closed-source"
    exclusion). [Decisions 3, 9]
  - Significance (ch1:208–219): recast the contribution as the **combination-logic
    characterization + open reproducible implementation**, dropping any
    novel-mechanism overclaim. [Decision 1]

- **Ch 2 — Review of Related Literature** _(not yet read; audit required)_
  - Its stated **gap narrative is now stale**: if the RRL builds toward a "no one
    does multi-axis round×client×layer quantization+KD" gap (the over-claim being
    retired) and elides DynFed, it will _contradict_ the honest novelty. Audit
    Ch2's gap statement for consistency with Decisions 1 & 9; **reframe the gap to
    the combination-logic + reproducibility contribution and name DynFed as the
    nearest neighbor.** (Outline _format_ is fine — only the gap _content_ is
    stale.)

- **Ch 3 — Theoretical Framework**
  - Define **"Multi-Adaptive" precisely as "multiple adaptive _signals_
    (resource/state/data)," not multiple precision axes.** [Decisions 1, KG-fix]
  - Multi-Adaptive Precision Scaling (ch3:162–234): re-derive and physically
    justify `c_unit` against the real ResNet-18 footprint. [Decisions 4, 8]
  - Quantization operator (ch3:129–160): make explicit that the client quantizer
    is **stochastic/unbiased**; connect the unbiasedness to the KD noise argument.
    [Decision 5]
  - Unified Codesign (ch3:290–334): state the **server loop explicitly** —
    parameter aggregation _then_ ensemble KD of participating teachers into the
    aggregated global student. Attribute **noise attenuation to the parameter-
    aggregation stage** (unbiased stochastic quantization + ~1/K_active averaging)
    and **KD to non-IID drift reconciliation** — do _not_ credit KD with noise
    cancellation (softmax nonlinearity breaks it). State the falsifiable
    α-dependence prediction. [Decision 5]

- **Ch 4 — Methodology**
  - Architecture (ch4:101): specify ResNet-18 + GroupNorm (CIFAR) / small CNN
    (FEMNIST) with the explicit BN→GroupNorm rationale. [Decision 8]
  - Pilot → **Formulation Study** (ch4:281–309): re-title and re-frame as a
    first-class experiment across α ∈ {0.1, 1.0} on CIFAR-10; state the main grid
    as the generalization test. [Decision 6]
  - Ablation (ch4:311–341): label the state-only+KD arm as the **DynFed-core
    reference**; optionally add the recursive-tracker arm. [Decision 3]
  - Grid/timeline (ch4:270, 346+): update run counts (+15 formulation runs, +
    optional arm) and note datacenter compute; keep the calendar realistic on
    _analysis/writing_ time, not compute. [Decisions 6, 7]

- **Ch 5 — Results (outline)** — structure into three reported blocks:
  (1) **Formulation study** (which combination logic wins, and its stability
  across skew); (2) **Main benchmark frontier** (accuracy-per-MB vs the 8
  reproducible baselines); (3) **Ablation** including the DynFed-core reference and
  per-signal isolation. Headline the **bits-to-accuracy frontier** (relative), not
  absolute top-1, so any CIFAR-100 under-convergence stays informative.

### B. Experiments code (`fedmaq-experiments/`, own session)

- **[HARD DEPENDENCY of the Ch3 causal claim]** Switch the FedMAQ client quantizer
  to **unbiased stochastic quantization**. _Verified 2026-07-10:_ the current FedMAQ
  quantizer uses FedPAQ-style **deterministic** round-to-nearest
  (`baselines/quantization.py` `FedPAQCompressionHook._quantize_elem`, via
  `np.round`), whose error is biased and does **not** cancel under FedAvg averaging.
  The Ch3 attribution (noise attenuation at the parameter-averaging stage, variance
  shrinking ~1/K_active) is *unsupported* until this switch lands. Note: the
  average-then-distill structure is **already in code** —
  `kd_utils.distill_ensemble_into_global` seeds the KD student from the FedAvg
  weighted average, then distills the teacher ensemble into it — so only the
  zero-mean-error premise is missing. Reuse the existing stochastic-rounding path
  (`DAdaQuantCompressionHook._quantize_elem`). Also update alignment rule #4, which
  currently groups FedMAQ with FedPAQ as "symmetric uniform quantization." [Decision 5]
- Swap ResNet-18 **BatchNorm → GroupNorm**; wire ResNet-18 for CIFAR-10/100.
  **This is a real code change, not a config tweak** — it touches the KD-student
  architecture, the gradient-norm probing path, the memory model, and likely the
  golden time/bytes tests. Budget accordingly. [Decision 8]
- Add the α=1.0 CIFAR-10 configs for the **formulation study** (now 5 formulations ×
  2 α × 3 seeds = 30 runs) **and** run all 5 net-new **ablation** arms across both α
  (5 × 2 α × 3 seeds = 30 runs). This brings the grid to **195 runs** (main 108 +
  FEMNIST 27 + formulation 30 + ablation 30). _Resolved 2026-07-10:_ running the
  quant-only arm (arm 5) at both α is **required** so the Ch3 α-dependence prediction
  (KD's marginal benefit larger at α=0.1 than α=1.0) is testable; the manuscript now
  states 195. (Optional) add the recursive-tracker ablation arm. [Decisions 6, 3]
- Re-derive `c_unit` from the ResNet-18 footprint; update `fedmaq.yaml` /
  memory-model constants. [Decisions 4, 8]
- Early **CIFAR-100 α=0.1 smoke-test** to confirm baselines separate before the
  full grid.

### C. Knowledge graph (`fedmaq-literature/kg/`, this repo)

- Dial down the over-claiming gap nodes to match the honest signal-based framing:
  - `gaps/adaptive-precision-scheduling.md` — drop "joint round × client × layer
    precision space"; reframe as **multi-signal combination** (client-level scalar,
    round only implicit, no layer-wise).
  - `gaps/multi-adaptive-q-kd-scarcity.md` — align the FedMAQ niche to the
    combination-logic contribution, not a multi-axis claim.
  - `gaps/heterogeneity-aware-quantization.md` — clarify that FedMAQ conditions on
    data _quantity_ + training state, **not** distributional-divergence/skew; it
    does not close the skew-aware-precision gap (KD, not quantization, carries the
    statistical-heterogeneity load). [Decisions 1, 9]

---

## Finalized thesis spine (the coherent narrative)

> Edge FL must trade **communication overhead** against **predictive utility**
> under **statistical + system heterogeneity**. FedMAQ (1) adaptively quantizes
> uplink gradients using three complementary signals — memory (a hard feasibility
> ceiling), gradient norm, and dataset size — _minimizing communication_; and
> (2) runs server-side ensemble KD over a tiny unlabeled proxy to _preserve
> utility_ lost to unbiased quantization noise and non-IID drift. The thesis's
> primary contributions are a **systematic study of how these signals should be
> combined** and an **open, reproducible DynFed-inspired testbed**, benchmarked on
> the bits-to-accuracy frontier against 8 reproducible baselines plus a DynFed-core
> reference arm.

---

## Verification

- **Manuscript compiles**: `latexmk main.tex` (in `fedmaq-manuscript/`) after
  edits; confirm the objectives, RQ, and significance read as one coherent
  overhead↔utility story end-to-end.
- **Coherence self-audit**: re-read Ch1 RQ → Ch1 Objectives → Ch3 mechanism → Ch4
  metrics and confirm no residual "system↔client / statistical↔server" pinning and
  no "round×client×layer" or "skew-aware quantization" claims survive.
- **Code correctness**: unit-check that the client quantizer is unbiased
  (E[Q(v)] ≈ v over many draws); confirm GroupNorm ResNet-18 trains; run the
  CIFAR-100 α=0.1 smoke-test and confirm baselines are separable before the grid.
- **DynFed-core arm**: confirm the state-only+KD config runs and is labeled as the
  reference point in telemetry.
- **KG consistency**: `grep` the gap nodes for "layer", "round × client", and
  "skew"/"divergence" to confirm the over-claims are removed.

## Open / deferred (not blocking)

- Recursive-tracker ablation arm (Decision 3) — include only if the direct-vs-
  recursive justification is worth the extra arm.
- Ch 6 (Conclusion) remains to be outlined after Ch 5 structure is locked.
