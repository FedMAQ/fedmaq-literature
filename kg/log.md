# Knowledge Graph Update Log

## 2026-07-09

- **Initialization**: Established the OKF bundle at `kg/` with the root [index](/index.md) (`okf_version: "0.1"`) and scaffolded sections `papers/`, `methods/`, `concepts/`, `findings/`, `gaps/`.
- **Creation**: Populated [papers/](/papers/) with 39 `type: Paper` nodes — the full FedMAQ thesis canon. 28 migrated from approved summaries; 11 hand-authored from source markdown (the 10 net-new baselines FedDyn, SCAFFOLD, MOON, FedProto, FedNova, QSGD, signSGD, Jeong FD+FAug, FedDF, FedGen, plus a re-authored English node for joseph-2026-air-quality whose prior summary was in Chinese).
- **Creation**: Generated [papers/index](/papers/index.md) — progressive-disclosure listing grouped by ingestion batch / theme.
- **Provenance**: Migration performed by `scripts/build_kg_papers.py`; hand-authored bodies retained under `scripts/kg_bodies/`.

### Phase 2 — Methods & Concepts

- **Scaffolding**: Codified `.claude/rules/okf-method-template.md` and `okf-concept-template.md` to fix the shape of Method and Concept nodes.
- **Creation (methods, batch 1)**: Authored 7 FL-core / heterogeneity Method nodes — [FedAvg](/methods/fedavg.md), [FedProx](/methods/fedprox.md), [SCAFFOLD](/methods/scaffold.md), [FedDyn](/methods/feddyn.md), [MOON](/methods/moon.md), [FedProto](/methods/fedproto.md), [FedNova](/methods/fednova.md) — sourced from the corresponding curated paper nodes.
- **Creation (methods, batch 2)**: Authored 6 quantization Method nodes — [QSGD](/methods/qsgd.md), [signSGD](/methods/signsgd.md), [FedPAQ](/methods/fedpaq.md), [DAdaQuant](/methods/dadaquant.md), [AdaGQ](/methods/adagq.md), [LAQ-HC](/methods/laq-hc.md).
- **Creation (methods, batch 3)**: Authored 6 knowledge-distillation Method nodes — [FedMD](/methods/fedmd.md), [FedDF](/methods/feddf.md), [FedGen](/methods/fedgen.md), [FedDistill](/methods/feddistill.md), [FedKD](/methods/fedkd.md), [FD+FAug](/methods/fd-faug.md).
- **Creation (methods, batch 4)**: Authored 5 joint quantization+distillation Method nodes — [CFD](/methods/cfd.md), [DynFed](/methods/dynfed.md), [FedDT](/methods/feddt.md), [AdaDQ-KD](/methods/adadq-kd.md), [AQFedAvg+FKD](/methods/quantized-kd.md). Methods layer complete (24 nodes); regenerated [methods/index](/methods/index.md) grouped by family.
- **Creation (concepts, batch 5)**: Authored 10 `type: Concept` nodes — [quantization](/concepts/quantization.md), [adaptive-bit-width](/concepts/adaptive-bit-width.md), [model-compression](/concepts/model-compression.md), [knowledge-distillation](/concepts/knowledge-distillation.md), [proxy-dataset-distillation](/concepts/proxy-dataset-distillation.md), [data-free-distillation](/concepts/data-free-distillation.md), [communication-efficiency](/concepts/communication-efficiency.md), [non-iid-heterogeneity](/concepts/non-iid-heterogeneity.md), [privacy-in-fl](/concepts/privacy-in-fl.md), [edge-iot-deployment](/concepts/edge-iot-deployment.md) — each linking out to its methods and papers. Regenerated [concepts/index](/concepts/index.md). Resolves all method→concept forward links; bundle-wide link check now green (no broken intra-bundle links across papers/, methods/, concepts/). Phase 2 complete.

### Phase 3 — Findings & Gaps

- **Creation (findings, batch 6)**: Authored 8 `type: Finding` nodes synthesizing claims across papers — [adaptive-quantization-beats-uniform](/findings/adaptive-quantization-beats-uniform.md), [distillation-cuts-uplink-needs-proxy](/findings/distillation-cuts-uplink-needs-proxy.md), [data-free-kd-removes-proxy-cost](/findings/data-free-kd-removes-proxy-cost.md), [drift-correction-tradeoff](/findings/drift-correction-tradeoff.md), [quantization-kd-complementary](/findings/quantization-kd-complementary.md), [comm-efficiency-tradeoff](/findings/comm-efficiency-tradeoff.md), [application-fl-validation](/findings/application-fl-validation.md), [no-unified-compression-heterogeneity-method](/findings/no-unified-compression-heterogeneity-method.md) — each with an evidence table and `/gaps/` forward links. Regenerated [findings/index](/findings/index.md). All non-gap links resolve; the 6 `/gaps/` forward-refs are created in batch 7.
- **Creation (gaps, batch 7)**: Authored 6 `type: Gap` nodes distilled from paper limitations and finding open-gaps — [adaptive-precision-scheduling](/gaps/adaptive-precision-scheduling.md), [heterogeneity-aware-quantization](/gaps/heterogeneity-aware-quantization.md), [quantization-drift-interaction](/gaps/quantization-drift-interaction.md), [proxy-data-dependence](/gaps/proxy-data-dependence.md), [multi-adaptive-q-kd-scarcity](/gaps/multi-adaptive-q-kd-scarcity.md), [application-eval-gaps](/gaps/application-eval-gaps.md) — each with problem / state-of-the-art / FedMAQ angle / sources. Regenerated [gaps/index](/gaps/index.md).
- **Finalization**: Updated root [index](/index.md) — dropped all `_(scaffolded)_` labels, added per-layer node counts (papers 39, methods 24, concepts 10, findings 8, gaps 6). Full-bundle verification green: 87 nodes total, every node carries a non-empty `type`, and every root-absolute intra-bundle link resolves. Phases 2 and 3 complete; the OKF bundle is fully populated.

## 2026-07-10

- **Lint**: Audited the bundle against `docs/llm-wiki.md` philosophy and `docs/okf.md` conformance. Structure, frontmatter, and link integrity were clean (0 broken links across 94 files, verified programmatically). Found two defects: (1) corrupted LaTeX subscripts in three paper nodes — [FedAvg](/papers/mcmahan-2017-fedavg.md), [DAdaQuant](/papers/honig-2022-dadaquant.md), [AdaDQ-KD](/papers/wang-2026-adadq-kd.md) — where `_{` had been mangled to `*{` during authoring, breaking equation rendering; (2) 9 of 39 paper nodes missing the `# Related` closing section required by `okf-paper-template.md`.
- **Fix**: Repaired all corrupted LaTeX subscripts in the three affected paper nodes. Authored `# Related` sections for the 9 nodes that lacked them — [cui-2026-laq-hc](/papers/cui-2026-laq-hc.md), [hinton-2015-distillation](/papers/hinton-2015-distillation.md), [joseph-2026-air-quality](/papers/joseph-2026-air-quality.md), [le-2024-compression-survey](/papers/le-2024-compression-survey.md), [mao-2023-power-load](/papers/mao-2023-power-load.md), [mcmahan-2017-fedavg](/papers/mcmahan-2017-fedavg.md), [reisizadeh-2020-fedpaq](/papers/reisizadeh-2020-fedpaq.md), [singh-2026-smart-campus](/papers/singh-2026-smart-campus.md), [wang-2026-adadq-kd](/papers/wang-2026-adadq-kd.md) — linking each to thematically adjacent papers already present in the corpus. Re-verified: 0 broken links, 0 missing `# Related` sections, 0 corrupted LaTeX remaining.
- **Revision (de-overclaim)**: Following the FedMAQ grilling pass, dialed the novelty framing in the gap/finding layer down from a "joint round × client × layer precision space" claim to the honest **multi-signal combination** contribution (client-level scalar bit-width; round-variation implicit; no layer-wise). Rewrote [adaptive-precision-scheduling](/gaps/adaptive-precision-scheduling.md) (retitled to the signal-combination question), [multi-adaptive-q-kd-scarcity](/gaps/multi-adaptive-q-kd-scarcity.md) (aligned niche to the combination-logic contribution over DynFed's resource+state design), and [heterogeneity-aware-quantization](/gaps/heterogeneity-aware-quantization.md) (clarified FedMAQ conditions on data *quantity* + training state, not distributional skew, and does **not** close the skew-aware-precision gap — distillation carries the statistical-heterogeneity load). Propagated the reframing to [no-unified-compression-heterogeneity-method](/findings/no-unified-compression-heterogeneity-method.md), [adaptive-quantization-beats-uniform](/findings/adaptive-quantization-beats-uniform.md), and [gaps/index](/gaps/index.md), removing residual "multi-axis"/"round × client × layer"/"skew-aware quantization" over-claims.

## 2026-07-11

- **Creation**: Authored [methods/fedmaq.md](/methods/fedmaq.md) — the thesis's own
  algorithm, with no `introduced_by` paper node (deviation from
  `okf-method-template.md`, documented here since FedMAQ is not introduced by a
  corpus paper; its specification lives in the manuscript, cited in the node body).
  Covers the two-tier precision-scaling design, the five-formulation study and its
  pre-registered winner rule (least cumulative communication to a 90%-of-FedAvg
  target accuracy, subject to an accuracy-floor guard), and the server-side
  two-stage aggregation (parameter averaging then equal-weight ensemble
  distillation), sourced from `chapter_3.tex` sections 3.3/3.5 and `chapter_4.tex`
  sections 4.2/4.4 post-"grill and polish" finalization. Regenerated
  [methods/index](/methods/index.md) (24 to 25 nodes) and the root
  [index](/index.md) count.
- **Fix (systemic de-overclaim sweep)**: The 2026-07-10 de-overclaim pass corrected
  the gap/finding layer only; the method and paper layers still described FedMAQ
  with the pre-scope-lock framing ("multi-adaptive across rounds and layers",
  "varying bit-widths per layer/round", combining quantization with
  sparsification/low-rank/pruning, and FedKD's mentee-mentor split as FedMAQ's
  "base framework"). Swept and corrected roughly 20 files against the locked
  `chapter_3.tex`/`chapter_4.tex` design: "multi-adaptive" means multiple adaptive
  *signals* (resource, training-state, data-richness) combined into one
  client-level, per-round scalar bit-width -- no layer axis, no sparsification, no
  local client-side distillation or mentor/mentee training (all error-mitigation is
  server-side, deferred entirely from the student-only client). Fixed nodes:
  [cfd](/methods/cfd.md), [fedkd](/methods/fedkd.md), [feddt](/methods/feddt.md),
  [dadaquant](/methods/dadaquant.md), [fedpaq](/methods/fedpaq.md),
  [qsgd](/methods/qsgd.md), [quantized-kd](/methods/quantized-kd.md),
  [adagq](/methods/adagq.md), [laq-hc](/methods/laq-hc.md),
  [dynfed](/methods/dynfed.md) (the last also removing the false claim that
  DynFed's active/uncertainty-based teacher selection is reusable in FedMAQ, which
  uses equal-weight ensemble averaging); concepts
  [quantization](/concepts/quantization.md) and
  [adaptive-bit-width](/concepts/adaptive-bit-width.md); and the "FedMAQ Thesis
  Relevance" sections of papers [liu-2023-adagq](/papers/liu-2023-adagq.md),
  [cui-2026-laq-hc](/papers/cui-2026-laq-hc.md),
  [wu-2022-fedkd](/papers/wu-2022-fedkd.md),
  [qu-2020-quantization-kd](/papers/qu-2020-quantization-kd.md),
  [alistarh-2017-qsgd](/papers/alistarh-2017-qsgd.md),
  [li-2020-fedprox](/papers/li-2020-fedprox.md),
  [honig-2022-dadaquant](/papers/honig-2022-dadaquant.md),
  [joseph-2026-air-quality](/papers/joseph-2026-air-quality.md),
  [sattler-2022-cfd](/papers/sattler-2022-cfd.md),
  [he-2025-dynfed](/papers/he-2025-dynfed.md),
  [he-2025-feddt](/papers/he-2025-feddt.md),
  [sater-2021-anomaly-detection](/papers/sater-2021-anomaly-detection.md), and
  [cajas-ordonez-2025-edge-computing-survey](/papers/cajas-ordonez-2025-edge-computing-survey.md).
  Verified via widened re-grep (`layer|axes|axis|mentee|mentor|sparsi|low-rank|pruning`
  scoped to FedMAQ-context sentences): zero remaining stale hits.
- **Enrichment**: Added the formulation-study winner rule and the equal-weight (not
  active-selection) ensemble-distillation detail to
  [adaptive-precision-scheduling](/gaps/adaptive-precision-scheduling.md) and
  [multi-adaptive-q-kd-scarcity](/gaps/multi-adaptive-q-kd-scarcity.md), each now
  linking to [methods/fedmaq.md](/methods/fedmaq.md) as the canonical description
  rather than re-paraphrasing FedMAQ's design independently. The bits-to-accuracy
  winner-rule metric was judged literature-derived (already the subject of
  [communication-efficiency](/concepts/communication-efficiency.md)) and folded
  there rather than given a separate Concept node; the formulation-study
  winner-rule procedure itself is FedMAQ-coined and lives in
  [methods/fedmaq.md](/methods/fedmaq.md) section 3 instead of a single-referent
  Concept node.

## 2026-07-11 (completeness pass)

- **Audit**: full structural sweep of the bundle — frontmatter required fields
  (all 5 node types), presence of `# Related` sections, root-absolute link
  resolution (88 unique links, 0 broken), incoming-link check (0 orphan nodes),
  `resource:` path existence (all 39 paper nodes), and node counts vs. the
  figures stated in `index.md` and each section's `index.md` (39/25/10/8/6, all
  consistent).
- **Fix**: `findings/index.md` still listed the pre-sweep title "No method
  jointly optimizes multi-axis compression and heterogeneity robustness" for
  [no-unified-compression-heterogeneity-method.md](/findings/no-unified-compression-heterogeneity-method.md),
  which the 2026-07-11 de-overclaim pass had already retitled to "No existing
  method combines multi-signal adaptive compression with heterogeneity
  robustness" inside the node itself — the index summary line was never synced.
  Corrected.
- **Verified, no change needed**: `concepts/index.md`'s "across rounds, clients,
  and layers" line for [adaptive-bit-width](/concepts/adaptive-bit-width.md) is
  a general-literature description (layer-wise is a real variant in FedDT), not
  a FedMAQ claim — consistent with the node's own frontmatter and its Section 2
  scoping. `gaps/index.md` and `papers/index.md` blurbs checked against their
  target nodes; no drift found.

## 2026-07-24

- **Ingest**: Added 3 newly converted papers to the bundle (39 to 42 paper
  nodes): [bonawitz-2019-fl-scale](/papers/bonawitz-2019-fl-scale.md) (Google's
  production FL system design — new "Systems & Infrastructure" section in
  [papers/index.md](/papers/index.md)), [cheng-2021-fedgems](/papers/cheng-2021-fedgems.md)
  (FedGEMS — larger-server-model FL via selective, entropy-weighted ensemble
  distillation; `baseline: FedGEMS`), and
  [sandler-2018-mobilenetv2](/papers/sandler-2018-mobilenetv2.md) (MobileNetV2 —
  mobile CNN backbone, new "Backbone Architectures" section; not an FL method,
  no baseline field).
- **Creation**: Authored [methods/fedgems.md](/methods/fedgems.md) (26 to 27
  method nodes) for the FedGEMS algorithm, cross-linked from
  [concepts/knowledge-distillation.md](/concepts/knowledge-distillation.md).
- **Cross-links**: Added inbound references from
  [concepts/communication-efficiency.md](/concepts/communication-efficiency.md)
  to Bonawitz (systems context for the unimplemented-compression gap it names)
  and from [concepts/model-compression.md](/concepts/model-compression.md) to
  MobileNetV2 (backbone reference, explicitly distinguished from communication
  compression). Updated node counts in [index.md](/index.md),
  [papers/index.md](/papers/index.md), and [methods/index.md](/methods/index.md).

## 2026-07-12

- **Fix (naming collision)**: [methods/feddistill.md](/methods/feddistill.md)
  documented only Song et al. 2024's group-distillation/de-biasing mechanism
  (`introduced_by: /papers/song-2024-feddistill.md`), but the FedMAQ codebase's
  actual "FedDistill" baseline (`.claude/project/baseline_registry.md`, Pure KD
  group) implements Jeong et al. 2023's per-label averaged-logit exchange (FD)
  instead — both papers legitimately use variants of the name "FedDistill" for
  unrelated mechanisms. Rewrote
  [methods/feddistill.md](/methods/feddistill.md) to document Jeong's FD
  mechanism (`introduced_by: /papers/jeong-2023-feddistill-aug.md`), scoped to
  the FD component only (no FAug/GAN augmentation, to avoid duplicating
  [methods/fd-faug.md](/methods/fd-faug.md), which already covered Jeong's full
  paper method). Created
  [methods/feddistill-debias.md](/methods/feddistill-debias.md) preserving
  Song's mechanism verbatim under a disambiguated title. Cross-linked each new
  node to its sibling for disambiguation. Repointed references in
  [concepts/non-iid-heterogeneity.md](/concepts/non-iid-heterogeneity.md),
  [concepts/data-free-distillation.md](/concepts/data-free-distillation.md),
  and [concepts/knowledge-distillation.md](/concepts/knowledge-distillation.md)
  to `feddistill-debias.md` (all three referenced Song's de-biasing/non-IID
  mechanism in context); left
  [methods/fedmaq.md](/methods/fedmaq.md) and [methods/fedkd.md](/methods/fedkd.md)
  pointing at the corrected `feddistill.md`, since those are generic
  "FedDistill baseline" references matching the codebase algorithm. Updated
  [methods/index.md](/methods/index.md) (25 to 26 nodes) with a row for the new
  node and a corrected description for `feddistill.md`. Left
  `log.md`'s 2026-07-09 batch-3 entry unedited (historical record of what was
  authored then; link still resolves).
