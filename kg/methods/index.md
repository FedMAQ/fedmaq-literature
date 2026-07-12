# Methods

`type: Method` nodes for concrete FL algorithms — one per algorithm, cross-linked to
the [papers](/papers/) that introduce or evaluate them and the
[concepts](/concepts/) they instantiate. Grouped by family below (26 nodes).

## This thesis

- [FedMAQ](/methods/fedmaq.md) - multi-adaptive (resource + training-state +
  data-richness) client-level quantization plus server-side proxy ensemble
  distillation. No `introduced_by` paper node — this thesis's own algorithm; see
  the node for its manuscript source sections.

## FL core, heterogeneity & aggregation

- [FedAvg](/methods/fedavg.md) - iterative model averaging over multiple local SGD epochs; the base algorithm and primary baseline.
- [FedProx](/methods/fedprox.md) - FedAvg plus a proximal term limiting local drift, with tolerance of partial work.
- [SCAFFOLD](/methods/scaffold.md) - server/client control variates that subtract client-drift (variance reduction).
- [FedDyn](/methods/feddyn.md) - dynamic regularization aligning local optima with the global optimum.
- [MOON](/methods/moon.md) - model-contrastive local regularizer aligning local and global representations.
- [FedProto](/methods/fedproto.md) - exchanges per-class prototype vectors instead of gradients; model-heterogeneous.
- [FedNova](/methods/fednova.md) - normalized averaging removing objective inconsistency from unequal local work.

## Quantization

- [QSGD](/methods/qsgd.md) - unbiased, tunable-level stochastic gradient quantization with a provable bits-vs-variance trade-off.
- [signSGD](/methods/signsgd.md) - 1-bit sign compression with majority-vote aggregation.
- [FedPAQ](/methods/fedpaq.md) - periodic averaging + partial participation + quantized model deltas, with guarantees.
- [DAdaQuant](/methods/dadaquant.md) - doubly-adaptive quantization: precision rises over rounds and is allocated across clients by weight.
- [AdaGQ](/methods/adagq.md) - adaptive resolution by loss-decrease rate + per-client bit-widths that equalize round time.
- [LAQ-HC](/methods/laq-hc.md) - lightweight tanh impact model selecting per-client precision by data quality and bandwidth.

## Knowledge distillation

- [FedMD](/methods/fedmd.md) - public-dataset logit sharing and consensus distillation for architecture-heterogeneous clients.
- [FedDF](/methods/feddf.md) - server-side ensemble distillation over proxy data, tolerating size/precision/architecture heterogeneity.
- [FedGen](/methods/fedgen.md) - data-free KD via a broadcast label-conditioned generator regularizing local training.
- [FedDistill](/methods/feddistill.md) - per-label averaged logit exchange (Jeong et al.), payload independent of model size; the codebase's "FedDistill" baseline.
- [FedDistill (De-Biasing / Song 2024)](/methods/feddistill-debias.md) - group distillation de-biasing local classifiers over true/rich/few-sample classes; unrelated mechanism sharing the "FedDistill" name.
- [FedKD](/methods/fedkd.md) - mentee-mentor mutual distillation plus dynamic SVD gradient compression.
- [FD + FAug](/methods/fd-faug.md) - per-label logit exchange (payload independent of model size) with GAN-based augmentation.

## Joint quantization + knowledge distillation

- [CFD](/methods/cfd.md) - soft-label quantization + delta coding + dual server distillation.
- [DynFed](/methods/dynfed.md) - resource/gradient-adaptive bit-widths fused by active multi-teacher distillation.
- [FedDT](/methods/feddt.md) - personalized-teacher distillation into a shared student, then trained ternary quantization.
- [AdaDQ-KD](/methods/adadq-kd.md) - dithering quantization doubling as DP noise, plus feature-level KD, under straggler-aware precision.
- [AQFedAvg + FKD](/methods/quantized-kd.md) - bandwidth-adaptive gradient quantization followed by federated distillation of a smaller student.
