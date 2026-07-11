# Findings

`type: Finding` nodes that synthesize evidence **across** papers into standalone,
citable claims for the FedMAQ manuscript — each stating a claim, an evidence table
of supporting papers, and links to the [gaps](/gaps/) it leaves open (8 nodes).

## Compression mechanisms

- [Adaptive bit-width beats uniform quantization under non-IID](/findings/adaptive-quantization-beats-uniform.md) - heterogeneity-aware precision reaches target accuracy in fewer total bits than a fixed bit-width.
- [Distillation cuts uplink but classically needs a proxy set](/findings/distillation-cuts-uplink-needs-proxy.md) - logit exchange decouples payload from model size, at the cost of a shared reference dataset.
- [Data-free KD removes the proxy requirement at an accuracy cost](/findings/data-free-kd-removes-proxy-cost.md) - generators and prototypes eliminate the shared-data dependency but trade accuracy.
- [Quantization and KD are complementary](/findings/quantization-kd-complementary.md) - jointly they exceed either alone; the technical premise of FedMAQ.

## Heterogeneity & objective

- [Drift correction aids convergence but adds state or communication](/findings/drift-correction-tradeoff.md) - regularization, control variates, and normalization each pay a distinct cost.
- [Communication efficiency is a bits-per-round vs rounds-to-converge trade-off](/findings/comm-efficiency-tradeoff.md) - the objective is the product, not raw single-round compression.

## Validation & positioning

- [Application deployments validate comm-efficient FL on real non-IID data](/findings/application-fl-validation.md) - energy, load, air-quality, and smart-city studies ground the thesis in realistic benchmarks.
- [No existing method combines multi-signal adaptive compression with heterogeneity robustness](/findings/no-unified-compression-heterogeneity-method.md) - the survey-and-corpus consensus that defines the FedMAQ niche.
