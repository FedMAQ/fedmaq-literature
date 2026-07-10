# Gaps

`type: Gap` nodes recording open problems in the literature that FedMAQ targets —
each stating what remains unresolved, the state of the art, how FedMAQ addresses
it, and links to the [findings](/findings/) and [papers](/papers/) that motivate
it (6 nodes).

## Quantization under heterogeneity

- [Combining multiple adaptive signals into a client-level precision budget](/gaps/adaptive-precision-scheduling.md) - existing methods key precision on one signal; how resource, training-state, and data-richness should combine is open.
- [Heterogeneity-aware quantization schedules](/gaps/heterogeneity-aware-quantization.md) - precision is allocated by bandwidth, rarely by statistical skew.
- [Quantization noise vs client drift interaction](/gaps/quantization-drift-interaction.md) - how compression error compounds with drift correction is uncharacterized.

## Distillation & the FedMAQ niche

- [Proxy-dataset dependence of federated KD](/gaps/proxy-data-dependence.md) - proxy-based KD needs public data; data-free alternatives pay an accuracy penalty.
- [Joint Q+KD omits data richness and a combination study](/gaps/multi-adaptive-q-kd-scarcity.md) - DynFed fuses resource+state quantization with multi-teacher KD but omits data richness and a study of the combination logic; the FedMAQ niche.

## Evaluation

- [Application-deployment evaluation gaps](/gaps/application-eval-gaps.md) - applied studies rarely report bits-to-accuracy or test adaptive compression.
