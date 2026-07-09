# Gaps

`type: Gap` nodes recording open problems in the literature that FedMAQ targets —
each stating what remains unresolved, the state of the art, how FedMAQ addresses
it, and links to the [findings](/findings/) and [papers](/papers/) that motivate
it (6 nodes).

## Quantization under heterogeneity

- [Joint precision scheduling across round, client, and layer](/gaps/adaptive-precision-scheduling.md) - existing methods adapt one axis; the multi-axis schedule is unsolved.
- [Heterogeneity-aware quantization schedules](/gaps/heterogeneity-aware-quantization.md) - precision is allocated by bandwidth, rarely by statistical skew.
- [Quantization noise vs client drift interaction](/gaps/quantization-drift-interaction.md) - how compression error compounds with drift correction is uncharacterized.

## Distillation & the FedMAQ niche

- [Proxy-dataset dependence of federated KD](/gaps/proxy-data-dependence.md) - proxy-based KD needs public data; data-free alternatives pay an accuracy penalty.
- [Scarcity of multi-adaptive Q + KD methods](/gaps/multi-adaptive-q-kd-scarcity.md) - joint methods fix single precision and single teacher; the FedMAQ niche.

## Evaluation

- [Application-deployment evaluation gaps](/gaps/application-eval-gaps.md) - applied studies rarely report bits-to-accuracy or test adaptive compression.
