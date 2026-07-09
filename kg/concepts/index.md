# Concepts

`type: Concept` nodes for cross-cutting ideas that span many papers and methods.
Each defines one idea once and links out to the [methods](/methods/) that
instantiate it and the [papers](/papers/) that study it, so shared manuscript
terminology resolves to a single authoritative node (10 nodes).

## Compression

- [Quantization](/concepts/quantization.md) - reducing update precision to fewer bits, trading representation error for payload.
- [Adaptive bit-width](/concepts/adaptive-bit-width.md) - allocating precision non-uniformly across rounds, clients, and layers.
- [Model compression](/concepts/model-compression.md) - the umbrella over quantization, distillation, pruning, and factorization.

## Knowledge transfer

- [Knowledge distillation](/concepts/knowledge-distillation.md) - transferring teacher behavior into a student via soft outputs or features.
- [Proxy-dataset distillation](/concepts/proxy-dataset-distillation.md) - federated KD through model outputs on a shared public/unlabeled set.
- [Data-free distillation](/concepts/data-free-distillation.md) - KD with no shared data, via a learned generator or class prototypes.

## Objectives & setting

- [Communication efficiency](/concepts/communication-efficiency.md) - the bits-per-round versus rounds-to-converge trade-off FedMAQ optimizes.
- [Non-IID heterogeneity](/concepts/non-iid-heterogeneity.md) - divergent client data and system capacity that biases the global model.
- [Privacy in federated learning](/concepts/privacy-in-fl.md) - DP and secure aggregation, and their interaction with compression.
- [Edge / IoT deployment](/concepts/edge-iot-deployment.md) - the constrained, heterogeneous setting that motivates communication-efficient FL.
