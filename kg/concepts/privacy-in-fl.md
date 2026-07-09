---
type: Concept
title: "Privacy in federated learning"
description: "The privacy guarantees and mechanisms — differential privacy, secure aggregation — layered on FL, and their interaction with compression."
tags: [privacy, differential-privacy, security]
timestamp: 2026-07-09T12:00:00Z
---

# Privacy in federated learning

## 1. Definition

Federated learning keeps raw data on-device, but shared gradients/updates can still
leak information, so formal privacy is added on top. The dominant mechanism is
differential privacy (DP): calibrated noise \(\mathcal N(0,\sigma^2)\) is added to
updates so that any single record's influence is bounded, giving an
\((\epsilon,\delta)\) guarantee. Complementary mechanisms include secure aggregation
(the server sees only the sum) and anomaly/attack detection at the aggregator.

## 2. Why it matters for FedMAQ

Privacy and compression interact rather than compose independently: quantization
already perturbs updates, and a dithering quantizer can be shown to inject noise
statistically equivalent to a DP mechanism — so one operation can serve compression
*and* privacy (AdaDQ-KD). FedMAQ treats this coupling as an opportunity, but must
track the accounting subtlety that adaptive per-client precision makes the per-round
privacy budget fluctuate.

## 3. Variants & dimensions

- **Differential privacy** — Gaussian/Laplace mechanisms; the accuracy-privacy-bits
  trilemma when noise doubles as compression error.
- **Secure aggregation** — cryptographic summation hiding individual updates.
- **Attack surface** — gradient-inversion and membership-inference threats motivating
  the above; anomaly detection as a defensive layer.
- **Quantization-as-privacy** — dithering/stochastic quantization noise reused as the
  DP mechanism.

## 4. Methods & papers

- Methods: [AdaDQ-KD](/methods/adadq-kd.md).
- Papers: [AdaDQ-KD](/papers/wang-2026-adadq-kd.md),
  [Sater 2021 anomaly detection](/papers/sater-2021-anomaly-detection.md).

# Related

- [Quantization](/concepts/quantization.md)
- [Adaptive bit-width](/concepts/adaptive-bit-width.md)
- [Edge / IoT deployment](/concepts/edge-iot-deployment.md)
