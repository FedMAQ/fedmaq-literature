---
type: Concept
title: "Edge / IoT deployment"
description: "The resource-constrained, heterogeneous, bandwidth-limited setting — smart meters, sensors, campuses, vehicles — where communication-efficient FL is applied in practice."
tags: [edge, iot, application, deployment, communication-efficiency]
timestamp: 2026-07-09T12:00:00Z
---

# Edge / IoT deployment

## 1. Definition

Edge and IoT deployment is the operating regime that motivates communication-efficient
FL: many low-power devices with limited compute and memory, connected over
constrained, unreliable, and asymmetric links, holding naturally non-IID local data
(each meter, sensor, or site observes a skewed slice of reality). Real applications in
the corpus include smart metering, electric-load and power-load forecasting,
air-quality sensing, smart campuses/cities, and spatiotemporal sensing.

## 2. Why it matters for FedMAQ

This is FedMAQ's target environment and its justification. The bandwidth and device
limits are exactly what make multi-adaptive quantization worthwhile, and the intrinsic
non-IID structure of sensor data is what makes heterogeneity-aware precision and
distillation necessary rather than optional. The application papers supply the
realistic non-IID benchmarks against which FedMAQ's communication savings are argued.

## 3. Variants & dimensions

- **Domain** — energy/load forecasting, environmental sensing, smart-city/campus
  infrastructure, anomaly detection.
- **Constraint emphasized** — bandwidth, device compute/memory, straggler delay, or
  intermittent connectivity.
- **Data structure** — spatiotemporal correlation and per-site distribution skew that
  sharpen the non-IID challenge.

## 4. Methods & papers

- Application papers: [Smart-meter](/papers/abdulla-2024-smart-meter.md),
  [Air-quality](/papers/joseph-2026-air-quality.md),
  [Power-load](/papers/mao-2023-power-load.md),
  [Electric-load](/papers/richter-2024-electric-load.md),
  [Energy-management](/papers/sravanthi-2025-energy-management.md),
  [Anomaly-detection](/papers/sater-2021-anomaly-detection.md),
  [Smart-campus](/papers/singh-2026-smart-campus.md),
  [Spatiotemporal-FL](/papers/thangakrishnan-2025-spatiotemporal-fl.md).
- Surveys: [Cajas-Ordonez 2025 edge survey](/papers/cajas-ordonez-2025-edge-computing-survey.md),
  [Alterkawi 2025 smart-cities review](/papers/alterkawi-2025-smart-cities-review.md).

# Related

- [Communication efficiency](/concepts/communication-efficiency.md)
- [Non-IID heterogeneity](/concepts/non-iid-heterogeneity.md)
- [Privacy in federated learning](/concepts/privacy-in-fl.md)
