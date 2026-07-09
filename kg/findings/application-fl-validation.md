---
type: Finding
title: "Application-domain deployments validate communication-efficient FL on real non-IID data"
description: "Energy, load-forecasting, air-quality, and smart-city studies confirm that FL over constrained edge links with intrinsically non-IID data is viable, grounding the thesis in realistic benchmarks."
tags: [application, edge, iot, non-iid, communication-efficiency]
timestamp: 2026-07-09T12:00:00Z
---

# Application deployments validate communication-efficient FL on real non-IID data

## Scope

Covers applied FL studies in the corpus — smart metering, electric- and power-load
forecasting, air-quality sensing, smart campuses/cities, anomaly detection, and
spatiotemporal sensing — as external validation of the methods layer.

## Claim

The communication-efficiency and heterogeneity problems the methods layer addresses
are not synthetic: real deployments run over bandwidth-constrained edge links and
observe naturally non-IID, spatiotemporally skewed data, and they report that FL is
viable there while data stays on-device. These studies supply the realistic non-IID
benchmarks and constraint profiles against which FedMAQ's savings are argued, and they
motivate the specific pressures — bandwidth, stragglers, per-site skew — that make
adaptive quantization and distillation worthwhile.

## Evidence

| Paper | Key result | Link |
| --- | --- | --- |
| Smart-meter | FL for smart-metering load under per-household distribution skew | [/papers/abdulla-2024-smart-meter.md](/papers/abdulla-2024-smart-meter.md) |
| Power-load | Federated power-load forecasting over distributed sites | [/papers/mao-2023-power-load.md](/papers/mao-2023-power-load.md) |
| Electric-load | Federated electric-load forecasting with non-IID client data | [/papers/richter-2024-electric-load.md](/papers/richter-2024-electric-load.md) |
| Air-quality | Federated air-quality prediction across spatially distributed sensors | [/papers/joseph-2026-air-quality.md](/papers/joseph-2026-air-quality.md) |
| Spatiotemporal-FL | FL exploiting spatiotemporal correlation in sensor networks | [/papers/thangakrishnan-2025-spatiotemporal-fl.md](/papers/thangakrishnan-2025-spatiotemporal-fl.md) |
| Smart-campus | Campus-scale FL deployment under real resource constraints | [/papers/singh-2026-smart-campus.md](/papers/singh-2026-smart-campus.md) |

## Open gaps

- These deployments seldom report bits-to-accuracy or evaluate adaptive compression:
  [/gaps/application-eval-gaps.md](/gaps/application-eval-gaps.md).

# Related

- [Edge / IoT deployment](/concepts/edge-iot-deployment.md)
- [Non-IID heterogeneity](/concepts/non-iid-heterogeneity.md)
- [Communication efficiency](/concepts/communication-efficiency.md)
