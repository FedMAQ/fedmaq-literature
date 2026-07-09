---
type: Gap
title: "Application deployments seldom evaluate adaptive compression or report bits-to-accuracy"
description: "Applied energy/sensor FL studies validate feasibility on real non-IID data but rarely measure communication-to-accuracy or test adaptive quantization plus distillation, leaving the thesis's target metric unbenchmarked in-domain."
tags: [application, evaluation, communication-efficiency, edge]
timestamp: 2026-07-09T12:00:00Z
---

# Application deployments seldom evaluate adaptive compression or bits-to-accuracy

## Problem

The application layer confirms that FL runs on real, non-IID edge data, but these
studies typically report task accuracy and sometimes a raw compression ratio, not the
bits-to-target-accuracy or round-time metrics that a communication-efficiency thesis
optimizes. Few apply adaptive quantization or distillation at all. The result is that
FedMAQ's central claim has no in-domain benchmark on the very deployments that motivate
it.

## State of the art

Smart-meter, power/electric-load, air-quality, smart-campus, and spatiotemporal FL
studies demonstrate feasibility and accuracy under heterogeneity, but their evaluation
protocols do not isolate communication-to-accuracy or compare compression schedules.
The methods layer, conversely, benchmarks compression mostly on vision/standard tasks
rather than these domains.

## FedMAQ's angle

FedMAQ should be evaluated with a communication-to-accuracy protocol on realistic
non-IID application data, bridging the methods layer's compression metrics and the
application layer's domains — turning this evaluation gap into the thesis's empirical
contribution.

## Sources

- Motivating findings: [Communication efficiency is a bits-per-round vs rounds trade-off](/findings/comm-efficiency-tradeoff.md),
  [Application deployments validate comm-efficient FL](/findings/application-fl-validation.md).
- Papers: [Smart-meter](/papers/abdulla-2024-smart-meter.md),
  [Power-load](/papers/mao-2023-power-load.md),
  [Electric-load](/papers/richter-2024-electric-load.md),
  [Spatiotemporal-FL](/papers/thangakrishnan-2025-spatiotemporal-fl.md).

# Related

- [Communication efficiency](/concepts/communication-efficiency.md)
- [Edge / IoT deployment](/concepts/edge-iot-deployment.md)
- [Non-IID heterogeneity](/concepts/non-iid-heterogeneity.md)
