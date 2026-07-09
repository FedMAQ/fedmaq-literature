---
type: Gap
title: "Federated distillation still depends on a proxy dataset or pays a data-free accuracy penalty"
description: "Proxy-dataset KD needs a distributionally relevant public set that may not exist in edge/IoT domains, while data-free alternatives that remove it lose accuracy — no method has both."
tags: [distillation, proxy-data, data-free, edge]
timestamp: 2026-07-09T12:00:00Z
---

# Federated distillation still depends on a proxy dataset or pays a data-free penalty

## Problem

The most communication-efficient federated KD relies on outputs over a shared public
or unlabeled reference set, but in energy, sensor, and IoT deployments no
distributionally relevant proxy set may exist. Data-free alternatives (generators,
prototypes, local augmentation) remove the dependency but generally lose accuracy and
add machinery. There is no method that achieves proxy-quality transfer without proxy
data.

## State of the art

FedMD, FedDF, FedKD, and CFD all assume a usable reference set; FedGen, FedProto, and
FD+FAug remove it but trade accuracy or restrict the transferred signal to
representations. Surveys note the proxy-set assumption as a defining, unresolved
constraint of federated KD.

## FedMAQ's angle

Because FedMAQ targets edge/IoT domains where public data is scarce, it must decide
which distillation regime it occupies and, ideally, narrow the proxy-vs-data-free gap
— for instance by using quantized soft labels or on-device synthesis so the
distillation channel remains cheap without a public corpus.

## Sources

- Motivating findings: [Distillation cuts uplink but needs a proxy set](/findings/distillation-cuts-uplink-needs-proxy.md),
  [Data-free KD removes the proxy requirement at a cost](/findings/data-free-kd-removes-proxy-cost.md).
- Papers: [FedMD](/papers/li-2019-fedmd.md), [FedDF](/papers/lin-2020-feddf.md),
  [FedGen](/papers/zhu-2021-fedgen.md), [FedProto](/papers/tan-2022-fedproto.md).

# Related

- [Proxy-dataset distillation](/concepts/proxy-dataset-distillation.md)
- [Data-free distillation](/concepts/data-free-distillation.md)
- [Edge / IoT deployment](/concepts/edge-iot-deployment.md)
- [FedMD](/methods/fedmd.md), [FedGen](/methods/fedgen.md)
