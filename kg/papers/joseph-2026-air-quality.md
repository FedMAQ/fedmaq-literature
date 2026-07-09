---
type: Paper
title: "Air Quality Prediction Using Communication-Efficient Federated Learning with Compressed Deep Learning Models"
description: "An FL-with-model-compression (FL-CM) framework for privacy-preserving air quality (PM2.5) prediction on IoT sensor networks, cutting communication overhead 71.7% versus standard FL without accuracy loss."
authors: "Joseph et al."
year: 2026
bibkey: joseph-2026-air-quality
tags: [application]
resource: markdown/joseph-2026-air-quality/paper.md
timestamp: 2026-06-21T10:17:35Z
---

## 1. Overview & Objectives

- **Core Problem**: Centralized air-quality forecasting raises **data-privacy**, **communication-overhead**, and **scalability** concerns across a federated sensor network. Plain FL preserves privacy but the frequent exchange of high-dimensional model parameters is costly in **bandwidth-constrained IoT/edge** deployments.
- **Main Objectives**:
  - Propose **FL-CM** (Federated Learning with Model Compression) for air-quality prediction: IoT sensing nodes train deep models locally on pollutant + meteorological data and transmit only **compressed model updates** to the server.
  - Reduce communication payload via compression (quantization / deep compression) **without sacrificing predictive accuracy**.
  - Show FL-CM beats traditional ML, centralized deep learning, and standard FL, cutting communication overhead by **71.70%** vs. FL-only.

## 2. Methodology & Key Innovations

- **Architecture**: distributed IoT nodes each train a deep sequence model (LSTM/CNN-style, suited to spatiotemporal PM2.5 dynamics) on local air-pollutant and weather data; a central server aggregates compressed updates into a global model.
- **Compression stage**: model-compression techniques (quantization and deep compression) shrink the uplink parameter payload each round — the mechanism delivering the 71.7% communication reduction.
- **Application framing**: an engineering integration of established components (FL + compression + deep temporal models) validated end-to-end on a distributed air-quality dataset, targeting real smart-city sensing deployments.
- **Evaluation**: benchmarked against traditional ML, centralized DL, and vanilla FL, reporting both predictive-accuracy metrics and communication savings.

## 3. Key Results & Setup

- **Communication**: **−71.70%** overhead relative to standard FL, attributed to compressed model updates.
- **Accuracy**: FL-CM reported to **outperform** traditional ML, centralized deep learning, and standard FL on the distributed air-quality dataset (accuracy preserved despite compression).
- **Task**: PM2.5 / air-quality-index forecasting from pollutant and meteorological time series across distributed sensing nodes.
- **Deployment target**: IoT-enabled, bandwidth- and resource-limited edge devices in smart cities.

## 4. Limitations & Constraints

- **Applied/empirical scope**: the contribution is an integration and experimental validation rather than a new compression algorithm or convergence theory.
- **Compression detail**: the specific quantization scheme and its adaptivity are described at an application level; there is no per-round or per-client adaptive bit-width mechanism.
- **Dataset specificity**: results are demonstrated on a particular distributed air-quality dataset; generalization across regions/pollutants is not exhaustively established.
- **No heterogeneity treatment**: non-IID sensor distributions and client/system heterogeneity are not a central focus.

## 5. FedMAQ Thesis Relevance

- **Target application + baseline**: this is a direct smart-city environmental-sensing use case for communication-efficient FL and a concrete application baseline for FedMAQ, alongside energy/load-forecasting applications such as [Mao (power load)](/papers/mao-2023-power-load.md) and [Richter (electric load)](/papers/richter-2024-electric-load.md).
- **Validates the FedMAQ premise on real data**: it empirically confirms that combining compression with FL yields large communication savings **without accuracy loss** in an IoT setting — the exact value proposition FedMAQ generalizes with *multi-adaptive* quantization and knowledge distillation.
- **Key insight to integrate**: FL-CM applies a **static** compression stage. FedMAQ's contribution over such application work is to make compression **adaptive** (per client/layer/round) and to add distillation — so this paper is a strong "before" baseline against which FedMAQ's adaptivity gains can be measured on spatiotemporal forecasting tasks.

# Related

- [Communication-Efficient Federated Learning for Power Load Forecasting in Electric IoTs](/papers/mao-2023-power-load.md)
- [Advancing Electric Load Forecasting: Leveraging Federated Learning for Distributed, Non-Stationary, and Discontinuous Time Series](/papers/richter-2024-electric-load.md)
- [Federated Learning for Smart Cities: A Thematic Review of Challenges and Approaches](/papers/alterkawi-2025-smart-cities-review.md)

# Citations

[1] Full-text conversion: [markdown/joseph-2026-air-quality/paper.md](markdown/joseph-2026-air-quality/paper.md)
[2] Source PDF: `papers/05 Applications/Joseph et al. - 2026 - Air Quality Prediction Using Communication-Efficient Federated Learning with Compressed Deep Learnin.pdf`
