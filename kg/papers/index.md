# Papers

OKF `type: Paper` concept documents for the 39 papers in the FedMAQ thesis canon.
One node per paper; each links to its full-text conversion under `markdown/{slug}/` and
cross-links related papers via `# Related`. Grouped by ingestion batch / theme.

## Surveys & Reviews

- [Federated Learning for Smart Cities: A Thematic Review of Challenges and Approaches](alterkawi-2025-smart-cities-review.md) — Core problem: Smart cities generate vast, distributed, and privacy-sensitive data across heterogeneous IoT devices (e.g., traffic sensors, smart meters, healthcare wearables).
- [Intelligent Edge Computing and Machine Learning: A Survey of Optimization and Applications](cajas-ordonez-2025-edge-computing-survey.md) — This survey addresses the critical challenges of deploying machine learning (ML) models on resource-constrained edge devices within next-generation network infrastructures.
- [Non-IID data in Federated Learning: A Survey with Taxonomy, Metrics, Methods, Frameworks and Future Directions](jimenez-2024-non-iid-survey.md) — This paper is a comprehensive survey on the challenges of Non-IID (non-independent and identically distributed) data in Federated Learning (FL).
- [A Survey of Model Compression and Its Feedback Mechanism in Federated Learning](le-2024-compression-survey.md) — The paper addresses the critical challenge of communication efficiency in Federated Learning (FL), where transmitting large neural network models between clients and servers creates significant bandwidth bottlenecks.
- [Knowledge Distillation in Federated Learning: A Survey on Long Lasting Challenges and New Solutions](qin-2025-kd-survey.md) — Traditional parameter-based Federated Learning (FL) suffers from several long-lasting challenges: privacy risks from sharing model parameters, performance degradation under non-IID (heterogeneous) data distributions,.
- [Knowledge distillation in federated learning a comprehensive survey](salman-2025-kd-survey.md) — This paper is a comprehensive survey on the integration of Knowledge Distillation (KD) within Federated Learning (FL) systems.

## Federated Learning & Heterogeneity

- [Federated Learning Based on Dynamic Regularization](acar-2021-feddyn.md) — FedDyn adds a per-device dynamic regularizer so that local minima align with the global optimum, decoupling communication rounds from statistical heterogeneity.  _(baseline: FedDyn)_
- [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](karimireddy-2020-scaffold.md) — SCAFFOLD uses server/client control variates (variance reduction) to correct client-drift, proving convergence unaffected by data heterogeneity or client sampling.  _(baseline: SCAFFOLD)_
- [Federated Optimization in Heterogeneous Networks](li-2020-fedprox.md) — Federated learning (FL) suffers from two key challenges: (i) systems heterogeneity – devices have varying computational, communication, and energy capabilities; (ii) statistical heterogeneity – data across devices is.  _(baseline: FedProx)_
- [Model-Contrastive Federated Learning](li-2021-moon.md) — MOON corrects local drift in non-IID FL by model-level contrastive learning, maximizing agreement between local and global model representations.  _(baseline: MOON)_
- [Communication-Efficient Learning of Deep Networks from Decentralized Data](mcmahan-2017-fedavg.md) — Training deep neural networks on data that is privacy-sensitive, large in volume, and naturally distributed across mobile devices (e.g., language models, image classifiers).  _(baseline: FedAvg)_
- [FedProto: Federated Prototype Learning across Heterogeneous Clients](tan-2022-fedproto.md) — FedProto communicates abstract per-class prototypes (mean feature representations) instead of gradients, tolerating both data and model heterogeneity at low communication cost.  _(baseline: FedProto)_
- [Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](wang-2020-fednova.md) — FedNova identifies objective inconsistency from heterogeneous local-update counts and fixes it via normalized averaging, converging to the true global objective.  _(baseline: FedNova)_

## Quantization

- [QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding](alistarh-2017-qsgd.md) — QSGD is a family of stochastic gradient quantization schemes with convergence guarantees that let each node smoothly trade communication bits for convergence time.  _(baseline: QSGD)_
- [signSGD: Compressed Optimisation for Non-Convex Problems](bernstein-2018-signsgd.md) — signSGD transmits only the sign of each stochastic gradient coordinate, achieving 1-bit compression with SGD-level convergence and majority-vote aggregation in the distributed setting.  _(baseline: signSGD)_
- [Lightweight Adaptive Quantization Algorithms for Federated Learning With Heterogeneous Clients](cui-2026-laq-hc.md) — In federated learning (FL) with heterogeneous edge clients, uniform quantization levels fail to account for variations in client data quality, communication bandwidth, and computational capabilities.
- [DAdaQuant: Doubly-adaptive quantization for communication-efficient Federated Learning](honig-2022-dadaquant.md) — Federated Learning (FL) suffers from high communication costs due to repeated transmission of model parameters between server and clients.  _(baseline: DAdaQuant)_
- [Communication-Efficient Federated Learning for Heterogeneous Edge Devices Based on Adaptive Gradient Quantization](liu-2023-adagq.md) — Federated Learning (FL) suffers from massive communication overhead due to the frequent upload of model updates (gradients) from clients to a central server.
- [FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization](reisizadeh-2020-fedpaq.md) — Core problem: Federated learning suffers from a severe communication bottleneck due to frequent, large-message exchanges between many edge devices and a central server.  _(baseline: FedPAQ)_

## Knowledge Distillation

- [Distilling the Knowledge in a Neural Network](hinton-2015-distillation.md) — The paper addresses the deployment challenge of large, cumbersome neural network models (e.g., ensembles or heavily regularized single models) that are computationally expensive and high-latency for production use.
- [Communication-Efficient On-Device Machine Learning: Federated Distillation and Augmentation under Non-IID Private Data](jeong-2023-feddistill-aug.md) — Federated Distillation exchanges per-label averaged logits (payload independent of model size) and pairs it with GAN-based Federated Augmentation to counter non-IID skew, cutting communication ~26x versus FL.
- [FedMD: Heterogenous Federated Learning via Model Distillation](li-2019-fedmd.md) — Standard federated learning (e.g., FedAvg) requires all participants to share a single global model architecture.  _(baseline: FedMD)_
- [Ensemble Distillation for Robust Model Fusion in Federated Learning](lin-2020-feddf.md) — FedDF replaces parameter averaging with ensemble distillation on unlabeled/generated data, fusing heterogeneous client models (differing size, precision, or architecture) in far fewer rounds.  _(baseline: FedDF)_
- [FedDistill: Global Model Distillation for Local Model De-Biasing in Non-IID Federated Learning](song-2024-feddistill.md) — Federated Learning (FL) suffers from severe performance degradation under non-IID data distributions.  _(baseline: FedDistill)_
- [Data-Free Knowledge Distillation for Heterogeneous Federated Learning](zhu-2021-fedgen.md) — FedGen performs data-free knowledge distillation by learning a lightweight generator from clients' prediction rules and broadcasting it to regularize local training, removing the proxy-dataset requirement.  _(baseline: FedGen)_

## Quantization + Knowledge Distillation

- [DynFed: Adaptive Federated Learning via Quantization-Aware Knowledge Distillation](he-2025-dynfed.md) — Federated Learning (FL) suffers from high communication overhead, client resource heterogeneity (varying memory/compute), and data heterogeneity (non-IID distributions).
- [FedDT: A Communication-Efficient Federated Learning via Knowledge Distillation and Ternary Compression](he-2025-feddt.md) — Federated learning (FL) suffers from three major challenges: (i) data heterogeneity (non-IID distributions) causing model drift and performance degradation; (ii) model heterogeneity (different client architectures).
- [Quantization and Knowledge Distillation for Efficient Federated Learning on Edge Devices](qu-2020-quantization-kd.md) — Communication is a critical bottleneck in federated learning (FL) due to the large size of deep neural network weights and gradients that must be transmitted between clients and server, especially over heterogeneous.
- [CFD: Communication-Efficient Federated Distillation via Soft-Label Quantization and Delta Coding](sattler-2022-cfd.md) — The primary challenge addressed is the communication bottleneck in Federated Learning (FL), which is exacerbated by the frequent exchange of large neural network models (e.g., millions of parameters) between clients.  _(baseline: CFD)_
- [AdaDQ-KD: An Adaptive Dithering Quantization with Knowledge Distillation in Privacy-Preserving Federated Learning](wang-2026-adadq-kd.md) — Federated Learning (FL) with Differential Privacy (DP) faces a critical trade-off between privacy, communication efficiency, and model accuracy.
- [Communication-efficient federated learning via knowledge distillation](wu-2022-fedkd.md) — Standard Federated Learning (FL) requires communicating large model updates (gradients/weights) between clients and server, leading to prohibitive communication costs, especially for modern deep models with billions.  _(baseline: FedKD)_

## Applications (Energy / Smart Cities / Sensing)

- [Smart meter-based energy consumption forecasting for smart cities using adaptive federated learning](abdulla-2024-smart-meter.md) — Short-term energy consumption forecasting in smart grids faces three key challenges: (i) privacy concerns from aggregating sensitive household consumption data centrally, (ii) scalability issues when thousands of.
- [Air Quality Prediction Using Communication-Efficient Federated Learning with Compressed Deep Learning Models](joseph-2026-air-quality.md) — An FL-with-model-compression (FL-CM) framework for privacy-preserving air quality (PM2.5) prediction on IoT sensor networks, cutting communication overhead 71.7% versus standard FL without accuracy loss.
- [Communication-Efficient Federated Learning for Power Load Forecasting in Electric IoTs](mao-2023-power-load.md) — Frequent communication of high-dimensional model updates between heterogeneous clients and a central server in Federated Learning (FL) leads to prohibitively high communication costs and privacy risks, especially in.
- [Advancing Electric Load Forecasting: Leveraging Federated Learning for Distributed, Non-Stationary, and Discontinuous Time Series](richter-2024-electric-load.md) — Renewable Energy Communities (RECs) with *dynamic portfolios* (daily member composition changes) produce electrical load time series that are simultaneously:.
- [A Federated Learning Approach to Anomaly Detection in Smart Buildings](sater-2021-anomaly-detection.md) — Anomaly detection in IoT-enabled smart buildings (e.g., lighting faults, energy consumption prediction) is traditionally performed using centralized machine learning models.
- [Edge-Assisted Smart Campus Energy Management using Federated Learning and Context-Aware Control](singh-2026-smart-campus.md) — University campuses consume large amounts of electricity with highly variable occupancy patterns.
- [Federated Learning-Based Energy Management Systems for Privacy-Preserving Demand Forecasting in Smart Cities](sravanthi-2025-energy-management.md) — Traditional centralized Energy Management Systems (EMS) for smart cities face significant challenges regarding user data privacy, scalability, and latency.
- [Spatiotemporal Federated Learning for Privacy-Preserving Load Forecasting and Appliance Scheduling in Smart City Homes](thangakrishnan-2025-spatiotemporal-fl.md) — Smart city home energy management faces three interconnected challenges: (1) privacy concerns from centralized collection of sensitive household energy data, (2) uncertainty in load forecasting and renewable.
