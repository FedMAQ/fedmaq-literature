<!-- image -->

Review

## Intelligent Edge Computing and Machine Learning: A Survey of Optimization and Applications

<!-- image -->

Sebastián A. Cajas Ordóñez, Jaydeep Samanta , Andrés L. Suárez-Cetrulo * and Ricardo Simón Carbajo

<!-- image -->

Ireland's Centre for Artificial Intelligence (CeADAR), University College Dublin, Belfield, D04 V2N9 Dublin, Ireland;

sebastian.cajasordonez@ucd.ie (S.A.C.O.); jaydeep.samanta@ucd.ie (J.S.); ricardo.simoncarbajo@ucd.ie (R.S.C.)

* Correspondence: andres.suarez-cetrulo@ucd.ie

## Abstract

Intelligent edge machine learning has emerged as a paradigm for deploying smart applications across resource-constrained devices in next-generation network infrastructures. This survey addresses the critical challenges of implementing machine learning models on edge devices within distributed network environments, including computational limitations, memory constraints, and energy-efficiency requirements for real-time intelligent inference. We provide comprehensive analysis of soft computing optimization strategies essential for intelligent edge deployment, systematically examining model compression techniques including pruning, quantization methods, knowledge distillation, and low-rank decomposition approaches. The survey explores intelligent MLOps frameworks tailored for network edge environments, addressing continuous model adaptation, monitoring under data drift, and federated learning for distributed intelligence while preserving privacy in next-generation networks. Our work covers practical applications across intelligent smart agriculture, energy management, healthcare, and industrial monitoring within network infrastructures, highlighting domain-specific challenges and emerging solutions. We analyze specialized hardware architectures, cloud offloading strategies, and distributed learning approaches that enable intelligent edge computing in heterogeneous network environments. The survey identifies critical research gaps in multimodal model deployment, streaming learning under concept drift, and integration of soft computing techniques with intelligent edge orchestration frameworks for network applications. These gaps directly manifest as open challenges in balancing computational efficiency with model robustness due to limited multimodal optimization techniques, developing sustainable intelligent edge AI systems arising from inadequate streaming learning adaptation, and creating adaptive network applications for dynamic environments resulting from insufficient soft computing integration. This comprehensive roadmap synthesizes current intelligent edge machine learning solutions with emerging soft computing approaches, providing researchers and practitioners with insights for developing next-generation intelligent edge computing systems that leverage machine learning capabilities in distributed network infrastructures.

Keywords: edge machine learning; edge AI; IoT; model optimization; MLOps; quantization; knowledge distillation; low-rank adaptation; federated learning; data drift; multimodal fusion; resource-constrained devices; system resilience; ethical AI; large language models

<!-- image -->

Academic Editors: Giovanni Pau and Fabio Arena Received: 19 August 2025 Revised: 4 September 2025 Accepted: 9 September 2025 Published: 11 September 2025

Citation: Cajas Ordóñez, S.A.; Samanta, J.; Suárez-Cetrulo, A.L.; Carbajo, R.S. Intelligent Edge Computing and Machine Learning: A Survey of Optimization and Applications. Future Internet 2025 , 17 , 417. https://doi.org/10.3390/fi 17090417

Copyright: © 2025 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https://creativecommons.org/ licenses/by/4.0/).

<!-- image -->

## 1. Introduction

Intelligent edge computing has emerged as a crucial research frontier that bridges artificial intelligence with resource-constrained network environments, driven by the exponential growth of IoT devices and the increasing demand for real-time data processing [1,2]. As edge computing becomes pervasive across next-generation networks, edge machine learning (Edge ML) aims to deliver low-latency, privacy-preserving, and energy-efficient inference through intelligent orchestration of heterogeneous devices spanning the IoT-Edge-Cloud continuum [3,4]. This paradigm shift from centralized cloud computing to distributed edge intelligence addresses critical limitations in bandwidth, latency, and privacy while enabling sophisticated AI applications in resource-constrained environments [5,6]. This survey focuses on emerging optimization strategies, soft computing techniques, and system-level integrations designed to enable intelligent network applications.

The deployment of sophisticated ML models on intelligent edge networks faces challenges due to the growing complexity of modern architectures and the inherent limitations of distributed network environments. While deep learning has advanced rapidly, particularly through transformer architectures [7], the associated computational demands remain prohibitive for next-generation network applications, especially for Large Language Models (LLMs) and Large Multimodal Models (LMMs). Despite the wide adoption of deep neural networks [7-9], very few leverage soft computing principles for adaptive edge environments. This challenge extends to multimodal data processing [10] and the deployment of Visual Language Models (VLMs) on mobile network hardware [11-13].

The proliferation of intelligent edge nodes deployed across diverse network infrastructures further compounds these challenges [6,14-16]. In parallel, federated learning has emerged as a promising paradigm for distributing intelligent model training while preserving data locality in network environments [17,18], though challenges persist in managing energy, model updates, and system robustness across network topologies [1,3,5,19,20]. Moreover, connectivity limitations in edge networks continue to pose challenges for intelligent update synchronization and streaming stability [2,21].

To address these network resource constraints, researchers have developed various model compression and soft computing optimization techniques. Model compression methods such as pruning, quantization, low-rank approximation, and distillation have gained significant traction for intelligent network applications [22,23]. Lightweight finetuning methods like LoRA [24], QA-LoRA [25], QLoRA [26], and DoRA [27] enable large models to run efficiently on embedded network systems by reducing latency and memory usage [13]. These approaches are often combined with hybrid and extreme quantization methods (e.g., BitNet, BitNet b1.58) [13,28-30] and knowledge distillation techniques for intelligent edge computing [31].

In practical next-generation network deployments, intelligent task offloading becomes essential, where computational tasks are distributed from central systems to edge devices [2,32] to support intelligent IoT scenarios [33,34], while maintaining data quality [15] and ethical safeguards [35]. Given the increasing adoption of intelligent Edge ML in critical network sectors like transportation, energy, and urban systems, it is vital to investigate how model adaptation, orchestration, and system resilience interact in practice through soft computing approaches.

Despite these advances, significant gaps remain in current intelligent edge computing research. While previous surveys have explored various aspects of fog and edge computing [4,36-39], they often lack coverage of soft computing techniques and intelligent network application trends. For example, Singh et al. [40] highlight edge task scheduling and caching but omit advances in lightweight ML techniques like quantization and soft computing approaches. Similarly, while Verbraeken et al. [41] survey distributed ML, they overlook intelligent edge-specific systems with orchestration and modular deployment. Yu et al. [42] provide foundational architectural perspectives, but do not address ethics, multimodal alignment, and intelligent deployment pipelines.

Recent surveys have investigated specific aspects such as green design principles [43], containerized orchestration systems [44-46], distributed streaming pipelines [47], multimodal fusion [48-51], transfer learning [52], hardware-aware ML [23,53,54], and privacypreserving ML via federated learning [55-57]. However, these optimization techniques remain underexplored in critical areas such as stream learning [58-61], anomaly detection [62], and distributed MLOps pipelines like dataClay [63]. Furthermore, quantization techniques lack full integration with orchestration frameworks [63,64], and most studies fail to incorporate lightweight tuning methods (e.g., BitNet) [28-30], stream learning under concept drift, and edge-ready anomaly detection or MLOps systems.

This survey addresses these gaps by providing a comprehensive review of modern intelligent Edge ML challenges, emphasizing real-world deployment in next-generation networks, modular system composition, and resilience under continuous data drift. We focus on dynamic, high-throughput settings such as intelligent smart grids [65-67], industrial monitoring [68,69], and autonomous mobility networks [70-72]. We also explore the intricacies of deploying intelligent ML systems in production environments with constrained network resources, advocating for effective monitoring and continual adaptation through soft computing principles [73,74].

To structure our survey, we begin with foundational optimization strategies and intelligent Edge ML techniques. We then review application areas and emerging toolchains for next-generation network applications. Finally, we highlight critical open questions to guide future research focused on balancing efficiency, trustworthiness, and sustainability in intelligent edge computing environments.

## 2. Machine Learning Background

The critical aspects for real-world implementation of intelligent edge machine learning (ML) models in next-generation networks include addressing model efficiency, robustness, footprint, AI degradation, and trust challenges within distributed computing environments. To tackle footprint concerns, deep compression techniques through pruning unnecessary connections and intelligent weight distribution have gained significant attention [22,23]. Collectively, these studies highlight how crucial it is to address these issues when deploying intelligent Edge ML models in network-aware practical applications. In this section, we cover essential concepts for intelligent model optimization and MLOps at the edge, model degradation, and data drift, as well as security and privacy for intelligent edge machine learning systems.

## 2.1. Intelligent Model Optimization at the Edge

This section introduces the main soft computing techniques to reduce GPU consumption, RAM utilization, computational needs by decreasing FLOPs count, and inference delay for intelligent network applications. This involves introducing sparsely connected deep neural networks for efficient deployment on resource-limited edge devices within next-generation network infrastructures, enhancing accessibility to training these networks with high-quality IoT datasets [75,76]. This approach is intended to reduce parameters and optimize networks for intelligent edge limitations in distributed environments.

Multiple soft computing techniques can be used for faster intelligent inference at the edge; these include parallelization across distributed network devices, memory offloading to speed up inference time by intelligently managing temporary data [40], and adaptive model optimization techniques such as (i) pruning, (ii) distillation, and (iii) quantiza- tion [77,78], as well as (iv) low-rank decomposition methods for intelligent language model optimization, which substantially reduce trainable parameters. These methods adjust neural network architectures to edge network restrictions such as latency and memory, allowing complex models to operate seamlessly on computationally limited network devices while achieving energy efficiency [22].

Table 1 provides a structured comparison of pruning, quantization, distillation, and low-rank factorization, emphasizing their impact on memory usage, accuracy impact, latency improvement, and best-suited use cases. This allows for informed decisions when selecting optimization techniques for resource-constrained devices.

Table 1. Comparison of neural network optimization techniques: ranges reflect variability across model architectures, datasets, and hardware platforms. Performance improvements are hardware-dependent and may not be additive when techniques are combined. Based on studies from [26,64,79-86].

| Technique                | Memory Reduction   | Accuracy Impact   | Latency Improvement           | Typical Use Case                          |
|--------------------------|--------------------|-------------------|-------------------------------|-------------------------------------------|
| Structured Pruning       | 2×-10× smaller     | 0.1-5% loss       | 1.2×-3× faster                | Hardware-friendly edge deployment         |
| Unstructured Pruning     | 5×-50× smaller     | 1-8% loss         | Limited improvement           | Memory-constrained scenarios              |
| INT8 Quantization        | 4× smaller         | 0.5-3% loss       | 1.5×-3× faster (edge devices) | Mobile inference optimization             |
| INT4/Binary Quantization | 8×-16× smaller     | 2-15% loss        | 2×-4× faster (specialized HW) | Ultra-low resource deployment             |
| Knowledge Distillation   | 2×-5× smaller      | 0.5-3% loss       | Proportional to compression   | Model compression with accuracy retention |
| Low-Rank Factorization   | 1.5×-4× smaller    | 0.1-2% loss       | 1.2×-2.5× faster              | Fine-tuning large models                  |

## 2.1.1. Intelligent Pruning Techniques

Pruning improves model performance by removing unnecessary connections via sparse connectivity for intelligent edge applications [79,80]. It maintains performance on edge nodes to allow high efficiency in network environments, reducing overparameterization while retaining accuracy. Excess components are removed using techniques such as weight, unit, and structural pruning, which are optimized for nextgeneration network applications. These methods optimize neural networks for resourceefficient, high-performance deployment in distributed intelligent systems. Pruning approaches can lower the parameters of trained networks by up to 90% [87].

Advanced techniques include local pruning for layer-wise optimization, global pruning for network-wide efficiency, and custom pruning with intelligent masking for specific network requirements [77]. Complementing these optimization approaches, frameworks like TensorFlow Lite have been specifically designed for scalable intelligent neural network deployment on resource-constrained network devices [88]. These frameworks provide optimized ML models for next-generation network applications, offering local data processing capabilities that eliminate cloud dependency, reduce latency through intelligent real-time operation, have offline functionality, and enhance data security for distributed edge environments.

## 2.1.2. Intelligent Quantization for Network Applications

Quantization is a fundamental soft computing optimization technique for converting tensors to lower precision, such as integers rather than floating-point values, enabling faster and more efficient computation in next-generation network environments [64,81]. An 8-bit integer-quantized version of a 32-bit floating-point model is approximately 4 × smaller in size and 1.5-4.0 × faster in computation. As a result, quantized models have a smaller memory footprint, making them ideal for deployment on resource-constrained edge devices in intelligent network infrastructures.

Deep learning has demonstrated significant performance improvements but at high computational cost, necessitating quantization tools for efficient and accurate intelligent inference schemes that address accuracy and latency in network applications [81]. TensorFlow Lite and PyTorch [85,89] enable efficient quantization with different methods, supporting customization for intelligent edge deployment in distributed networks.

There are two primary quantization approaches for creating low-bit models in intelligent network applications:

- Post-training quantization: This approach reduces the precision of weights and activations after model training, supporting various quantization levels including 8-bit variants [90], 4-bit variants [91,92], 2-bit quantization [93], and 1-bit quantization (BitNet variants) [13,28] that replace matrix multiplication with integer addition for intelligent edge applications. While simple to implement for network deployment, this method might result in accuracy loss. It includes
1. Quantizing only weights;
2. Quantizing both weights and activations [94].
- Quantization-aware training: This employs quantization during model training, achieving better accuracy for intelligent network applications. This technique incorporates simulated quantization operations using automated tools from the TensorFlow and PyTorch libraries [89].

## 2.1.3. Knowledge Distillation for Intelligent Edge Networks

Knowledge distillation enables smaller models to emulate larger ones through soft computing approaches, optimizing model size and accuracy for application-specific and resource-constrained network environments.

These approaches follow an intelligent teacher-student learning strategy, utilizing larger teacher networks to train compact student networks with minimal accuracy loss for edge deployment. Combined with transfer learning, network distillation significantly reduces model size without compromising performance in intelligent network applications [85]. Transferring knowledge from ensemble models or large regularized models into smaller, distilled models achieves comparable performance for next-generation network applications.

Knowledge distillation reduces training times considerably, facilitating easier deployment in intelligent network environments [84]. As shown in Figure 1, the teacher model transfers knowledge to the student model through intelligent distillation processes.

Figure 1. The intelligent teacher-student framework for knowledge distillation in edge networks [31].

<!-- image -->

Distillation techniques for intelligent edge applications, illustrated in Figure 2, include three categories:

- Response-based knowledge: The student model learns from teacher predictions, with distillation loss reducing logit differences for intelligent network optimization [95].
- Feature-based knowledge: Intermediate layers reduce feature discrepancies between models, enabling students to emulate teacher neuron activations in distributed environments [96].
- Relational knowledge: This evaluates feature maps and similarity matrices, understanding feature correlations across multiple representations for intelligent edge applications [97].

Figure 2. The three primary knowledge forms in intelligent distillation models for network applications [31].

<!-- image -->

Advanced distillation algorithms for intelligent network applications include (i) adversarial distillation, which uses generative principles for improved data representation [98]; (ii) multi-teacher distillation, which collects knowledge from multiple models; (iii) cross-modal distillation, which is used for multimodal network applications; and (iv) DistilGPT2, which demonstrates effective language model compression for intelligent edge deployment [99,100].

Additional techniques including attention-based, data-free, quantized, and lifelong distillation have emerged as valuable strategies for intelligent edge computing [101-103]. These methods enable more efficient and adaptable deep learning models for intelligent network inference while maintaining an optimal accuracy-latency balance.

## 2.1.4. Low-Rank Decomposition Methods for Intelligent Networks

Neural network models in next-generation network applications necessitate vast amounts of data and typically consist of millions to billions of parameters, creating significant computa- tional challenges for intelligent edge deployment [7]. This reliance on expensive computational resources highlights the importance of devising specialized soft computing optimization techniques to effectively minimize costs in distributed network environments.

Research has demonstrated that over-parameterized models reside on low intrinsic dimensions [104,105], leading to the development of Low-Rank Adaptation (LoRA) [24]. LoRA focuses on training only parameter subsets from dense layers, optimizing resources and enhancing energy efficiency for intelligent network applications within transformer architectures. This method freezes pre-trained model weights while incorporating trainable rank decomposition matrices into each layer, significantly reducing trainable parameters for efficient intelligent edge deployment.

LoRA has enabled rapid development of advanced adaptations for next-generation network applications:

- QLoRA [26] optimizes weight parameters by reducing the 32-bit format to 4-bit quantization space, significantly reducing memory usage for intelligent edge networks while maintaining training effectiveness through dynamic precision switching.
- QA-LoRA [25] combines quantization and fine-tuning of LoRA parameters, balancing adapter and quantization parameters through group-wise operators for distributed network optimization.
- DoRA [27] enhances LoRA by decomposing pre-trained weights into magnitude and direction components, focusing on directional adaptation to improve scalability and learning capacity while reducing training overhead for intelligent network applications.

Low-rank matrix factorization plays a pivotal role in enhancing adaptability and efficiency for intelligent edge computing in network environments. Parameter-Efficient Fine-Tuning frameworks provide accessible deployment avenues for LoRA variations, enabling model optimization for next-generation network applications [106].

While optimization techniques reduce resource consumption, managing these models at scale introduces new challenges, as outlined in the following section on MLOps, which discusses the context where MLOps operates, particularly in real-world scenarios, highlighting its main significance in edge machine learning deployment:

## 2.2. Intelligent MLOps at the Edge

Recent developments in machine learning have focused on building intelligent models for diverse domains including health, finance, defense, entertainment, and commerce within next-generation network environments. These models, developed by domain experts and data scientists, constantly evolve as new data becomes available to enhance their capabilities for resource-constrained network devices. AI frameworks provide powerful toolkits for building complex predictive systems in distributed environments. However, real-world deployment involves maintenance costs that, if overlooked, may lead to technical debt in intelligent ML systems [107]. Unlike standard software design, ML solutions require data relationships that challenge long-term maintenance in network environments [108]. This necessitates complex pipelines for framework enhancement, making it challenging to retrain and deploy intelligent machine learning models into production [73]. This complex pipeline constitutes Machine Learning Operations or MLOps for intelligent edge computing [74].

Running intelligent MLOps at the edge becomes crucial with increasing IoT demand and the benefits provided by edge devices supporting ML frameworks in next-generation network infrastructures compared to traditional cloud-based approaches [109]. With the proliferation of IoT devices and edge data availability, leveraging intelligent edge computing for inference has become standard for efficient information processing and insight gathering from distributed network devices [3].

Recent frameworks have addressed intelligent MLOps deployment. Raj et al. [110] proposed synchronizing the development, deployment, and monitoring of ML models for real-time automation across cloud and edge operations in network environments. John et al. [111] developed frameworks helping organizations integrate intelligent MLOps into existing software development practices for next-generation network applications.

Since intelligent ML operations span all AI application stages, AI software sustainability presents significant risks. Recent research addresses the challenges and trends for sustainable intelligent ML operations in distributed network environments [112].

## 2.2.1. Intelligent MLOps Pillars and Goals

Successful intelligent MLOps implementation depends on four critical pillars for next-generation network applications, which are illustrated in Figure 3. This framework emphasizes intelligent MLOps technologies in accelerating the complete machine learning lifecycle for edge networks.

- Intelligent Model Deployment and Experimentation: Simplifies model creation and deployment by optimizing data procedures and verifying that intelligent models function as intended in real-world network environments.
- Intelligent Model Monitoring: Monitors model performance across various network situations, recognizing data drift and limiting risks associated with incorrect predictions in distributed environments.
- Intelligent Production Deployment: Automates critical operations including model upgrades, troubleshooting, approval, updates, and scalability for seamless integration into operational network settings.
- Preparation for Intelligent Production Release: Includes version control, automated documentation, update tracking, and risk assessment, ensuring seamless model releases in network environments.

Figure 3. Intelligent MLOps pillars and associated roles for next-generation network applications [113].

<!-- image -->

## 2.2.2. Intelligent MLOps Tools for Network Applications

Building, deploying, and managing intelligent machine learning models in production requires efficient MLOps tools that deliver high-quality models and enhance collaboration between development and operations teams in next-generation network environments.

Numerous open-source platforms support intelligent ML project management and deployment. MLFlow [114] provides tracking, project management, and model registry functions for distributed networks. Kubeflow [115] enables scalable intelligent ML model deployments using Kubernetes for network infrastructures. MLReef [113] utilizes gitbased repositories for collaborative intelligent ML development. ZenML [116] simplifies the setup and maintenance of intelligent ML pipelines for edge networks. MLRun [117] provides standardized interfaces for different ML libraries and data storage systems in distributed environments. Seldon Core [118] offers standardized interfaces for updating and deploying intelligent ML models in practical network settings. Tables 2 and 3 compare these frameworks for next-generation network applications looking into features such as data versioning, hyperparameter tuning, experiment and pipeline versioning, continuous integration and delivery, and model deployment and performance monitoring (Table 2), as well as their ease of use, scalability, edge compatibility, and use cases (Table 3).

Table 2. Comparison of key features across leading MLOps platforms, highlighting capabilities such as data versioning, experiment tracking, pipeline orchestration, and model deployment. Adapted from [113].

| Platform             | DV   | HT   | MEV   | PV   | CI/CD   | MD   | PM   |
|----------------------|------|------|-------|------|---------|------|------|
| AWS SageMaker        | ✓    | ✓    | ✓     | ✓    | ✓       | ✓    | ✓    |
| MLFlow               | ✓    | ✓    | ✓     | ✓    | ✓       | ✓    |      |
| Kubeflow             |      | ✓    | ✓     |      | ✓       | ✓    | ✓    |
| DataRobot            |      | ✓    | ✓     |      |         | ✓    | ✓    |
| Iterative Enterprise | ✓    |      | ✓     |      | ✓       | ✓    | ✓    |
| ClearML              | ✓    |      | ✓     | ✓    | ✓       | ✓    | ✓    |
| MLReef               | ✓    | ✓    | ✓     | ✓    | ✓       | ✓    | ✓    |
| Streamlit            | ✓    |      | ✓     |      |         | ✓    | ✓    |

Abbreviations-DV: Data Versioning; HT: Hyperparameter Tuning; MEV: Experiment Versioning; PV: Pipeline Versioning; CI/CD: Continuous Integration/Delivery; MD: Model Deployment; PM: Performance Monitoring.

Table 3. Comparison of existing MLOps platforms adapted from [74,119].

| Tool       | Ease of Use   | Scale     | Edge Compat   | Best Use Case                                 |
|------------|---------------|-----------|---------------|-----------------------------------------------|
| MLflow     | Moderate      | Good      | Variable      | Strong tracking/registry. Edge: model format. |
| W&B        | High          | Excellent | Variable      | Excellent viz/tracking. Edge: model format.   |
| CometML    | High          | Excellent | Variable      | Robust tracking. Edge: model format.          |
| Kubeflow   | Complex       | Excellent | Moderate      | K8s-native, powerful but complex.             |
| BentoML    | Moderate      | Good      | Good          | Optimized serving; edge-suitable.             |
| SageMaker  | Mod-High      | Excellent | Good          | Comprehensive suite, edge manager.            |
| Databricks | Mod-High      | Excellent | Variable      | Big data scaling. Edge: model format.         |
| Streamlit  | High          | Moderate  | Variable      | Quick dashboards, interactive viz.            |
| MLReef     | Moderate      | Good      | Good/Var.     | Full-stack: deploy and monitor models.        |
| DVC        | Moderate      | Excellent | Limited       | Git-like versioning, reproducible ML.         |
| DataRobot  | High          | Excellent | Good/Var.     | End-to-end AutoML, explainability.            |

## 2.3. Intelligent AI Degradation and Data Drifts in Network Environments

Various factors contributing to data distribution changes or environmental variations in next-generation network infrastructures can cause intelligent model deterioration, significantly impacting predictive accuracy in distributed edge computing environments [59,120].

In intelligent network applications, concept drift represents 'a change in data distribution and evolution of relationships between attributes and target features over time, or transitions between generative processes in network data streams. These transitions occur with different speeds, severity, and distribution patterns" [59].

Shifts and drifts may occur when data streams from edge devices in network infrastructures evolve, particularly when computational jobs change or environmental conditions surrounding sensors shift in intelligent IoT deployments [121]. Anomalies and data shifts can be monitored using statistical functions over sliding windows of data statistics and model performance metrics in distributed network environments [58,122].

Furthermore, intelligent models can be trained to adapt to data drifts through soft computing approaches utilizing forgetting mechanisms [123], offering better balance between performance and inference time compared to traditional algorithms in next-generation network applications [124]. Model explainability can be integrated with these mechanisms to increase transparency and comprehend intelligent model behavior in network environments, identifying improvement areas for enhanced performance in distributed edge computing systems [118].

## 2.4. Intelligent Federated Learning at the Edge

For real-world implementation in next-generation network applications, addressing intelligent Edge ML concerns of security, privacy, robustness, and trust is crucial. Federated learning (FL) is essential for enhancing privacy and security in distributed network environments. FL protects data privacy while producing accurate results by enabling edge devices to jointly train intelligent models without sharing raw data across network infrastructures. This approach creates the trust necessary for intelligent Edge ML to reach its potential in diverse network applications. Additionally, soft computing methods like bias reduction and anomaly identification contribute to the fairness and robustness of intelligent edge machine learning models in distributed environments. Open-source federated learning frameworks enable joint training of intelligent edge models, fostering transparency and trust while preserving privacy in network deployments.

Federated learning is an intelligent machine learning technique that enables model training on decentralized data sources across distributed network devices while optimizing privacy in next-generation network environments [1,17]. This strategy maximizes data utilization for training while maintaining data privacy in network infrastructures. Advantages of this method include data privacy preservation, scalability via distributed datasets, efficiency through minimized data transfer, and increased model resilience from varied data sources, improving robustness and generalizability for collaborative learning in intelligent network applications.

Figure 4 illustrates intelligent federated learning where data owners train local data within clusters A, B, and C, ultimately sharing updates through aggregation methods to update the global model in distributed network architectures.

Figure 4. Intelligent client-server federated learning architecture for next-generation networks [125].

<!-- image -->

Several foundational approaches have been developed for intelligent federated learning to address various challenges in network environments. FedAvg [17] enables clients to update local models and aggregate at servers. FedProx [126] uses regularization terms to reduce weight discrepancies between local and global models in heterogeneous networks. FedPer [127] trains personalized models with shared base layers and individualized local layers for network clients.

Advanced approaches include FedOpt [128], which uses adaptive optimization methods for intelligent device selection, SplitFL [129], which employs dual models for local updates, and FedTL [130], which leverages pre-trained global models for device-specific refinement. FedEL [131] utilizes evolutionary algorithms for intelligent federated training. These strategies advance FL by addressing specific challenges in distributed network environments.

Alternative privacy protection approaches include hashing encryption techniques based on data fusion [132], exploring spatial-temporal privacy exposure risks in distributed intelligent edge environments.

Three categories of open-source intelligent federated learning frameworks support next-generation network applications [133]:

- All-in-one frameworks: Comprehensive solutions like FATE [134] and FedML [135] provide secure collaborative FL on decentralized network data.
- Horizontal-only frameworks: User-friendly APIs like Flower and FLUTE [136] emphasize simplicity for intelligent network applications.
- Specialized frameworks: Goal-specific solutions can also be used, like CrypTen [137] for secure multi-party computing and FedTree [138] for federated decision tree training in network environments.

GPU-accelerated tools, including Nvidia FLARE SDK [139] and PyTorch-based solutions like PySyft [140] enable scalable intelligent FL deployment. These frameworks offer diverse functionalities for various network applications, advancing intelligent federated learning adoption in next-generation network infrastructures.

## 2.5. Performance Evaluation Metrics for Intelligent Edge AI

Evaluating the performance of intelligent edge AI systems often requires a number of factors listed below, consisting of a multi-dimensional assessment across computational efficiency, resource utilization, model quality, and system-level characteristics, among others, to ensure reliable deployment in next-generation network environments [141,142].

- Computational metrics form the foundation for evaluating edge AI performance, with latency measured as the time from input to output completion, mathematically expressed as L = t end -tstart , where processing delays are critical for real-time applications. Throughput quantifies system capacity as the number of inference operations completed per unit time: T = Noperations t elapsed . Inference time specifically measures the duration required for model prediction on input data [143,144].
- Resource utilization metrics assess system efficiency through energy consumption measurement, typically expressed as energy per inference operation Einference = Pavg × t in f erence Ninferences , where Pavg represents average power consumption [145,146]. Memory utilization is quantified as Mutil = Mused Mtotal × 100%, while CPU and GPU utilization percentages indicate processing resource efficiency [147,148].
- Model quality metrics ensure intelligent edge systems maintain acceptable accuracy levels. These metrics are highly dependent on the task performed by the AI models. An example for classification tasks would be classification-accuracy-related metrics. For instance, Accuracy = TP + TN TP + TN + FP + FN , Precision = TP TP + FP , Recall = TP TP + FN , and f 1 score = 2 × Precision × Recall Precision + Recall , where TP, TN, FP, and FN represent true positives, true negatives, false positives, and false negatives, respectively [3,6].
- System-level metrics evaluate operational characteristics including availability measured as Availability = MTBF MTBF + MTTR × 100%, where MTBF is mean time between failures and MTTR is mean time to repair. Scalability metrics assess system performance under varying loads, while reliability quantifies system stability over extended operation periods [37].
- Standardized evaluation frameworks provide consistent benchmarking approaches, with, for instance, MLPerf serving as the industry standard for measuring AI inference performance across diverse hardware platforms, supporting edge-specific benchmarks including MLPerf Inference Edge and MLPerf Mobile for comprehensive system evaluation [141,149]. These frameworks enable fair comparison across different edge AI implementations while supporting reproducible research and development efforts in intelligent edge computing environments.

## 3. Intelligent Edge ML Use Cases and Application Domains for Next-Generation Networks

In this section, we aggregate real-world use cases for intelligent edge machine learning in next-generation network applications that address the constraints of distributed edge deployment [109,150]. Table 4 provides a comprehensive comparison of edge AI characteristics, requirements, and challenges across major application domains, including agriculture, energy, healthcare, manufacturing, transportation, retail, smart cities, and finance.

We begin by introducing intelligent cloud offloading and highlighting real-time visionassisted applications for network environments. We emphasize the need to reduce latency in next-generation network applications, particularly in scenarios like intelligent video offloading, where data streaming is essential for optimizing network consumption. For in- stance, smart home scenarios in network infrastructures aim to reduce power usage and CO2 emissions through intelligent edge computing [66,151]. In these and other use cases, collaborative intelligent edge computing [1,152] enables data communication across distributed network nodes and geographically dispersed intelligent systems.

This section examines applications including (i) intelligent smart cities and smart agriculture using edge computing for enhanced decision-making while minimizing environmental impact in network environments [67,153]; (ii) remote patient vital sign monitoring in intelligent healthcare networks [154,155]; (iii) intelligent internet of vehicles enabling autonomous driving and optimized routing in next-generation network infrastructures [156,157]; and (iv) intelligent smart industry approaches addressing failure detection and predictive maintenance in distributed network environments [158]. The domain-specific requirements and challenges outlined in Table 4 highlight the complexity of deploying edge AI solutions across these diverse applications. Finally, we discuss the optimization of intelligent operating systems and smart environments for next-generation network applications.

Table 4. Characteristics, requirements, and challenges of edge AI across different application domains. Sources [66,151,153-158].

| Domain         | Key Characteristics                                             | Requirements                                                       | Main Challenges                                                |
|----------------|-----------------------------------------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------|
| Agriculture    | Precision farming, crop tracking, weather prediction            | Low power/wide coverage, weather resistance, real-time data        | Rural connectivity, harsh conditions, cost                     |
| Energy         | Smart grid, predictive maintenance, load balancing              | High reliability, real-time decisions, system integration          | Safety, regulatory compliance, scalability                     |
| Healthcare     | Patient monitoring, diagnostics, wearables, emergency response  | Ultra-low latency, high accuracy, privacy                          | Data privacy, life-critical accuracy, device size              |
| Manufacturing  | Quality control, predictive maintenance, robotics, supply chain | Real-time processing, high precision, system integration           | Harsh environments, legacy systems, minimal downtime           |
| Transportation | Autonomous vehicles, traffic management, fleet optimization     | Ultra-low latency, high reliability/safety, real-time coordination | Safety, regulatory approval, infrastructure integration        |
| Retail         | Inventory, analytics, recommendations, checkout                 | Customer privacy, real-time analytics, scalability, cost           | Privacy concerns, behavior patterns, POS integration           |
| Smart Cities   | Traffic/environmental monitoring, public safety                 | Wide area deployment, interoperability, scalability                | Infrastructure complexity, data integration, public acceptance |
| Finance        | Fraud detection, trading, risk assessment, automation           | Ultra-low latency, high security, real-time processing             | Regulatory demands, security threats, high-frequency decisions |

## 3.1. Intelligent Energy Management for Network Applications

Intelligent AI systems can automate energy usage and storage in residential environments within next-generation network infrastructures, serving as crucial elements for decreasing costs, minimizing CO2 emissions, and enhancing Quality of Service (QoS) [151,159]. The availability of micro-generation systems, electric vehicles, heat pumps, home energy storage, and smart meters provides essential data granularity for intelligent AI models that enable prediction, maintenance, and prevention of energy waste while reducing CO2 emissions in distributed network environments [160].

Energy prediction in intelligent network applications depends on factors like building type, weather, structural features, and subsystem operations through soft computing approaches [65]. Occupancy behavior enhances statistical and ML models for consumption forecasting in smart network environments. Research demonstrates that simple intelligent models like logistic regression effectively support policy decisions for sustainable energy use in next-generation networks [67]. Real-time electricity usage data provision reduces residential consumption with robust results in intelligent network deployments [66]. Additional efforts include optimizing energy-efficient communication for video streaming in network applications [159].

## 3.2. Intelligent Smart Agriculture in Network Environments

Intelligent smart agriculture has gained attention for automating traditional farming processes, including precision agriculture monitoring, wildlife tracking, environmental detection, forest fire monitoring, pollution monitoring, and flood monitoring through next-generation network applications [161]. Nanotechnology advancement has enabled compact, inexpensive sensors for diverse applications from beekeeping to precision farming in distributed network infrastructures [162].

Intelligent smart agriculture utilizes modern technology including sensors mounted on farm machines and in fields to collect real-time data about planting, spraying, produce, soil types, and weather in network environments [153]. This data is analyzed using IoT and intelligent edge computing to inform decision-making and increase productivity while reducing environmental damage in next-generation network applications. Recent studies have proposed various systems using intelligent edge computing, fog computing, and cloud infrastructure to collect and analyze data, reduce response time, and improve overall efficiency [163-165].

Research focuses on decreasing energy consumption costs within heterogeneous networks for processing sensor information while enhancing energy efficiency through novel architectures, enabling real-time operations across multiple network layers [153].

## 3.3. Intelligent Smart Cities for Next-Generation Networks

Intelligent smart city technologies utilize data from devices and sensors to provide near-real-time information for addressing urban challenges in network environments. Data is collected, processed, and analyzed at intelligent edge, fog, or cloud computing layers to provide time-sensitive solutions using wireless sensor networks [154,161]. Large training datasets are essential for intelligent machine learning methods, particularly deep learning architectures in next-generation network applications [156].

Two pivotal aspects for intelligent smart cities include

- Quantity and quality of available data for intelligent city networks;
- Low-latency requirements for city edge nodes, varying by the specific functions required for each network application [37].

Intelligent applications encompass smart homes enabling remote control of energy and water consumption while functioning as security systems [166], smart lighting optimizing energy consumption based on city conditions [167], smart roads contributing to driver safety and traffic management [168,169], and intelligent wireless parking sensors facilitating parking space location while reducing congestion [170].

Location awareness enables faster processing times by allowing intelligent edge networks to collect and process data independently of physical location in distributed environments [109]. Research demonstrates that intelligent fog computing decreases processing time, hop traversal, and bandwidth usage compared to cloud approaches [171]. Mobile edge models for intelligent smart cities leverage network devices for computing tasks, re- ducing latency for real-time applications like autonomous vehicles and emergency response systems [172].

Advanced approaches include 5G-based smartphone gateways for intelligent healthcare in smart cities, improving energy efficiency while reducing service response time in next-generation network infrastructures [173]. Given resource constraints and the dynamic nature of edge devices in intelligent smart cities, efficient resource distribution is crucial to avoid delays and QoS degradation, particularly as latency increases with device proliferation [6,174].

To solve these problems, one approach is to bring computing resources closer to the edge devices for enabling optimized data processing (see Figure 5). Hong et al. [6] collected multiple architectures and algorithms for resource management at the edge, considering the fog-edge computing model (see Figure 6).

Figure 5. Afog/edge computing model encompassing cloud resources at the edge of the network [6].

<!-- image -->

## 3.4. Intelligent Healthcare Networks

In intelligent healthcare applications, edge and fog computing enable real-time patient data collection and local analysis, facilitating faster, more accurate diagnoses and treatment opportunities in next-generation network environments [154,155]. Intelligent sensors monitor patient vital signs, including blood pressure and heart rate. Connecting edge devices with cloud infrastructure enables cost-effective telemedicine with improved response times, reduced latency, and streamlined workflows in distributed healthcare networks [175].

Intelligent edge and fog computing enable remote monitoring and telemedicine, allowing patients to track their health from home while participating in real-time video conferences or teleoperations in network environments [176]. Research has established metrics for evaluating medical indicators including transmission, retrieval, encryption, and authentication in intelligent network applications [177]. IoT solutions using Raspberry Pi sensors enable measurement of vital body parameters with high accuracy and cost-effectiveness [178]. Specialized systems target diabetes monitoring using continuous insulin pumps and glucose monitors connected to intelligent fog layers for cloud data processing [176]. Edge ML-enabled IoT healthcare monitoring systems emphasize efficient data collection, processing, and distribution while minimizing latency in next-generation network infrastructures [154].

Figure 6. Architectures and algorithms for resource management on a fog/edge computing model [6].

<!-- image -->

## 3.5. Intelligent Smart Industry

Industry 4.0 benefits from intelligent AI solutions across multiple domains including predictive maintenance, waste reduction, quality control, sustainability, process optimization, production planning, safety, and human-robot collaboration, enhancing machine downtime, costs, and production quality in next-generation network environments [158].

Predictive maintenance (PM) plays a crucial role in manufacturing for monitoring and detecting equipment failure during operation in intelligent network applications. Manufacturers utilize predictive maintenance to improve production line stability [68]. PM detects equipment malfunctions before complete failure, enabling systems to take the necessary preventive action. With advancement in intelligent edge IoT and inference capabilities, real-time data collection and monitoring for detecting potential faults and abnormalities within distributed systems is now possible [69]. PM objectives include detecting abnormalities before occurrence, reducing maintenance costs, failure rates, and downtime while improving system reliability in intelligent network infrastructures.

Recent approaches utilize intelligent edge for industrial IoT applications [179,180]. Real-time frameworks using intelligent deep learning for manufacturing inspection detect defects, improve efficiency, and provide defect information in fog computing environments for next-generation network applications [181].

## 3.6. Intelligent Internet of Vehicles

The intelligent Internet of Vehicles (IoV), enabled by edge computing devices, facilitates vehicle monitoring to enhance road safety, efficient communication, congestion avoidance, traffic maintenance, and parking optimization in next-generation network infrastructures [157]. Intelligent IoV edge devices provide efficient road congestion estimation while ensuring user location privacy [70]. Advanced Driver Assistance Systems built on intelligent edge platforms enable efficient cloud network communication, providing low-latency real-time driving assistance, including weather prediction and smart navigation [71].

Research demonstrates that offloading autonomous driving services via intelligent edge computing improves autonomous driving QoS, as edge devices process large amounts of sensor data for safe and reliable decision-making in distributed network environments [72,182].

## 3.7. Intelligent Smart Environment

Environmental safety challenges require intelligent smart environment technologies to provide safe, healthy environments and propose innovative solutions for efficient energy conservation in next-generation network applications [183]. Real-time air quality monitoring systems using intelligent sensor data improve indoor and outdoor air quality through distributed computing frameworks interacting across cloud and edge networks [184]. Raspberry Pi-based intelligent IoT edge solutions enable air quality monitoring and prediction [185].

Water quality monitoring utilizes IoT-based real-time frameworks for intelligent water quality management, monitoring, and alert generation based on contamination and toxic parameter levels in network environments [186]. Intelligent monitoring and prediction using drones and UAVs addresses geological hazards including landslides, earthquakes, and forest fires [187,188]. Efficient garbage collection management employs intelligent IoT edge devices with robotic systems for automated cleaning in next-generation network infrastructures [189].

## 3.8. Intelligent Operating Systems for Network Applications

Accommodating the resource limitations of low-end devices, including limited memory, computational resources, and power supply, requires selecting operating systems that optimize available resources effectively for next-generation network applications. This includes traditional systems like Linux [32], with Linux-Docker-based frameworks enhancing IoT security in network environments [190].

Operating systems for intelligent edge devices must be compact and efficient to accommodate limited processing and storage capabilities in distributed networks. Many OS systems aid intelligent edge application creation, addressing real-time performance, connectivity protocols, power management, and security. Key systems include TinyOS for sensor networks [191], EOS for telecommunications edge infrastructure [192], and ThingSpire OS for coherent processing across IoT devices and the cloud in network environments [193].

Specialized systems include Zephyr for minimal-footprint operations [194], RIOT and NuttX for real-time performance and low power consumption [195], Mbed OS, which emphasizes connectivity [196], and real-time systems like FreeRTOS and Azure RTOS ThreadX that have a reduced memory footprint [197,198].

The IoT2Cloud Operating System (ICOS) prioritizes device diversity, infrastructure virtualization, network diversity, scalability, privacy, security, and data sharing for intelligent edge market scenarios in next-generation network continuum paradigms.

## 4. State-of-the-Art Intelligent Edge ML Solutions for Next-Generation Network Applications

The future of intelligent machine learning at the edge is shaped by cloud offloading techniques, distributed learning approaches, context-awareness, soft computing model compression techniques, specialized hardware, and security and privacy via design principles [199-202]. These trends enable truly ubiquitous intelligent edge computing that can be popularized across AI applications in next-generation network infrastructures [1].

This section examines the existing technologies and methods currently used to address practical use cases across different application domains in intelligent network environments. These represent state-of-the-art solutions deployed today to meet specific needs in edge environments rather than addressing future research challenges. For each case, we evaluate technical maturity by incorporating concise assessments in the concluding paragraphs that categorize solutions based on their deployment readiness and commercial availability. These assessments distinguish between production-ready technologies with established commercial platforms, pilot-phase solutions requiring specialized expertise, and researchphase approaches still under development [203,204]. These technologies for next-generation network applications are depicted in Figure 7.

Figure 7. Solutions and technologies for edge machine learning.

<!-- image -->

## 4.1. Intelligent Cloud Offloading for Network Applications

Intelligent cloud offloading serves as a pivotal mechanism within next-generation edge cloud frameworks, enabling mechanisms that minimize latency, enhance energy efficiency, and optimize cost-effectiveness in distributed network environments. This approach streamlines and fine-tunes the coordination of computational resources, contributing to efficient operation of intelligent edge cloud systems by making them more responsive, sustainable, and economically viable for network applications.

Many intelligent cloud offloading technologies explore energy-performance tradeoffs in IoT environments, focusing on real-time applications like vision-aided games, augmented reality, connected health, and vehicular multi-access edge computing networks [4,150,205]. Intelligent edge cloud offloading addresses centralized cloud computing limitations by relocating computation and storage resources closer to network devices, supporting resource-intensive applications in next-generation network infrastructures [4].

While intelligent edge computing reduces latency and improves energy efficiency through local data storage, systems like CloneCloud transform mobile applications to leverage cloud resources for specific tasks within edge devices [206]. Intelligent fog nodes drastically reduce latency between applications and the centralized cloud, improving QoS in network environments [150]. The primary goal is optimizing offloading for low latency and energy efficiency in computational offloading for intelligent network applications.

Advanced approaches utilize deep learning models for IoT applications with intelligent edge computing [207], while optimization techniques employ DQN for task offloading and wireless resource allocation to maximize network data acquisition and analysis capabilities [208,209]. The Internet of Vehicles exemplifies the necessity of real-time data collection and resource allocation optimization, highlighting the crucial synergy between AI and intelligent edge computing in next-generation network applications [210]. These cloud offloading approaches have reached commercial maturity, with widespread deployment across major cloud platforms including AWS, Azure, and Google Cloud, making them readily accessible for production implementations in next-generation network infrastructures.

## 4.2. Intelligent Edge Caching for Network Environments

Intelligent edge caching technology enables scenarios like object detection in edge device video analysis within smart city contexts, supporting AIoT platforms that collect video data from personal edge devices and transform it into valuable information for IoT and intelligent network applications [211]. However, video surveillance data analytics presents privacy concerns and unauthorized data exposure risks when uploading videos without user consent. Implementing intelligent video analytics with offloading approaches accelerates processing while reducing latency and resource consumption in network environments [109].

Internet of Vehicles applications demonstrate intelligent edge caching where multiple moving agents share data across networks, utilizing reinforcement learning for efficient networking in next-generation infrastructures. Research shows how intelligent edge caching with reinforcement learning facilitates computational workload offloading in 6G-enabled IoV environments [212].

Collaborative caching approaches among agents for sharing multimedia content minimize content access latencies and improve caching resource utilization in intelligent network applications [213]. Deep learning-based cloud video recommendation systems enhance accuracy through user profiles and video descriptions, incorporating federated learning for collaborative training across distributed cloud servers [214]. Federated video offloading via intelligent edge networks utilizes federated learning algorithms to optimize accuracy while reducing offloading latency [215].

Intelligent edge caching enables real-time data analysis and decision-making closer to data resources, offering faster processing, reduced latency, and resource efficiency for next-generation network applications. This provides viable solutions for federated learning involving heterogeneous edge devices through cache-driven learning approaches [216,217]. Self-trained and auto-tuned intelligent edge caching systems utilize deep reinforcement learning techniques for autonomous decision-making in network environments [217]. While basic edge caching has achieved production maturity through CDN providers like Cloudflare [218], intelligent caching approaches incorporating federated learning and deep reinforcement learning remain in pilot-testing phases, requiring specialized expertise for deployment in network infrastructures.

## 4.3. Intelligent Data Stream Processing for Network Applications

Intelligent data stream processing has gained attention across various fields, particularly for real-time inference use cases including streaming data display on mobile devices and the transfer of data mining results over limited-bandwidth wireless networks in nextgeneration infrastructures [219]. Selecting appropriate data streaming frameworks while maximizing the real-time processing of high-volume heterogeneous data streams remains challenging in intelligent network environments [47].

Intelligent edge analytics systems process data dynamically at edges and in the cloud in real-time, working in distributed contexts where analytical latency depends on the users, applications, and data from multiple regions [220]. Traffic video analytics systems produce high-quality analytical results while maintaining minimal resource consumption, including public cloud options and private edge nodes with various hardware for media processing and algorithm execution in intelligent network applications [221].

The rapid processing of continuous data streams within constrained timeframes emphasizes utilizing resource elasticity features from cloud computing, enabling dynamic system scaling in response to demand conditions in next-generation network environments [222].

Data streams in continuous ML scenarios leverage the intelligent stream learning literature, where algorithms continuously adapt to incoming data changes for sustained performance [59]. Common solutions for AI degradation include model retraining, early stopping mechanisms, and constant ground-truth data access [121]. Alternative solutions utilize online or adaptive machine learning techniques, with forgetting mechanisms enabling faster real-time inference in intelligent network applications [59]. These data stream processing approaches exhibit varying technical maturity levels. Traditional streaming frameworks like Apache Kafka [223] have achieved production readiness, while adaptive machine learning techniques with forgetting mechanisms are still primarily in the research and pilot phases and require further development for robust network deployment.

## 4.4. Intelligent Distributed Machine Learning for Networks

Intelligent distributed machine learning at the edge enables knowledge interchange without centralized data storage, which is essential for resource-intensive training and inference of complex models in next-generation network environments [1,34]. Parallel computing methods demonstrate the superiority of GPUs over CPUs in latency reduction for intelligent network applications [224].

Intelligent distributed learning enhances user privacy protection across communication networks through training and inference on local data, reducing exposure to attacks in network environments [225]. Methods including model partitioning, federated learning baselines, and spatiotemporal data fusion techniques reduce private data exposure, particularly when applied to heterogeneous edge devices [19,132].

Privacy issues are addressed through access policies detecting anomalies or harmful threats, demonstrating that communication protocols and advanced techniques like data anonymization provide potential solutions for IoT security challenges in intelligent network infrastructures [150].

Recent approaches include distributed edge/cloud paradigms for lightweight virtualization using Docker [226]. These paradigms combine deep learning applications with edge optimizations across inference, computation, and training [39] and explore comprehensive Edge ML aspects including caching, training, inference, and offloading [227]. AI-to-edge architectures investigate intelligent Edge ML across domains, considering sensors, analytics, and ML across edge and fog layers for next-generation network applications [1].

Collaborative intelligent edge technologies facilitate ad hoc networking among stakeholders to coordinate collaborative edge devices and servers for processing geographically distributed data [1,228]. Advanced approaches include Variational Recurrent Neural Networks for distributed cooperative task offloading among multiple agents [229] and collaborative edge computing linking social relationships to physical domains through auction, coalition games, and federated learning solutions addressing incentive compatibility and security challenges in intelligent network environments [152]. Distributed machine learning technologies demonstrate mixed maturity levels, with federated learning frameworks like Pysift [140] reaching pilot deployment stages at major technology companies, and advanced approaches involving variational networks and game-theoretic solutions remaining primarily in the research phase, i.e., requiring substantial development before network-scale implementation.

## 4.5. Intelligent Efficient Modeling for Network Edge Applications

Intelligent machine learning pipelines must operate with stringent energy efficiency constraints in next-generation network edge environments, where computational resources are limited and power consumption directly impacts device battery life and operational costs. This challenge is particularly acute when training occurs locally or when minimizing data transmission overhead is critical for bandwidth optimization in intelligent network applications [34]. Large data volumes can significantly degrade Quality of Service (QoS)

and increase latency, creating cascading effects on pipeline management and scalability in distributed network environments.

Researchers have developed complementary strategies focusing on hardware optimization, algorithmic efficiency, and architectural design to address energy efficiency constraints in intelligent network applications. Key approaches include deploying energy-efficient hardware accelerators, implementing communication-efficient algorithms for distributed AI model training on edge nodes [2], and developing lightweight models specifically optimized for low-power edge inference in next-generation networks.

The intelligent lightweight model ecosystem has produced notable architectures designed for energy-constrained deployments, including ShuffleNet [230], ShuffleNet V2 [231], MobileNet [232], and SqueezeNet [233]. These models prioritize computational efficiency while maintaining acceptable accuracy levels for intelligent network applications. Supporting frameworks facilitate deployment in distributed network environments.

Recent advances have highlighted significant progress in intelligent energy management and optimization for next-generation networks. Self-learning energy management frameworks using Soft Actor-Critic algorithms achieve substantial energy reduction and battery life improvement through context-aware power optimization [234]. Enhanced IoT routing approaches utilize social attributes and energy awareness for intelligent network applications [235], while Rainbow DQN-based task offloading frameworks improve energy efficiency, reduce latency, and increase utility, demonstrating reinforcement learning effectiveness for energy-aware intelligent edge computing [236].

Beyond energy optimization, fault tolerance remains essential for maintaining reliable intelligent edge operations in network environments. Containerized architectures for fault-tolerant IoT applications [237], intelligent agent-based fault tolerance models [238], and clustering-based approaches [239] complement energy efficiency objectives in nextgeneration network applications. Efficient modeling techniques exhibit high technical maturity. Model compression methods like quantization and pruning are widely adopted in production frameworks including PyTorch, while fault tolerance approaches through containerization have reached commercial deployment via Docker and Kubernetes. However, intelligent agent-based fault tolerance models remain in the early stages of development.

## 4.6. Intelligent Specialized Hardware for Network Applications

Technologies in intelligent edge computing depend upon the specialized hardware used for computation in next-generation network environments. Devices with inbuilt CPUs and GPUs provide suitable alternatives for fast processing, model inference, and training in distributed networks. In addition to traditional CPU and GPU architectures, Neural Processing Units (NPUs) have emerged as specialized hardware accelerators specifically designed for AI and machine learning workloads, offering optimized computational architectures for neural network operations with superior energy efficiency compared to general-purpose processors [240-243]. The potential of intelligent edge machine learning is achieved through edge devices equipped with specialized hardware accelerators, including ARM-inbuilt GPUs, TPUs, NPUs, and FPGAs for network applications [199]. NPUs excel at the parallel processing of neural network computations through specialized architectures for matrix operations and support low-precision arithmetic operations such as INT4, INT8, and FP16 to maximize computational efficiency while minimizing power consumption. Modern NPUs integrate high-bandwidth memory architectures and work in heterogeneous computing paradigms alongside CPUs and GPUs, dynamically allocating computational tasks for optimal performance [241,242].

These intelligent hardware accelerators enable effective model inference while drastically reducing latency and power consumption, providing exceptional speed for next- generation network applications [244-246]. This approach reduces dependencies on cloud servers and helps reduce operational costs since intelligent edge devices become powerful enough to handle heavy workloads independently in network environments. Moreover, these devices are significantly more power efficient than large servers, which require sophisticated cooling systems.

Intelligent systems provide enhanced security levels as data is processed and stored locally within edge devices for model training and serving in network applications. The local processing capabilities of these specialized processors support privacy-preserving applications in the healthcare, automotive, and industrial domains, where data confidentiality is paramount. This enables edge devices to build more the complex ML-based algorithms required for intrusion detection systems, cryptography, and quantum computing applications [247,248].

Specialized hardware in intelligent edge computing creates a greater need for novel methods to train high-quality models within devices, addressing on-device instruction challenges in network environments [249]. The spectrum of intelligent IoT applications continues to expand with GPU-enabled microcontroller capabilities, opening new realtime use cases and facilitating deployment of complex models in next-generation network infrastructures [245]. Specialized hardware technologies show varied maturity levels. Traditional CPUs and GPUs have achieved full production readiness across multiple vendors, while NPUs have reached commercial deployment in mobile devices and some edge systems through companies like Qualcomm and Apple. However, broader enterprise adoption and the standardization of NPU architectures need further development.

## 5. Research Challenges and Future Directions in Edge Machine Learning

One of the primary goals of this survey is to identify the key challenges and future directions of machine learning (ML) at the edge to expand applications across resourceconstrained devices. While the AI space has experienced rapid progress in Large Language Models (LLMs) that extend to other branches of AI, it has become increasingly difficult to adapt such models within constrained settings and sparse data environments.

Figure 8 presents the major research challenges that span three critical dimensions: data-level challenges, including heterogeneity and label scarcity, optimized multimodal AI, and multimodal data fusion; systems-level challenges encompassing efficient orchestration and energy efficiency; and societal-level challenges addressing ethics in the Artificial Intelligence of Things and AI trustworthiness. These interconnected research areas demonstrate how advances in one domain directly influence progress in others, emphasizing the need for holistic approaches to Edge ML development.

Figure 8. Future research areas for machine learning at the edge. The challenges span data (heterogeneity, label scarcity, multimodal fusion), systems (energy efficiency, orchestration), and societal dimensions (trustworthiness, ethics in AIoT).

<!-- image -->

## The following subsections cover these challenges in more detail.

## 5.1. Intelligent Heterogeneity and Label Scarcity in Network Environments

A critical challenge in intelligent edge computing for next-generation networks is the limited availability of labeled data, especially within sparse data environments in distributed network infrastructures [43]. Manual data annotation is resource-intensive and typically infeasible in distributed, resource-constrained edge scenarios across network applications.

Several promising strategies have emerged to address this challenge in intelligent network environments. Transfer learning reduces the need for extensive labeled data by leveraging model weights trained on larger datasets for next-generation network applications [52]. Transfer learning combined with monitoring shows promising results in network deployments [250], while data distillation techniques train student models using smaller label amounts by inheriting knowledge from larger teacher models [251]. Semi-supervised learning techniques, combining limited labeled data with abundant unlabeled samples, have emerged as effective strategies for intelligent edge computing. Active learning methods optimize annotation efforts by selectively labeling only the most informative data samples, significantly reducing costs in network environments [252].

Beyond label scarcity, hardware and communication variations in intelligent edge computing profoundly influence data quality, impacting AI model accuracy in next-generation networks. Factors including data noise, incompleteness, heterogeneous hardware capabilities, and remote deployments exacerbate annotation-related challenges. Edge device diversity often necessitates developing and managing multiple tailored machine learning models, potentially resulting in underutilized architectures across network stakeholders [15,73].

Heterogeneity concerns persist regarding fragmented hardware and software ecosystems, complicating compatibility with intelligent MLOps workflows in network environments. Proposed solutions include containerized applications to standardize edge device environments [73] and strategically distributing training workloads between cloud and edge infrastructures according to the available network resources [253].

Integrating these approaches into intelligent Edge ML frameworks can significantly improve model performance while minimizing dependency on extensive, annotated datasets in next-generation network applications. Future research should emphasize developing lightweight, resource-conscious semi-supervised and active learning methods specifically adapted for heterogeneous intelligent edge environments.

## 5.2. Intelligent Optimized Multimodal AI for Network Applications

The rapid advancement of Large Multimodal Models shows promise across various fields, particularly in resource-constrained settings dominated by mobile devices in next-generation network infrastructures. Vision Language Models and Vision Language Pretraining hold significant potential for applications in real scenarios including mobile devices, self-driving cars, and embedded systems within intelligent network environments [12]. However, they present challenges for the IoT and constrained devices due to larger datasets and computational demands in network applications [254].

Deploying LMMs effectively on the intelligent IoT and edge devices while maintaining generalization and efficiency is essential for practical network applications. While VLMs exhibit exceptional reasoning abilities and effectively integrate multimodal data [50], they remain expensive, demand significant computational resources, and rely on vast datasets. Notable models include OPT [255], Flan-T5 [256], LlaVA [257], BLIP-2 [258], CogVLM [254], and Llama Series [259] for intelligent network applications.

Model optimization techniques for resource-constrained network devices involve complexity reduction via parameter optimization and quantization approaches [13,28,90].