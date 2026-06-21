<!-- image -->

Review

## Federated Learning for Smart Cities: A Thematic Review of Challenges and Approaches

<!-- image -->

<!-- image -->

Academic Editors: Demetris Trihinas and Alexandros Karakasidis

Received: 21 October 2025

Revised: 12 November 2025

Accepted: 13 November 2025

Published: 28 November 2025

Citation: Alterkawi, L.; Dib, F.K. Federated Learning for Smart Cities: AThematic Review of Challenges and Approaches. Future Internet 2025 , 17 , 545. https://doi.org/10.3390/ fi17120545

Copyright: © 2025 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https://creativecommons.org/ licenses/by/4.0/).

<!-- image -->

Laila Alterkawi 1, * and Fadi K. Dib 2, *

- 1 Department of Computer Engineering and Cybersecurity, International University of Kuwait, Kuwait City 70070, Kuwait
- 2 Computer Science Department, Center of Applied Mathematics and Bioinformatics (CAMB), Gulf University for Science and Technology, Mubarak Al-Abdullah 32093, Kuwait
* Correspondence: laila.alterkawi@iuk.edu.kw (L.A.); deeb.f@gust.edu.kw (F.K.D.)

## Abstract

Federated Learning (FL) offers a promising way to train machine learning models collaboratively on decentralized edge devices, addressing key privacy, communication, and regulatory challenges in smart city environments. This survey adopts a narrative approach, guided by systematic review principles such as PRISMA and Kitchenham, to synthesize current FL research in urban contexts. Unlike prior domain-focused surveys, this work introduces a challenge-oriented taxonomy and integrates an explicit analysis of reproducibility, including datasets and deployment artifacts, to assess real-world readiness. The review begins by examining how FL supports the privacy-preserving analysis of environmental and mobility data. It then explores strategies for resource optimization, including load balancing, model compression, and hierarchical aggregation. Applications in anomaly and event detection across power grids, water infrastructure, and surveillance systems are also discussed. In the energy sector, the survey emphasizes the role of FL in demand forecasting, renewable integration, and sustainable logistics. Particular attention is given to security issues, including defenses against poisoning attacks, Byzantine faults, and inference threats. The study identifies ongoing challenges such as data heterogeneity, scalability, resource limitations at the edge, privacy-utility trade-offs, and lack of standardization. Finally, it outlines a structured roadmap to guide the development of reliable, scalable, and sustainable FL solutions for smart cities.

Keywords: federated learning; Internet of Things (IoT); smart cities; edge intelligence; privacy-preserving learning; distributed machine learning; resource optimization; security and privacy

## 1. Introduction

Smart cities are increasingly envisioned as data-driven ecosystems that integrate largescale Internet of Things (IoT) infrastructures, distributed sensors, and advanced analytics to improve urban services such as transportation, energy management, public safety, and environmental monitoring [1,2]. By 2050, nearly 70% of the global population is projected to reside in urban areas, and the number of IoT connections is expected to exceed 30 billion by 2030, generating unprecedented volumes of heterogeneous data [3,4]. These data streams, ranging from smart meters and surveillance feeds to electric vehicle (EV) chargers and solar inverters, are inherently distributed and highly sensitive. Although centralizing such data can enable powerful machine learning (ML) models, it also raises critical concerns related to privacy, scalability, interoperability, and compliance with regulations such as the General Data Protection Regulation (GDPR) [5]. The distributed nature of urban governance, where utilities, transportation departments and municipal agencies operate independently, further complicates data sharing, particularly when raw datasets are involved. Aggregating realtime streams from thousands of edge devices also imposes substantial communication burdens on network infrastructures.

<!-- image -->

Smart city data are inherently non-independent and identically distributed (nonIID) due to localized behaviors, diverse sensor types, and temporal variations [6]. These conditions undermine the assumptions of centralized ML models, which typically presuppose uniform data distributions. At the same time, many urban stakeholders require maintaining ownership of their data while still benefiting from collaborative analytics. Cloud-centric AI struggles with latency and bandwidth limitations, while purely local AI lacks coordination across heterogeneous devices. Beyond these infrastructural and analytical challenges, smart cities also encompass a cultural and experiential dimension, where technology shapes community interaction, citizen participation, and urban identity. Recent perspectives highlight how digital infrastructures, from smart sports to tourism platforms, co-create cultural value and collective experience within cities [7]. Federated learning can further this vision by enabling cross-domain collaboration while respecting privacy, inclusivity, and local autonomy.

FL, first introduced through FedAvg by McMahan et al. [8], offers a compelling 'middle path.' FL enables decentralized model training across edge nodes by distributing a global model to participating clients (e.g., sensors, substations, or edge devices). Each client performs local updates and returns only the model parameters or gradients, which are aggregated centrally to form an improved global model, without transferring raw data. To strengthen privacy, techniques such as secure aggregation, homomorphic encryption, and differential privacy are often incorporated, although their scalability and effectiveness in city-scale deployments remain open challenges.

The decentralized and privacy-preserving nature of FL aligns well with the constraints of smart city environments. Over the past five years, FL has been applied to various urban use cases, including predictions of EV charging demand, solar energy forecasting, adaptive traffic control, anomaly detection in utilities, and fraud identification. Variants such as clustered FL, reinforcement learning-based FL, and ensemble FL have emerged to handle non-IID distributions, device heterogeneity, and personalized service requirements. These advances highlight both the promise and the complexity of bringing FL into operational urban intelligence systems.

Recent bibliographic reviews reveal a sharp increase in research activity on federated learning within the smart city domain. Between 2019 and 2024, approximately 170-200 peerreviewed articles have been published that explicitly address FL applications in urban or IoT-based infrastructures, corresponding to an annual growth rate exceeding 30% [9,10]. These studies cover a wide spectrum of topics, including transportation management, renewable energy forecasting, environmental monitoring, and public safety. The steady expansion of this literature highlights both the maturity and fragmentation of the field, reinforcing the need for a comprehensive and challenge-oriented synthesis such as the one presented in this paper.

## 1.1. Positioning Against Prior Surveys

Several recent FL surveys have provided useful overviews of the field. Pandya et al. [10] reviewed general FL applications across industries, Zhaohua et al. [11] examined security and privacy mechanisms, Jia et al. [12] synthesized algorithmic improvements, and Rahdari et al. [13] surveyed cross-domain deployments. Although valuable, these works primarily categorize FL by domain (e.g., healthcare, finance, transportation) or by technical dimension (e.g., aggregation methods, privacy techniques). None provides a comprehensive challengeoriented synthesis specifically tailored to the multifaceted requirements of smart cities. In particular, cross-cutting issues such as interoperability between municipal agencies, integration with 5G/6G infrastructures, reproducibility of experiments, and readiness for real-world deployment remain underexplored.

Methodologically, our review follows a narrative-driven synthesis guided by systematic principles, including PRISMA [14] and Kitchenham's software engineering review guidelines [15], which ensure transparency and rigor.

## 1.2. Our Contributions

The key contributions of this survey are as follows.

- We introduce a challenge-oriented taxonomy of FL in smart cities, organized around privacy and security, resource optimization, event detection and situational awareness, and energy sustainability.
- We provide a cross-domain synthesis of recurring trade-offs, highlighting how challenges such as heterogeneity, communication cost, latency, and fairness manifest across multiple domains.
- We systematically evaluate the reproducibility and readiness for real-world deployment of existing studies, identifying gaps in open datasets, code availability, and benchmarking practices.
- We compare algorithmic extensions and adaptations, including clustered FL, reinforcement learning-based FL, personalized FL, and graph-based FL, that enhance applicability in city-scale systems.
- We outline a forward-looking research agenda, addressing scalability in ultra-dense IoT, integration with 6G and edge-cloud architectures, robustness against adversarial threats, and ethical considerations such as transparency, fairness, and explainability.

## 1.3. Paper Organization

The remainder of this paper is structured as follows. Section 3 outlines our methodology for identifying and selecting relevant literature. Section 4 reviews privacy-preserving strategies in FL, including differential privacy and secure multiparty computation. Section 5 explores FL techniques for resource optimization, such as communication efficiency, model compression, and distributed scheduling. Section 6 focuses on FL applications in event detection and situational awareness. Section 7 presents research on energy management and sustainability, while Section 8 highlights other related works. Section 9 synthesizes the open challenges and research directions, and Section 10 concludes with broader implications for federated learning in smart cities.

By framing FL for smart cities through a challenge-oriented taxonomy, this survey not only synthesizes the state of the art, but also sets a forward-looking agenda toward building trustworthy, scalable, and sustainable urban intelligence.

## 2. Background

This section introduces the foundational concepts required to understand the intersection of FL and smart cities. We first review FL paradigms and their variants, then highlight enabling technologies that support FL in urban environments, and finally discuss the distinctive characteristics of smart city data.

## 2.1. Federated Learning Paradigms

FL is a decentralized machine learning paradigm that enables multiple participants to collaboratively train a shared model without exchanging raw data [8,16]. Instead, model updates, such as gradients or parameters, are transmitted to a central server or aggregator that combines them into a global model. Over the years, several paradigms have emerged to adapt FL to different data distribution and system scenarios.

In horizontal FL (also known as cross-device FL), clients share the same feature space but hold different data samples. This approach is particularly suitable for mobile devices or IoT edge nodes, where each client observes only a portion of the overall population. By contrast, vertical FL (cross-silo FL) applies when clients share the same user IDs but differ in their feature spaces. A typical example would be a collaboration between a hospital, which holds medical records, and a bank, which holds financial records, for joint predictive analytics.

Federated transfer learning (FTL) addresses scenarios where datasets differ in both samples and features, but there is a small overlap between them. In such cases, transfer learning techniques are used to bridge heterogeneous data sources and allow knowledge sharing across domains. Finally, hierarchical or clustered FL groups clients into clusters or introduces multi-layered architectures where edge servers aggregate updates before passing them to the cloud [17,18]. This approach reduces communication costs and captures locality in non-IID settings, making it especially relevant for city-scale deployments. These paradigms are illustrated in Figure 1.

Figure 1. Main paradigms of Federated Learning. Horizontal FL (cross-device) assumes clients share the same feature space but hold different data samples, as in mobile or IoT networks. Vertical FL (cross-silo) involves organizations with shared user identifiers but disjoint feature spaces, such as collaborations between healthcare and finance institutions. FTL applies when datasets differ in both features and samples, with limited overlap enabling cross-domain knowledge transfer. Clustered or hierarchical FL organizes clients into multi-layer architectures (edge → cloud), reducing communication costs and improving performance under non-IID data distributions.

<!-- image -->

## 2.2. Enabling Technologies for Smart Cities

The deployment of FL in smart cities depends on several enabling technologies. Edge and cloud computing form the backbone of this ecosystem: edge computing brings computation closer to IoT devices, reducing latency and bandwidth costs, while cloud infrastructures provide centralized coordination and long-term storage [19]. Building on this, mobile edge computing (MEC) integrates network and computing capabilities at the base-station level, enabling low-latency aggregation and localized model updates [20].

Next-generation wireless networks also play a vital role. The features of the anticipated 5G and 6G technologies, such as network slicing, ultra-reliable low-latency communications (URLLC), and massive machine-type communications (mMTC), are essential for scaling FL deployments across millions of devices [21]. Trusted execution environments (TEEs), such as Intel SGX or ARM TrustZone, provide hardware-based secure enclaves for sensitive model computations, reducing the risk of information leakage through gradient sharing [22]. Finally, blockchain and other distributed ledger technologies are increasingly integrated with FL to enhance transparency and trust, record model updates, reward contributions, and mitigate poisoning attacks [23]. These key enabling technologies are summarized in Table 1.

Table 1. Key enabling technologie s that support Federated Learning in smart cities.

| Technology     | Relevance to FL in Smart Cities                                                          |
|----------------|------------------------------------------------------------------------------------------|
| Edge Computing | Reduces latency, supports distributed analytics close to data sources.                   |
| MEC            | Integrates computing and communication at base stations for localized model aggregation. |
| 5G/6G Networks | Provides bandwidth and reliability for dense IoT and vehicular networks.                 |
| TEEs           | Protect sensitive gradients during model updates.                                        |
| Blockchain     | Ensures transparency, trust, and incentivization in multi-stakeholder environments.      |

## 2.3. Smart City Data Characteristics

Smart city environments generate large-scale, multimodal and heterogeneous datasets, which present unique challenges for FL. The heterogeneity of the sources is particularly striking: transportation networks rely on traffic sensors and GPS traces, healthcare involves wearables and electronic health records, and energy management depends on smart meters, solar inverters, and EV chargers. Each of these data streams comes with distinct formats, resolutions, and sampling rates, which complicates the design of unified models. Representative examples of these data domains and their implications for FL are summarized in Table 2.

In addition to heterogeneity, smart city data is often non-IID. Localized behaviors, such as neighborhood-specific traffic patterns, seasonal variations in energy demand, and varying device sampling policies, introduce distribution skews that can slow convergence and reduce model accuracy [24]. Privacy sensitivity further constrains data use: mobility traces, health records, and household energy consumption are regulated by strict data protection laws, making direct data centralization infeasible. Communication limitations also play a role, since many edge devices operate under restricted bandwidth or intermittent connectivity, limiting their ability to frequently transmit model updates. Finally, governance fragmentation remains a barrier, as datasets are often siloed across different agencies (utilities, transport authorities, healthcare providers), each with distinct policies and incentives.

In summary, FL offers a promising paradigm for integrating heterogeneous, privacysensitive, and distributed smart city data into cohesive learning models. However, the diversity of data types, strict privacy regulations, and system-level constraints necessitate specialized adaptations, which motivates the challenge-oriented taxonomy presented in the next section.

Table 2. Representative smart city data sources, their characteristics, and associated challenges for FL.

| Domain         | Data Sources                                       | Characteristics                                  | Challenges for FL                              |
|----------------|----------------------------------------------------|--------------------------------------------------|------------------------------------------------|
| Transportation | Traffic sensors, GPS traces, vehicular networks    | Spatio-temporal, high frequency, multimodal      | Non-IID, real-time latency constraints         |
| Energy         | Smart meters, solar inverters, EV chargers         | Periodic, household-level, sensitive usage data  | Privacy, seasonal variation, adversarial risks |
| Healthcare     | Wearables, hospital EHRs, emergency services       | Multimodal, irregular sampling, highly sensitive | Regulatory compliance, heterogeneity           |
| Public Safety  | Surveillance feeds, incident reports, police IoT   | Video, text, event-driven data                   | Large-scale streaming, privacy, security risks |
| Environment    | Weather stations, pollution sensors, waste systems | Continuous, geographically distributed           | Missing values, spatial heterogeneity          |

## 3. Methodology

Building on the conceptual background outlined in the previous section, we now describe the methodology that guided our review. Since smart city research in FL spans diverse domains, architectures, and evaluation practices, a quantitative meta-analysis was neither feasible nor appropriate. Instead, we adopted a narrative-driven synthesis structured around systematic review principles, allowing us to identify common patterns, highlight technical trade-offs, and assess deployment readiness across heterogeneous studies. Our approach was based, in particular, on the PRISMA checklist [14] and the systematic review guidelines established in software engineering research [15,25], ensuring transparency and reproducibility.

## 3.1. Search Strategy and Scope

We conducted keyword-based searches across IEEE Xplore, ACM Digital Library, and Scopus, covering the period from January 2015 to May 2025. Although FL was formally introduced in 2016 by McMahan et al. [8], earlier work on distributed ML and privacy-preserving analytics began to gain traction in 2015. Including this initial phase allowed us to capture the evolution of enabling technologies that later converged under the FL paradigm. Extending the time window through 2025 ensured coverage of the latest advances, including early-access articles and arXiv preprints that have not yet appeared in indexed databases.

Our search terms combined phrases such as 'federated learning,' 'FedAvg,' 'smart city,' 'urban IoT,' 'smart grid,' 'edge computing,' and 'privacy-preserving AI.' The initial search returned 246 records: 80 from IEEE, 52 from ACM, and 108 from Scopus. We supplemented these with six additional sources of high practical relevance, including four arXiv preprints (2024-2025) and two reports on real-world FL pilot projects. These were screened and analyzed using the same framework as peer-reviewed work to ensure methodological consistency.

According to the PRISMA and Kitchenham guidelines, we explicitly report the search strings used to retrieve the studies. The queries were adapted to the syntax of each database to ensure a complete and reproducible coverage. All searches were performed between January 2019 and September 2025.

The generic Boolean structure of the query was:

('federated learning' OR 'collaborative learning' OR 'distributed machine learning') AND ('smart city' OR 'urban computing' OR 'intelligent transportation' OR 'IoT' OR 'cyber-physical system' OR 'smart grid')

The corresponding database-specific queries are listed below:

## IEEE Xplore:

```
(``federated learning'' OR ``collaborative learning'') AND (``smart city'' OR ``IoT'' OR ``cyber-physical system'') Scopus: TITLE-ABS-KEY((``federated learning'' OR ``collaborative learning'' OR ``distributed machine learning'') AND (``smart city'' OR ``urban computing'' OR ``intelligent transportation'' OR ``IoT'' OR ``smart grid'')) Web of Science: TS=(``federated learning'' OR ``collaborative learning'' OR ``smart
```

```
OR ``distributed machine learning'') AND {TS=(``smart city'' OR ``IoT'' OR ``cyber-physical system'' grid'')}
```

The search process was performed iteratively to capture newly published studies within the review period, with duplicates automatically removed before manual screening. The final set of eligible records was then filtered according to the predefined inclusion and exclusion criteria and is summarized in Table 3.

To further contextualize reproducibility and evaluation practices, we acknowledge that several reviewed works were conducted on simulated or emulated infrastructures rather than real deployments. To support scientific validity and facilitate standardized comparisons, recent benchmarking and emulation frameworks have emerged for federated and edge learning research. FedBed [26] provides a reproducible benchmarking suite for evaluating federated learning over virtualized edge testbeds, enabling controlled experimentation across diverse network and client configurations. Similarly, Fogify [27] offers a fog computing emulation framework that can reproduce large-scale IoT deployments under realistic latency and bandwidth constraints. Integrating such tools into future smart city FL studies can enhance reproducibility, enable cross-study comparability, and reduce dependence on purely simulation-based results.

Table 3. Summary of record screening following PRISMA guidelines.

| Stage                 |   Records | Excluded   | Notes                                                                      |
|-----------------------|-----------|------------|----------------------------------------------------------------------------|
| Identified            |       246 | -          | IEEE: 80; ACM: 52; Scopus: 108; 4 arXiv (2024-2025); 2 real-world projects |
| After deduplication   |       176 | 70         | Duplicate titles/DOIs removed                                              |
| Abstract screening    |       116 | 60         | Not FL: 26; Not smart city: 20; Not substantive: 14                        |
| Included in synthesis |       116 | -          | 110 peer-reviewed; 4 arXiv; 2 projects                                     |

## 3.2. Handling Non-IID Data in Smart City Federated Learning

Non-identically distributed (non-IID) data is one of the central methodological challenges in federated learning, particularly within heterogeneous smart city environments. Urban devices and sensors generate diverse data distributions due to differences in sampling frequency, sensing modalities, and contextual factors such as location and infrastructure type. To ensure consistency and fairness in collaborative training, this subsection integrates key definitions, causes, and mitigation strategies discussed in the reviewed studies.

Non-IID data can arise from feature imbalance, label distribution skew, quantity skew, or temporal variation across clients. Mitigation strategies reported in the literature include clustering-based FL, personalized FL, transfer learning, and meta-learning. Clustering approaches group clients with similar data characteristics before model aggregation; personalization introduces local adaptation layers or fine-tuning; transfer and meta-learning leverage shared representations to improve generalization across clients. These strategies enhance convergence stability and fairness while maintaining privacy constraints. Crossreferences to corresponding application domains (e.g., energy, mobility, anomaly detection) are provided where relevant to maintain narrative coherence across sections.

## 3.3. Screening and Inclusion Criteria

After deduplication, there were 176 unique records. The titles and abstracts were then selected for relevance to FL and smart city contexts. Articles were excluded if they did not address federated learning (26 cases), did not have a clear connection to urban applications (20 cases) or did not present substantive findings (14 cases). This left 110 articles for full-text review. Following validation, our final corpus consisted of 116 works: 110 peer-reviewed studies, 4 arXiv preprints, and 2 project reports. Although we clearly distinguish non-peerreviewed contributions in our analysis, including them enabled us to capture cutting-edge developments and deployment insights in this fast-moving area.

Studies were prioritized if they developed or evaluated FL algorithms tailored to smart city settings, implemented FL on real-world datasets or urban-scale testbeds, applied FL to urban challenges such as mobility optimization, energy management, anomaly detection, or disaster response, or examined practical constraints like data heterogeneity, privacy preservation, communication efficiency, and resource limitations at the edge. To strengthen coverage in underrepresented domains such as EV infrastructure, disaster resilience, and grid optimization, we also conducted backward and forward citation tracking from foundational works [28].

To ensure consistency and transparency, the screening and inclusion process was conducted in multiple stages and coherence was verified in all records. Discrepancies or borderline cases were resolved through iterative review and consensus, rather than through formal inter-rater scoring, following best practices for narrative synthesis, where interpretative judgment is prioritized over quantitative agreement metrics. No formal quality assessment procedure was applied, as this review adopts a thematic and methodological synthesis rather than a quantitative meta-analysis. Instead, the assessment of methodological rigor was integrated within the synthesis itself, emphasizing reproducibility indicators such as access to the dataset, code availability, and validation of the deployment. This approach is consistent with the PRISMA 2020 [14] and Kitchenham guidelines, providing methodological transparency without imposing scoring frameworks unsuitable for heterogeneous study designs.

Quantitatively, the reviewed corpus spans publications from 2019 to 2024, reflecting the rapid expansion of research interest in this domain. The number of relevant studies increased from fewer than ten in 2019 to more than 40 in 2023, with an overall annual growth rate of approximately 35%. Among the 116 included works, 38% focus on transportation and mobility optimization, 22% on energy and grid management, 18% on anomaly detection and cybersecurity, and the remainder on cross-domain or infrastructural frameworks. This distribution provides a clear quantitative overview of the literature analyzed and highlights the interdisciplinary character of federated learning research in smart city contexts.

## 3.4. Data Extraction and Analysis

For each selected article, we extracted information on the application domain (e.g., traffic, energy, public safety), the FL architecture employed (centralized, hierarchical, peerto-peer), the learning paradigm adopted (supervised, reinforcement, ensemble), and the evaluation metrics used. We also recorded privacy mechanisms such as differential privacy and secure multiparty computation, as well as indicators of reproducibility, including dataset accessibility and code availability. This structured extraction enabled consistent cross-comparison across studies with otherwise heterogeneous setups.

To support reproducibility, we compile a consolidated dataset and source table (Table A1, provided at the end of this article) as a benchmarking resource for future research. Although placed at the end of the paper for readability, this table is a central contribution of our survey.

Our analysis was organized around the taxonomy introduced in Figure 2, which groups FL applications in smart cities into four overarching themes: Privacy and Security, Resource Optimization, Event Detection and Situational Awareness, and Energy Management and Sustainability. Subtopics such as robustness and architectural variants are treated as cross-cutting issues within these categories rather than separate dimensions. The complete screening process is summarized in Table 3, with a PRISMA-style flow diagram provided in Figure 3.

Figure 2. Challenge-oriented taxonomy of FL applications in smart cities, illustrating the main thematic pillars reviewed in this study.

<!-- image -->

Figure 3. PRISMA-style flow diagram illustrating the systematic review process. From an initial 246 identified records across IEEE Xplore, ACM Digital Library, Scopus, and complementary sources, deduplication and relevance screening resulted in a final corpus of 116 studies included in the synthesis.

<!-- image -->

## 3.5. Architectural and Algorithmic Innovations

To meet the diverse and evolving demands of smart cities, FL has expanded beyond its original centralized form into a spectrum of architectural paradigms, including hierarchical and fully decentralized (peer-to-peer) designs [29]. Hierarchical FL is particularly attractive for city-scale deployments, as it reduces communication latency and distributes workloads across cloud, fog, and edge layers. This layered structure allows for scalable and responsive learning in environments characterized by device heterogeneity and intermittent connectivity.

On the algorithmic front, federated reinforcement learning (FRL) has gained momentum as a framework for adaptive control in smart cities, supporting applications such as traffic-signal optimization, energy dispatch, and large-scale resource allocation [30]. Ensemble-based FL and emerging 'federated X-learning' frameworks have also been proposed to improve robustness, personalization, and generalization. These include client clustering, meta-learning, and knowledge fusion approaches [31], which help systems adapt to non-IID data and client diversity. Recent surveys of FL system design further enrich the methodological landscape, covering client selection, aggregation strategies, and multi-tier coordination mechanisms [32]. Collectively, these architectural and algorithmic innovations provide the foundation for the deployment of next-generation FL systems in intelligent and resilient urban infrastructures.

## 3.6. Deployment Readiness and Real-World Pilots

To complement the methodological synthesis, we further evaluated each study in terms of its readiness to be deployed in real-world smart city contexts. This assessment aimed to distinguish conceptual or simulation-based work from implementations demonstrating operational feasibility. Six criteria were used to guide this classification: (1) data scale: the volume and diversity of local and global datasets used for training; (2) device count: the number of participating edge or client nodes; (3) duration: the temporal span of training or evaluation; (4) failure reporting: documentation of issues such as communication loss, model divergence, or hardware limitations; (5) privacy and security controls: explicit integration of mechanisms such as differential privacy, secure aggregation, or homomorphic encryption; and (6) artifact availability: public release of code, configuration scripts, or datasets supporting reproducibility.

Based on these criteria, the reviewed works were categorized into three tiers of deployment maturity: (a) simulation-only studies (62% of the corpus), which focus primarily on algorithmic validation using synthetic or partitioned datasets; (b) prototype or small-scale pilots (27%), typically evaluated on limited physical testbeds or short-term federated experiments; and (c) real-world or production-level deployments (11%), involving operational FL systems deployed across municipal or industrial infrastructures. These real-world pilots commonly feature heterogeneous sensing devices, longer monitoring durations, and explicit reporting of privacy compliance and failure management.

This classification provides a structured understanding of the technological maturity in the surveyed studies and situates our subsequent synthesis within a clearly defined methodological context. The following subsection outlines how the scope and organization of this review compare with related federated learning surveys.

## 3.7. Comparison with Related Surveys

Several recent surveys have examined federated learning in smart cities or related contexts, but they differ in scope, organization, and technical depth. Pandya et al. [10] provided a broad overview of FL applications in domains such as transportation, healthcare, UAVs, and governance. Although valuable for mapping the landscape, their review is largely narrative and lacks a structured taxonomy or technical granularity. Zheng et al. [11] focused on privacy and architectural aspects, organizing the literature by application domains. However, their domain-based taxonomy makes it difficult to analyze crosscutting challenges shared between urban systems. Jia et al. [12] narrowed the scope to communication-efficient FL in mobile edge computing, offering rich technical insights into compression and aggregation but with limited applicability to broader urban contexts.

In contrast, our survey adopts a challenge-oriented taxonomy that emphasizes systemic issues, privacy-utility trade-offs, resource constraints, event detection, and sustainability, rather than siloed application domains. We also place special focus on methodological innovations (e.g., reinforcement and ensemble FL), reproducibility practices, and deployment readiness, aspects often underexplored in earlier reviews. Table 4 summarizes how our work extends and complements these related efforts.

Out of 116 studies analyzed, 72 (62%) relied solely on simulation-based evaluation, 31 (27%) used publicly available datasets, and only 13 (11%) provided partial or complete open-source code. These quantitative indicators highlight the limited reproducibility and transparency that currently characterize FL implementations in smart city contexts.

Table 4. Comparison of this survey with related FL surveys in smart city contexts.

| Aspect   | This Survey                                                                                        | Pandya et al. (2023) [10]                                    | Zheng et al. (2021) [11]                                                  | Jia et al. (2025) [12]                           |
|----------|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------|---------------------------------------------------------------------------|--------------------------------------------------|
| Scope    | Cross-cutting smart city challenges: privacy, optimization, event detection, energy sustainability | Domains such as transportation, healthcare, UAVs, governance | Broad range of applications: healthcare, transportation, energy, security | Communication- efficient FL in edge environments |

Table 4. Cont.

| Aspect                    | This Survey                                                                              | Pandya et al. (2023) [10]                                         | Zheng et al. (2021) [11]                | Jia et al. (2025) [12]                                   |
|---------------------------|------------------------------------------------------------------------------------------|-------------------------------------------------------------------|-----------------------------------------|----------------------------------------------------------|
| Structure                 | Organized around four urban challenges                                                   | Application-driven, domain-based                                  | Application-specific taxonomy           | Communication- focused framework                         |
| Technical Focus           | Architectural strategies, FL variants (ensemble, reinforcement), system-level challenges | Conceptual overview with emphasis on privacy and decentralization | FL architectures and privacy mechanisms | Compression, over-the-air aggregation, client scheduling |
| Deployment Insights       | Emphasis on real-world datasets, testbeds, and reproducibility                           | Illustrative case studies with less technical depth               | Limited deployment benchmarking         | Focus on optimization, not deployment                    |
| Security and Privacy      | Analysis of DP, secure aggregation, and adversarial threats                              | High-level discussion of privacy                                  | Broad treatment of privacy issues       | Communication/ security threat mitigation                |
| Energy and Sustainability | EVs, demand forecasting, energy-efficient FL                                             | Brief mention of smart grids                                      | Part of broader application scope       | Communication energy optimization only                   |
| Research Directions       | Benchmarks, explainability, scalability, fairness                                        | High-level opportunities across domains                           | Broad research themes                   | Suggestions for efficient architectures                  |
| Reproducibility           | Strong focus on datasets, code, and standardization                                      | Not emphasized                                                    | Not structured explicitly               | Technical, but not reproducibility- focused              |

## 4. Privacy and Security

This section marks the first thematic pillar of the taxonomy introduced in Figure 2. Each pillar is developed with comparable analytical depth and follows a consistent paragraph structure, where individual studies are discussed in terms of their motivation, methodological approach, key findings, and limitations. This uniform format ensures structural balance and readability across the survey.

With the rapid growth of smart cities, powered by dense networks of sensors, mobile devices, and interconnected infrastructure, protecting privacy and securing data have become essential priorities. FL provides a natural foundation by keeping raw data local to edge devices, yet its deployment in real-world urban systems remains vulnerable to threats such as inference attacks, model poisoning, and systemic weaknesses in aggregation protocols. These vulnerabilities are especially critical in domains like transportation, energy, and public safety, where compromised models could have direct societal consequences.

To mitigate these risks, research has advanced along multiple fronts. Algorithmic defenses, including robust aggregation, anomaly detection, and secure multi-party computation, aim to preserve model integrity, while architectural strategies such as data localization and blockchain integration reinforce trust and compliance with governance requirements. In parallel, differential privacy has emerged as a central tool for balancing confidentiality with model utility. Complementing these technical contributions, survey studies have synthesized existing work and underscored reproducibility gaps, highlighting the ongoing need for standardized benchmarks and real-world validation.

Federated learning in smart city environments operates under a diverse threat landscape encompassing both data-level and model-level attacks. Inference attacks, including membership and model inversion [33,34], attempt to recover sensitive local information from shared gradients; these are typically mitigated using differential privacy or secure multiparty computation [35]. Poisoning attacks compromise model integrity by injecting malicious updates, countered through robust aggregation or anomaly detection that filters abnormal gradients before global averaging [36,37]. Backdoor attacks embed hidden triggers within local models, defended through gradient sanitization and model auditing techniques [38]. In addition, Byzantine failures arising from faulty or unreliable clients can be addressed using reputation-based or coordinate-median aggregation [39]. Communication-level threats such as eavesdropping or replay attacks are prevented through homomorphic encryption and secure channel protocols [40]. These mechanisms collectively strengthen privacy and resilience but also introduce trade-offs between computational overhead and model accuracy, which remain open challenges for large-scale smart city deployments.

This section reviews key developments in federated learning for smart cities with a focus on privacy and security. We organize the discussion into five focal areas: Mitigating Attacks, Data Localization, Differential Privacy, Secure Multi-Party Computation, and Architectural and Domain Adaptations, followed by an overview of related Surveys and Research Outlook. While techniques such as differential privacy and secure multiparty computation provide algorithmic privacy guarantees, architectural strategies like data localization and blockchain integration respond to legal and governance requirements. By treating these dimensions separately, we capture both the technical depth and the practical deployment challenges facing real-world urban systems.

## 4.1. Mitigating Attacks

Although FL provides inherent privacy advantages by keeping raw data local, it remains vulnerable to a wide range of adversarial threats. These threats exploit the very mechanisms that make FL attractive, such as distributed updates and decentralized training, to compromise system integrity. The most prominent categories include data poisoning, backdoor insertion, Byzantine failures, and inference-based attacks. Addressing these threats is especially critical in smart cities, where compromised models could directly impact transportation safety, energy grid stability, or public surveillance systems. Research efforts to mitigate these vulnerabilities have progressed in three main directions: robust aggregation, anomaly and hybrid defenses, and application-oriented intrusion detection.

The first stream of work focuses on aggregation rules that are resilient to malicious contributions. Blanchard et al. [37] proposed the Byzantine-resilient Krum algorithm, which reduces the influence of adversarial clients by selecting the update closest to the majority. Although effective in limiting the impact of outliers, Krum struggles in highly heterogeneous data environments typical of urban IoT systems. To address this, Mhamdi et al. [41] introduced the Hidden Autoencoder Defense, which reconstructs updates and uses reconstruction error to identify poisoned contributions. Although these methods improve robustness, they often introduce computational overhead, raising concerns for deployment on resource-constrained devices such as sensors and edge gateways. Scalability also remains a challenge, as simple aggregation rules can falter when scaled to the thousands of clients present in city-wide deployments.

Asecond body of research emphasizes anomaly detection and hybrid strategies that monitor update streams for suspicious behavior. Bagdasaryan et al. [38] revealed the risk of backdoor poisoning, where a malicious client embeds hidden triggers to force targeted misclassifications, underscoring the inadequacy of aggregation alone. Building on this line of inquiry, Sun et al. [42] conducted one of the earliest systematic analyzes of backdoor vulnerabilities in federated learning, showing that secure aggregation alone cannot fully prevent such attacks and motivating the development of hybrid detection-based defenses. Their experiments highlight how backdoors can persist even under differential privacy or clipping mechanisms, emphasizing the need for stronger anomaly-based protections.

To support rigorous evaluations, Croce and Hein [43], Croce et al. [44], and Zhang et al. [45] introduced parameter-free attacks, benchmarks, and standardized metrics. These contributions are especially important for smart cities, where traditional ML security benchmarks often fail to capture the complexity of heterogeneous IoT data and non-IID distributions. However, anomaly-based methods remain sensitive to parameter tuning and may be circumvented by adaptive adversaries who mimic benign update patterns.

The third research direction involves tailoring defenses to specific smart city domains. Arya et al. [46] designed a federated intrusion detection system for Vehicular Ad Hoc Networks (VANETs), leveraging heterogeneous neural networks to preserve privacy while achieving strong detection performance. Priyadarshini [47] introduced a hybrid framework that combines FL with split learning to secure IoT networks, balancing privacy preservation with adaptability to new threats. Matheu et al. [48] extended this idea by incorporating Threat Intelligence and Manufacturer Usage Description (MUD) files into FL-based intrusion detection, strengthening both preventive and mitigative defenses. Similarly, Djenouri and Belbachir [49] proposed a trusted authority within a federated intrusion detection system, demonstrating improved scalability and privacy protection on the NSL-KDD dataset.

Together, these approaches highlight the diverse strategies that are being developed to strengthen FL against adversaries in smart cities. Aggregation-based methods provide lightweight robustness but remain fragile under non-IID data. Anomaly-driven and hybrid defenses offer stronger resilience, but with higher overhead that can hinder IoT deployments. Domain-specific intrusion detection frameworks demonstrate practical potential, yet most evaluations rely on controlled datasets rather than operational infrastructures. Reproducibility remains a persistent gap, as implementations are rarely open-source and benchmarking practices remain fragmented. Looking ahead, key priorities include developing standardized adversarial benchmarks tailored to heterogeneous IoT and VANET settings, balancing robustness with energy and latency constraints, and exploring explainable defenses that can build trust among diverse stakeholders in urban ecosystems.

## Case Summary: Federated Intrusion Detection for VANETs [46,49]

Scope: Privacy-preserving intrusion detection across distributed vehicular networks.

Infrastructure: Edge-based FL among on-board units and roadside gateways.

Devices: 50-100 simulated vehicles in field test; heterogeneous sensors.

Metrics: Detection accuracy, precision/recall, communication latency.

Artifacts: Dataset (NSL-KDD) publicly available; implementation not released.

## 4.2. Data Localization

Acentral motivation for adopting federated learning in smart cities is the need to avoid centralizing raw data, which can create risks of breaches, leakage, and unauthorized access. Unlike traditional machine learning systems that depend on large aggregated repositories, FL allows model training to take place directly on edge devices or user terminals. In this design, raw data never leave their local sources; instead, only model updates are transmitted to an aggregator, often with encryption. This decentralized structure reduces the attack surface while also aligning with privacy regulations and governance frameworks that restrict cross-agency data sharing.

Several studies have demonstrated the benefits of data localization in smart-city domains. Kim et al. [50] proposed a personalized federated transfer learning framework for building-energy forecasting in heterogeneous sensing environments, demonstrating improved prediction accuracy across diverse buildings using real energy-consumption datasets without requiring raw data sharing. Nadeem and Jaber [51] extended this line of work by incorporating differential privacy into FL-based energy systems, allowing stronger user participation without sacrificing the utility of the model. Beyond the energy domain, Jiang et al. [9] examined FL as a replacement for centralized sensing in urban environments, showing how local training can protect privacy in citizen-generated data while highlighting practical obstacles such as unstable communication, the absence of trust frameworks, and limited stakeholder engagement.

Taken together, these works suggest that data localization is not only a technical choice but also a governance necessity in smart cities, where raw data often cannot be pooled across agencies such as utilities, transport, and healthcare. Although existing studies confirm bandwidth savings and privacy advantages, their validation typically remains limited to controlled experiments, with few openly available implementations or datasets. This lack of reproducibility makes it difficult to assess how well proposed systems scale in heterogeneous city-wide deployments. Moreover, a fundamental trade-off persists: reducing data transfers improves compliance and efficiency, but also exposes systems to communication bottlenecks and reliability concerns when thousands of devices are involved. As a result, most of the current efforts remain proof-of-concept demonstrations rather than integrated operational pilots into urban infrastructure.

## Case Summary: Personalized Federated Transfer Learning for Building-Energy Forecasting [50]

Scope: The study investigates personalized federated transfer learning for energyconsumption forecasting across heterogeneous buildings using real-world campus energy datasets. Although the system is not deployed in a live commercial smart-city infrastructure, the use of actual building-level data provides practical relevance for smart-building applications.

Infrastructure: Asimulated federated learning environment is constructed to emulate multiple building clients with diverse sensing conditions. Personalization is achieved through model-ensemble strategies and multi-level masking to adapt to client heterogeneity.

Devices/Data: Real energy-consumption traces collected from multiple campus buildings, representing different usage patterns, temporal behaviors, and sensing characteristics. No live FL deployment was used; data were processed offline.

Metrics: The proposed personalized FTL approach improves forecasting accuracy and robustness compared with global FL baselines, especially under heterogeneous sensing environments.

Artifacts: The authors state that the datasets are available upon request. No open-source code or supplementary software artifacts were publicly released.

## 4.3. Differential Privacy (DP)

Although data localization in FL reduces the risk of raw data exposure, it cannot fully prevent inference attacks such as model inversion or membership inference. DP complements this architectural safeguard by injecting calibrated noise into model updates, providing formal privacy guarantees at the cost of some utility loss. In smart city contexts, DP has been explored in mobility, energy, IoT, and healthcare systems, with methodological variations tailored to different deployment scenarios.

He et al. [52] proposed a federated clustered learning framework that integrates adaptive local differential privacy (LDP) for heterogeneous IoT data. By dynamically adjusting privacy budgets and using discrete cosine transform-based compression, their approach maintained high model accuracy under strong privacy constraints.

In the energy domain, Ashraf et al. [53] integrated DP with ensemble-based aggregation to detect theft of electricity in smart grids, demonstrating effective anomaly detection without compromising privacy. Baligodugula and Amsaad [54] further optimized DP for resource-constrained devices, showing that hardware-aware designs can significantly reduce computational overhead and make DP more practical for IoT nodes.

Building on these foundations, a second strand of research explores localized and personalized DP. Local LDP has been applied to vehicular networks, where each node perturbs updates before sharing, improving resilience against model inversion and eavesdropping [55]. Correlated DP (CDP) adapts noise injection to data sensitivity, allowing personalized privacy guarantees for autonomous driving systems [56]. Jiang et al. [57] advanced this idea with Fed-MPS, a parameter-selection method that combines LDP with resource efficiency for large-scale IoT deployments. These adaptations illustrate how DP can go beyond generic protections to context-aware safeguards that address the heterogeneity of smart city data.

A third line of work focuses on hybrid DP models that integrate privacy with trust and incentive mechanisms. Blockchain-enabled frameworks [58,59] combine tamper-proof auditability with DP guaranties, ensuring transparent aggregation for smart grids and healthcare collaborations. Zhang et al. [60] introduced a contract-theoretic DP framework for connected vehicles, balancing privacy and utility while incentivizing honest participation among stakeholders. Similarly, Ahmed et al. [61] explored adaptive DP in healthcare, dynamically adjusting the privacy budget to support timely and privacy-preserving diagnosis during emergencies. These approaches highlight DP's flexibility in multi-stakeholder smart city systems where trust, incentives, and transparency are as important as privacy.

Healthcare, although not traditionally part of urban infrastructure, has served as a proving ground for DP-enabled FL. Shukla et al. [62] demonstrated that DP-FL can support privacy-preserving cancer diagnosis in hospitals, achieving high diagnostic accuracy while protecting sensitive records. Such studies underscore DP's broader relevance to urban health systems and the potential to extend lessons from clinical collaborations to cityscale deployments.

In general, DP has emerged as a cornerstone for balancing privacy and utility in FL for smart cities. Results generally show modest accuracy losses under moderate privacy budgets, but greater degradation in stricter settings. Deployment challenges remain: most implementations are evaluated in simulation or small-scale pilots, with reproducibility uneven due to missing datasets or code. Future work should prioritize (i) benchmarking DP-enhanced FL in real-world urban datasets, (ii) advancing scalable and hardware-aware implementations for resource-constrained IoT devices, and (iii) developing standardized trade-off metrics that jointly assess privacy strength, system latency, and utility in operational environments [63].

## Case Summary: DP-Enhanced FL for Electricity-Theft Detection [53]

Scope: Detecting abnormal energy-consumption patterns under differential privacy guarantees.

Infrastructure: Central aggregator with distributed smart-meter clients.

Devices: 100 smart meters in a regional utility pilot.

Metrics: F1-score, privacy loss ( ε ), communication overhead.

Artifacts: Partial dataset (utility records); code unavailable.

## 4.4. Secure Multi-Party Computation (SMPC)

Another line of research focuses on secure multi-party computation (SMPC), which enables collaborative model training while ensuring that individual client updates remain confidential. In this paradigm, aggregation is performed over encrypted or secret-shared contributions, allowing the server to compute a global model without ever accessing raw updates. Bonawitz et al. [40] introduced a widely adopted framework for secure aggregation based on pairwise masking and homomorphic encryption, establishing a cornerstone for privacy-preserving FL. Building on this direction, Kanagavelu et al. [64] proposed a two-phase MPC-enabled framework that integrates secure aggregation and secret sharing, achieving strong privacy guaranties and scalability across distributed smartmanufacturing nodes without compromising convergences.

SMPChasalso been explored in combination with other privacy-preserving techniques. Byrd and Polychroniadou [65] presented a protocol that integrates SMPC with differential privacy for financial services. Using a credit card fraud dataset, they demonstrated that strong confidentiality can be preserved while maintaining predictive accuracy, though at the cost of additional computational overhead. These hybrid approaches illustrate how SMPC can be adapted to domain-specific constraints, balancing the need for rigorous privacy guaranties with practical performance requirements.

Despite its strong mathematical foundations, SMPC remains computationally demanding, which limits its scalability for city-scale deployments involving thousands of IoT devices. Encryption and masking introduce latency and energy costs, raising concerns for real-time systems such as transportation networks or emergency response platforms. Most evaluations remain confined to controlled experiments, often with synthetic data, and reproducibility is weak due to the limited release of open-source implementations or standardized benchmarks. For smart city applications, the challenge lies in making SMPClightweight enough to operate within resource-constrained edge environments while retaining its formal guaranties of confidentiality and robustness.

In summary, SMPC provides one of the most rigorous approaches to secure FL, ensuring that no single party gains access to the contributions of another. However, the trade-off between privacy strength and system efficiency is significant, and widespread adoption in smart cities will depend on reducing computational costs, improving benchmarking practices, and validating these systems in real-world urban infrastructures rather than simulated settings.

## 4.5. Architectural and Domain Adaptations

Beyond algorithmic defenses, a growing body of research has adapted federated learning architectures to the requirements of specific smart city domains. These adaptations often combine privacy-preserving mechanisms with structural modifications to address constraints such as device heterogeneity, communication reliability, and regulatory demands. By extending the core FL framework, studies have sought to balance performance with trust and accountability across diverse urban infrastructures.

In healthcare, Rieke et al. [66] and Vu Khanh et al. [67] demonstrated the viability of cross-institutional diagnostic models that preserve patient confidentiality while achieving accuracy comparable to centralized approaches. These systems typically adopt centralized or hierarchical aggregation, but their reproducibility is limited due to proprietary datasets and restricted access to implementation details. In vehicular networks and smart grids, researchers have customized FL models for location prediction [68], secure data sharing among distributed energy assets [69], and energy-aware client participation [51]. These studies highlight how privacy considerations influence evaluation metrics, prioritizing latency, energy efficiency, and reliability over raw accuracy.

At the architectural level, several enhancements focus on strengthening trust and auditability. The integration of blockchain in FL workflows has been explored as a means to ensure transparency and tamper-resistance [70,71], while frameworks such as FL-DABE-BC [72] extend this idea with decentralized authentication and fine-grained access control. Wang et al. [73] advanced this direction by proposing a decentralized FL framework for IoT devices, where blockchain-based consensus mechanisms replace the central server and a reputation system ranks the reliability of the client. Their evaluation of synthetic IoT datasets demonstrated improvements in accuracy, communication integrity, and resilience to malicious participants, though still in a simulated environment.

Despite these promising innovations, most real-world and academic prototypes continue to rely on centralized or hierarchical FL architectures. Fully decentralized or peer-topeer variants remain rare, particularly in privacy-sensitive applications, due to challenges such as trust management without central authority, synchronization overhead, and the risk of cascading failures under adversarial conditions. Even when blockchain or reputation systems are introduced, they often increase communication costs and reduce efficiency, creating new trade-offs between security and scalability.

In summary, architectural adaptations have expanded the reach of FL in healthcare, transportation, and energy systems while introducing mechanisms for stronger transparency and accountability. However, these advances remain largely experimental: evaluations often use synthetic data, reproducibility is weak due to the lack of open-source implementations, and the efficiency of decentralized trust models remains an open question. Bridging these gaps is essential for moving from proof-of-concept prototypes to operational deployments in smart city infrastructures.

## Case Summary: Cross-Institutional Healthcare FL [66,67]

Scope: Collaborative diagnostic modeling among hospitals while preserving patient privacy.

Infrastructure: Central server coordinating hospital-level FL clients.

Devices: 5-10 hospitals, GPU-enabled servers.

Metrics: AUC, convergence rate, communication cost.

Artifacts: Clinical datasets under license; no public code.

## 4.6. Surveys and Research Outlook

Alongside technical advances, several survey papers have examined privacy and security in federated learning, providing valuable overviews while also exposing persistent gaps. Rahdari et al. [13] and Mathew and Panchami [74] offer comprehensive reviews of threat models, defensive mechanisms, and privacy-preserving techniques in FL for smart cities. Both emphasize adversarial resilience as a central challenge, but also reveal a common shortcoming: limited reproducibility. Most of the studies discussed in these surveys do not release implementation code or datasets, making rigorous benchmarking between solutions difficult. This lack of openness continues to impede fair comparisons and slows progress toward deployable frameworks.

Expanding beyond generic analysis, Al-Huthaifi et al. [75] provide a domain-oriented perspective, reviewing FL applications across transportation, healthcare, and communication. Their work underscores FL's role as a privacy-preserving alternative to centralized systems, while also calling for adversarial testing under realistic threat conditions. Importantly, they outline the technological risks and practical limitations that currently prevent large-scale deployment and propose a roadmap for strengthening privacy and robustness. This emphasis on practical constraints distinguishes their contribution from earlier surveys that focused primarily on conceptual frameworks.

Other surveys have investigated narrower domains. Zhao et al. [76], for instance, explore how local differential privacy (LDP) can mitigate threats in Internet of Vehicles (IoV) systems, particularly against location tracking and identity inference. Although limited to vehicular networks, their insights illustrate the importance of lightweight, decentralized privacy mechanisms in mobility-focused smart city applications, where centralized trust is often infeasible.

Taken together, existing surveys confirm FL's promise for privacy-preserving analytics but repeatedly highlight weaknesses in adversarial robustness, deployment readiness, and reproducibility. Compared with these works, our survey contributes a challengeoriented taxonomy that cuts across application domains and emphasizes shared technical issues such as communication bottlenecks, heterogeneity, and regulatory constraints. A distinctive feature of our work is the explicit evaluation of data set availability and code availability, which addresses reproducibility as a first-class concern rather than a secondary observation. In doing so, we respond directly to the limitations of previous surveys and position reproducibility and deployment readiness as essential next steps for academic and industrial research agendas.

A comparative synthesis of privacy and security approaches in FL for smart cities is presented in Table 5, which summarizes representative methods, key trade-offs, reproducibility gaps, and their relevance to urban applications.

Table 5. Summary of Privacy and Security Strategies in Federated Learning for Smart Cities.

| Theme                                                                                              | Representative Approaches Key                                                                                                                                 | Smart City Relevance                                                                                                                                         | Insights/Trade-Offs Reproducibility Status   |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| Attacks Byzantine-robust aggregation (Krum) [37]; analysis and hybrid defense intrusion            | backdoor motivation [42]; detection [46] Strengthens resilience against poisoning and backdoor threats, though increases computational and communication cost | VANETs, and detection for infrastructure Primarily simulation-based; few standardized benchmarks; reproducibility remains limited                            | Mitigating IoT, intrusion critical           |
| building forecasting [50]; privacy-preserving energy FL [51];                                      | energy Reduces data transfer and enhances privacy compliance; limited large-scale validation                                                                  | Energy forecasting, urban sensing, multi-agency governance Few public datasets; most implementations evaluated on real data but simulated federated settings | Data Localization                            |
| Adaptive clustered LDP [52]; LDP [55]; CDP [56]; DP + Blockchain [58,59]; FedDP [53]; Fed-MPS [57] | Balances privacy with accuracy; adaptive budgets mitigate utility loss under strict constraints                                                               | IoT networks, VANETs, smart grids, healthcare pilots Mixed: some public datasets (FedDP, Fed-MPS); clustered LDP remains simulation-based                    | Differential Privacy                         |

Table 5. Cont.

| Theme                                 | Representative Approaches                                                                                      | Key Insights/Trade-Offs                                                                                                    | Smart City Relevance                                                          | Reproducibility Status                                                                |
|---------------------------------------|----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| Secure Multi-Party Computation (SMPC) | Secure aggregation [40]; two-phase MPC framework [64]; hybrid SMPC + DP [65]                                   | Provides strong confidentiality via secret sharing and encrypted aggregation, but increases latency and computational cost | IoT deployments, smart manufacturing, financial services, sensitive analytics | Rarely open-sourced; benchmarking inconsistent; industrial datasets typically private |
| Architectural and Domain Adaptations  | Blockchain-enabled FL [73]; hierarchi- cal/decentralized FL; FL-DABE-BC [72]; domain-specific adaptations [67] | Enhances trust and auditability, but adds communication overhead                                                           | Smart grids, vehicular networks, healthcare collaborations                    | Reliant on proprietary datasets; limited open implementations                         |
| Surveys &Outlook                      | Privacy/security reviews [13,74-76]                                                                            | Identify gaps in adversarial robustness, trust, and reproducibility                                                        | Broad smart city domains (transportation, healthcare, IoT)                    | Rarely provide datasets or benchmarks; highlight need for standardization             |

Together, these studies establish privacy and security as the foundational requirements for federated learning in smart cities. Approaches such as differential privacy, secure multiparty computation, and blockchain integration strengthen confidentiality and trust, while robust aggregation and anomaly detection aim to counter adversarial threats. However, reproducibility remains weak, with few open datasets or standardized benchmarks available. Most implementations are validated only in simulations or limited pilots, raising questions about their scalability in real-world deployments. Future work must therefore prioritize benchmarking under realistic urban conditions, balancing robustness with efficiency, and ensuring that privacy-preserving FL can be deployed at city scale.

## 5. Resource Optimization

As FL moves from theoretical frameworks to real-world smart city deployments, optimizing resource usage becomes crucial. Smart city environments are often constrained by energy, bandwidth, memory, and computational resources. Consequently, FL research has increasingly focused on improving energy efficiency, minimizing communication overhead, enabling computational offloading, and enhancing system scalability. These optimization strategies are critical not only to reduce cost and energy consumption, but also to maintain real-time responsiveness and scalability in smart city operations.

This section categorizes key research efforts into four main areas: GreenFL, Enhanced Scalability, Reduced Communication Overhead, and Computational Offload to Edge. Each subsection highlights representative methodologies, system architectures, evaluation metrics, and urban use cases, while also noting reproducibility gaps.

## 5.1. GreenFL

Optimizing energy consumption is vital for sustainable FL deployment in resourceconstrained urban infrastructures, such as edge devices and IoT sensors prevalent in smart cities.

Yu et al. [77] proposed an energy-aware device scheduling strategy for joint federated learning in edge-assisted Internet-of-Agriculture environments. By adaptively selecting clients based on residual energy and model contribution, their approach reduced total energy consumption and communication delay without compromising accuracy, demonstrating the importance of energy-efficient coordination in edge-based FL systems.

Arouj and Abdelmoniem [78] introduced EAFL, an energy-sensitive client selection scheme that prioritizes devices with higher battery reserves. This strategy improved the accuracy of the model by up to 85% and reduced the rates of client dropout by a factor of 2.45. Although the system also relied on a centralized FL architecture, the availability of datasets or the implementation code was not specified.

Chen and Liu [79] addressed the challenge of reducing energy consumption in MEC environments by proposing a federated deep reinforcement learning framework (FL-DDPG). Their method formulates a joint optimization problem for task offloading and resource allocation, allowing IoT devices to collaboratively learn energy-efficient policies without compromising data privacy. The two-time-scale Deep Deterministic Policy Gradient (DDPG) algorithm significantly reduced energy usage in simulated smart city environments, demonstrating the effectiveness of the model for edge-based FL intelligence.

In general, energy-aware FL methods show promising gains in reducing device strain and extending operational lifetimes in smart city infrastructures such as street lighting and IoT sensors. The trade-off lies in balancing accuracy with device longevity, as aggressive energy savings can slow convergence. Reproducibility is weak, few works share code or datasets, limiting validation beyond simulations. Real-world pilots, such as those in urban lighting, are rare but essential to confirm feasibility at the city scale.

Given these energy-driven trade-offs, the next challenge is scalability, specifically extending FL frameworks from dozens of energy-constrained devices to thousands or even millions of smart city nodes without collapsing under communication or computation strain.

## 5.2. Enhanced Scalability

Scalability is essential for large-scale FL deployments on heterogeneous smart city devices and networks.

McMahan et al. [8] introduced FedAvg, a foundational FL algorithm that reduces global communication rounds by allowing local model updates before aggregation. Evaluated on non-IID decentralized image and text datasets, it demonstrated strong performance and remains the most widely adopted baseline for scalable FL systems.

Sattler et al. [80] addressed the scalability challenges of FL in non-IID client settings by combining sparse ternary compression (STC) with adaptive update strategies. STC reduces the size of the model update by encoding gradients in ternary values ( {-1, 0, + 1 } ), significantly reducing communication costs. To maintain learning performance, the authors incorporated momentum correction, error feedback, and update sparsification, enabling efficient convergence even with severely imbalanced data distributions. Their approach achieved a reduction of up to 99% in communication bandwidth while maintaining the accuracy of the competitive model. The results were validated in simulated environments; however, the study did not provide publicly available reproducibility resources.

Tian et al. [81] introduced FedFOR, a stateless FL framework that integrates firstorder regularization to address the convergence challenges posed by the heterogeneity of the clients. Unlike traditional FL methods that rely on frequent state synchronization, FedFOR allows clients to perform updates independently without storing prior states between communication rounds. Experimental results on image classification tasks with non-IID data distributions show that FedFOR outperforms FedAvg and FedProx in both convergence speed and final accuracy.

Li et al. [82] proposed DART, a robust evaluation framework for decentralized FL that facilitates benchmarking among heterogeneous clients. These frameworks emphasize generalization and stability across large-scale decentralized infrastructures, although public code is often unavailable.

Khan et al. [83] developed a dispersed federated learning (DFL) framework for cognitive IoT in smart industries. The model formulates an integer linear programming problem to minimize FL cost, which is decomposed into sub-problems for association and resource allocation. These are relaxed into convex forms and solved via iterative rounding algorithms. Experimental results show that DFL outperforms random association schemes, improving convergence and scalability by eliminating reliance on central coordination.

Scalability remains a defining challenge for FL in smart cities, where millions of devices may contribute updates under highly non-IID conditions. Techniques like sparse compression, stateless updates, and dispersed coordination improve efficiency but introduce trade-offs in accuracy or system stability. Despite strong simulation results, reproducibility is limited by the absence of shared code and large-scale urban benchmarks. Without city-scale pilots, scalability remains more a theoretical goal than a demonstrated capability.

While scalability focuses on coordinating vast population of devices, another central bottleneck is communication: reducing the volume, frequency, and cost of updates is vital for resource-limited networks typical of urban infrastructures.

## Case Summary: Sparse Ternary Compression for Scalable FL [80]

Scope: Communication-efficient model compression for large-scale FL.

Infrastructure: Decentralized simulation using image/text datasets under non-IID conditions.

Devices: 100 simulated clients with heterogeneous data partitions.

Metrics: Bandwidth reduction ( ≈ 99%), convergence accuracy, latency.

Artifacts: Simulation code not released; results validated experimentally.

## 5.3. Reduced Communication Overhead

Minimizing communication costs is critical in smart city environments with constrained networks.

Konen et al. [6] laid the early groundwork for communication-efficient FL using structured updates and subsampling techniques to reduce uplink bandwidth. Although privacy was not the main focus, these strategies indirectly supported it by minimizing the volume of transmitted model information. The methods were demonstrated in decentralized datasets in mobile environments, setting a precedent for later bandwidth-conscious protocols.

Liu et al. [84] proposed an adaptive client selection strategy for 5G/B5G vehicular networks, where federated learning assists in balancing client participation under varying link conditions, reducing communication overhead, and improving training stability.

Jia et al. [12] provided a comprehensive survey of communication-efficient strategies for FL of the mobile edge, covering techniques such as model quantification, gradient sparsification, asynchronous updates, and adaptive aggregation. Although the paper does not propose new algorithms, it offers a taxonomy of methods tailored for heterogeneous edge networks. The survey highlights peer-to-peer FL as a promising direction, although still uncommon due to synchronization complexity and trust management.

Seon Hong et al. [85] conducted an extensive analysis of resource optimization in federated wireless learning, focusing in particular on computational, communication, and power limitations. The authors presented a convergence analysis tailored to wireless settings and introduced a collaborative framework that facilitates participation from communication resource-deficient devices. Their solution includes joint resource and power allocation strategies, validated through analytical modeling and simulation.

Manju et al. [86] proposed a Hierarchical Federated Learning (HFL) framework that uses fog nodes to mitigate communication and latency challenges in smart city networks. The system incorporates multilevel model aggregation, reducing communication overhead by up to 50% while maintaining accuracy and accelerating convergence. Simulations carried out on real-world smart city datasets demonstrated enhanced scalability and reduced energy consumption at the edge, positioning HFL as a practical architecture for efficient and privacy-aware urban intelligence.

Asha et al. [87] introduced a federated learning-based network-slicing framework for 6G autonomous-vehicle communications. In their design, edge nodes collaboratively train local slice models that optimize bandwidth and latency for vehicle-to-everything (V2X) interactions while preserving data privacy. The simulation results on an OMNeT++ testbed showed improvements of nearly 18% in throughput and 22% in latency reduction compared to conventional static slicing approaches, highlighting the potential of FL-enabled orchestration for intelligent vehicular networks.

Communication-efficient FL has advanced from structured updates to adaptive aggregation, reducing bandwidth usage in constrained urban networks. The main trade-off is between compression and accuracy, with stronger reductions often impacting model precision. Most works remain on simulation or controlled wireless testbeds with limited reproducibility. Real-world projects, such as the NSF effort on 5G devices, illustrate the potential, but remain isolated pilots rather than standardized practice.

Reducing communication overhead improves participation and scalability, but efficient training in smart cities also depends on where computations are performed. Edge offloading has thus emerged as a complementary approach to balance central coordination with localized responsiveness.

## Case Summary: Hierarchical FL for Smart City Networks [86]

Scope: Multilevel aggregation via fog nodes to mitigate communication and latency bottlenecks.

Infrastructure: Hierarchical FL with edge-fog-cloud coordination.

Devices: 200 simulated IoT nodes across edge clusters.

Metrics: Communication cost ( -50%), accuracy, energy consumption.

Artifacts: Real smart-city dataset; reproducibility materials unavailable.

## 5.4. Computational Offload to Edge

Edge computing in FL enables localized processing, preserves data locality, and improves responsiveness in latency-sensitive smart city applications.

Ji et al. [88] developed a computation-offloading framework for edge-assisted federated learning, where devices collaborate with nearby edge servers to minimize latency and energy consumption. By jointly optimizing local training and task-offloading decisions, their system reduced end-to-end delay by up to 40% and device energy usage by about 35%, while increasing the task-completion rate by 20%. The architecture preserved data locality and achieved a 25% faster model-convergence time in simulated multi-edge environments, though real-world reproducibility details were not provided.

Fu and Di [89] extended edge-based computation offloading to urban traffic systems through a federated reinforcement learning framework. Each intersection acted as an edge node, performing local training, and periodically transmitting model parameters to a central coordinator. This distributed setup reduced communication load and achieved a 15% decrease in average vehicle delay, highlighting the efficiency of offloading learning tasks to edge intersections in latency-sensitive environments.

Offloading computation to edge servers enhances responsiveness in latency-sensitive domains such as traffic management and 5G slice allocation. These strategies reduce load variance and improve system stability, but often rely on simulated settings without reproducibility artifacts. A recurring challenge is trust and coordination between local