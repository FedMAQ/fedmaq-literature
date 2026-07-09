---
type: Paper
title: "Non-IID data in Federated Learning: A Survey with Taxonomy, Metrics, Methods, Frameworks and Future Directions"
description: "This paper is a comprehensive survey on the challenges of Non-IID (non-independent and identically distributed) data in Federated Learning (FL)."
authors: "Jimenez et al."
year: 2024
bibkey: jimenez-2024-non-iid-survey
tags: [survey, heterogeneity]
resource: markdown/jimenez-2024-non-iid-survey/paper.md
timestamp: 2026-06-21T08:28:38Z
---

### 1. Overview & Objectives

This paper is a comprehensive survey on the challenges of **Non-IID (non-independent and identically distributed) data** in Federated Learning (FL). The core problem is that when data across different clients is heterogeneous, standard FL algorithms like FedAvg suffer from **slower convergence, reduced model accuracy, and increased communication overhead**.

The main objectives of the survey are to:

- Provide a detailed **taxonomy** of non-IID data types.
- Catalog and analyze **partition protocols** used to simulate non-IID scenarios.
- Review **metrics** for quantifying data heterogeneity.
- Summarize state-of-the-art **solutions** to address non-IID data.
- Evaluate **standardized FL frameworks** for their support of heterogeneous data.

### 2. Methodology & Key Innovations

As a survey, the paper's methodology is a systematic literature review following a PRISMA-like process, analyzing 235 papers. Its key innovations are not a single algorithmic contribution but a structured synthesis of the field. The main contributions are:

- **Comprehensive Taxonomy:** It introduces a detailed taxonomy of data heterogeneity, including **Label Skew**, **Attribute Skew**, **Quantity Skew**, **Spatiotemporal Heterogeneity**, and **Participation Skew**. It also introduces **Modality Skew** as a novel category.
- **Systematic Categorization:** It provides a structured overview of partition protocols (e.g., Dirichlet, Sharding) and non-IID metrics (e.g., Earth Mover's Distance, Heterogeneity Index).
- **Framework Analysis:** It compares popular FL frameworks (e.g., FEDML, Flower, TFF) based on their support for non-IID data handling.

### 3. Mathematical Formulation

The paper does not propose a new algorithm but formalizes the FL problem and the concept of data skew.

- **Global Objective Function:** The core FL objective is defined as minimizing a weighted average of local loss functions:

  $$
  \min_{w} l(w) := h(L_k(w))
  $$

  where $w$ are the global model parameters, $L_k(w)$ is the local objective for client $k$, and $h$ is the aggregation function.

- **Federated Averaging (FedAvg):** The standard aggregation function is defined as:

  $$
  h(L_k(w)) = \sum_{k=1}^{K} \frac{n_k}{n} L_k(w)
  $$

  where $n_k$ is the size of client $k$'s dataset and $n$ is the total number of samples.

- **Formal Definition of Data Skew:** The paper defines data skew as a difference in the underlying probability density functions (pdfs) of client data:

  $$
  f^{(i)} \neq f^{(j)} \text{ for some } i, j \in \{1, \dots, K\}
  $$

  This is further broken down into **Label Skew** (differences in $f_Y$ or $f_{Y|X}$) and **Attribute Skew** (differences in $f_X$ or $f_{X|Y}$).

- **Non-IID Metrics:** The paper catalogs several metrics, including:
  - **Heterogeneity Index (HI):** A class-based metric for label skew:

    $$
    HI = 1 - \frac{1}{(C_{\max} - 1)} \cdot (c - 1)
    $$

    where $c$ is the max number of classes per client and $C_{\max}$ is the total number of classes.

  - **Imbalance Ratio (IR):** A class-based metric for label skew (measuring label distribution imbalance within the classes present on the clients):
    $$
    IR(\xi) = \frac{\max_i \xi_i}{\min_j \xi_j}
    $$
    where $\xi_i$ is the frequency of class $i$. It quantifies the imbalance between majority and minority classes rather than differences in total data volume across clients.

### 4. Limitations & Constraints

The survey identifies several key limitations and constraints in the current state of non-IID FL research:

- **Lack of Consensus:** There is no agreed-upon standard for classifying or quantifying non-IID data, making it difficult to compare different studies.
- **Single-Skew Focus:** Most research focuses on a single type of skew (e.g., label skew) while ignoring the interdependence of different skew types (e.g., label skew often introduces quantity skew).
- **Low Framework Adoption:** Only 14.2% of reviewed papers use standardized FL frameworks, hindering reproducibility.
- **Understudied Areas:** **Modality skew** and **Spatiotemporal heterogeneity** are significantly under-researched.
- **Communication Bottleneck:** The paper explicitly lists **communication inefficiency** as a downstream effect of non-IID data, noting that more communication rounds are required to reach convergence, increasing network overhead.

### 5. FedMAQ Thesis Relevance

This survey is highly relevant to the FedMAQ thesis (Communication-Efficient FL via Multi-Adaptive Quantization and Knowledge Distillation) in several ways:

- **Problem Context:** The survey provides a strong theoretical and practical foundation for the core problem FedMAQ aims to solve: **communication inefficiency caused by non-IID data**. It confirms that non-IID data is a primary driver of increased communication rounds.
- **Baseline Identification:** The paper can serve as a **critical resource for identifying baselines**. It reviews popular non-IID solutions like **FedProx**, **SCAFFOLD**, and **FedNova**. These are standard baselines against which FedMAQ's performance should be compared.
- **Technique Integration:**
  - **Knowledge Distillation (KD):** The survey explicitly identifies **Knowledge Distillation** as a key solution for non-IID data. This directly validates the KD component of FedMAQ. The paper notes that KD can reduce communication costs by exchanging soft labels (logits) instead of full model parameters.
  - **Quantization & Compression:** The paper's "Future Directions" section explicitly calls for "designing advanced compression techniques tailored to model updates in non-IID environments." This directly aligns with the **quantization** component of FedMAQ. The survey provides the motivation for why adaptive quantization is needed: because the degree of heterogeneity varies, a one-size-fits-all compression strategy is suboptimal.
- **Gap Analysis:** The survey highlights a gap in the literature: a lack of solutions that simultaneously address multiple types of skew. FedMAQ, by combining adaptive quantization (to handle varying update importance) and KD (to handle label/feature distribution differences), can be positioned as a holistic approach that tackles both communication efficiency and data heterogeneity.

**In summary, this survey is not a baseline algorithm itself, but it is an essential reference for the FedMAQ thesis. It provides the problem motivation, identifies relevant baselines (FedProx, SCAFFOLD), validates the use of KD, and points to the need for adaptive compression techniques, which is the core innovation of FedMAQ.**

# Related

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](/papers/mcmahan-2017-fedavg.md)
- [Federated Optimization in Heterogeneous Networks](/papers/li-2020-fedprox.md)
- [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](/papers/karimireddy-2020-scaffold.md)
- [Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](/papers/wang-2020-fednova.md)

# Citations

[1] Full-text conversion: [markdown/jimenez-2024-non-iid-survey/paper.md](markdown/jimenez-2024-non-iid-survey/paper.md)
[2] Source PDF: `papers/00 Surveys/Jimenez et al. - 2024 - Non-IID data in Federated Learning A Survey with Taxonomy, Metrics, Methods, Frameworks and Future.pdf`
