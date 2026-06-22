# CFD: Communication-Efficient Federated Distillation via Soft-Label Quantization and Delta Coding

## 1. Overview & Objectives

**Core Problem:** The primary challenge addressed is the **communication bottleneck** in Federated Learning (FL), which is exacerbated by the frequent exchange of large neural network models (e.g., millions of parameters) between clients and a central server. This is particularly problematic in bandwidth-constrained environments like mobile and IoT networks.

**Main Objectives:** The paper aims to:
1.  Investigate and analyze the communication properties of **Federated Distillation (FD)** , a paradigm that exchanges model outputs (soft-labels) rather than model parameters.
2.  Develop a novel, highly communication-efficient FD method called **Compressed Federated Distillation (CFD)** .
3.  Demonstrate that CFD can reduce communication by **more than two orders of magnitude** compared to standard FD and **more than four orders of magnitude** compared to Federated Averaging (FedAvg), while maintaining or improving model accuracy.

## 2. Methodology & Key Innovations

The proposed **Compressed Federated Distillation (CFD)** method is a systematic framework that integrates five key techniques to minimize communication in both upstream (client-to-server) and downstream (server-to-client) directions.

**Core System Model:** The CFD framework operates within a standard Federated Distillation setup, where a public, unlabeled dataset ($X^{pub}$) is used for distillation. The core innovation is the compression of the soft-labels ($Y_i$) exchanged during this process.

**Key Innovations:**
1.  **Distill Data Curation:** A fixed, random subset of the public distillation data is selected for the entire training process, avoiding the computational overhead of active learning strategies.
2.  **Upstream Quantization:** The bit-width of client soft-labels is reduced using **constrained uniform quantization**.
3.  **Delta Coding (Lossless Compression):** An entropy coding technique (e.g., CABAC) is applied to the quantized soft-labels. Furthermore, **delta coding** is used to exploit the temporal correlation between consecutive rounds, communicating only the changes in predictions.
4.  **Dual Distillation (Downstream):** To handle scenarios with less than 100% client participation, a **dual distillation** step is performed on the server. The server distills an aggregated model from the previous round's soft-labels, and then sends its own quantized soft-labels to the clients. This prevents model desynchronization.
5.  **Downstream Quantization:** The server's soft-labels (from the dual distillation step) are also quantized before being sent to the clients.

## 3. Mathematical Formulation

### 3.1. Core Communication Model

The total communication per round ($b_{total}$) is defined as:
$$b_{total} = |X^{pub}| \times (H(Y_i) + \eta)$$
where:
- $|X^{pub}|$ is the size of the distillation dataset.
- $H(Y_i)$ is the entropy of the soft-labels.
- $\eta$ is the coding inefficiency.

### 3.2. Constrained Uniform Quantization

The quantization operator $\mathcal{Q}_b$ maps a probability vector $p$ to a quantized vector $q$ with $b$ bits:
$$q = \mathcal{Q}_b(p) = \arg \min_{q_i \in \{\frac{1}{2^b - 1}, \iota \in 0, \dots, 2^b - 1\}} \|q - p\|_1 \quad \text{subject to} \quad \sum_i q_i = 1$$
For $b=1$, this is equivalent to the maximum vote:
$$\mathcal{Q}_1(p)_i = \begin{cases} 1 & \text{if } i = \arg \max(p) \\ 0 & \text{else} \end{cases}$$

### 3.3. Delta Coding

The delta-coded soft-label $\hat{Y}^t$ for round $t$ is defined as:
$$\hat{Y}^t_l = \begin{cases} \tilde{Y}^t_l & \text{if } \tilde{Y}^t_l \neq \tilde{Y}^{t-1}_l \\ 0 & \text{else} \end{cases} \quad \forall l$$
where $\tilde{Y}^t$ is the quantized prediction from round $t$.

### 3.4. Dual Distillation (Server-Side)

The server performs a distillation step on its own model $\theta_S^{t-1}$:
$$\theta_S^t \leftarrow \text{train}(\theta_S^{t-1}, X^{pub}, Y^{pub})$$
where $Y^{pub}$ is the aggregated soft-labels from the clients. The server then computes new soft-labels:
$$Y_S^{pub} = \{f_{\theta_S}(x) | x \in X^{pub}\}$$
These are sent to the clients, who then distill from them:
$$\theta \leftarrow \text{train}(\theta_0, X^{pub}, Y_S^{pub})$$

### 3.5. Algorithm (Compressed Federated Distillation)

The full algorithm is described in **Algorithm 1** of the paper, which outlines the steps for each communication round:
1.  **Client Selection:** Select a subset of clients.
2.  **Downstream:** Send quantized server soft-labels to clients.
3.  **Client Update:** Clients distill from server soft-labels, then train on local data.
4.  **Upstream:** Clients compute and quantize their soft-labels, apply delta coding, and send to server.
5.  **Server Aggregation:** Server aggregates and performs dual distillation.

## 4. Limitations & Constraints

- **Statistical Assumptions:** The method assumes the availability of a **public, unlabeled dataset** ($X^{pub}$) that is roughly similar in distribution to the private client data. This is a strong assumption that may not hold in all FL scenarios.
- **System Constraints:** The **dual distillation** step adds computational overhead on the server, which is assumed to have strong resources. This may be a limitation in resource-constrained server environments.
- **Communication Bottleneck:** While CFD drastically reduces communication, it does not eliminate it