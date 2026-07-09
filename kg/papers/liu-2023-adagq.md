---
type: Paper
title: "Communication-Efficient Federated Learning for Heterogeneous Edge Devices Based on Adaptive Gradient Quantization"
description: "Federated Learning (FL) suffers from massive communication overhead due to the frequent upload of model updates (gradients) from clients to a central server."
authors: "Liu et al."
year: 2023
bibkey: liu-2023-adagq
tags: [quantization, adaptive]
resource: markdown/liu-2023-adagq/paper.md
timestamp: 2026-06-21T05:39:14Z
---

## 1. Overview & Objectives

**Core Problem:**  
Federated Learning (FL) suffers from massive communication overhead due to the frequent upload of model updates (gradients) from clients to a central server. Existing gradient quantization methods use a fixed, uniform number of quantization levels for all clients throughout training, which cannot adapt to changing gradient norms or heterogeneous client capabilities, leading to suboptimal wall‑clock training time.

**Main Objectives:**  
- Minimize the total wall‑clock training time of FL under heterogeneous edge devices.
- Achieve this via two novel mechanisms:
  - **Adaptive quantization** – dynamically adjust the quantization resolution per round based on the gradient norm.
  - **Heterogeneous quantization** – assign different quantization resolutions to clients based on their computation and communication resources, reducing straggler delays.

## 2. Methodology & Key Innovations

**Proposed Algorithm: AdaGQ**  
AdaGQ operates in a synchronous FL setting with a central server and $n$ clients. Each round consists of:

1. Server broadcasts aggregated gradients from the previous round.
2. Clients compute local losses under two different quantizations (to estimate loss decrease rates) and report them along with timing information.
3. Server derives the *average* quantization level $\bar{s}_k$ via an online learning rule that considers loss decrease rate and gradient norm changes.
4. Server computes *per‑client* quantization levels $s_{i,k+1}$ so that expected per‑round times of all clients are approximately equal.
5. Clients quantize their local gradients with the assigned $s_{i,k+1}$ and upload them.

**Key Innovations:**
- **Online adaptation of average quantization:** Uses the sign of the derivative of a loss‑decrease‑rate function to update $\bar{s}_k$, then calibrates with the observed change in gradient norm.
- **Heterogeneous assignment:** Explicitly models the relationship between quantization bits and per‑client communication time to align the slowest client with the others, mitigating the straggler effect.
- **Lightweight estimation:** Only requires clients to send two loss values and timing data (local computation time, download/upload times) per round; no additional model training.

## 3. Mathematical Formulation

### 3.1 Global Aggregation with Quantized Gradients
Let $g_k^{(i)}$ be the stochastic gradient of client $i$ at round $k$. After quantization $Q(\cdot)$, the server updates the global model $w$ as:

$$
w_{k+1} = w_k - \eta_k \sum_{i=1}^n p_i \, Q\!\left(g_k^{(i)}\right),
$$

where $p_i$ is the data fraction of client $i$.

### 3.2 Stochastic Uniform Quantization (QSGD)
For a gradient vector $v \in \mathbb{R}^d$ with norm $\|v\|_2$, the $j$-th component $v_j$ is quantized to $s$ levels:

$$
Q_s(v_j) = \|v\|_2 \cdot \text{sign}(v_j) \cdot \zeta_j(v,s),
$$

where $\zeta_j(v,s)$ is a random variable such that $\mathbb{E}[Q_s(v_j)] = v_j$:

$$
\zeta_j(v,s) = 
\begin{cases}
l/s, & \text{with probability } 1 - \left( \frac{|v_j|}{\|v\|_2}s - l \right) \\
(l+1)/s, & \text{otherwise}
\end{cases}
$$

with $l = \lfloor \frac{|v_j|}{\|v\|_2} s \rfloor$.

### 3.3 Adaptive Quantization (Average Level $\bar{s}_k$)

Define the loss decrease rate at round $k$:

$$
R_k = \frac{L_{k-1} - L_k}{T_{k-1,k}},
$$

where $L_k$ is the average loss after round $k$, and $T_{k-1,k}$ is the round duration (worst‑case client time plus server time).  
AdaGQ optimizes $\bar{s}_k$ by online gradient descent on the objective $f(\bar{s}_k) = R_k^* - R_k$, using the sign of the derivative:

$$
\text{sign}\!\big(\nabla f(\bar{s}_k)\big) = \text{sign}\!\left(\frac{R_k' - R_k}{\bar{s}_k - \bar{s}_k'}\right),
$$

where $\bar{s}_k' = \lfloor \bar{s}_k / 2 \rfloor$ (one fewer bit).  
The update rule for $\hat{\bar{s}}_{k+1}$ is:

$$
\hat{\bar{s}}_{k+1} = 
\begin{cases}
\bar{s}_k - \lambda_1, & \text{if } \text{sign}(\nabla f(\bar{s}_k)) = +1 \\
\bar{s}_k + \lambda_2, & \text{if } \text{sign}(\nabla f(\bar{s}_k)) = -1
\end{cases}
$$

with $\lambda_1 = \bar{s}_k/2$ (reduces bits by 1) and $\lambda_2 = \bar{s}_k$ (increases bits by 1).  
Finally, calibrate with gradient norm change:

$$
\bar{s}_{k+1} = \hat{\bar{s}}_{k+1} + \lambda_g \big( \log_2 \|g_k\| - \log_2 \|g_{k-1}\| \big).
$$

### 3.4 Heterogeneous Quantization (Per‑Client Levels $s_{i,k}$)

The expected local time of client $i$ in round $k+1$ is:

$$
\mathbb{E}[t_{i,k+1}^r] \approx \mathbb{E}[t_{i,k+1}^{cp}] + b_{i,k+1} \cdot \mathbb{E}\!\left[\frac{P}{r_{i,k+1}^{\text{trans}}}\right],
$$

where $b_{i,k} = \lfloor \log_2 s_{i,k} \rfloor + 1$ (quantization bits), $P$ is the number of parameters (same for all clients), and $r_{i,k}^{\text{trans}}$ is the client's transmission rate.

To equalize expected times across clients, for any two clients $i,j$:

$$
b_{j,k+1} = \frac{1}{\mathbb{E}[P/r_{j,k+1}]} \Big( \mathbb{E}[t_{i,k+1}^{cp}] - \mathbb{E}[t_{j,k+1}^{cp}] + b_{i,k+1} \cdot \mathbb{E}[P/r_{i,k+1}] \Big).
$$

Practical estimation uses historical averages and the approximation $\mathbb{E}[P/r_{i,k+1}] \approx t_{i,k}^{cm}/b_{i,k}$.  
Combined with the constraint $\frac{1}{n}\sum_i s_{i,k+1} = \bar{s}_{k+1}$, the server solves for all $b_{i,k+1}$ and obtains $s_{i,k+1} = 2^{b_{i,k+1}}-1$.

### 3.5 Round Time Estimation

The actual round time (under $\bar{s}_k$) is:

$$
T_{k-1,k} = \max_i\{ t_{i,k}^{cp} + t_{i,k}^{cm} + t_{i,k}^{down} \} + t_k^{\text{server}}.
$$

For the alternative quantization $\bar{s}_k'$, the communication time is scaled:

$$
t_{i,k}^{'cm} = \frac{\lfloor \log_2 s_{i,k}' \rfloor + 1}{\lfloor \log_2 s_{i,k} \rfloor + 1} \cdot t_{i,k}^{cm}.
$$

### 3.6 Algorithm Skeleton (Algorithm 1 in Paper)

1. **Initialization:** $w_0$, $s_{i,0}=s_0$, $\forall i$.
2. **For each round $k$:**
   - Server broadcasts $g_k$ and auxiliary $s_{i,k}'$.
   - Clients compute losses $L_{i,k}$ and $L_{i,k}'$ under $s_{i,k}$ and $s_{i,k}'$, and send them with $t_{i,k}^{down}, t_{i,k}^{cp}, t_{i,k}^{cm}$.
   - Server estimates $R_k$, $R_k'$ via Eq. (16), updates $\bar{s}_{k+1}$ via Eq. (9)–(10).
   - Server computes $s_{i,k+1}$ via Eq. (13) and sends to clients.
   - Clients perform local SGD, quantize gradients with $s_{i,k+1}$, and upload.
   - Server aggregates to $g_{k+1}$ and prepares $s_{i,k+1}'$.

## 4. Limitations & Constraints

- **Extra overhead:** Each client must perform two forward passes per round (to obtain losses under current and one‑bit‑lower quantization), increasing local computation slightly.
- **Estimation reliance:** The heterogeneous quantization relies on accurate estimation of future training time and transmission rates; high variance in these channels may degrade performance.
- **Synchronous assumption:** All clients must finish within the same round; despite alignment, the server still waits for all clients (no asynchronous operation).
- **Scalability:** The algorithm requires the server to solve a linear relationship for all $n$ clients each round, which is feasible but may become a bottleneck for very large $n$.
- **Not evaluated on extremely large models** (e.g., Transformer‑based architectures) or extreme non‑IID settings beyond $\sigma_d=0.8$.
- **Fixed quantization range** – the QSGD quantizer itself does not adapt the bucket boundaries beyond the norm scaling; very skewed gradient distributions may still cause errors.

## 5. FedMAQ Thesis Relevance

**Direct Baseline:** AdaGQ is a strong baseline for a thesis on **FedMAQ (Communication‑Efficient Federated Learning via Multi‑Adaptive Quantization and Knowledge Distillation)**. AdaGQ already demonstrates that *adaptive* and *heterogeneous* quantization can significantly reduce training time. FedMAQ can use AdaGQ as a comparison point to highlight any additional gains from incorporating multiple adaptation strategies (e.g., layer‑wise quantization, sparsity) and knowledge distillation.

**Techniques that can be integrated into FedMAQ:**

- **Adaptive resolution rule** based on gradient norm (Eq. 10) – FedMAQ can adopt a similar online learning approach for its quantizers.
- **Heterogeneous assignment formula** (Eq. 13) – can be reused or extended to incorporate better time estimators (e.g., using reinforcement learning or more robust online averaging).
- **Loss‑decrease‑rate driven adaptation** (Eq. 5–9) – provides a principled way to trade off per‑round time vs. convergence speed; FedMAQ could unify this with knowledge distillation objectives.
- **Per‑client auxiliary quantization overhead** – the two‑quantization trick to obtain the derivative sign is light; FedMAQ can adopt it for any parameter that affects communication efficiency.

**Contrast:** While AdaGQ focuses solely on gradient quantization, FedMAQ could combine multiple compression dimensions (quantization, sparsification, low‑rank) and add knowledge distillation to further reduce communication rounds. AdaGQ’s method of aligning client times via quantization bit assignment can serve as a plug‑in module for the “multi‑adaptive” component of FedMAQ.

# Related

- [QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding](/papers/alistarh-2017-qsgd.md)

# Citations

[1] Full-text conversion: [markdown/liu-2023-adagq/paper.md](markdown/liu-2023-adagq/paper.md)
[2] Source PDF: `papers/02 Quantization/Liu et al. - 2023 - Communication-Efficient Federated Learning for Heterogeneous Edge Devices Based on Adaptive Gradient.pdf`
