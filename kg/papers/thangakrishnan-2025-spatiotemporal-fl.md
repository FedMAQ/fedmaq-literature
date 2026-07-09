---
type: Paper
title: "Spatiotemporal Federated Learning for Privacy-Preserving Load Forecasting and Appliance Scheduling in Smart City Homes"
description: "Smart city home energy management faces three interconnected challenges: (1) privacy concerns from centralized collection of sensitive household energy data, (2) uncertainty in load forecasting and renewable..."
authors: "Suresh Thangakrishnan et al."
year: 2025
bibkey: thangakrishnan-2025-spatiotemporal-fl
tags: [application]
resource: markdown/thangakrishnan-2025-spatiotemporal-fl/paper.md
timestamp: 2026-06-21T09:44:27Z
---

## 1. Overview & Objectives

**Core Problem:** Smart city home energy management faces three interconnected challenges: (1) privacy concerns from centralized collection of sensitive household energy data, (2) uncertainty in load forecasting and renewable generation that degrades scheduling reliability, and (3) the need for multi-objective optimization balancing cost, user comfort, and grid stability.

**Main Objectives:**

- Develop a privacy-preserving federated learning framework for short-term load forecasting using smart home energy data
- Design a robust appliance scheduling mechanism that accounts for forecast uncertainty
- Achieve simultaneous reduction in energy costs, user discomfort, and peak-to-average ratio (PAR)
- Maintain high demand response compliance (>92%) while preserving data privacy

## 2. Methodology & Key Innovations

### System Architecture: Edge-Fog-Cloud Hierarchy

The framework employs a three-tier architecture:

1. **Edge Layer (Smart Homes):** Each home $i \in \{1,2,\ldots,N\}$ contains $M_i$ smart appliances, photovoltaic (PV) units, and battery energy storage (BESS). Raw data remains local; only intermediate activations are transmitted upstream.

2. **Fog Layer:** Intermediate processing nodes that complete GRU model inference and training, and execute the hybrid metaheuristic scheduler for appliance optimization.

3. **Cloud Layer:** Global model aggregation using Federated Averaging (FedAvg).

### Key Innovation 1: Split Federated Learning (Split-FL) with GRU

The GRU network is split between Edge and Fog layers:

- **Edge:** Executes initial GRU layers (partial forward pass), computes intermediate activations $f_t^{(i)}$
- **Fog:** Completes forward pass, computes loss, and backpropagates gradients to Edge
- **Privacy Guarantee:** Raw consumption data never leaves the Edge device

### Key Innovation 2: Hybrid IGWO-TLBO Optimization

Combines Improved Grey Wolf Optimizer (IGWO) for global search with Teaching-Learning-Based Optimization (TLBO) for local refinement, applied to multi-objective appliance scheduling under uncertainty.

### Key Innovation 3: Ellipsoidal Uncertainty Modeling

Forecast errors are modeled using ellipsoidal uncertainty sets to ensure schedule robustness against prediction inaccuracies.

### Key Innovation 4: Hypergraph-Based Conflict Resolution

Appliance scheduling conflicts are resolved via hypergraph coloring, where nodes represent appliance requests and hyper-edges represent shared resource competition.

## 3. Mathematical Formulation

### 3.1 Net Load Model

For each smart home $i$ at time $t$:

$$
L_i(t) = \sum_{m=1}^{M_i} s_{i,m}(t) P_{i,m}^{\text{rated}} + P_{\text{grid},i}(t) - P_{\text{PV},i}(t) - P_{\text{bat-dis},i}(t) + P_{\text{bat-chg},i}(t)
$$

where $s_{i,m}(t) \in \{0,1\}$ indicates appliance on/off status.

### 3.2 Feature Vector

At each time step $t$, smart home $i$ collects:

$$
\mathbf{x}_t = \left[ P_{i,m}(t),\; P_{\text{PV},i}(t),\; E_{\text{bat},i}(t),\; \lambda(t) \right]
$$

where $\lambda(t)$ is the dynamic electricity pricing signal.

### 3.3 GRU Gating Mechanisms (Edge-Side Computation)

The partial forward pass computes:

**Update Gate:**

$$
z_t = \sigma(W_z x_t + U_z h_{t-1})
$$

**Reset Gate:**

$$
r_t = \sigma(W_r x_t + U_r h_{t-1})
$$

**Candidate Hidden State:**

$$
\tilde{h}_t = \tanh(W x_t + U(r_t \odot h_{t-1}))
$$

**Final Hidden State (Intermediate Activation):**

$$
f_t = h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

where $\sigma(\cdot)$ is the sigmoid function, $\tanh(\cdot)$ is the hyperbolic tangent, and $\odot$ denotes element-wise multiplication.

### 3.4 Split-FL Training: Backpropagation Across Layers

**Step 1 - Fog-Side Gradient Computation:**

$$
\nabla_{\theta^{\text{Fog}}} = \frac{\partial \mathcal{L}^{(i)}}{\partial \theta_i^{\text{Fog}}}
$$

**Step 2 - Activation Gradient:**

$$
\nabla_{f_t} = \frac{\partial \mathcal{L}^{(i)}}{\partial f_t^{(i)}}
$$

**Step 3 - Edge-Side Gradient (via Chain Rule):**

$$
\frac{\partial \mathcal{L}^{(i)}}{\partial \theta_i^{\text{Edge}}} = \nabla_{f_t} \cdot \frac{\partial f_t^{(i)}}{\partial \theta_i^{\text{Edge}}}
$$

**Step 4 - Local Parameter Updates (learning rate $\eta$):**

At Edge:

$$
\theta_i^{\text{Edge}} \leftarrow \theta_i^{\text{Edge}} - \eta \cdot \frac{\partial f_t^{(i)}}{\partial \theta_i^{\text{Edge}}}
$$

At Fog:

$$
\theta_i^{\text{Fog}} \leftarrow \theta_i^{\text{Fog}} - \frac{\partial \mathcal{L}^{(i)}}{\partial \theta_i^{\text{Fog}}}
$$

**Step 5 - Local Model Formation:**

$$
\theta_i = \theta_i^{\text{Edge}} \cup \theta_i^{\text{Fog}}
$$

### 3.5 Federated Aggregation (FedAvg)

Global model update:

$$
\theta_{\text{global}} = \sum_{i=1}^{N} \frac{n_i}{n} \theta_i, \quad \text{where } n = \sum_{i=1}^{N} n_i
$$

### 3.6 Multi-Objective Scheduling Objective

Minimize total cost $J_i$ for each home $i$:

$$
\min_{s_i, m_i} J_i = \omega_1 \sum_{t=1}^{T} \lambda_t P_{\text{grid},it} + \omega_2 \sum_{m \in A_i} P_{\text{penalty}}^{A_{i,m}} - \omega_3 \max_{i \in T} L_{it}
$$

where:

- $\omega_1, \omega_2, \omega_3$ are weights summing to 1.
- $\lambda_t$ is the real-time electricity price.
- $P_{\text{penalty}}^{A_{i,m}}$ quantifies user discomfort from scheduling delays.
- $\max_{i \in T} L_{it}$ represents the peak load (minimizing peak improves grid stability). Note that the subscript $i \in T$ is reproduced exactly as written in Equation (12) of the original paper; this is a notation typo in the source, as it should logically be $t \in T$ to represent the maximum over all time slots $t$ in the scheduling horizon $T$.

### 3.7 Ellipsoidal Uncertainty Set

True load $L_i(t)$ is modeled within an ellipsoidal uncertainty set:

$$
\mathcal{W}\left(\hat{L}_i(t), \Delta_i\right) = \left\{ L : \left(L - \hat{L}_i(t)\right)^\top \Delta_i^{-1} \left(L - \hat{L}_i(t)\right) \leq 1 \right\}
$$

where $\hat{L}_i(t)$ is the predicted load and $\Delta_i$ is a positive definite covariance matrix.

## 4. Limitations & Constraints

### Statistical Assumptions

- **Data Heterogeneity:** Households have varying numbers and types of appliances, leading to non-IID data distributions across clients
- **Forecast Uncertainty:** Load predictions are subject to model errors and volatility (e.g., solar PV variation), requiring robust optimization
- **Temporal Dependencies:** Energy consumption patterns exhibit strong temporal correlations that GRU must capture

### System Constraints

- **Communication Bottleneck:** While Split-FL reduces raw data transmission, intermediate activations and gradients still require bandwidth; the paper mentions "compression of model parameters, selective gradient sharing, and asynchronous updates" as mitigations
- **Computational Limitations:** Edge devices have limited processing power, necessitating partial GRU computation (only initial layers)
- **Latency Requirements:** Real-time scheduling demands low-latency communication between Edge and Fog layers
- **Scalability:** The hierarchical aggregation strategy aims to reduce network congestion, but large-scale deployments (thousands of homes) may still face bandwidth constraints

### Privacy Considerations

- Split-FL provides privacy-by-design, but gradient leakage attacks remain a potential vulnerability
- The framework assumes honest-but-curious adversaries; malicious attacks are not addressed

## 5. FedMAQ Thesis Relevance

### Baseline Comparison

This paper **cannot serve as a direct baseline** for FedMAQ because:

- It does not employ multi-adaptive quantization or knowledge distillation techniques
- Communication efficiency is mentioned qualitatively but not quantified (no compression ratios, bit-widths, or communication cost metrics)
- The focus is on energy management rather than fundamental FL communication optimization

### Techniques for Integration

The following components from this paper could be **integrated into FedMAQ**:

| Technique                            | Integration Potential                                                                                                  |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| **Split-FL Architecture**            | FedMAQ could adopt the Edge-Fog split to reduce client-side computation while applying quantization at the split layer |
| **GRU for Temporal Forecasting**     | FedMAQ's KD framework could use GRU-based teacher-student models for time-series tasks                                 |
| **Ellipsoidal Uncertainty Modeling** | Could inform adaptive quantization levels based on prediction confidence                                               |
| **Hypergraph Conflict Resolution**   | Applicable to FedMAQ's multi-client coordination in resource-constrained environments                                  |

### Key Gaps for FedMAQ

- **No quantization scheme** is proposed or evaluated
- **No knowledge distillation** mechanism is employed
- **Communication overhead** is not measured or optimized
- **Gradient compression** is mentioned but not formalized

### Recommended Mapping

This paper serves as a **domain-specific application benchmark** for FedMAQ. FedMAQ could:

1. Apply its multi-adaptive quantization to the Split-FL gradient transmissions in this framework
2. Use KD to compress the GRU model before distribution to Edge devices
3. Evaluate communication savings against the baseline performance reported here (32% cost reduction, 51% discomfort reduction, 92% DRC)

# Related

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](/papers/mcmahan-2017-fedavg.md)

# Citations

[1] Full-text conversion: [markdown/thangakrishnan-2025-spatiotemporal-fl/paper.md](markdown/thangakrishnan-2025-spatiotemporal-fl/paper.md)
[2] Source PDF: `papers/05 Applications/Suresh Thangakrishnan et al. - 2025 - Spatiotemporal Federated Learning for Privacy-Preserving Load Forecasting and Appliance Scheduling i.pdf`
