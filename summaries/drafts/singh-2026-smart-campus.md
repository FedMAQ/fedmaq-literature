# Research Summary: Singh - Edge-Assisted Smart Campus Energy Management

## 1. Overview & Objectives

**Core Problem:** University campuses consume large amounts of electricity with highly variable occupancy patterns. Conventional building management systems rely on fixed schedules and centralized analytics, which fail to adapt to local comfort needs and raise privacy concerns when occupant traces are collected at scale.

**Main Objectives:**
- Develop a privacy-preserving energy management system for multi-building campuses
- Combine short-term load forecasting with context-aware control
- Reduce energy consumption and peak demand while maintaining occupant comfort
- Enable collaborative learning across heterogeneous buildings without sharing raw data

## 2. Methodology & Key Innovations

**System Architecture (Four Layers):**
1. **Sensing Layer:** Collects meter readings, indoor temperature/humidity, occupancy estimates, and timetable events
2. **Edge Analytics Layer:** Runs short-term prediction models locally inside each building
3. **Federated Coordination Layer:** Aggregates model updates at scheduled intervals
4. **Control Layer:** Translates forecasts into actions (adjusting set points, dimming lights, delaying noncritical loads)

**Key Innovations:**
- **Federated Forecasting:** Each building trains a gated recurrent model locally; only parameter updates are shared with the coordinator
- **Context-Aware Control:** Minimizes a cost function including energy cost, peak penalty, and comfort deviation with hard constraints
- **Load Classification:** Separates campus loads into critical, flexible, and deferrable groups for demand response scheduling
- **Privacy Preservation:** Raw occupancy traces never leave local sites; only bounded model updates are transmitted

## 3. Mathematical Formulation

### Federated Aggregation Rule

The global parameter vector is updated as a weighted average of local model parameters:

$$
\theta^{r+1} = \sum_{b=1}^{B} \frac{|D_b|}{|D|} \cdot \theta_b^{r+1}
$$

Where:
- $\theta^{r+1}$: Global model parameters at round $r+1$
- $\theta_b^{r+1}$: Local model parameters from building $b$
- $|D_b|$: Size of local dataset at building $b$
- $|D|$: Total dataset size across all buildings
- $B$: Number of buildings

### Local Forecasting Model

For each building $b$, the edge node maintains a local dataset $D_b$ containing recent context and load observations. A gated recurrent model estimates the next-horizon load $L_{t+k}$ from the feature sequence $X_{t-n:t}$.

### Context-Aware Control Objective

The control layer minimizes a cost function:

$$
\min \left( \text{Energy Cost} + \text{Peak Penalty} + \text{Comfort Deviation} \right)
$$

Subject to hard constraints:
- Safety limits
- Equipment cycling restrictions
- Laboratory ventilation requirements

### Demand Response Scheduling

Actions are ranked by predicted energy benefit and comfort risk, with the least disruptive choices executed first during peak tariff intervals.

## 4. Limitations & Constraints

**Statistical Assumptions:**
- Buildings share broad academic and seasonal trends despite different operational rhythms
- Local datasets are sufficiently large for meaningful local training
- Occupancy patterns follow predictable academic schedules

**System Constraints:**
- Sensor calibration affects forecast quality
- Legacy electrical systems may not support fine-grained control
- Model drift during semester breaks requires periodic validation and retraining
- Communication bottleneck is partially addressed (update traffic <6% of raw telemetry volume)

**Privacy Limitations:**
- Only bounded model updates are shared, but no differential privacy mechanism is implemented
- Secure transport and update clipping are applied, but formal privacy guarantees are not provided

## 5. FedMAQ Thesis Relevance

**Connection to FedMAQ (Communication-Efficient FL via Multi-Adaptive Quantization and Knowledge Distillation):**

This paper serves as a **baseline application scenario** for FedMAQ rather than a direct competitor. Key relevance points:

| Aspect | Singh (2026) | FedMAQ Potential |
|--------|--------------|------------------|
| **Communication** | Transmits full model parameters (no compression) | Could benefit from multi-adaptive quantization |
| **Aggregation** | Standard weighted averaging | Could incorporate KD-based distillation |
| **Privacy** | No differential privacy | Could integrate DP mechanisms |
| **Heterogeneity** | Handles building-level heterogeneity | Could improve via adaptive quantization levels |

**Integration Opportunities:**
1. **Quantization:** The 6% communication overhead could be further reduced by applying FedMAQ's adaptive quantization to the model updates transmitted between edge nodes and coordinator
2. **Knowledge Distillation:** The federated forecasting model could benefit from KD to compress the gated recurrent model while maintaining prediction accuracy
3. **Multi-Adaptive Strategy:** Different buildings could use different quantization levels based on their data size ($|D_b|$) and network conditions

**Limitations as Baseline:**
- No explicit communication efficiency techniques are employed
- Model updates are transmitted without compression
- No formal privacy guarantees beyond architectural separation of data

**Recommendation:** This paper provides a realistic application domain and baseline performance metrics (17.6% energy reduction, 21.3% peak demand reduction) against which FedMAQ's communication-efficiency gains can be benchmarked while maintaining comparable forecasting accuracy (8.7% MAPE).