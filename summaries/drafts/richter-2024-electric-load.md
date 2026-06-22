# Research Summary: Advancing Electric Load Forecasting via Federated Learning

## 1. Overview & Objectives

**Core Problem:**  
Renewable Energy Communities (RECs) with *dynamic portfolios* (daily member composition changes) produce electrical load time series that are simultaneously:

- **Non-stationary** (statistical properties change over time)
- **Discontinuous** (bounded observations with abrupt state changes)
- **NON-IID** across different communities (heterogeneous consumption patterns)

Traditional forecasting models fail to generalize across RECs, and privacy concerns prevent centralized data collection.

**Main Objectives:**

1. Develop a federated learning (FL) framework that can learn *domain-invariant features* across distributed RECs.
2. Handle both non-stationarity and discontinuity by incorporating the REC's time-dependent resident composition vector \(\vec{r}_t\) into the model.
3. Achieve transferability to new, unseen RECs (out-of-sample generalization).
4. Demonstrate that FL can match the performance of a centrally trained model while preserving privacy.

---

## 2. Methodology & Key Innovations

### System Model Overview

The proposed framework consists of four key components:

1. **Data Synthesis & Preprocessing**  
   - Generate synthetic REC time series (RECTS) by aggregating district-level electricity consumption time series (DECTS) from 10 ACORN subgroups.
   - Each REC has a dynamic portfolio defined by a resident composition vector \(\vec{r}_t = [N_{1,t}, \dots, N_{10,t}]\) (Equation 1), where \(N_{i,t}\) is the number of households from subgroup \(i\) at time \(t\).
   - ARIX-based feature engineering: past (I), present (AR), and future (F) input arrays are constructed using calendar, weather, and composition variables.

2. **Forecast Model Architecture** (FNN with three input branches)  
   - Three separate *Dense* layers process AR, I, and F inputs independently.
   - Latent features are combined (concatenation or multiplication) and passed to an output layer.
   - 30 neurons per input layer, linear activation functions.

3. **Federated Learning with Data Sharing**  
   - Uses **FederatedAveraging** (FedAvg) for weight aggregation.
   - **Key innovation**: Shares a small number of full RECTS (0, 1, or 2) with every client to help extract *relational behavior* between different composition vectors and load shapes.
   - Local training: 1 epoch per FL round, stochastic gradient descent (SGD).

4. **Experiments to Combat Weight Divergence**  
   - Tests different batch sizes (16, 64), learning rates (0.001, 0.0001), and number of shared time series (0, 2).
   - Best configuration (M6): BS=16, LR=0.001, STS=2.

---

## 3. Mathematical Formulation

### 3.1 ARIX Process Equations for Input Construction

Three input arrays are built from the variables in Table 4:

**AutoRegressive (AR) component – present observations:**

\[
X_{AR}(t) = \sum_{j=1}^{n} \sum_{i=1}^{p} \bigl( \alpha_{j,i} \cdot x_j(t-i) \bigr) \quad \text{(Equation 5)}
\]

**Integrated (I) component – past reference observations (with shift \(\tau\) depending on day type):**

\[
X_I(t) = \sum_{j=1}^{n} \bigl( \beta_j \cdot x_j(t-\tau) \bigr) \quad \text{(Equation 6)}
\]

**Future (F) component – future exogenous variables:**

\[
X_F(t) = \sum_{j=1}^{n} \bigl( \gamma_j \cdot x_j(t) \bigr) \quad \text{(Equation 7)}
\]

**Complete ARIX process (without differencing filter):**

\[
y(t) = X_{AR} + X_I + X_F \quad \text{(Equation 8)}
\]

where:
- \(p = 2\) (number of lags, determined via partial autocorrelation)
- \(n\) = number of variables (target, calendar, weather, \(\vec{r}_t\))
- \(\tau\) = shift: Monday→3 days, Tuesday–Friday→1 day, Saturday/Sunday→7 days, holidays→last Sunday

### 3.2 Data Scaling

**Min-max normalization to \([-1, 1]\):**

\[
x' = a + \frac{(x - \min(x)) \times (b - a)}{\max(x) - \min(x)}, \quad a = -1, \; b = 1 \quad \text{(Equation 3)}
\]

**Inverse transformation:**

\[
x = \min(x) + \frac{x' \times (\max(x) - \min(x)) - a}{b - a} \quad \text{(Equation 4)}
\]

### 3.3 FederatedAveraging (FedAvg)

**Algorithm 2 (simplified):**

\[
\mathbf{w}_{t+1} \leftarrow \sum_{k=1}^{K} \frac{n_k}{n} \mathbf{w}_k^{t+1}
\]

where \(n_k\) is the number of samples at client \(k\), \(n = \sum n_k\), and \(\mathbf{w}_k^{t+1}\) are the locally updated weights after \(e = 1\) epoch of SGD.

### 3.4 Loss Function

The paper states: *"while maintaining a certain loss function Equation (9) with \(\alpha = 0.9\) accounting for bias and strong outliers"*. The explicit formula for Equation (9) is not provided in the extracted text, but the evaluation uses **MAE** and **MAPE**:

\[
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \quad \text{(Equation 10)}
\]

\[
\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \quad \text{(Equation 11)}
\]

### 3.5 Cyclical Encoding for Calendar Features

For a cyclical variable \(h \in [1, H]\) (e.g., hour of day, day of week):

\[
\sin\left( \frac{2\pi h}{H} \right), \quad \cos\left( \frac{2\pi h}{H} \right) \quad \text{(Definition 13)}
\]

### 3.6 Residents Composition Vector

\[
\vec{r}_t = [N_{1,t}, \dots, N_{10,t}]
\]

where \(N_{i,t}\) = number of households from ACORN subgroup \(i\) at day \(t\).

### 3.7 Synthetic Data Generation (Algorithm 1)

Key equations for generating non-stationary and discontinuous RECTS:
- Linear progression of \(\vec{r}_t\) from start to end vector: \(\vec{r}_{t=i} = \text{int}(\vec{r}_{t=i-1} \times \text{grad})\)
- Stochastic perturbation: add or remove one household from a random subgroup with probability based on Gaussian noise \(z \sim \mathcal{N}(0,1)\)
- Probability \(p \in \{0.1, 0.2, \dots, 0.8\}\) that \(N_{i,t}\) is zero

---

## 4. Limitations & Constraints

### Statistical / Data Assumptions

1. **Synthetic data only**: RECTS are generated from ACORN-based DECTS; real-world validation is missing.
2. **Availability of \(\vec{r}_t\)**: Assumes each REC knows the exact composition vector daily—impractical without sophisticated classification.
3. **Stationarity of DECTS**: Base district time series are stationary; non-stationarity arises only from dynamic aggregation. Real data may have additional intrinsic non-stationarity.
4. **Fixed number of DECTS**: Only 300 per ACORN subgroup, limiting diversity.

### System / Communication Constraints

1. **No explicit compression**: The paper does *not* apply quantization, sparsification, or knowledge distillation. All model weights are transmitted with full precision.
2. **Synchronous aggregation**: Uses FedAvg without handling stragglers or heterogeneous hardware.
3. **Data sharing overhead**: Sharing full time series (STS=2) with all clients violates strict privacy and increases communication load.
4. **Fixed architecture**: Only FNN with 3×30 neurons tested; no hyperparameter tuning for activation functions, number of layers, or optimizer.

### Remaining Challenges

- Residual autocorrelation persists (lag 1 and 48 significant in Figure 13).
- Bartlett's test shows non-uniform frequency distribution → some information remains unexploited.
- Model convergence sensitive to batch size, learning rate, and STS; no adaptive strategies.
- No defense against data poisoning or anomaly attacks in FL.

---

## 5. FedMAQ Thesis Relevance

### Comparison to FedMAQ (Communication-Efficient FL via Multi-Adaptive Quantization and Knowledge Distillation)

| Aspect | This Paper | FedMAQ (Target) | Relationship |
|--------|------------|------------------|--------------|
| **Communication efficiency** | None (full-precision weight transmission) | Core focus (adaptive quantization, gradient compression) | Can serve as a **baseline** – performance without any compression |
| **Knowledge distillation** | Not used | Integral part (client-side distillation, ensemble learning) | Could be integrated: distilled lightweight models could replace full model transmission |
| **Quantization** | Not used | Multi-adaptive quantization (dynamic bit-width) | Potential integration: apply adaptive quantization to client model updates |
| **Weight divergence mitigation** | Data sharing + small batch/high LR | Regularization (FedProx/FedDyn) + quantization noise | Data sharing is complementary; quantization noise may also reduce overfitting |
| **Non-stationarity handling** | Separate input branches (AR, I, F) | Not explicit – general FL | FedMAQ can adopt this architecture for time series tasks |
| **Domain adaptation** | \(\vec{r}_t\) as auxiliary input + multi-client aggregation | Not explicit – could use distillation to transfer domain knowledge | FedMAQ's KD can enhance cross-client feature extraction |

### Recommendations for FedMAQ Integration

1. **Replace FedAvg with FedMAQ's adaptive aggregation** – combine weight averaging with quantization-aware updates to reduce communication.
2. **Use knowledge distillation** – compress each client's model into a smaller student model before transmission, or use ensemble distillation to share soft targets.
3. **Adopt the three-branch input architecture** – for time series forecasting, the AR/I/F separation is effective for non-stationarity.
4. **Quantize the shared time series** – if data sharing is kept, apply differential privacy and quantization to minimize leakage.
5. **Employ adaptive batch size/learning rate** – extend FedMAQ's multi-adaptive principle to hyperparameters based on data heterogeneity.

**Conclusion:** This paper is a **strong baseline** for federated time series forecasting under dynamic, non-stationary conditions, but lacks communication-efficient mechanisms. Its key innovations (composition-aware inputs, data sharing, ARIX feature engineering) can be **readily integrated into FedMAQ** to handle distributed non-stationary time series while significantly reducing communication overhead.