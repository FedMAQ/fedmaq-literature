# Research Summary: Adaptive Federated Learning for Smart Meter Energy Forecasting

## 1. Overview & Objectives

**Core Problem:** Short-term energy consumption forecasting in smart grids faces three key challenges: (i) privacy concerns from aggregating sensitive household consumption data centrally, (ii) scalability issues when thousands of smart meters participate, and (iii) model degradation over time due to concept drift in time-series data.

**Main Objectives:**
- Design a privacy-preserving, decentralized forecasting framework using federated learning (FL) and edge computing.
- Improve forecast accuracy by integrating adaptive learning to detect and respond to data drifts.
- Reduce communication overhead and training time compared to centralized architectures.
- Evaluate performance using real-world smart meter data (London households) with both univariate (SISO) and multivariate (MISO) input-output configurations.

## 2. Methodology & Key Innovations

**Proposed Framework:**
- **Architecture:** Three-tier system consisting of smart meters (data collection), edge computing nodes (local model training and storage), and a central aggregator (model fusion and redistribution) – see Fig. 4 of the paper.
- **Base Models:** Two LSTM variants are compared:
  - **Stacked LSTM** (4 hidden layers, 50 neurons each, dropout 20%, ReLU activation).
  - **Bidirectional LSTM (BiLSTM)** (simple structure, two directions).
- **Federated Learning:** Uses the **Federated Averaging (FedAVG)** algorithm [McMahan et al., 2017] where each client performs multiple local SGD epochs before sending weight updates to the server. Server aggregates weighted by number of local samples.
- **Adaptive Learning Mechanism:** A drift detection trigger based on RMSE. After each prediction period (e.g., one day), the RMSE of current predictions is compared against a running average of recent RMSE (e.g., 2-week window). If the daily RMSE exceeds the average by a threshold (10%), the model is retrained locally using the most recent historical data.

**Key Innovations:**
- First work to combine **adaptive learning + FL** for energy forecasting.
- Evaluates both SISO (only past energy consumption) and MISO (past consumption + calendar + weather features) architectures.
- Demonstrates that adaptive FL reduces training time by ~80% and RMSE by 8% compared to centralized adaptive learning.

## 3. Mathematical Formulation

### Evaluation Metrics
**Mean Absolute Error (MAE):**
$$
MAE = \frac{\sum_{i=1}^{n} |y_i - \hat{y}_i|}{n}
$$

**Root Mean Squared Error (RMSE):**
$$
RMSE = \sqrt{\frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{n}}
$$
where $y_i$ is the observed value, $\hat{y}_i$ is the forecast value, and $n$ is the number of observations.

### Federated Averaging (FedAVG)
Let $K$ be the number of clients, each with local dataset $\mathcal{D}_k$ of size $n_k$. In round $t$:
1. Server sends global model $w_t$ to a subset of clients.
2. Each client $k$ performs $E$ local epochs of SGD on $\mathcal{D}_k$ starting from $w_t$ to obtain $w_{t+1}^k$.
3. Server aggregates: 
   $$
   w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_{t+1}^k
   $$
   where $n = \sum_k n_k$.

### Adaptive Retraining Condition
Let $\text{RMSE}_{\text{day}}$ be the RMSE on predictions for the current day, and $\overline{\text{RMSE}}_{2\text{wk}}$ be the average RMSE over the past two weeks. Retrain if:
$$
\text{RMSE}_{\text{day}} > \overline{\text{RMSE}}_{2\text{wk}} \times 1.1
$$

If triggered, the local model is retrained on the most recent historical data (e.g., last month's data) before the next prediction round.

### Model Hyperparameters (Table 4)
- Neurons: 50
- Dense units: 1
- Dropout: 20%
- Activation: ReLU
- Loss: MSE
- Optimizer: Adam
- Epochs: 50
- Batch size: 100

## 4. Limitations & Constraints

**Statistical/System Assumptions:**
- Data is time-series with hourly granularity; tested only on one year (2013) with a single-month test period (December).
- Assumes that 10% RMSE increase threshold is a sensible drift indicator, but no formal statistical test for concept drift is used.
- Non-IID data distribution is not explicitly addressed; the FedAVG aggregation assumes client data sizes are known and used as weights.

**Communication Bottleneck Concerns:**
- The study measures training time and data transmitted but does **not provide detailed analysis of communication cost** per round or total bytes exchanged.
- Adaptive retraining can increase communication when drift is frequent; the paper notes that in winter (test period) drift is minimal, so the benefit may be overestimated.
- No model compression, quantization, or knowledge distillation techniques are applied to reduce weight transmission size.

**Other Limitations:**
- Only 50 clients maximum; scalability to hundreds/thousands not tested.
- BiLSTM underperformed Stacked LSTM; no hyperparameter tuning for BiLSTM was reported.
- No comparison with other FL variants (e.g., FedProx, SCAFFOLD) or with communication-efficient methods.

## 5. FedMAQ Thesis Relevance

**Connection to FedMAQ (Communication-Efficient FL via Multi-Adaptive Quantization and Knowledge Distillation):**

| Aspect | This Paper | FedMAQ Potential Integration |
|--------|------------|------------------------------|
| **Communication strategy** | Assumes full-precision weight exchange; no quantization or compression. | Could apply multi-adaptive quantization (e.g., varying bit-width across layers or clients) to reduce uplink message size. |
| **Model adaptation** | Adaptive retraining triggered by RMSE drift. | FedMAQ could extend this by adaptively choosing quantization levels based on detected drift severity – more quantization when no drift, less when retraining needed. |
| **Knowledge Distillation** | Not used. | Distillation could be employed to compress the global model into a smaller local student model, further reducing communication and enabling heterogeneous client capabilities. |
| **Baseline role** | This paper can serve as a **strong baseline** for performance (RMSE, training time) in energy forecasting FL with adaptivity. | FedMAQ should compare against this adaptive FL baseline to demonstrate gains in communication efficiency while maintaining accuracy. |
| **Techniques to adopt** | The RMSE-based drift detection and retriggering mechanism can be integrated into FedMAQ's adaptive quantization scheduler to decide when to increase precision (e.g., during retraining rounds). | |

**Conclusion:** The paper provides an effective application of FL with adaptive learning for smart grid forecasting but lacks explicit communication optimization. FedMAQ can leverage its drift detection framework and combine it with multi-adaptive quantization and KD to achieve greater communication savings without sacrificing forecast accuracy.