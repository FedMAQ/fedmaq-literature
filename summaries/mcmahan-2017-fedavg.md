Below is a structured, professional Markdown summary of McMahan et al. (2017), focusing on methodology, formulas, and relevance to communication efficiency—specifically for the FedMAQ context.

---

# Communication-Efficient Learning of Deep Networks from Decentralized Data (McMahan et al., 2017)

## 1. Overview & Objectives

- **Core Problem**: Training deep neural networks on data that is **privacy-sensitive**, **large in volume**, and naturally **distributed across mobile devices** (e.g., language models, image classifiers). Centralizing such data is infeasible due to privacy risks and high communication costs.
- **Main Objectives**:
  - Introduce **Federated Learning** as a paradigm that keeps training data on devices and learns a shared model by aggregating locally computed updates.
  - Propose the **FedAvg** (FederatedAveraging) algorithm that reduces the number of required communication rounds by **10–100×** compared to synchronous SGD.
  - Demonstrate robustness to **non-IID** and **unbalanced** data distributions – inherent characteristics of real-world federated settings.

## 2. Methodology & Key Innovations

- **Key Idea**: Instead of communicating raw gradients every round, each client performs **multiple local SGD updates** (over one or more local epochs) before sending only its updated model back to the server. The server then averages these models, weighted by the number of local data points.
- **Algorithm (FedAvg)**:
  - **Server** initializes global model \( w_0 \).
  - For each round \( t \):
    1. Select a random fraction \( C \) of clients (at least 1).
    2. Broadcast current global model \( w_t \) to selected clients.
    3. Each selected client runs **ClientUpdate(\( k, w_t \))**:
       - Split local data \( \mathcal{P}\_k \) into batches of size \( B \).
       - Perform \( E \) local epochs of SGD: \( w \leftarrow w - \eta \nabla \ell(w; b) \) for each batch \( b \).
       - Return new local model \( w\_{t+1}^k \) to server.
    4. Server aggregates: \( w*{t+1} = \sum*{k=1}^K \frac{n*k}{n} w*{t+1}^k \).
- **Parameters controlling local computation**:
  - \( C \): fraction of clients used per round (parallelism).
  - \( E \): number of local epochs (training passes over local data).
  - \( B \): local minibatch size. Setting \( B=\infty \) (full local batch) and \( E=1 \) recovers **FedSGD** (synchronous gradient averaging).

## 3. Mathematical Formulation

- **Global objective** (finite-sum over all data points):

\[
\min*{w \in \mathbb{R}^d} f(w) \quad \text{where} \quad f(w) = \frac{1}{n} \sum*{i=1}^n f_i(w).
\]

- **Decomposition over clients** (each client \( k \) has dataset \( \mathcal{P}\_k \) of size \( n_k \)):

\[
f(w) = \sum*{k=1}^K \frac{n_k}{n} F_k(w), \quad F_k(w) = \frac{1}{n_k} \sum*{i \in \mathcal{P}\_k} f_i(w).
\]

- **Local update on client \( k \)** (for one minibatch \( b \) of size \( B \)):

\[
w \leftarrow w - \eta \nabla \ell(w; b).
\]

- **Server aggregation** (weighted average of returned models):

\[
w*{t+1} = \sum*{k=1}^K \frac{n*k}{n} w*{t+1}^k.
\]

- **Special case FedSGD** (\( E=1, B=\infty \)): each client computes full-batch gradient \( g_k = \nabla F_k(w_t) \), then:

\[
w*{t+1} = w_t - \eta \sum*{k=1}^K \frac{n_k}{n} g_k = w_t - \eta \nabla f(w_t).
\]

- **Local computation per round per client**: \( u_k = E \cdot \frac{n_k}{B} \) (expected updates; expected across clients: \( u = \frac{n}{K} \cdot \frac{E}{B} \)).

## 4. Limitations & Constraints

- **Statistical assumptions violated in practice**:
  - Data is **non-IID** across clients (local distributions \( F_k \) are poor approximations of global \( f \)).
  - Data is **unbalanced** (client dataset sizes vary greatly).
  - **Massively distributed**: number of clients can be much larger than examples per client.
- **System constraints**:
  - **Communication is the bottleneck** (slow/expensive links, clients often offline).
  - **Synchronous rounds** assumed; stragglers or dropped clients not handled.
- **Algorithmic limitations**:
  - **No formal convergence guarantees** for non-convex objectives (though empirical robustness shown).
  - **Large \( E \) can cause divergence or plateauing** (e.g., Shakespeare LSTM; Fig. 3). May require decaying local computation.
  - **Model averaging may fail** if clients start from different initializations (but works when starting from shared init; Fig. 1).
  - **No differential privacy or secure aggregation** in basic FedAvg (mentioned as future work).

## 5. FedMAQ Thesis Relevance

- **FedAvg is the direct baseline** for FedMAQ. The paper demonstrates that increasing local computation reduces communication rounds, but **does not compress the communicated model updates** themselves.
- **FedMAQ can extend FedAvg by**:
  - **Multi‑adaptive quantization**: reducing the bitwidth of each model update (e.g., gradients or weight deltas) to lower per‑round communication cost.
  - **Knowledge distillation**: sending a compact “student” model or soft labels instead of full model parameters, further saving bandwidth.
  - **Adaptive communication**: dynamically choosing quantization levels or whether to distill based on non‑IID-ness or round progress.
- **Key insights to integrate**:
  - FedAvg shows that additional computation is “free” relative to communication – FedMAQ can trade increased local computation for even lower communication.
  - The non‑IID and unbalanced nature of data (observed in MNIST non‑IID, Shakespeare) informs how aggressive quantization/distillation can be without harming convergence.
  - The empirical methodology (e.g., measuring rounds to target accuracy, tuning \( C, E, B \)) should be adopted for evaluating FedMAQ variants.

---

_This summary is intended for researchers building on FedAvg for communication‑efficient federated learning, especially in the context of FedMAQ._
