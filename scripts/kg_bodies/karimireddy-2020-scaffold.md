description: SCAFFOLD uses server/client control variates (variance reduction) to correct client-drift, proving convergence unaffected by data heterogeneity or client sampling.

## 1. Overview & Objectives

- **Core Problem**: On heterogeneous (non-IID) data, **FedAvg suffers from "client-drift"**: multiple local SGD steps pull each client toward its own local optimum, so the averaged update is biased away from the global descent direction. The paper proves this drift persists even with full-batch gradients and full participation, causing slow and unstable convergence.
- **Main Objectives**:
  - Derive **tight convergence rates** for FedAvg (upper and matching lower bounds) that formally expose client-drift.
  - Propose **SCAFFOLD** (Stochastic Controlled Averaging), which uses **control variates** to cancel drift, converging in far fewer rounds and provably **unaffected by heterogeneity or client sampling**.
  - Show SCAFFOLD can exploit inter-client similarity for even faster rates — the first result quantifying the benefit of local steps in distributed optimization.

## 2. Methodology & Key Innovations

- **Key Idea**: Maintain a **server control variate** \(c\) (estimate of the global update direction) and a **per-client control variate** \(c_i\) (estimate of client \(i\)'s update direction). The difference \((c - c_i)\) estimates the client-drift and is subtracted from each local gradient step, correcting the local trajectory back toward the global direction.
- **Corrected local update** (client \(i\), step \(k\), learning rate \(\eta_l\)):
  - \(y_i \leftarrow y_i - \eta_l\big(g_i(y_i) - c_i + c\big)\), where \(g_i\) is a stochastic gradient.
- **Control-variate update**: after local training, \(c_i\) is refreshed (e.g. via the last gradient or the option-II estimate \(c_i^+ = c_i - c + \tfrac{1}{K\eta_l}(x - y_i)\)); the server aggregates \(\Delta c_i\) into \(c\).
- **Interpretation**: SCAFFOLD is "client-variance reduction" (SAGA/SVRG-style) applied across clients rather than across data points, which is also why it is robust to client sampling.

## 3. Mathematical Formulation

- **Objective**: \(\min_{x\in\mathbb{R}^d} f(x) := \tfrac{1}{N}\sum_{i=1}^N f_i(x)\), \(f_i(x)=\mathbb{E}_{\zeta_i}[f_i(x;\zeta_i)]\), each \(f_i\) \(\beta\)-smooth.
- **Bounded gradient dissimilarity (A1)**: \(\exists\, G\ge0, B\ge1\) with \(\tfrac{1}{N}\sum_i \lVert\nabla f_i(x)\rVert^2 \le G^2 + B^2\lVert\nabla f(x)\rVert^2\); heterogeneity is captured by \(G, B\).
- **Local step with control variates**:

\[
y_{i,k} = y_{i,k-1} - \eta_l\big(g_i(y_{i,k-1}) - c_i + c\big).
\]

- **Key result**: SCAFFOLD's convergence rate is **independent of \(G\)** (the heterogeneity constant), whereas FedAvg's rate degrades with \(G\) — the formal statement that control variates remove drift.

## 4. Limitations & Constraints

- **Doubled uplink/state**: each client must communicate and store a control variate \(c_i\) of the same dimension as the model, increasing per-round communication and requiring **persistent client state** — problematic in cross-device settings where clients participate once.
- **Stale control variates**: with low participation, \(c_i\) can become stale, weakening drift correction.
- **No payload compression**: SCAFFOLD reduces rounds but transmits full-precision models *and* control variates, so raw bits per round increase.
- **Convex-leaning theory**: the strongest rate guarantees assume smoothness/(strong) convexity; non-convex guarantees are weaker.

## 5. FedMAQ Thesis Relevance

- **Canonical drift-correction baseline**: SCAFFOLD is the reference variance-reduction method for non-IID FL, complementary to the regularization approaches of [FedProx](/papers/li-2020-fedprox.md) and [FedDyn](/papers/acar-2021-feddyn.md). It sets the accuracy-vs-rounds bar FedMAQ should approach under heterogeneity.
- **Tension with compression**: SCAFFOLD *adds* a full-model-sized control variate to every message — the opposite of FedMAQ's goal. This makes it a compelling target for FedMAQ's quantization: can control variates be transmitted at low bit-width without destroying drift correction? The interaction of quantization noise with variance reduction is a concrete open question.
- **Key insight to integrate**: SCAFFOLD formalizes that heterogeneity, not just communication, governs convergence. FedMAQ's adaptive quantization schedule should account for heterogeneity level (as SCAFFOLD's rate does), quantizing less aggressively when client dissimilarity \(G\) is high.
