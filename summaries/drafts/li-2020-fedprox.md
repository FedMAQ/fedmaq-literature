# Research Summary: FedProx – Federated Optimization in Heterogeneous Networks

## 1. Overview & Objectives

**Core Problem:** Federated learning (FL) suffers from two key challenges: (i) **systems heterogeneity** – devices have varying computational, communication, and energy capabilities; (ii) **statistical heterogeneity** – data across devices is non‑identically distributed (non‑IID). The de‑facto method, FedAvg, does not explicitly address these issues: it forces all devices to perform the same amount of local work (dropping stragglers) and can diverge under non‑IID data.

**Main Objectives:**
- Propose a generalized framework, **FedProx**, that tolerates variable amounts of local work (partial solutions) and adds a proximal term to stabilize training.
- Provide convergence guarantees for non‑convex objectives under both statistical and systems heterogeneity.
- Demonstrate empirically that FedProx improves robustness and accuracy over FedAvg, especially in highly heterogeneous settings (up to 22% absolute test accuracy improvement).

## 2. Methodology & Key Innovations

**FedProx** modifies FedAvg in two critical ways:

1. **Tolerating Partial Work:** Instead of requiring all selected devices to complete a fixed number of local epochs \(E\), FedProx allows each device \(k\) to perform a variable amount of local computation (parameterized by \(\gamma_k^t\)). This accommodates stragglers by incorporating their partial updates rather than dropping them.

2. **Proximal Term:** Each device minimizes a regularized local objective:
   \[
   \min_w h_k(w; w^t) = F_k(w) + \frac{\mu}{2} \|w - w^t\|^2,
   \]
   where \(\mu > 0\) is a penalty parameter. This term restricts local updates from deviating too far from the current global model \(w^t\), mitigating the negative effects of statistical heterogeneity and variable local work.

**Key Definitions:**
- **\(\gamma\)-inexact solution** (Definition 1): For a function \(h(w; w_0) = F(w) + \frac{\mu}{2}\|w - w_0\|^2\), a point \(w^*\) is a \(\gamma\)-inexact minimizer if
  \[
  \|\nabla h(w^*; w_0)\| \le \gamma \|\nabla h(w_0; w_0)\|.
  \]
  Smaller \(\gamma\) means higher accuracy. This quantifies the amount of local computation.
- **\(\gamma_k^t\)-inexactness** (Definition 2): Extends the above to device‑specific and iteration‑specific accuracy, enabling analysis of systems heterogeneity.

**Algorithm 2 (FedProx):**
- Server selects a subset \(S_t\) of \(K\) devices.
- Each device \(k \in S_t\) finds a \(\gamma_k^t\)-inexact minimizer of \(h_k(w; w^t)\).
- Devices send \(w_k^{t+1}\) back; server aggregates: \(w^{t+1} = \frac{1}{K}\sum_{k \in S_t} w_k^{t+1}\).

FedAvg is a special case of FedProx with \(\mu=0\), SGD as local solver, and constant \(\gamma\) across devices/rounds.

## 3. Mathematical Formulation

### Global Objective
\[
\min_w f(w) = \sum_{k=1}^N p_k F_k(w) = \mathbb{E}_k[F_k(w)], \quad p_k = \frac{n_k}{n}.
\]

### Local Subproblem (FedProx)
\[
h_k(w; w^t) = F_k(w) + \frac{\mu}{2} \|w - w^t\|^2.
\]

### B‑Local Dissimilarity (Definition 3)
\[
\mathbb{E}_k[\|\nabla F_k(w)\|^2] \le B^2 \|\nabla f(w)\|^2,
\]
where \(B(w) = \sqrt{\frac{\mathbb{E}_k[\|\nabla F_k(w)\|^2]}{\|\nabla f(w)\|^2}}\) for \(\|\nabla f(w)\| \neq 0\). This measures statistical heterogeneity; \(B=1\) for IID data, larger \(B\) indicates higher dissimilarity.

### Convergence Theorem (Non‑convex, Theorem 4)
Assume \(F_k\) are \(L\)-smooth, \(\nabla^2 F_k \succeq -L_- I\), \(\bar{\mu} = \mu - L_- > 0\), and \(B(w^t) \le B\). If \(\mu, K, \gamma\) satisfy
\[
\rho = \left( \frac{1}{\mu} - \frac{\gamma B}{\mu} - \frac{B(1+\gamma)\sqrt{2}}{\bar{\mu}\sqrt{K}} - \frac{L B(1+\gamma)}{\bar{\mu}\mu} - \frac{L(1+\gamma)^2 B^2}{2\bar{\mu}^2} - \frac{L B^2 (1+\gamma)^2}{\bar{\mu}^2 K}(2\sqrt{2K}+2) \right) > 0,
\]
then at iteration \(t\):
\[
\mathbb{E}_{S_t}[f(w^{t+1})] \le f(w^t) - \rho \|\nabla f(w^t)\|^2.
\]

### Convergence Rate (Theorem 6)
Under the same conditions, after \(T = O\left(\frac{\Delta}{\rho \epsilon}\right)\) iterations, \(\frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[\|\nabla f(w^t)\|^2] \le \epsilon\), where \(\Delta = f(w^0) - f^*\).

### Variable \(\gamma\) (Corollary 9)
If \(\gamma^t = \max_{k \in S_t} \gamma_k^t\), the same decrease holds with \(\rho^t\) replacing \(\rho\) and \(\gamma^t\) in the expression.

### Bounded Variance Equivalence (Corollary 10)
If \(\mathbb{E}_k[\|\nabla F_k(w) - \nabla f(w)\|^2] \le \sigma^2\), then \(B_\epsilon \le \sqrt{1 + \frac{\sigma^2}{\epsilon}}\) for \(\|\nabla f(w)\|^2 > \epsilon\).

## 4. Limitations & Constraints

- **Assumptions:** The analysis requires \(L\)-smoothness, a lower bound on the Hessian (\(\nabla^2 F_k \succeq -L_- I\)), and the bounded dissimilarity assumption (Assumption 1). While reasonable for many FL tasks, these may not hold for all non‑convex models.
- **Parameter Tuning:** The proximal parameter \(\mu\) must be chosen carefully – too large slows convergence, too small may not help. The paper suggests a heuristic adaptive scheme but no automatic tuning method.
- **Communication Bottleneck:** FedProx does not directly address communication efficiency (e.g., compression or quantization). It reduces communication by allowing more local work, but the cost per round remains the same as FedAvg (full model transmission).
- **Device Sampling:** The analysis assumes uniform sampling with probabilities \(p_k\); practical implementations often use uniform device sampling with weighted averaging, which is not fully covered by the theory.
- **Convex Case:** For convex \(F_k\) with exact minimization (\(\gamma=0\)), the rate becomes \(O(LB^2 \Delta / \epsilon)\), which can be worse than SGD if \(B\) is large.

## 5. FedMAQ Thesis Relevance

**FedProx as a Baseline:** FedProx is a direct baseline for FedMAQ. Both target heterogeneous FL, but FedMAQ additionally focuses on **communication efficiency** via multi‑adaptive quantization and knowledge distillation. FedProx does not compress model updates; it only reduces communication by increasing local computation. FedMAQ can be seen as a complementary approach that adds compression on top of a robust optimization framework.

**Integration Opportunities:**
- **Proximal Term for Stability under Compression:** The proximal term \(\frac{\mu}{2}\|w - w^t\|^2\) can be incorporated into FedMAQ to stabilize training when aggressive quantization or distillation is applied. Quantization errors may cause local updates to drift; the proximal term mitigates this drift, similar to how it handles statistical heterogeneity.
- **Variable Local Work:** FedMAQ can adopt FedProx’s tolerance of partial work (variable \(\gamma_k^t\)) to handle devices with limited bandwidth or energy – e.g., devices that can only afford a few local steps or lower quantization levels.
- **Theoretical Framework:** FedProx’s convergence analysis under bounded dissimilarity provides a foundation for analyzing FedMAQ. The bounded dissimilarity assumption can be extended to account for additional errors from quantization and distillation.
- **Adaptive \(\mu\):** The heuristic for tuning \(\mu\) based on loss behavior can be adapted to FedMAQ to dynamically balance the proximal effect against compression noise.

**Key Differences:** FedMAQ’s core innovations – multi‑adaptive quantization (varying bit-widths per layer/round) and knowledge distillation (using a teacher model to guide local training) – are not present in FedProx. FedProx focuses on optimization stability, while FedMAQ focuses on communication reduction. Combining both could yield a highly communication‑efficient and robust FL system.