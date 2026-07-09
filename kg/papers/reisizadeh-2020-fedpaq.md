---
type: Paper
title: "FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization"
description: "Core problem: Federated learning suffers from a severe communication bottleneck due to frequent, large-message exchanges between many edge devices and a central server."
authors: "Reisizadeh et al."
year: 2020
bibkey: reisizadeh-2020-fedpaq
baseline: FedPAQ
tags: [quantization]
resource: markdown/reisizadeh-2020-fedpaq/paper.md
timestamp: 2026-06-21T05:08:38Z
---

## 1. Overview & Objectives

**Core problem:** Federated learning suffers from a severe communication bottleneck due to frequent, large-message exchanges between many edge devices and a central server. Additionally, the system must scale to millions of devices, many of which may be unavailable at any given time.

**Main objectives:**  
- Reduce the total communication load while preserving convergence accuracy.  
- Design a provably efficient algorithm that simultaneously handles: (i) infrequent communication via *periodic averaging*, (ii) realistic *partial device participation*, and (iii) *compressed (quantized) message passing*.  
- Provide near-optimal convergence guarantees for both strongly convex and non‑convex loss functions.

The resulting algorithm is **FedPAQ** (**Fed**erated learning with **P**eriodic **A**veraging and **Q**uantization).

---

## 2. Methodology & Key Innovations

FedPAQ integrates three complementary modules:

1. **Periodic averaging (local‑SGD style)**  
   Devices perform `$\tau$` local SGD updates before synchronising with the server. This reduces the number of communication rounds from `$T$` to `$K = T/\tau$`.

2. **Partial node participation**  
   In each communication round, only a random subset `$\mathcal{S}_k$` of `$r \le n$` devices is active. This reduces the per‑round communication load and mirrors real‑world device availability.

3. **Quantized message passing**  
   Each participating node quantises the difference between its local model (after `$\tau$` updates) and the received server model before uploading. A low‑precision quantizer `$Q_{\text{LP}}$` (Alistarh et al., 2017) is used, controlled by a level parameter `$s$`.

The server aggregates the quantised updates and updates the global model.

---

## 3. Mathematical Formulation

### 3.1 Problem Setting

We minimise the expected loss over `$n$` nodes:

$$
\min_{x} \; f(x) = \frac{1}{n} \sum_{i=1}^{n} f_i(x), \qquad 
f_i(x) = \mathbb{E}_{\xi \sim \mathcal{P}^i}\big[\ell(x,\xi)\big].
$$

All nodes share the same data distribution (i.i.d. setting).

### 3.2 Quantization Operator

The low‑precision quantizer (Example 1):

$$ Q_i^{\text{LP}}(x) = \|x\| \cdot \operatorname{sign}(x_i) \cdot \xi_i(x,s), \quad i\in[p], $$

where `$\xi_i(x,s)$` is a random variable taking value `$(l+1)/s$` with probability `$\frac{|x_i|}{\|x\|}s - l$` and `$l/s$` otherwise. Here `$l$` satisfies `$\frac{|x_i|}{\|x\|} \in [l/s, (l+1)/s)$`.

The quantizer is **unbiased** and satisfies

$$ \mathbb{E}[Q(x)|x] = x, \qquad \mathbb{E}[\|Q(x)-x\|^2 \;|\; x] \le q \|x\|^2, $$

for some constant `$q>0$` (Assumption 1).

### 3.3 Algorithm Update (Simplified Pseudocode)

**Algorithm 1 (FedPAQ):**  
For each period `$k = 0,\dots,K-1$`:

1. Server picks a random set `$\mathcal{S}_k$` of size `$r$`.
2. Server sends current model `$x_k$` to all nodes in `$\mathcal{S}_k$`.
3. **Local updates** (each node `$i\in\mathcal{S}_k$`):
   - Initialise `$x_{k,0}^{(i)} = x_k$`.
   - For `$t = 0,\dots,\tau-1$`:
     $$ x_{k,t+1}^{(i)} = x_{k,t}^{(i)} - \eta_{k,t} \widetilde{\nabla} f_i\!\big(x_{k,t}^{(i)}\big). $$
   - Compute the update `$\Delta_k^{(i)} = x_{k,\tau}^{(i)} - x_k$` and send `$Q(\Delta_k^{(i)})$` to server.
4. **Server aggregation**:

   $$ x_{k+1} = x_k + \frac{1}{r} \sum_{i \in \mathcal{S}_k} Q\!\big(x_{k,\tau}^{(i)} - x_k\big). $$

The step size `$\eta_{k,t}$` is chosen as a decreasing sequence (see convergence results).

### 3.4 Key Assumptions

- **A1 (Quantizer):** Unbiased, variance bounded by `$q\|x\|^2$`.
- **A2 (Smoothness):** Each `$f_i$` is `$L$`‑smooth: `$\|\nabla f_i(x) - \nabla f_i(\hat{x})\| \le L \|x-\hat{x}\|$`.
- **A3 (Stochastic gradient):** Unbiased, bounded variance: `$\mathbb{E}[\widetilde{\nabla}f_i(x)] = \nabla f_i(x)$` and `$\mathbb{E}[\|\widetilde{\nabla}f_i(x) - \nabla f_i(x)\|^2] \le \sigma^2$`.
- **A4 (Strong convexity, for Theorem 1):** Each `$f_i$` is `$\mu$`‑strongly convex.

### 3.5 Convergence Results

**Theorem 1 (Strongly convex losses):**  
Let `$B_1 = 2L^2\!\left(\frac{q}{n} + \frac{n-r}{r(n-1)} 4(1+q)\right)$`. With a suitable decreasing step size `$\eta_k = \frac{4}{\mu(k\tau+1)}$` and for `$k\ge k_0$` large enough, we have

$$
\mathbb{E}\|x_k - x^*\|^2 \le \frac{(k_0\tau+1)^2}{(k\tau+1)^2}\|x_{k_0} - x^*\|^2 
+ C_1\frac{\tau}{k\tau+1} + C_2\frac{(\tau-1)^2}{k\tau+1} + C_3\frac{\tau-1}{(k\tau+1)^2},
$$

where `$C_1, C_2, C_3$` depend on `$\sigma, \mu, L, n, r, q$` (given in the paper).  
Order‑wise, for `$T=K\tau$`:

$$ \mathbb{E}\|x_K - x^*\|^2 = \mathcal{O}\!\left(\frac{\tau}{T}\right) + \mathcal{O}\!\left(\frac{\tau^2}{T^2}\right). $$

Thus `$\tau = o(\sqrt{T})$` retains the optimal `$\mathcal{O}(1/T)$` rate.

**Theorem 2 (Non‑convex losses):**  
Let `$B_2 = \frac{q}{n} + \frac{4(n-r)}{r(n-1)}(1+q)$`. Under conditions `$T\ge 2$` and `$\tau \le \frac{\sqrt{B_2^2+0.8} - B_2}{8}\sqrt{T}$`, and with step size `$\eta = 1/(L\sqrt{T})$`, we obtain

$$
\frac{1}{T}\sum_{k=0}^{K-1}\sum_{t=0}^{\tau-1}\mathbb{E}\|\nabla f(\bar{x}_{k,t})\|^2 
\le \frac{2L(f(x_0)-f^*)}{\sqrt{T}} + N_1\frac{1}{\sqrt{T}} + N_2\frac{\tau-1}{T},
$$

where `$N_1, N_2$` are constants.  
This yields an `$\mathcal{O}(1/\sqrt{T})$` rate to a stationary point, even when `$\tau = \mathcal{O}(\sqrt{T})$` – i.e., with only `$\mathcal{O}(\sqrt{T})$` communication rounds.

---

## 4. Limitations & Constraints

- **Statistical homogeneity:** The analysis assumes i.i.d. data across nodes. Non‑i.i.d. settings are not covered (a known limitation of many theoretical FL works).
- **Unbiased quantizer:** Results rely on unbiasedness and a variance bound linear in `$\|x\|^2$`. Biased compressors (e.g., top‑`$k$` sparsification) are not analysed.
- **Assumption of smoothness:** Loss functions must be `$L$`‑smooth; non‑smooth objectives are excluded.
- **Constant device availability probability:** The active set `$\mathcal{S}_k$` is chosen uniformly at random. In practice, availability may follow a non‑uniform or correlated pattern.
- **Communication‑computation trade‑off:** While `$\tau$` can be as large as `$\sqrt{T}$` in theory, the optimal choice depends on problem constants (e.g., `$L,\mu,q,n,r$`) and requires tuning.
- **No privacy guarantees:** The paper does not address differential privacy or other privacy‑enhancing techniques.

---

## 5. FedMAQ Thesis Relevance

**FedPAQ is a key baseline** for FedMAQ (Communication‑Efficient FL via Multi‑Adaptive Quantization and Knowledge Distillation). Its relevance includes:

- **Periodic averaging** (local‑SGD) is a core technique that FedMAQ can directly inherit or compare against. FedMAQ may adapt the period `$\tau$` dynamically, whereas FedPAQ uses a fixed `$\tau$`.
- **Quantization** in FedPAQ is a single‑level, fixed‑precision operation (e.g., low‑precision quantizer). FedMAQ extends this to *multi‑adaptive quantization*, where different parts of the model or different communication rounds may use different precision levels, potentially improving the communication‑accuracy trade‑off.
- **Knowledge distillation** is not used in FedPAQ. FedMAQ can incorporate KD (e.g., to align local models or compress global knowledge) on top of FedPAQ’s framework, providing an orthogonal enhancement.
- **Theoretical guarantees** of FedPAQ (strongly convex `$\mathcal{O}(1/T)$`, non‑convex `$\mathcal{O}(1/\sqrt{T})$` with `$\tau=\mathcal{O}(\sqrt{T})$`) serve as a benchmark. FedMAQ should aim to achieve at least the same rates while improving empirical performance through adaptive quantization and KD.
- **Partial participation** is already handled in FedPAQ; FedMAQ can adopt the same random‑subset model or extend to more sophisticated participant selection.

In summary, FedPAQ is a natural starting point for FedMAQ: it addresses two of the three communication‑saving pillars (periodic averaging and quantization) and provides rigorous convergence analysis. FedMAQ can build upon this by adding multi‑level adaptive quantization and knowledge distillation, targeting further communication reductions while maintaining or improving model accuracy.

# Related

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](/papers/mcmahan-2017-fedavg.md)
- [QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding](/papers/alistarh-2017-qsgd.md)
- [DAdaQuant: Doubly-adaptive quantization for communication-efficient Federated Learning](/papers/honig-2022-dadaquant.md)

# Citations

[1] Full-text conversion: [markdown/reisizadeh-2020-fedpaq/paper.md](markdown/reisizadeh-2020-fedpaq/paper.md)
[2] Source PDF: `papers/02 Quantization/Reisizadeh et al. - 2020 - FedPAQ A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization.pdf`
