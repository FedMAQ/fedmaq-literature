description: FedGen performs data-free knowledge distillation by learning a lightweight generator from clients' prediction rules and broadcasting it to regularize local training, removing the proxy-dataset requirement.

## 1. Overview & Objectives

- **Core Problem**: KD-based FL (e.g., [FedDF](/papers/lin-2020-feddf.md), [FedMD](/papers/li-2019-fedmd.md)) mitigates non-IID model drift by distilling client knowledge, but **depends on a proxy dataset** on the server — often unavailable in practice. Moreover, prior KD refines only the **global** model, leaving client-side heterogeneity unaddressed.
- **Main Objectives**:
  - Propose **FedGen**: a **data-free** knowledge-distillation method where the server learns a **lightweight generator** that captures the ensemble of clients' prediction rules, with no external data.
  - Broadcast the generator to clients so it **regularizes local training** (an inductive bias), directly addressing client-side drift rather than only the global model.
  - Achieve better generalization with **fewer communication rounds**, and support settings where sharing full parameters is impractical (only the prediction layer is needed).

## 2. Methodology & Key Innovations

- **Key Idea**: The server trains a generator \(G\) that, given a **target label** \(y\), produces **latent feature representations** \(z\) whose induced predictions (through the clients' shared classifier heads) agree with the **ensemble** of client predictions for \(y\). This distills "what the clients collectively know" into \(G\) **without any real data**.
- **Latent-space generation**: \(G\) outputs features in a low-dimensional latent space \(Z\) (\(d < p\)), so the generator is **lightweight**, adding minimal overhead.
- **Client-side regularization**: each client downloads \(G\) and, during local training, samples \(G(\cdot\mid y)\) to augment learning over the latent space — injecting peer knowledge as an inductive bias that counteracts local-data bias.
- **Minimal sharing**: knowledge extraction needs only the **predictor (output layer)** of client models, enabling use where transmitting whole models is infeasible for privacy/communication reasons.

## 3. Mathematical Formulation

- **Model decomposition**: \(\theta = [\theta_f; \theta_p]\) — feature extractor \(f: X\to Z\) and predictor \(h: Z\to \triangle_Y\).
- **Generator objective** (schematic): learn \(G\) so that sampled latent features maximize agreement with the aggregated client predictors for the target label:

\[
\max_{G}\; \mathbb{E}_{y\sim p(y),\, z\sim G(\cdot\mid y)}\; \log \sum_{k} w_k\, h_{\theta_p^k}(z)_y ,
\]

i.e., latent samples should be confidently classified as \(y\) by the ensemble of client heads.
- **Client local loss**: supervised risk \(L_{T_k}(\theta)=\mathbb{E}_{x\sim D_k}[\ell(h(f(x)), c^*(x))]\) **plus** a distillation/regularization term using \(G\)-sampled latent features for the ideal (IID-like) label distribution.

## 4. Limitations & Constraints

- **Shared classifier-head assumption**: knowledge extraction assumes clients share a compatible predictor/latent space, constraining full model heterogeneity.
- **Generator quality risk**: if client heads are biased by scarce local data, the distilled generator inherits that bias, capping the benefit.
- **Extra server training**: learning \(G\) each round adds server computation and a generator broadcast (though latent-space \(G\) is small).
- **Classification framing**: the label-conditioned latent generator is formulated for multi-class classification.

## 5. FedMAQ Thesis Relevance

- **Data-free KD baseline**: FedGen removes the proxy-data dependency that limits [FedDF](/papers/lin-2020-feddf.md) and public-dataset KD like [FedMD](/papers/li-2019-fedmd.md), making it a key reference for FedMAQ's distillation component in settings lacking server-side data.
- **Latent-channel knowledge transfer**: the lightweight generator is an extremely compact knowledge carrier — conceptually a compression of the ensemble's knowledge. This aligns with FedMAQ's aim to move maximal knowledge under minimal communication, and pairs naturally with quantized parameter updates.
- **Key insight to integrate**: FedGen regularizes **local** training with distilled global knowledge, a hook FedMAQ can exploit to **stabilize clients under aggressive quantization** — using generator-sampled inductive bias to compensate for information lost when weight updates are heavily quantized.
