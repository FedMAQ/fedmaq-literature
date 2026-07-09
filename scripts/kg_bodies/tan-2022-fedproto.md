description: FedProto communicates abstract per-class prototypes (mean feature representations) instead of gradients, tolerating both data and model heterogeneity at low communication cost.

## 1. Overview & Objectives

- **Core Problem**: Aggregating client knowledge **in the gradient/parameter space** breaks down when clients differ not only in data distribution but also in **model architecture, input/output spaces, and network latency**. Gradient averaging assumes homogeneous models and incurs high communication cost.
- **Main Objectives**:
  - Propose **FedProto**, where clients and server exchange **class prototypes** — abstract per-class mean feature representations — instead of gradients or model weights.
  - Support **model-heterogeneous** FL: clients may have different architectures yet still share knowledge through a common representation space.
  - Improve communication efficiency and provide a **convergence guarantee under non-convex objectives**; introduce a benchmark tailored to heterogeneous FL.

## 2. Methodology & Key Innovations

- **Key Idea**: A **prototype** for class \(j\) on client \(i\) is the mean of the embedding-layer representations of that client's samples labeled \(j\). The server aggregates local prototypes into **global prototypes** and broadcasts them; clients regularize local training so their per-class representations stay close to the corresponding global prototype.
- **Local objective** combines (a) supervised classification loss on local data and (b) a **prototype-alignment regularizer** pulling local class means toward global prototypes:
  - \(\ell = \ell_{sup}(\phi_i, x, y) + \lambda\, \lVert C_i^{(j)} - \bar{C}^{(j)} \rVert\), where \(C_i^{(j)}\) is client \(i\)'s prototype for class \(j\) and \(\bar C^{(j)}\) the global one.
- **Server aggregation** is a weighted average of prototypes per class (dimension = embedding size, **independent of model size**), which is why communication is cheap and architecture-agnostic.

## 3. Mathematical Formulation

- **Local prototype**: \(C_i^{(j)} = \tfrac{1}{|D_{i,j}|}\sum_{(x,y)\in D_{i,j}} f_{\phi_i}(x)\), the mean representation of class-\(j\) samples on client \(i\).
- **Global prototype** (weighted over clients holding class \(j\)):

\[
\bar{C}^{(j)} = \frac{1}{\sum_{i} |D_{i,j}|}\sum_{i} |D_{i,j}|\, C_i^{(j)} .
\]

- **Regularized local loss**:

\[
\ell = \ell_{sup} + \lambda \sum_{j} \lVert C_i^{(j)} - \bar{C}^{(j)} \rVert_2 .
\]

- The paper derives a **non-convex convergence rate**, bounding the expected gradient norm across rounds.

## 4. Limitations & Constraints

- **Shared representation dimension required**: clients may differ in architecture but must map to a **common embedding space** for prototypes to be comparable.
- **Class-coverage dependence**: prototypes are only meaningful for classes a client observes; extreme label skew (missing classes) limits the regularization signal.
- **Personalized, not a single global model**: FedProto yields per-client models regularized by shared prototypes rather than one deployable global network, which may not fit all deployment needs.
- **Representation-only transfer**: knowledge exchange is limited to class-mean statistics; fine-grained decision boundaries are not directly shared.

## 5. FedMAQ Thesis Relevance

- **Extreme-compression reference point**: FedProto is an aggressive form of communication reduction — transmitting only class-mean vectors instead of models. It bounds the "how little can we send" question that FedMAQ's quantization+distillation also targets, from a different (prototype) direction.
- **KD kinship**: like FedMAQ's distillation component, FedProto transfers *representations* rather than *parameters*, and it explicitly contrasts itself with public-dataset KD methods ([FedDF](/papers/lin-2020-feddf.md)-style) that FedMAQ also builds on. It is a natural non-parametric baseline for the KD axis.
- **Key insight to integrate**: FedProto shows heterogeneity can be handled by aligning compact representation statistics. FedMAQ could combine prototype-style semantic anchors with quantized parameter updates, using prototypes to stabilize accuracy when bit-widths are pushed low.
