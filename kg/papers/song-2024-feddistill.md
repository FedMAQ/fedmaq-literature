---
type: Paper
title: "FedDistill: Global Model Distillation for Local Model De-Biasing in Non-IID Federated Learning"
description: "Federated Learning (FL) suffers from severe performance degradation under non-IID data distributions."
authors: "Song et al."
year: 2024
bibkey: song-2024-feddistill
baseline: "FedDistill (name collision only — NOT the implemented FedMAQ baseline; see /methods/feddistill.md for that)"
tags: [kd]
resource: markdown/song-2024-feddistill/paper.md
timestamp: 2026-06-21T05:24:36Z
---

## 1. Overview & Objectives

**Core Problem:**  
Federated Learning (FL) suffers from severe performance degradation under non-IID data distributions. Local models trained on imbalanced datasets exhibit a *forgetting* phenomenon—they lose generalization ability for underrepresented classes due to insufficient positive gradient updates. Existing knowledge distillation (KD) methods that use the global model as a teacher fail to address this class imbalance effectively.

**Main Objectives:**  
- Identify the root cause of local model forgetting: inadequate positive feedback for few-sample classes during local training.  
- Propose a framework that enhances knowledge transfer from the global model to local models without extra communication overhead or privacy risks.  
- Mitigate the bias induced by imbalanced local data, improving both accuracy and convergence speed.

## 2. Methodology & Key Innovations

**FedDistill Framework** (Figure 4 in paper):  
- **Group Distillation (GD) Loss:** Instead of standard KL divergence, the distillation loss is decomposed into three components targeting *true-class*, *rich-sample classes*, and *few-sample classes*. A threshold \(\gamma\) (set to \(1/|\mathcal{C}|\)) determines whether a class is rich or few-sample based on its proportion in the local dataset.  
- **Model Decomposition:** The global model is split into a feature extractor \(E_g\) and a classifier \(FC_g\). Local models have analogous components \(E_l\) and \(FC_l\). Four prediction paths are used:  
  - \(\hat{y}_{gg}\): global feature + global classifier  
  - \(\hat{y}_{ll}\): local feature + local classifier  
  - \(\hat{y}_{lg}\): local feature + global classifier  
  - \(\hat{y}_{gl}\): global feature + local classifier  
- **Three Loss Components:**  
  - \(\mathcal{L}_L = GD(\hat{y}_{gg} \| \hat{y}_{ll})\) – aligns local model output with global model output.  
  - \(\mathcal{L}_E = CE(\hat{y}_{lg}, y)\) – ensures local feature extractor generalizes by using the unbiased global classifier.  
  - \(\mathcal{L}_{FC} = GD(\hat{y}_{gg} \| \hat{y}_{gl})\) – de-biases the local classifier by preserving ranking information from the global feature extractor.  
- **Composite Loss:**  
  \[
  \mathcal{L} = CE(\hat{y}_{ll}, y) + \beta_L \mathcal{L}_L + \beta_E \mathcal{L}_E + \beta_{FC} \mathcal{L}_{FC}
  \]

## 3. Mathematical Formulation

### 3.1 Standard FL Aggregation (FedAvg)
\[
\theta_g^{(t)} = \frac{\sum_{s \in S^{(t)}} |\mathcal{D}_s| \theta_s^{(t)}}{\sum_{s' \in S^{(t)}} |\mathcal{D}_{s'}|}
\]

### 3.2 Cross-Entropy Gradient Analysis
For a model with feature extractor \(E\) and classifier weights \(w_c\):
\[
q_c(x) = \frac{\exp(w_c^T E(x))}{\sum_k \exp(w_k^T E(x))}
\]
Gradient for true class \(c\):
\[
\frac{\partial \log(q_c)}{\partial w_c} = (1 - q_c) E(x) \quad (\text{positive})
\]
Gradient for non-true class \(\bar{c}\):
\[
\frac{\partial \log(q_c)}{\partial w_{\bar{c}}} = - q_{\bar{c}} E(x) \quad (\text{negative})
\]
This differential causes forgetting of underrepresented classes.

### 3.3 Group Distillation Loss
Let \(\mathcal{C}_r\) = rich-sample classes, \(\mathcal{C}_f\) = few-sample classes, \(t\) = true class. Define:
\[
\tilde{q}_i = \frac{q_i}{p_{\setminus t}}, \quad p_{\setminus t} = \sum_{k \neq t} q_k
\]
\[
\tilde{p}_f = \sum_{i \in \mathcal{C}_f \setminus \{t\}} \tilde{q}_i, \quad \tilde{p}_r = \sum_{i \in \mathcal{C}_r \setminus \{t\}} \tilde{q}_i
\]

**TC-KD (True Class):**
\[
TC\text{-KD}(q^g \| q^l) = q_t^g \log\left(\frac{q_t^g}{q_t^l}\right) + p_{\setminus t}^g \log\left(\frac{p_{\setminus t}^g}{p_{\setminus t}^l}\right)
\]

**RC-KD (Rich-Sample Classes):**
\[
RC\text{-KD}(q^g \| q^l) = \sum_{i \in \mathcal{C}_r \setminus \{t\}} \tilde{q}_i^g \log\left(\frac{\tilde{q}_i^g}{\tilde{q}_i^l}\right) + \tilde{p}_f^g \log\left(\frac{\tilde{p}_f^g}{\tilde{p}_f^l}\right)
\]

**FC-KD (Few-Sample Classes):**
\[
FC\text{-KD}(q^g \| q^l) = \sum_{i \in \mathcal{C}_f \setminus \{t\}} \tilde{q}_i^g \log\left(\frac{\tilde{q}_i^g}{\tilde{q}_i^l}\right) + \tilde{p}_r^g \log\left(\frac{\tilde{p}_r^g}{\tilde{p}_r^l}\right)
\]

**Overall GD Loss:**
\[
GD(q^g \| q^l) = \alpha_t \, TC\text{-KD} + \alpha_r \, RC\text{-KD} + \alpha_f \, FC\text{-KD}
\]
When \(\alpha_t = \alpha_r = \alpha_f = 1\), GD reduces to standard KL divergence.

### 3.4 Forgetting Measure
\[
\mathcal{F} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \left( \max_{t \in \{1,\dots,T-1\}} \mathcal{A}_c^t - \mathcal{A}_c^T \right)
\]
where \(\mathcal{A}_c^t\) is accuracy on class \(c\) at round \(t\).

## 4. Limitations & Constraints

- **Hyperparameter Sensitivity:** The method introduces several hyperparameters (\(\alpha_t, \alpha_r, \alpha_f, \beta_L, \beta_E, \beta_{FC}, \gamma\)) that require careful tuning. The authors acknowledge this complexity and suggest future self-adaptive learning.
- **Assumption of Global Model Generalization:** The framework relies on the global model being more generalized than local models. In extreme non-IID settings, the global model itself may be biased, limiting the effectiveness of distillation.
- **No Communication Reduction:** FedDistill does not reduce communication overhead; it only modifies local training. The number of communication rounds is reduced (faster convergence), but per-round cost remains the same as FedAvg.
- **Threshold Selection:** The few-sample threshold \(\gamma\) is set empirically to \(1/|\mathcal{C}|\). This may not be optimal for all datasets or client distributions.
- **Scalability to Many Classes:** The group distillation loss requires per-class computation, which may become expensive for datasets with hundreds of classes (e.g., CIFAR100 with 100 classes).

## 5. FedMAQ Thesis Relevance

FedDistill is directly relevant to the FedMAQ thesis (Communication-Efficient FL via Multi-Adaptive Quantization and Knowledge Distillation) in the following ways:

- **Baseline for KD-based Methods:** FedDistill represents a state-of-the-art KD approach for non-IID FL. It can serve as a strong baseline for FedMAQ, especially for the knowledge distillation component. The paper shows FedDistill outperforms FedNTD, MOON, etc., making it a natural candidate for comparison.
- **Integration Potential:**  
  - **Group Distillation + Quantization:** The GD loss could be combined with adaptive quantization of model updates. For example, the \(\alpha\) weights for different class groups could be used to prioritize quantization precision for few-sample classes.  
  - **Model Decomposition for Compression:** The separation of feature extractor and classifier could enable selective compression (e.g., quantize the classifier more aggressively since it is more biased).  
  - **Communication Efficiency:** FedDistill already reduces the number of rounds to reach target accuracy (Table III). FedMAQ could further reduce per-round communication by applying multi-adaptive quantization to the model parameters exchanged between clients and server.
- **Complementary Techniques:** FedDistill focuses on local training improvements; FedMAQ can add communication compression on top without interfering with the distillation process. The composite loss in FedDistill is independent of the aggregation mechanism, so quantization-aware training could be incorporated.

**Conclusion:** FedDistill is a strong KD-based method that addresses non-IID forgetting. It can be used as a baseline for FedMAQ and its techniques (group distillation, model decomposition) can be integrated with adaptive quantization to achieve both accuracy and communication efficiency.

# Related

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](/papers/mcmahan-2017-fedavg.md)
- [Model-Contrastive Federated Learning](/papers/li-2021-moon.md)

# Citations

[1] Full-text conversion: [markdown/song-2024-feddistill/paper.md](markdown/song-2024-feddistill/paper.md)
[2] Source PDF: `papers/03 KD/Song et al. - 2024 - FedDistill Global Model Distillation for Local Model De-Biasing in Non-IID Federated Learning.pdf`
