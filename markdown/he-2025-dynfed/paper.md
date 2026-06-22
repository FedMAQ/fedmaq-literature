## DynFed: Adaptive Federated Learning via Quantization-Aware Knowledge Distillation

## Nan He

henancs@mail.tsinghua.edu.cn Department of Computer Science and Technology, Tsinghua University Beijing, China

## Yiming chen

chenym22@mails.tsinghua.edu.cn Tsinghua University Beijing, China

## Song Yang

s.yang@bit.edu.cn School of Computer Science, Beijing Institute of Technology Beijing, China

## Zheng Jiang

jz24@mails.tsinghua.edu.cn Computer Science, Tsinghua University Beijing, China

## Lifeng Sun ∗

sunlf@tsinghua.edu.cn Department of Computer Science and Technology, Tsinghua University Beijing, China

## Keywords

Federated learning, Knowledge distillation, Quantization

## ACMReference Format:

Nan He, Yiming chen, Zheng Jiang, Song Yang, and Lifeng Sun. 2025. DynFed: Adaptive Federated Learning via Quantization-Aware Knowledge Distillation. In Proceedings of the 33rd ACM International Conference on Multimedia (MM '25), October 27-31, 2025, Dublin, Ireland. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3746027.3755451

## 1 Introduction

Federated Learning (FL) has emerged as a promising approach for collaboratively training machine learning models across decentralized entities without sharing raw data, thus addressing critical privacy concerns [43]. By enabling local training on distributed data, FL reduces the communication burden typically associated with centralized data collection while preserving data privacy [44]. Despite these advantages, FL faces significant challenges due to the inherent heterogeneity of client devices, which vary in computational power, storage capacity, and network connectivity [27]. These disparities can lead to inefficiencies in model training and aggregation, ultimately degrading the performance of the global model [20]. Furthermore, extensive parameter exchanges between the server and clients introduce additional communication overhead, constrained by bandwidth and network latency, which impedes the efficiency of the learning process [21]. To address this, various methods, such as model pruning and reducing aggregation frequency, have been proposed to mitigate communication overhead [16, 32].

However, existing methods primarily focus on reducing communication overhead while overlooking significant differences in resource capacities among clients [13, 14]. This imbalance can lead to issues such as client dropout during training, affecting overall training efficiency and model quality [42]. Current research often ignores this heterogeneity by adopting identical model structures for the server and clients, failing to fully utilize the cloud's storage and computational capabilities and neglecting the limited resources of client devices. This limitation restricts the application of FL to small-scale datasets and specific scenarios [31]. Thus, designing federated learning systems requires a simultaneous consideration of both communication overhead and storage capacity.

## Abstract

Federated Learning (FL) has become a powerful technique for collaborative model training across decentralized entities while preserving data privacy. Despite its potential, FL faces significant challenges, including communication overhead, resource heterogeneity, and data heterogeneity. Existing solutions fall short in addressing disparities in client resources and the errors introduced by direct model aggregation across heterogeneous clients. To tackle these issues, we propose DynFed, a novel federated learning framework that incorporates dynamic quantization bit-width allocation and multi-teacher knowledge distillation for model aggregation. DynFed dynamically adjusts quantization bit-widths to clients based on their resource heterogeneity, adapting these allocations according to variations in the local loss function during training. This adaptive quantization strategy optimizes resource utilization while preserving model performance. For model aggregation, DynFed utilizes a dynamic multi-teacher knowledge distillation approach, assigning the most suitable teacher model to each data sample based on a comprehensive evaluation score, thereby ensuring effective knowledge transfer even in the presence of quantization-induced errors. This method not only mitigates the negative effects of heterogeneous bit-widths but also leverages client model diversity to enhance the robustness of the global model. Extensive experimental results demonstrate the superiority of DynFed over state-of-the-art methods.

## CCS Concepts

· Do Not Use This Code → Generate the Correct Terms for Your Paper ; Generate the Correct Terms for Your Paper ; Generate the Correct Terms for Your Paper; Generate the Correct Terms for Your Paper.

∗ Corresponding author.

<!-- image -->

This work is licensed under a Creative Commons Attribution 4.0 International License. MM'25, Dublin, Ireland

© 2025 Copyright held by the owner/author(s).

ACM ISBN 979-8-4007-2035-2/2025/10

https://doi.org/10.1145/3746027.3755451

Additionally, client data heterogeneity introduces further challenges [7]. Direct aggregation of models trained on diverse local datasets can introduce errors that significantly impact federated learning performance [32]. Existing research often overlooks errors from heterogeneous network aggregation [15]. Therefore, this paper aims to address communication overhead, device performance imbalance, and data heterogeneity in federated learning design.

To address communication overhead, model quantization has been proposed to reduce the cost associated with model training and updates, thereby accelerating the training process while maintaining performance [1]. Quantization represents model parameters with lower precision, reducing communication costs. However, existing methods typically assume uniform bit-widths across clients, ignoring resource heterogeneity [14]. Aggregating models with heterogeneous bit-widths presents challenges, as differing bit-widths affect parameter representation, leading to decreased performance and potential information loss. Current approaches attempt to address this through weighted aggregation but do not fully resolve error accumulation issues [11].

Knowledge distillation has emerged as a promising solution to handle client heterogeneity [18]. In FL, knowledge distillation can mitigate the negative impacts of quantization and heterogeneity [40]. However, existing research focuses mainly on distillation across networks with different structures, with limited exploration of distillation for heterogeneous bit-widths [36]. This paper addresses these challenges by designing a federated learning framework that incorporates dynamic bit-width allocation and aggregation of heterogeneous bit-width models.

We propose DynFed, which leverages multi-teacher knowledge distillation to address challenges of heterogeneous client allocation and quantization in FL. DynFed dynamically adjusts quantization bit-widths based on client capabilities and local loss changes, ensuring effective aggregation and robust model performance. Instead of directly aggregating heterogeneous bit-widths, DynFed employs a multi-teacher knowledge distillation approach, selecting the most suitable teacher model for each sample based on class uncertainty, enhancing knowledge transfer and improving model accuracy. This approach yields a superior global model compared to one aggregated solely from client models. Extensive experiments validate the effectiveness of DynFed in heterogeneous environments. The main contributions are summarized as follows:

- We introduce a method for dynamically adjusting quantization bit-widths based on client resources and local loss, improving model efficiency and accuracy.
- We propose a multi-teacher knowledge distillation approach for model aggregation, effectively managing heterogeneous bit-widths and enhancing global model performance.
- We incorporate active learning to select the most suitable teacher models, optimizing knowledge transfer and improving model aggregation quality in federated learning.
- We conduct extensive experiments to validate the effectiveness of the proposed DynFed algorithm in heterogeneous client scenarios, demonstrating its superiority over state-ofthe-art methods.

## 2 Related Work

## 2.1 Federated Learning with Client Heterogeneity

Federated learning faces significant challenges due to client heterogeneity, which manifests in two main categories. The first category is Heterogeneity in Non-IID data, and several approaches have been proposed to address it. For example, FedNH [5] improves local and global model performance by using class prototypes to handle class imbalance, enhancing both personalization and generalization across various datasets. FedFed [41] mitigates data heterogeneity by sharing performance-sensitive features globally while keeping performance-robust features local, balancing privacy and model effectiveness. Classifier Calibration with Virtual Representations (CCVR) [24]addresses layer-specific biases by calibrating classifiers with virtual representations, achieving state-of-the-art results on federated learning benchmarks. Virtual Homogeneity Learning (VHL) [34] introduces a virtual homogeneous dataset generated from pure noise to rectify data heterogeneity, improving convergence speed and generalization performance.

Considering the resource heterogeneity, the capabilities of devices among participants may vary significantly, potentially leading to failures and inactivity in certain nodes, commonly referred to as straggler nodes. There are several methods developed to deal with this challenge at different stages. To address the challenges posed by heterogeneous resources, the Split-Mix FL [10] offers a novel approach that allows in-situ customization of model sizes and robustness levels, effectively improving communication, storage, and inference efficiency. Marfoq et al. [25] propose a personalized Federated Learning (PFL) approach, such as the one that leverages local memorization through deep neural network representations, demonstrates significantly higher accuracy and fairness by tailoring models to individual client data distributions while still utilizing shared global knowledge. Different from these methods, we allocate different model bit-widths based on device capacity, aiming to reduce storage overhead and communication time without compromising model performance.

## 2.2 Quantization and Knowledge Distillation in FL

Model quantization is a crucial technique to reduce the communication cost and resource requirements in FL. Unlike the fixed bit-width assumption in DAdaQuant [12], where each client operates with a predetermined bit-width, Liu et al. [23] propose a method that allocates different bit-widths based on the client's loss levels. However, in contrast to their approach, our method avoids static bit-width allocation. Instead, we initialize bit-widths based on the resource conditions of heterogeneous clients and introduce adaptive and precise regulation of local models by the global model during each aggregation.

Knowledge distillation, initially proposed by Hinton et al. [9], involves transferring knowledge from a larger, well-trained teacher model to a smaller student model. In the context of FL, Zhu et al. [46] utilize knowledge distillation (KD) to mitigate the impact of data heterogeneity on global model performance, while SKDBERT [6] combines KD with quantization for compression. Unlike these approaches, our method employs knowledge distillation to facilitate collaboration between local and global models. This is particularly important for compensating the errors introduced by heterogeneous bit-widths, ensuring that high-precision local models contribute effectively to the global model.

Recent studies have further explored the synergy between knowledge distillation, quantization, and client heterogeneity. On the one hand, methods such as PeFAD [39], AeroRec [37], and FedTAD [45] focus on improving robustness and convergence under nonIID conditions via synthetic data, anomaly detection, or topologyaware distillation. On the other hand, approaches like FedPFT [28], MFL-AKD [4], and LG-FGAD [3] demonstrate the potential of layerwise, multimodal, and graph-level distillation to enhance scalability and personalization in complex federated environments. Moreover, emerging works such as Quantized LoRA [17] and [35] further combine efficient fine-tuning, quantization, and public-data-aided distillation to balance utility and privacy in LLM-scale FL tasks.

Building upon these advancements, our proposed method integrates dynamic, resource-aware quantization-aware training with a multi-teacher distillation strategy, which is carefully tailored to the resource profiles and learning dynamics of individual clients. This design not only reduces communication and computation overheads, but also ensures that high bit-width clients can effectively guide global learning through reliable knowledge transfer. Distinct from prior approaches, our framework emphasizes client-server synergy via targeted distillation, which serves as a key mechanism to mitigate performance degradation caused by quantization heterogeneity. As a result, our method enables a more balanced and robust global model aggregation across diverse client configurations.

## 3 Preliminaries

## 3.1 Federated Learning

In this paper, we consider a standard FL setting-a decentralized machine learning paradigm that enables multiple participants to collaboratively train a shared global model while keeping their raw data stored locally. Formally, let there be 𝑁 distributed clients, each with a local dataset denoted as {D 𝑖 } 𝑁 𝑖 = 1 . The objective is to learn a global model while preserving the privacy of users' local data. Federated learning enables this by training a shared global model without directly accessing clients' raw data, aiming to minimize the following objective:

$$
\begin{aligned}
\min _ { w } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathcal { L } _ { i } ( w ; \mathcal { D } _ { i } ) , & \quad \text {where} \mathcal { L } _ { i } ( w ; \mathcal { D } _ { i } ) = \frac { 1 } { n _ { k } } \sum _ { ( x , y ) \in \mathcal { O } } l _ { w } ( x , y ) \\ & \quad \text {where} \, y \subset \mathbb { R } ^ { d } \text {, donot} \, \text {the global model parametors to be anti}
\end{aligned}
$$

where w ∈ R 𝑑 denotes the global model parameters to be optimized,

Each sample ( 𝑥,𝑦 ) ∈ D 𝑖 corresponds to an input-label pair drawn from client 𝑖 's data. The function ℓ ( w ; 𝑥,𝑦 ) denotes the loss incurred by model w on sample ( 𝑥,𝑦 ) , typically implemented as a classification loss such as cross-entropy. Accordingly, L 𝑖 ( w ; D 𝑖 ) is the empirical loss of model w evaluated over the local dataset D 𝑖 of client 𝑖 .

<!-- image -->

- (a) Impact of Extreme Bit-Width Disparity on Model Convergence

(b) Nonlinear Aggregation Error Induced by Bit-Width Disparity

<!-- image -->

Figure 1: Effects of heterogeneous quantization bit-widths in DynFed: (a) demonstrates convergence challenges under extreme bit-width differences, while (b) analyzes aggregation errors caused by nonlinear bit-width interactions.

## 3.2 Motivation

As demonstrated in Fig. 1a, aggregating models with extreme bitwidth disparities (e.g., 2-bit and 8-bit clients) reduces global accuracy by 5.8% compared to homogeneous 4-bit baselines, with persistent divergence after 40 communication rounds. This degradation stems from the inherent limitations of naive parameter averaging, which propagates quantization-induced errors from low-bitwidth clients to the global model. Fig. 1b further quantifies this phenomenon, revealing a nonlinear correlation between bit-width disparity and model corruption: a six-bit disparity amplifies aggregation mean squared error (MSE) by 625%, accompanied by an accelerated growth in accuracy loss. These findings underscore the unsuitability of existing methods-resource-agnostic quantization squanders the potential of high-capacity devices, while error-agnostic aggregation mechanisms (e.g., FedAvg) treat heterogeneous models equivalently, disregarding their divergent noise profiles.

To overcome these dual limitations, a paradigm shift is imperative. Effective federated learning under resource heterogeneity demands (1) adaptive bit-width allocation that dynamically aligns quantization precision with client capabilities and training dynamics, and (2) error-resilient aggregation that disentangles meaningful knowledge from quantization noise. Current approaches lack the flexibility to prevent harmful bit-width combinations (Fig.1a) or mitigate error propagation (Fig.1b) , necessitating a unified framework that transforms resource diversity into robustness. Our proposed DynFed addresses this gap through two synergistic innovations: dynamic bit-width adaptation guided by local resource constraints and loss trajectories, and multi-teacher knowledge distillation that selectively transfers knowledge from optimal client-model pairs.

## 3.3 Overview of DynFed

The model training of proposed framework, DynFed, typically involves a certain number of rounds. The framework dynamically assigns quantization bit-widths to clients based on their memory capacity. It then adjusts these bit-widths according to the change in the loss function during training. For model aggregation, DynFed uses a dynamic multi-teacher knowledge distillation approach, where each sample is assigned the most suitable teacher model based on data similarity. As shown in Fig. 2, each round in DynFed consists of the following phases:

Figure 2: An overview of the training procedure proposed by DynFed.

<!-- image -->

(1) Dynamic bit-width allocation Strategy . Initially, clients are assigned quantization bit-widths based on their resources. This phase aims to balance the trade-off between communication efficiency and model performance. Bit-widths are dynamically adjusted during training to optimize gradient norms, ensuring that quantization does not degrade model performance significantly. (2) Local Training . Each client trains its local model using the assigned quantization bit-width. The local training process incorporates techniques to handle quantization-induced losses, ensuring that the local models remain robust and effective. (3) Model Aggregation . For model aggregation, DynFed employs a dynamic multi-teacher knowledge distillation approach. Each sample in the global model is assigned to the most suitable teacher model based on data similarity, ensuring effective knowledge transfer. Entropy smoothing is used to handle high entropy in logits due to data heterogeneity, selecting appropriate entropy levels and higher bit-width clients from similar datasets as teachers for distillation.

## 4 Method

## 4.1 Dynamic bit-width allocation Phase

In this phase, clients dynamically adjust their quantization bitwidths based on the client capacity and the change of loss function. This adaptive approach ensures that the quantization levels are optimal for each client's current training state, minimizing quantizationinduced losses and enhancing overall model performance.

Initialization of quantization bit-width allocation . Initially, the quantization bit-width is set based on the memory capabilities of each client, which can be estimated from factors such as processor performance and capacity size. The quantization process employs Quantized Stochastic Gradient Descent (QSGD) [2], ensuring efficiency while considering each client's resource constraints.

Let p = [ 𝑝 1 , 𝑝 2 , . . . , 𝑝 𝑛 ] denote the gradient vector to be quantized. Each client 𝑘 is characterized by a specific memory capacity denoted as c 𝑘 . The quantization level for each client, 𝑞 𝑘 , is determined by these capacity constraints, with the quantizer 𝑄 𝑞 𝑘

dividing the interval [ 0 , 1 ] into 𝑞 𝑘 bins. First, normalize the gradient vector p to ensure all elements fall within the range [ 0 , 1 ] , i.e., p norm = p ∥ p ∥ 2 , where ∥ p ∥ 2 represents the Euclidean norm of the gradient vector. Calculate the quantization level 𝑞 𝑘 based on the client's capacity:

$$
q _ { k } = \min \left ( c _ { \max } , \left \lfloor \frac { c _ { k } } { c _ { p } } \right \rfloor \right ) \quad ( 2 )
$$

where 𝑐 𝑚𝑎𝑥 is the maximum quantization level. c 𝑝 denotes the capacity required per quantization bin. Apply the quantizer 𝑄 𝑞 𝑘 to each element of the normalized gradient vector p norm:

$$
Q _ { q _ { k } } ( p _ { i } ) = s i g ( p _ { i } ) \cdot \left ( \frac { \lfloor q _ { k } | p _ { i } | \rfloor + \lfloor U n i ( 0 , 1 ) < ( q _ { k } | p _ { i } | - \lfloor q _ { k } | p _ { i } | ) \rfloor } { q _ { k } } \right ) \quad ( 3 )
$$

where I [·] is the indicator function that introduces stochasticity to the quantization process. This approach ensures that quantization is performed efficiently within each client's resource constraints, optimizing the federated learning process across heterogeneous client devices.

Dynamic adjustment of quantization bit-width . During training, the bit-widths are dynamically adjusted based on the gradient norms measured over a unit time interval. Clients with larger gradient norms are assigned higher bit-widths to capture more precise updates, while clients with smaller gradient norms are assigned lower bit-widths to reduce communication costs. The bit-width can be calculated by:

$$
b _ { i } ^ { ( t ) } = b _ { i } ^ { ( t - 1 ) } + \eta \cdot \left ( \frac { | \nabla F _ { i } ( w , t ) | } { \max ( | \nabla F _ { i } ( w , t ) | ) } - \frac { b _ { i } ^ { ( t - 1 ) } } { B _ { \max } } \right ) \quad ( 4 )
$$

where 𝜂 is the learning rate for bit-width adjustment and ∥∇ 𝐹 𝑖 ( 𝑤 )∥ is the gradient norm of the local objective function for client 𝑖 measured over a unit time interval.

After each training iteration, the quantization bit-width is adjusted based on the observed change in the loss function. This adaptive strategy dynamically tunes the bit-width according to the training state and resource characteristics of each client, thereby enhancing the effectiveness and efficiency of federated learning.

## 4.2 Model Aggregation Phase

During model aggregation, DynFed employs a multi-teacher knowledge distillation approach. Each sample is matched with the most suitable teacher model, based on data similarity, to ensure effective knowledge transfer. Entropy smoothing techniques are applied to manage high entropy in logits, selecting appropriate entropy levels and higher bit-width clients as teachers. In federated learning with heterogeneous quantization bit-widths across clients, aggregating model updates while preserving accuracy is challenging. To address this, we propose a method leveraging active learning-based knowledge distillation.

Active learning-based teacher selection . In this section, we describe our centralized active learning approach for federated learning, which involves uncertainty evaluation using Markov Chain Monte Carlo sampling (MCMC) sampling, client model evaluation with a comprehensive scoring model, diversity consideration, and knowledge distillation to aggregate client models into a global model.

4.2.1 Uncertainty evaluation. Given an unlabeled dataset 𝐷 𝑢 = { 𝑥 𝑖 } 𝑁 𝑖 = 1 , we use MCMC sampling to evaluate the uncertainty of each sample. For each sample 𝑥 𝑖 , the global model M 𝐺 produces a probability distribution over classes, 𝑝 ( 𝑦 | 𝑥 𝑖 , M 𝐺 ) . We perform 𝑀 Monte Carlo samples to obtain 𝑀 probability distributions { 𝑝 𝑚 ( 𝑦 | 𝑥 𝑖 , M 𝐺 )} 𝑀 𝑚 = 1 . The uncertainty 𝑈 ( 𝑥 𝑖 ) is then computed as the entropy of the average distribution:

$$
p ( y | x _ { i } , \mathcal { M } _ { G } ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } p _ { m } ( y | x _ { i } , \mathcal { M } _ { G } ) , \quad \ \ ( 5 ) \quad \ \
$$

$$
U ( x _ { i } ) = - \sum _ { c = 1 } ^ { C } p ( y = c | x _ { i } , \mathcal { M } _ { G } ) \log p ( y = c | x _ { i } , \mathcal { M } _ { G } ) , \quad ( 6 ) \quad \text {all } p \colon \quad \text {Algebra}
$$

where 𝐶 is the number of classes. We select the top 𝐾 models with lower entropy uncertainty for sample prediction.

4.2.2 Client model evaluation. For the 𝐾 selected models, the server loads each dataset sample 𝑥 𝑖 , 𝑖 = 1 , . . . , 𝑁 and performs inference to obtain the probability distributions { 𝑝 ( 𝑦 | 𝑥 𝑖 , M 𝑘 )} 𝐾 𝑘 = 1 . The comprehensive score 𝑆 𝑗𝑘 for client model 𝑗 on sample 𝑥 𝑘 is computed by combining quantization bit-width and confidence:

$$
S _ { i } ^ { k } = \alpha \cdot b ( x _ { i } , \mathcal { M } _ { k } ) + \beta \cdot \min p ( y | x _ { i } , \mathcal { M } _ { k } ) , \quad ( 7 )
$$

where 𝛼 and 𝛽 are weighting factors, and 𝑏 ( 𝑥 𝑖 , M 𝑘 ) is the quantization bit-width of M 𝑘 on a validation set similar to 𝑥 𝑖 .

4.2.3 Diversity consideration. To avoid bias towards any single model, we ensure diversity in model selection. For each sample 𝑥 𝑘 , we rank the clients based on their scores 𝑆 𝑘 𝑖 and select a diverse set of models. We employ a diversity penalty term 𝐷 𝑘 , which increases with the frequency of model 𝑘 being selected:

$$
S _ { i } ^ { k ^ { \prime } } = S _ { i } ^ { k } - \lambda D _ { k } ,
$$

where 𝜆 is a diversity weighting factor, the final client model for each sample is chosen to maximize 𝑆 𝑘 ′ 𝑖 .

## Knowledge distillation with quantization awareness .

Once the most suitable client models for each sample are selected, we use knowledge distillation to aggregate these models into a global model. Let M 𝑇 denote the set of selected client models, and M 𝐺 the global model to be trained. The soft labels from M 𝑇 are used as targets for training M 𝐺 . For a sample 𝑥 𝑖 , the soft label 𝜏 𝑖 is given by:

$$
\tau _ { i } = \frac { 1 } { | \mathcal { M } _ { T } | } \sum _ { \mathcal { M } _ { j } \in \mathcal { M } _ { T } } p ( y | x _ { i } , \mathcal { M } _ { j } ) , \quad \qu
$$

The distillation loss 𝐿 𝑑𝑖𝑠𝑡𝑖𝑙𝑙 for updating the global model M 𝐺 is defined as:

$$
\begin{aligned}
L _ { d i s t i l l } = - \sum _ { i = 1 } ^ { K } \sum _ { c = 1 } ^ { C } \tau _ { i } ( c ) \log p ( y = c | x _ { i } , \mathcal { M } _ { G } ) , \quad ( 1 0 )
\end{aligned}
$$

where 𝜏 𝑖 ( 𝑐 ) is the probability of class 𝑐 for sample 𝑥 𝑖 given by the average soft label. The final loss function 𝐿 for training M 𝐺 combines the distillation loss and a cross-entropy loss 𝐿 𝐶𝐸 on data 𝐷 𝑙 :

$$
L = \gamma L _ { d i s t i l l } + ( 1 - \gamma ) L _ { C E } ,
$$

where 𝛾 is a balancing factor. The global model M 𝐺 is updated to minimize this loss.

Note that in our approach, knowledge distillation is performed entirely on the server side. This strategy: (1) Reduces Client Burden: Clients focus on local training, avoiding additional computational tasks associated with distillation. (2) Centralizes Knowledge Integration: The server aggregates knowledge from all client models, leading to a more comprehensive global model. (3) Optimizes Resource Usage: The server's powerful hardware handles the complex distillation process more efficiently than client devices. The overall process of DynFed for Heterogeneous Clients is described in Algorithm 1.

## Algorithm 1 DynFed for Heterogeneous Clients

Require: Global training data D , client set K , memory capabilities 𝑐 𝑘 , total number of rounds 𝑇 .

- 1: Initialize local model parameters w 0 𝑘 for each client;
- 2: for t = 1 to T do
- 3: // Clients executes:
- 4: for each client 𝑘 ∈ K do
- 5: Compute quantization bit-width of clients based on Memory capabilities 𝑐 𝑘 by Eq. (3);
- 6: Train client model w 𝑘 on local data D 𝑘 ;
- 7: Update client model parameters w 𝑘 to w 𝑞,𝑡 + 1 𝑘 , using quantization with bitwidth adaptively determined by Eq. (4);
- 8: Compute gradients ∇ 𝐹 𝑘 ( w 𝑞,𝑡 + 1 𝑘 ) ;
- 9: Send ∇ 𝐹 𝑘 ( w 𝑞,𝑡 + 1 𝑘 ) to server;
- 10: end for
- 11: // Server executes:
- 12: for each sample x 𝑖 ∈ D do
- 13: Select the most suitable client model 𝑘 ∗ based on Eq. (8);
- 14: Distill knowledge from selected client 𝑘 ∗ to server model by Eq. (11).
- 15: end for
- 16: end for

## 4.3 Convergence Analysis

In this section, we present a theoretical analysis of the convergence properties of the DynFed framework, demonstrating that it effectively converges to a stationary point of the global objective function, even in heterogeneous federated learning environments. By dynamically adjusting quantization bit-widths and utilizing knowledge distillation for aggregation, we show that our approach remains robust despite the presence of quantization and distillation errors.

Let w ∗ denote the optimal global model parameters that minimize the global objective function 𝐹 ( w ) , which is defined as:

$$
F ( w ) = \sum _ { k = 1 } ^ { K } \frac { | \mathcal { D } _ { k } | } { \sum _ { j = 1 } ^ { K } | \mathcal { D } _ { j } | } F _ { k } ( w )
$$

where 𝐹 𝑘 ( w ) = 1 | D 𝑘 | ˝ 𝑖 ∈D 𝑘 ℓ ( w ; 𝑥 𝑖 , 𝑦 𝑖 ) is the local objective function for client 𝑘 .

Assumptions . To analyze the convergence, we make the following standard assumptions:

1. Smoothness: The local objective functions 𝐹 𝑘 ( w ) are 𝐿 -smooth, i.e., ∇ 𝐹 𝑘 ( w 1 ) - ∇ 𝐹 𝑘 ( w 2 )∥ ≤ 𝐿 ∥ w 1 -w 2 ∥ , for any w 1 , w 2 .

2. Bounded Variance: The variance of the stochastic gradients is bounded, i.e., E ∥∇ 𝐹 𝑘 ( w ; 𝜉 ) - ∇ 𝐹 𝑘 ( w )∥ 2 ≤ 𝜎 2 , for any 𝑘, w , where 𝜉 denotes the stochasticity in the gradient due to sampling.

3. Bounded Quantization and distillation errors: The quantization error is bounded by 𝜖 𝑞 , i.e., ∥ w 𝑞 𝑘 -w 𝑘 ∥ ≤ 𝜖 𝑞 , the distillation error at the server is bounded by 𝜖 𝑑 , i.e., ∥ w d -w agg ∥ ≤ 𝜖 𝑑 , where w d is the model after distillation, and w agg is the aggregated model before distillation.

Theorem 1 . Given the assumptions above, we can derive the following convergence result for the DynFed framework,Under the assumptions of smoothness, bounded variance, and bounded quantization error, the global model w in DynFed converges to a stationary point of the global objective function 𝐹 ( w ) . Specifically, after 𝑇 rounds of training, we have:

$$
\mathbb { E } [ F ( w _ { T } ) - F ( w ^ { * } ) ] \leq \frac { L \| w _ { 0 } - w ^ { * } \| ^ { 2 } } { 2 T \eta } + \frac { \eta L \sigma ^ { 2 } } { 2 } + L \epsilon _ { q } + L \epsilon _ { d } \quad ( 1 3 )
$$

where 𝜂 is the learning rate, w 0 is the initial global model, and w 𝑇 is the global model after 𝑇 rounds.

Proof . Local Update Analysis . Each client 𝑘 performs local updates using SGD on its quantized model w 𝑞 𝑘 . The local update rule is:

$$
w _ { k } ^ { q , t + 1 } = w _ { k } ^ { q , t } - \eta \nabla F _ { k } ( w _ { k } ^ { q , t } ; \xi _ { k } ^ { t } ) , \quad \quad ( 1 4 )
$$

where 𝜉 𝑡 𝑘 represents the stochastic gradient at iteration 𝑡 for client 𝑘 .

The quantization error introduces an additional term:

$$
w _ { k } ^ { q , t + 1 } = w _ { k } ^ { t } - \eta \nabla F _ { k } ( w _ { k } ^ { t } ; \xi _ { k } ^ { t } ) + q _ { k } ^ { t } ,
$$

where q 𝑡 𝑘 = w 𝑞,𝑡 𝑘 -w 𝑡 𝑘 represents the quantization error.

Error Bound for Local Updates . Using the smoothness assumption, we can bound the difference between the objective function values before and after the local update:

$$
\begin{aligned}
F _ { k } ( w _ { k } ^ { q , t + 1 } ) & \leq F _ { k } ( w _ { k } ^ { t } ) + \langle \nabla F _ { k } ( w _ { k } ^ { t } ) , w _ { k } ^ { q , t + 1 } - w _ { k } ^ { t } \rangle \\ & + \frac { L } { 2 } \| w _ { k } ^ { q , t + 1 } - w _ { k } ^ { t } \| ^ { 2 } .
\end{aligned}
$$

s u b t i t u t i n g t h o w d e t o r u l y w e g t a t

$$
\begin{aligned}
& \quad \ \end{aligned}
$$

Substituting the update rule, we get:

$$
\begin{aligned}
F _ { k } ( w _ { k } ^ { q , t + 1 } ) & \leq F _ { k } ( w _ { k } ^ { t } ) - \eta \| \nabla F _ { k } ( w _ { k } ^ { t } ) \| ^ { 2 } \\ & + \eta \langle \nabla F _ { k } ( w _ { k } ^ { t } ) , q _ { k } ^ { t } \rangle \\ & + \frac { L \eta ^ { 2 } } { 2 } \| \nabla F _ { k } ( w _ { k } ^ { t } ) \| ^ { 2 } + \frac { L } { 2 } \| q _ { k } ^ { t } \| ^ { 2 } . \\ \text {Taking the expectation over the stochastic gradients and using} & \quad \text {gen}
\end{aligned}
$$

Taking the expectation over the stochastic gradients and using the bounded variance assumption, we have:

$$
\begin{aligned}
\mathbb { B } [ F _ { k } ( w _ { k } ^ { q , t + 1 } ) ] & \leq \mathbb { E } [ F _ { k } ( w _ { k } ^ { t } ) ] - \eta \left ( 1 - \frac { L \eta } { 2 } \right ) \| \nabla F _ { k } ( w _ { k } ^ { t } ) \| ^ { 2 } \\ & + \eta \mathbb { E } \left [ \langle \nabla F _ { k } ( w _ { k } ^ { t } ) , \mathbf q _ { k } ^ { t } \rangle \right ] + \frac { L } { 2 } \epsilon _ { q } ^ { 2 } .
\end{aligned}
$$

Global Model Aggregation . After each round, the server aggregates the local models from the clients:

$$
w ^ { t + 1 } = \sum _ { k = 1 } ^ { K } \frac { | \mathcal { D } _ { k } | } { \sum _ { j = 1 } ^ { K } | \mathcal { D } _ { j } | } w _ { k } ^ { q , t + 1 } .
$$

Next, the server performs knowledge distillation w 𝑡 + 1 = w d ,𝑡 + 1 . . Due to the distillation error, we have:

$$
\| w ^ { t + 1 } - w ^ { a g g , t + 1 } \| \leq \epsilon _ { d } .
$$

Using the smoothness of the global objective function, we can bound the difference between 𝐹 ( w 𝑡 + 1 ) and 𝐹 ( w agg ,𝑡 + 1 ) :

$$
\begin{array} { c c } \det & F ( w ^ { t + 1 } ) \leq F ( w ^ { a g g , t + 1 } ) + \langle \nabla F ( w ^ { a g g , t + 1 } ) , w ^ { t + 1 } - w ^ { a g g , t + 1 } \rangle \\ \\ & + \frac { L } { 2 } \| w ^ { t + 1 } - w ^ { a g g , t + 1 } \| ^ { 2 } . \end{array} \ ( 2 1 ) \\
$$

$$
^ { 2 }
$$

$$
\begin{aligned}
+ \frac { t } { 2 } \| w ^ { t + 1 } - w ^ { a g g , t + 1 } \| ^ { 2 } . \\ \text {Since } \| w ^ { t + 1 } - w ^ { a g g , t + 1 } \| \leq \epsilon _ { d } , \, w e \, get \colon \\ F ( w ^ { t + 1 } ) \leq F ( w ^ { a g g , t + 1 } ) + \frac { L } { 2 } \epsilon _ { d } ^ { 2 } .
\end{aligned}
$$

Combining Errors . Combining the error bounds from the local updates and the global aggregation, and summing over 𝑇 rounds, we obtain:

$$
^ { w _ { T } } _ { \log _ { 0 } } \mathbb { E } [ F ( w _ { T } ) ] - F ( w ^ { * } ) \leq \frac { L \| w _ { 0 } - w ^ { * } \| ^ { 2 } } { 2 T \eta } + \frac { \eta L \sigma ^ { 2 } } { 2 } + L \epsilon _ { q } + L \epsilon _ { d } . \ \ ( 2 2 )
$$

The convergence result indicates that the DynFed framework achieves convergence to a stationary point of the global objective function despite the quantization errors. The rate of convergence is influenced by the learning rate 𝜂 , the variance of the stochastic gradients 𝜎 2 , and the quantization error 𝜖 𝑞 . By dynamically adjusting the quantization bit-widths based on gradient norms, DynFed minimizes the quantization error, thereby improving the convergence rate and overall model performance.

## 5 EXPERIMENTS

## 5.1 Experimental Settings

Models and Datasets In our experimental setup, we use the following datasets in our experiments:(1) FMNIST [38]: a dataset of 70,000 grayscale images of fashion items, consisting of 60,000 training images and 10,000 test images. Each image is 28x28 pixels and belongs to one of 10 classes, (2) CIFAR-10 [19]: A dataset of 60,000 32x32 color images in 10 classes, with 50,000 training images and 10,000 test images. (3) CIFAR-100: Similar to CIFAR-10, but with 100 classes containing 600 images each. There are 50,000 training images and 10,000 test images. We used ResNet10 [8] as the base model for the FMNIST dataset, MobileNet [30] for the CIFAR-10 dataset, and GoogLeNet [33] for the CIFAR-100 dataset.

We employed a Dirichlet distribution to simulate data heterogeneity among clients, with Dirichlet coefficients set to 0.1, 1, and 10. These coefficients control the level of data distribution skewness among clients, with 0.1 representing a highly imbalanced distribution, 1 indicating a moderately imbalanced distribution, and 10 corresponding to a more uniform distribution. For the public datasets, we selected 200 unlabeled samples that do not overlap with the client samples.

Table 1: Performance comparison on FMNIST, CIFAR-10 and CIFAR-100 datasets. The percentages represent the level of quantization applied. For example, DynFed (10%) indicates that 90% of the clients maintain a lower quantization precision, while 10% of the clients use a higher quantization precision.

| Method       |   FMNIST |   FMNIST |   FMNIST |   CIFAR-10 |   CIFAR-10 |   CIFAR-10 |   CIFAR-100 |   CIFAR-100 |   CIFAR-100 |
|--------------|----------|----------|----------|------------|------------|------------|-------------|-------------|-------------|
| Method       |      0.1 |        1 |       10 |        0.1 |          1 |         10 |         0.1 |           1 |          10 |
| FedAvg       |    75.04 |    79.32 |    83.73 |      62.24 |      76.03 |      80.11 |       49.21 |       59.53 |       60.81 |
| FedProx      |    82.31 |    86.04 |    90.56 |      69.67 |      78.54 |      82.31 |       57.23 |       62.41 |       67.09 |
| FedKD        |    84.67 |    88.22 |    91.86 |      76.51 |      80.23 |      84.26 |       60.31 |       65.23 |       70.32 |
| FedPAQ       |    78.56 |    82.09 |    86.31 |      69.34 |      73.41 |      78.39 |       52.01 |       57.19 |       62.46 |
| DAdaQuant    |    83.74 |    86.31 |    89.46 |      74.14 |      78.07 |      82.87 |       58.08 |       63.08 |       68.09 |
| DynFed(10%)  |    79.83 |    83.34 |    87.12 |      71.33 |      75.05 |      79.21 |       54.41 |       60.37 |       64.59 |
| DynFed(50%)  |    82.02 |    86.07 |    89.08 |      74.29 |      78.34 |      82.31 |       58.25 |       63.41 |       68.31 |
| DynFed (90%) |    84.34 |    87.21 |    90.01 |      76.03 |      80.21 |      84.14 |       59.18 |       64.25 |       70.08 |

Training Details All experiments were conducted using PyTorch on NVIDIA GeForce RTX 3090 GPUs. The default experimental configuration includes 100 rounds with 20 clients. The local batch size is set to 10, with client memory sizes randomly generated within the range of (2048, 16384). The local learning rate is set to 0.005, with a decay factor of 0.99.

Baselines We compare DynFed with several baseline methods, including traditional federated learning approaches, knowledge distillation methods, and other quantization-aware techniques. The baseline methods are as follows:

- FedAvg [26]: the classic FL algorithm that performs weighted averaging of model updates from different clients to obtain a global model.
- FedProx [22]: an extension of FedAvg that introduces a proximal term to the local objective function on each client.
- FedKD [36]: a federated learning approach that integrates knowledge distillation. In FedKD, selected clients act as teacher models, distilling their knowledge into a global student model.
- FedPAQ [29]: FedPAQ combines partial model averaging with quantization techniques to enhance communication efficiency. In FedPAQ, clients selectively share only a subset of their model parameters, which are then quantized before being transmitted to the server. The quantization levels is set to be 8 bit.
- DAdaQuant [11]: a dynamic and adaptive quantization algorithm that adjusts the quantization bit-widths of model updates based on time training dynamics.

## 5.2 Results

Table 1 shows the comparative results. The results indicate that DynFed consistently outperforms other methods across all datasets and Dirichlet coefficients. Compared to traditional FL methods like FedAvg and FedProx, which struggle under high data heterogeneity, DynFed demonstrates improvements, particularly in handling imbalanced distributions. Methods incorporating knowledge distillation, such as FedKD, show enhanced performance over basic approaches, especially under more skewed data distributions. However, DynFed further surpasses these methods on three datasets, showcasing its robustness and effectiveness in mitigating the challenges posed by both data and resource heterogeneity. In result, DynFed's ability to dynamically adjust quantization bit-widths and integrate knowledge distillation enables it to deliver superior performance across diverse federated learning scenarios, proving less sensitive to the degree of data heterogeneity than other approaches.

Table 2: Ablation study of major components in DynFed.

| RQ   | AQ   | ATS   |   Acc. | Overhead   | Totel Time   |
|------|------|-------|--------|------------|--------------|
| ×    | ×    | ×     |  81.87 | 20.625 G   | 738 s        |
| ✓    | ×    | ×     |  78.71 | 5.228 G    | 517 s        |
| ✓    | ✓    | ×     |  79.83 | 4.028 G    | 554 s        |
| ✓    | ✓    | ✓     |  82.39 | 5.656 G    | 738 s        |

Ablation study . We conduct an ablation study to assess the impact of three essential components in our federated learning algorithm: resource-aware quantization (RQ), adaptive quantization based on loss dynamics (AQ), and Active Teacher Selection (ATS) in server-side distillation. The ablation study is carried out on CIFAR10 using models MobileNet. The baseline without any component reverts to a standard federated learning approach that simply aggregates client models on the server. Table 2 illustrates that each component-RQ, AQ, and ATS-contributes uniquely to balancing the trade-offs between storage efficiency, computation time, and model accuracy in federated learning. RQ is highly effective in reducing storage overhead but at the expense of accuracy. AQ helps recover some accuracy while maintaining efficiency, and ATS is critical for achieving the highest accuracy, albeit with increased storage and time costs. The combination of all three components yields the best overall performance, with improved accuracy and moderate storage overhead, making it a promising approach for optimizing federated learning in heterogeneous environments.

Comparison of storage and time cost under different algorithms . Fig. 3a compares storage and total time overhead under different algorithms. FedPAQ and DAdaQuant show the lowest storage overhead, with DAdaQuant being the most efficient due to its adaptive quantization. DynFed also achieves low storage overhead.

<!-- image -->

<!-- image -->

- (b) Accuracy vs. client heterogeneity

<!-- image -->

(c)

Adaptive quantization

memory

(d) Teacher selection strategies

<!-- image -->

Figure 3: Performance evaluation of DynFed and comparison of training strategies.

In terms of total time, DAdaQuant is the fastest, followed by FedPAQ, while DynFed is slightly longer but still efficient. FedProx and FedKD, though requiring more communication, are better suited for scenarios prioritizing model accuracy. Overall, DynFed effectively balances storage and time, making it suitable for environments with varying resource constraints.

Comparison of Accuracy Across Varying Levels of Client Heterogeneity . This experiment investigated the impact of client heterogeneity on the accuracy of federated learning for image classification, specifically using the CIFAR-100 dataset and the GoogLeNet. When the Dirichlet coefficients was set to 0.1, we considered scenarios with 𝐸 = 2, 𝐸 = 4, and 𝐸 = 8, corresponding to low, medium, and high proportions of clients with limited memory resources, respectively, across 20 clients. The results, as illustrated in Fig. 3b, indicate that as the value of E increases, the model's accuracy is noticeably affected. When 𝐸 = 2, the accuracy increased rapidly with the number of training rounds and stabilized at a high level, demonstrating strong global model performance when most clients have sufficient resources. For 𝐸 = 8, despite slower accuracy improvement and greater fluctuations, the model still achieved a satisfactory level of convergence. This suggests that the proposed method can effectively address client heterogeneity to some extent. Overall, the experiment demonstrates that increasing E exacerbates the negative impact of heterogeneity on accuracy and stability in federated learning. However, the proposed method shows considerable adaptability in mitigating these challenges, proving its effectiveness in handling client heterogeneity.

Adaptive Quantization Bit-Width Allocation . To evaluate the effectiveness of the proposed dynamic quantization strategy, we visualize the relationship between client memory resources and the allocated quantization bit-widths, as shown in Fig. 3c. Each point in the plot represents an individual client, with memory resources ranging from 2,048 MB to 16,384 MB. The assigned bit-widths are positively correlated with available resources, indicating that clients with higher memory capacity are assigned higher bit-widths for improved model expressiveness. Notably, the fitted polynomial trend line reveals a clear nonlinear allocation pattern, reflecting the adaptive nature of our method. Unlike fixed-bit or random quantization strategies, which fail to leverage client heterogeneity, our approach dynamically adjusts quantization precision according to client-side resource constraints. This ensures that resource-limited clients are not overloaded while maximizing model quality on high-resource nodes.

Comparison of Teacher Selection Strategies in Federated Distillation To validate the effectiveness of the proposed Active Teacher Selection (ATS) in server-side distillation, we compare it with three commonly used teacher selection baselines: Random-KD, TopLoss-KD (selecting clients with lowest local loss), and All-KD (using all clients as teachers). As shown in Fig.3d, DynFed (w/ ATS) achieves the highest global accuracy of 81.3%, significantly outperforming TopLoss-KD (78.5%) and All-KD (79.0%). This demonstrates that uncertainty-based teacher selection leads to more effective global knowledge transfer. In terms of distillation quality, ATS yields the lowest soft label MSE (0.021), indicating superior generalization and consistency of teacher guidance on unseen data. Notably, ATS also excels in distillation efficiency: it achieves optimal performance using only 20% of clients as teachers, reducing training time and communication overhead. In contrast, All-KD involves all clients and incurs significantly higher cost (95s training time). As a result, our ATS method strikes an balance between effectiveness and efficiency, making it suitable for real-world federated learning applications under resource constraints.

## 6 Conclusion

In this paper, we introduced DynFed, a novel approach designed to address the challenges of federated learning with heterogeneous clients. DynFed dynamically adjusts quantization bit-widths based on client resources and local loss, thereby enhancing both model efficiency and accuracy. We also proposed a multi-teacher knowledge distillation strategy for model aggregation, which effectively manages heterogeneous bit-widths and boosts the performance of the global model. Additionally, by incorporating active learning to select the most appropriate teacher models, we optimized knowledge transfer and improved the quality of model aggregation in federated learning. Experiments and comprehensive ablation analysis have demonstrated the superior performance of DynFed in heterogeneous environments.

## Acknowledgments

This work is supported by the National Key Research and Development Program of China No.2022ZD0115903, the China Postdoctoral Science Foundation No.2024M751689, the National Natural Science Foundation of China (NSFC, No. 62172038, No. 62472028), in part by Beijing Natural Science Foundation (No. 4232033), and the Beijing Key Laboratory of Networked Multimedia.

by

## References

- [1] Ahmed M Abdelmoniem and Marco Canini. 2021. Towards mitigating device heterogeneity in federated learning via adaptive model quantization. In Proceedings of the 1st Workshop on Machine Learning and Systems . 96-103.
- [2] Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic. 2017. QSGD: Communication-efficient SGD via gradient quantization and encoding. Advances in neural information processing systems 30 (2017).
- [3] Jinyu Cai, Yunhe Zhang, Jicong Fan, and See-Kiong Ng. 2024. Lg-fgad: an effective federated graph anomaly detection framework. In Proceedings of the International Joint Conference on Artificial Intelligence .
- [4] Jinqian Chen, Haoyu Tang, Junhao Cheng, Ming Yan, Ji Zhang, Mingzhu Xu, Yupeng Hu, and Liqiang Nie. 2024. Breaking barriers of system heterogeneity: straggler-tolerant multimodal federated learning via knowledge distillation. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI) .
- [5] Yutong Dai, Zeyuan Chen, Junnan Li, Shelby Heinecke, Lichao Sun, and Ran Xu. 2023. Tackling data heterogeneity in federated learning with class prototypes. In AAAI , Vol. 37. 7314-7322.
- [6] Zixiang Ding, Guoqing Jiang, Shuai Zhang, Lin Guo, and Wei Lin. 2023. SKDBERT: Compressing BERT via stochastic knowledge distillation. In AAAI , Vol. 37. 74147422.
- [7] Xiuwen Fang, Mang Ye, and Xiyuan Yang. 2023. Robust heterogeneous federated learning under data corruption. In Proceedings of the IEEE/CVF International Conference on Computer Vision . 5020-5030.
- [8] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition . 770-778.
- [9] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531 (2015).
- [10] Junyuan Hong, Haotao Wang, Zhangyang Wang, and Jiayu Zhou. 2022. Efficient Split-Mix Federated Learning for On-Demand and In-Situ Customization. In ICLR .
- [11] Robert Hönig, Yiren Zhao, and Robert Mullins. 2022. DAdaQuant: Doublyadaptive quantization for communication-efficient federated learning. In International Conference on Machine Learning . PMLR, 8852-8866.
- [12] Robert Hönig, Yiren Zhao, and Robert Mullins. 2022. DAdaQuant: Doublyadaptive quantization for communication-efficient Federated Learning. In Proceedings of the 39th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 162) , Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (Eds.). PMLR, 8852-8866. https://proceedings.mlr.press/v162/honig22a.html
- [13] Xiaopeng Jiang and Cristian Borcea. 2023. Complement sparsification: Lowoverhead model pruning for federated learning. In AAAI , Vol. 37. 8087-8095.
- [14] Zhida Jiang, Yang Xu, Hongli Xu, Zhiyuan Wang, Jianchun Liu, Qian Chen, and Chunming Qiao. 2023. Computation and communication efficient federated learning with adaptive model pruning. IEEE Transactions on Mobile Computing 23, 3 (2023), 2003-2021.
- [15] Zhida Jiang, Yang Xu, Hongli Xu, Zhiyuan Wang, and Chen Qian. 2023. Heterogeneity-Aware Federated Learning with Adaptive Client Selection and Gradient Compression. In IEEE INFOCOM 2023-IEEE Conference on Computer Communications . IEEE, 1-10.
- [16] Zhida Jiang, Yang Xu, Hongli Xu, Zhiyuan Wang, Chunming Qiao, and Yangming Zhao. 2022. Fedmp: Federated learning through adaptive model pruning in heterogeneous edge computing. In ICDE . IEEE, 767-779.
- [17] Zhu JianHao, Changze Lv, Xiaohua Wang, Muling Wu, Wenhao Liu, Tianlong Li, Zixuan Ling, Cenyuan Zhang, Xiaoqing Zheng, and Xuan-Jing Huang. 2024. Promoting Data and Model Privacy in Federated Learning through Quantized LoRA. In Findings of the Association for Computational Linguistics: EMNLP 2024 . 10501-10512.
- [18] Young Geun Kim and Carole-Jean Wu. 2021. Autofl: Enabling heterogeneityaware energy efficient federated learning. In MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture . 183-198.
- [19] Alex Krizhevsky, Geoffrey Hinton, et al. 2009. Learning multiple layers of features from tiny images. (2009).
- [20] Sunwoo Lee, Tuo Zhang, and A Salman Avestimehr. 2023. Layer-wise adaptive model aggregation for scalable federated learning. In AAAI , Vol. 37. 8491-8499.
- [21] Ang Li, Jingwei Sun, Pengcheng Li, Yu Pu, Hai Li, and Yiran Chen. 2021. Hermes: an efficient federated learning framework for heterogeneous mobile clients. In Proceedings of the 27th Annual International Conference on Mobile Computing and Networking . 420-437.
- [22] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. 2020. Federated optimization in heterogeneous networks. Proceedings of Machine learning and systems 2 (2020), 429-450.
- [23] Heting Liu, Fang He, and Guohong Cao. 2023. Communication-Efficient Federated Learning for Heterogeneous Edge Devices Based on Adaptive Gradient Quantization. In IEEE INFOCOM . 1-10.
- [24] Mi Luo, Fei Chen, Dapeng Hu, Yifan Zhang, Jian Liang, and Jiashi Feng. 2021. No fear of heterogeneity: Classifier calibration for federated learning with non-iid
25. data. Advances in Neural Information Processing Systems 34 (2021), 5972-5984.
- [25] Othmane Marfoq, Giovanni Neglia, Richard Vidal, and Laetitia Kameni. 2022. Personalized federated learning through local memorization. In International Conference on Machine Learning . PMLR, 15070-15092.
- [26] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. 2017. Communication-Efficient Learning of Deep Networks from Decentralized Data. In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (Proceedings of Machine Learning Research, Vol. 54) , Aarti Singh and Jerry Zhu (Eds.). PMLR, 1273-1282.
- [27] Matias Mendieta, Taojiannan Yang, Pu Wang, Minwoo Lee, Zhengming Ding, and Chen Chen. 2022. Local learning matters: Rethinking data heterogeneity in federated learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition . 8397-8406.
- [28] Zhaopeng Peng, Xiaoliang Fan, Yufan Chen, Zheng Wang, Shirui Pan, Chenglu Wen, Ruisheng Zhang, and Cheng Wang. 2024. FedPFT: federated proxy finetuning of foundation models. arXiv preprint arXiv:2404.11536 (2024).
- [29] Amirhossein Reisizadeh, Aryan Mokhtari, Hamed Hassani, Ali Jadbabaie, and Ramtin Pedarsani. 2020. Fedpaq: A communication-efficient federated learning method with periodic averaging and quantization. In International conference on artificial intelligence and statistics . PMLR, 2021-2031.
- [30] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and LiangChieh Chen. 2018. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition . 4510-4520.
- [31] Xiaoxin Su, Yipeng Zhou, Laizhong Cui, and Jiangchuan Liu. 2023. On model transmission strategies in federated learning with lossy communications. IEEE Transactions on Parallel and Distributed Systems 34, 4 (2023), 1173-1185.
- [32] Wen Sun, Yong Zhao, Wenqiang Ma, Bin Guo, Lexi Xu, and Trung Q Duong. 2023. Accelerating convergence of federated learning in MEC with dynamic community. IEEE Transactions on Mobile Computing 23, 2 (2023), 1769-1784.
- [33] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. 2015. Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition . 1-9.
- [34] Zhenheng Tang, Yonggang Zhang, Shaohuai Shi, Xin He, Bo Han, and Xiaowen Chu. 2022. Virtual homogeneity learning: Defending against data heterogeneity in federated learning. In International Conference on Machine Learning . PMLR, 21111-21132.
- [35] Boxin Wang, Yibo Jacky Zhang, Yuan Cao, Bo Li, H Brendan McMahan, Sewoong Oh, Zheng Xu, and Manzil Zaheer. 2023. Can public large language models help private cross-device federated learning? arXiv preprint arXiv:2305.12132 (2023).
- [36] Chuhan Wu, Fangzhao Wu, Lingjuan Lyu, Yongfeng Huang, and Xing Xie. 2022. Communication-efficient federated learning via knowledge distillation. Nature communications 13, 1 (2022), 2032.
- [37] Tengxi Xia, Ju Ren, Wei Rao, Qin Zu, Wenjie Wang, Shuai Chen, and Yaoxue Zhang. 2024. Aerorec: an efficient on-device recommendation framework using federated self-supervised knowledge distillation. In IEEE INFOCOM 2024-IEEE Conference on Computer Communications . IEEE, 121-130.
- [38] Han Xiao, Kashif Rasul, and Roland Vollgraf. 2017. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747 (2017).
- [39] Ronghui Xu, Hao Miao, Senzhang Wang, Philip S Yu, and Jianxin Wang. 2024. PeFAD: a parameter-efficient federated framework for time series anomaly detection. In Proceedings of the 30th ACM SIGKDD conference on knowledge discovery and data mining . 3621-3632.
- [40] Zirui Xu, Fuxun Yu, Jinjun Xiong, and Xiang Chen. 2021. Helios: Heterogeneityaware federated learning with dynamically balanced collaboration. In 2021 58th ACM/IEEE Design Automation Conference (DAC) . IEEE, 997-1002.
- [41] Zhiqin Yang, Yonggang Zhang, Yu Zheng, Xinmei Tian, Hao Peng, Tongliang Liu, and Bo Han. 2024. FedFed: Feature distillation against data heterogeneity in federated learning. Advances in Neural Information Processing Systems 36 (2024).
- [42] Abbas Yazdinejad, Ali Dehghantanha, Hadis Karimipour, Gautam Srivastava, and Reza M Parizi. 2024. A robust privacy-preserving federated learning model against model poisoning attacks. IEEE Transactions on Information Forensics and Security (2024).
- [43] Rui Ye, Mingkai Xu, Jianyu Wang, Chenxin Xu, Siheng Chen, and Yanfeng Wang. 2023. Feddisco: Federated learning with discrepancy-aware collaboration. In International Conference on Machine Learning . PMLR, 39879-39902.
- [44] Feilong Zhang, Xianming Liu, Shiyi Lin, Gang Wu, Xiong Zhou, Junjun Jiang, and Xiangyang Ji. 2023. No one idles: Efficient heterogeneous federated learning with parallel edge and server computation. In International Conference on Machine Learning . PMLR, 41399-41413.
- [45] Yinlin Zhu, Xunkai Li, Zhengyu Wu, Di Wu, Miao Hu, and Rong-Hua Li. 2024. FedTAD: topology-aware data-free knowledge distillation for subgraph federated learning. arXiv preprint arXiv:2404.14061 (2024).
- [46] Zhuangdi Zhu, Junyuan Hong, and Jiayu Zhou. 2021. Data-free knowledge distillation for heterogeneous federated learning. In International conference on machine learning . PMLR, 12878-12889.