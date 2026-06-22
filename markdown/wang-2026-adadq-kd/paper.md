## AdaDQ-KD: An Adaptive Dithering Quantization with Knowledge Distillation in Privacy-Preserving Federated Learning

Gang Wang, Member, IEEE , Qi Qi, Haomiao Yu, Rui Han, Member, IEEE , Lin Bai, Senior Member, IEEE , and Jinho Choi, Fellow, IEEE

Abstract -Dithering quantization (DQ) is a promising Differential Privacy (DP) approach designed for Federated Learning (FL) to prevent privacy leakage of clients. However, DQ-FL inevitably exacerbates the decline in model accuracy in FL, particularly under conditions characterized by straggler effects and statistical heterogeneity across clients. In this paper, we propose a novel Adaptive Dithering Quantization with Knowledge Distillation (AdaDQ-KD) algorithm to improve the training efficiency within a privacy-preserving FL framework. Specifically, an adaptive dithering quantization scheme is developed, which dynamically tunes the dithering quantization intervals to decrease communication overhead based on the clients' local delays. Moreover, we introduce a novel training strategy through a knowledge distillation module, which integrates a feature distillation loss into the training loss function to mitigate the accuracy loss induced by noise injection and gradient quantization. Theoretical analysis is conducted to evaluate the privacy guarantees, efficiency and convergence properties of the proposed AdaDQ-KD algorithm. Experimental results demonstrate that the AdaDQ-KD algorithm outperforms state-of-the-art (SOTA) methods across various settings, achieving a better trade-off among privacy, training efficiency, and model accuracy.

Index Terms -Federated Learning, Differential Privacy, Nonindependent identical distribution (Non-IID), Knowledge Distillation

## I. INTRODUCTION

F EDERATED Learning (FL) [1] is a distributed learning paradigm that enables multiple parties to collaboratively train models while keeping data localized, gaining increasing attention due to data scale expansion and privacy concerns. In a typical FL scenario, a parameter server (PS) coordinates a group of participating clients to collaboratively train a common model. Clients upload just updated local parameters instead of raw data for model aggregating in the server's side. In this way, the model can be trained without direct access to sensitive local data among clients. Thereby the privacy of client can be protected.

Received 9 October 2025; revised 20 December 2025 and 12 March 2025. Gang Wang is with the State Key Laboratory of CNS/ATM, the School of Cyber Science and Technology, Beihang University, Beijing 100191, China (e-mail: gwang@buaa.edu.cn).

Qi Qi is with the School of Electronics and Information Engineering, Beihang University, Beijing 100191, China (e-mail: qi qi@buaa.edu.cn).

Haomiao Yu, Rui Han, and Lin Bai are with the School of Cyber Science and Technology, Beihang University, Beijing 100191, China (e-mail: Haomiao@buaa.edu.cn; ruihan@buaa.edu.cn; l.bai@buaa.edu.cn).

Jinho Choi is with the School of Electrical and Mechanical Engineering, The University of Adelaide, Adelaide, SA 5005, Australia (e-mail: jinho.choi@adelaide.edu.au).

It has been shown that even simple model updates can leak substantial private information under appropriate attack strategies [2]-[4]. A representative example is the gradient reconstruction attack [5], where an adversary iteratively optimizes dummy inputs such that their induced gradients match the genuine gradients uploaded by a client, thereby enabling the recovery of original training samples. This demonstrates that shared gradients may reveal information about local data. To mitigate such risks, DP has been widely adopted in federated learning [6], [7]. DP-based FL bounds gradient sensitivity via clipping and injects calibrated noise before transmission, ensuring that the contribution of any individual sample has a negligible influence on the aggregated updates and preventing reliable inference of private information. However, the noise introduced by DP can degrade model performance, especially in communication-constrained settings. To alleviate this issue, DQ [8] has been proposed to reduce performance loss while preserving privacy guarantees. When applying DQ to FL, known as FL-DQ, it is of vital importance to solve the challenge the stragglers and heterogeneity of clients of FL training in practice, which has always been a hotpot in FLDQ.

Due to the distributed nature of FL, participating clients are often resource-constrained, making training efficiency a critical issue, especially in DP-enabled FL systems. Meanwhile, modern models are typically large-scale and overparameterized, resulting in significant computation and communication overhead [9]. Consequently, extensive efforts have focused on gradient sparsification and quantization to exploit model redundancy and reduce training costs [10]-[14]. Beyond system constraints, client heterogeneity further degrades FL efficiency. Variations in computation, communication, and data distributions often lead to imbalanced training latency, where stragglers can delay synchronous updates and cause substantial idle waiting. Therefore, works on straggler tolerance strategies [15]-[19] are also advantageous to enhancement of FL training efficiency. Furthermore, client selection and allocation of communication resources [20]-[22] are primary means for improving communication performance as well.

In regulated domains involving sensitive data, such as healthcare and finance, stringent privacy regulations necessitate the use of privacy-preserving techniques such as DP-FL. However, when DP-FL is deployed on resource-constrained devices like smartphones and edge devices, it faces a significant communication bottleneck, as communication latency often dominates local computation time. This challenge is further exacerbated by both statistic and system heterogeneity across clients, including differences in local datasets, computational resources, and network conditions, thereby reducing training efficiency and slowing convergence. These challenges jointly call for an algorithm that can simultaneously reduce communication overhead, protect sensitive data, and remain robust to heterogeneity in FL.

To overcome these limitations, we propose a DQ-FL algorithm that jointly enhances training efficiency and model utility by integrating adaptive dithering-based gradient quantization and knowledge distillation (KD). The proposed method uses dithering quantization (DQ) for both privacy assurance and communication compression, adaptively allocates communication loads to mitigate straggler effects, and incorporates KD to improve model utility under data heterogeneity.

Our main contributions are as follows:

- We propose an adaptive DQ with KD (AdaDQ-KD) algorithm, which utilizes adaptive DQ to achieve model compression according to the status of clients, with its compression error providing DP guarantee exactly. Furthermore, KD is introduced to pre-tune the gradients, thereby alleviating the adverse effects induced by quantization and noise injection.
- A rigorous theoretical analysis of the proposed AdaDQKD algorithm is presented, where a precision coefficient is introduced into the DQ to explicitly characterize the relationship between communication overhead and DP noise amplitude under privacy constraints. The privacy guarantee, communication efficiency, and convergence properties of the algorithm are systematically analyzed.
- Extensive simulation results are provided to validate the effectiveness of the proposed AdaDQ-KD algorithm and benchmark it against well-known schemes. The results demonstrate significant performance gains in training efficiency and model utility, indicating that the proposed method achieves a favorable trade-off among privacy protection, communication efficiency, and model accuracy.

The remainder of this paper is organized as follows. In section II, we present a brief introduction of related works about training efficiency enhancement, straggler tolerance strategies and KD. In section III, we give the problem formulation, and describe the motivation and work flow of our proposed AdaDQ-KD algorithm in detail. Then, section IV introduces the theoretical analysis about privacy and convergence properties of AdaDQ-KD algorithm. Simulation experiments are shown in section V. Finally, we draw the conclusions of our work in section VI.

## II. RELATED WORK

## A. Gradient Quantization

Gradient sparsification or quantization is a widely used method for reducing the communication burden for each client, thereby enhancing training efficiency. A method called lazilyaggregated quantization (LAQ) is proposed in [13], where only partial important quantized gradients, identified through a threshold, are collected and aggregated. This approach further improves efficiency compared to conventional random quantization. In [14], a randomized quantization mechanism (RQM) is introduced to quantify local gradients with two-stage randomness. Specifically, quantization bins are first chosen randomly, and then the random quantization is applied to the original gradient. This mechanism offers a Renyi DP (RDP) guarantee. A novel DQ method is presented in [8], which differs from conventional quantization by its measurable quantization error. This error can be leveraged to support DP guarantees, providing a practical scheme to avoid manual noise injection and benefiting training efficiency.

Considering gradient heterogeneity among clients and training rounds, [23] introduces a DP-FL scheme with adaptive noise. The clip threshold for the gradient of each client is determined by the historical gradient norm, and the variance of the DP noise is adjusted throughout the iterations. A private federated edge learning with sparsification (PFELS) algorithm is designed in [24], which supports DP assurance and reduces communication overhead in FL. The key idea is to exploit the inherent characteristics of over-the-air computation for secure aggregation in FL through power-aligned analog signal transmission. Additionally, it uses the natural noise in wireless channel transmission instead of manually added noise to provide the DP guarantee, achieving high performance and low energy consumption under the same configuration. Furthermore, a sparse client selection strategy is proposed, where the number of clients is projected from a high-dimensional space to a lower-dimensional one via a random selection matrix.

In this work, we propose a modified DQ algorithm with an optional quantization precision for access to subsequent optimization adjustment and a better trade-off between efficiency and utility, while keeping the inherent advantages.

## B. Straggler Tolerance Strategies

Extensive research has been conducted on straggler tolerance schemes to improve FL training efficiency due to their significant negative impact. A scheme is presented in [15], where the statistical characteristics of clients' data are considered to determine client involvement. Specifically, clients with greater computational and communication capacities are prioritized for participation in training, while less capable ones are gradually incorporated as the process progresses. To address the neglect of stragglers' contributions in the previous work, two novel algorithms are introduced in [16] to enhance straggler resiliency, targeting privacy leakage and model inversion attacks.

Works on client selection and resource allocation can also enhance the training efficiency of FL on group level. For example, a client selection algorithm to improve straggler tolerance is introduced in [20], where a commonly shared representation is used while accounting for differences in computing speeds among clients. To address the heterogeneity in clients' local data, an adaptive client selection scheme is proposed in [21], which prioritizes significant updates from clients through adaptive quantization, thereby mitigating model deviation and improving robustness. Additionally, joint resource optimization involving channel assignment, power allocation, and CPU cycle frequencies of clients is explored in [22]. The resulting optimization problem is solved through sub-problem decomposition.

TABLE I COMPARISON OF IMPLEMENTATION METHODS FOR THE SAME GOAL ACROSS DIFFERENT STUDIES

| Surveys                   | [29]                   | [30]                   | Ours                            |
|---------------------------|------------------------|------------------------|---------------------------------|
| Communication efficiency  | Knowledge Distillation | Quantization           | Dithering Quantization          |
| Privacy preservation      | Gaussian Noise         | -                      | Dithering Quantization          |
| Statistical heterogeneity | -                      | -                      | Knowledge Distillation          |
| System heterogeneity      | Knowledge Distillation | Adaptive Quantization  | Adaptive Dithering Quantization |
| Training accuracy         | -                      | Knowledge Distillation | Knowledge Distillation          |

In this work, we consider dynamically allocating the communication load among clients based on their delays by adjusting the DQ precision, thereby mitigating the impact of client heterogeneity.

## C. Knowledge Distillation in Federated Learning

KD, initially introduced by [25], facilitates model compression by training a student model to emulate the behavior of a teacher model. This approach is straightforward and userfriendly, requiring only the alignment of output probabilities, making it particularly well-suited for classification tasks. However, it may fail to capture the knowledge encoded in the intermediate layers of the teacher model. Furthermore, the choice of temperature parameters can significantly influence the efficacy of the distillation process.

Additionally, KD has found extensive applications in FL, enhancing model performance across distributed systems. FedTL [26] enables clients to transfer knowledge to other clients through model fine-tuning and domain adaptation. However, FedTL incurs significant computation and communication overhead, particularly in large-scale FL contexts. In contrast, FedKD [27] transfers soft labels instead of full model parameters, significantly reducing communication overhead. Specifically, FedKD introduces an innovative scheme where a large-scale complex model, referred to as the teacher model, guides a relatively small model, called the student model. FitNets [28] focuses on aligning the intermediate feature representations of the teacher and student models. The student model gains a more robust identifier by learning from deep-level features extracted from intermediate layers. Unlike FedKD, FitNets can effectively compensate for losses caused by quantization errors. Even in the presence of sparsification, FitNets reduces the accumulation of sparsity errors by aligning the layers.

While the applicability of the above methods is limited, in this work, we aim to enhance the robustness of the model by adaptively adjusting the communication load based on the local delay of clients and by improving the generalization ability of local models through feature extraction under privacy constraints. Furthermore, we conduct a detailed comparison with two studies that are closely related to our research. This comparison not only demonstrates the novelty of our approach but also better contextualizes our contribution with respect to prior works. As summarized in Table I, [29] seeks to improve communication efficiency by transmitting soft labels, an approach derived from KD-based FL. This strategy is different from the conventional aggregation mechanism in FL. By contrast, [30] improves communication efficiency through gradient quantization. Similarly, our proposed algorithm also improves communication efficiency through gradient quantization. However, our method differs from [29] in that it remains within the classical FL framework, where clients upload quantized gradients for aggregation. Compared with [30], the main distinction of our approach lies in the direct integration of DP into the quantization process via DQ mechanism, which not only compresses gradients for efficient transmission but also provides differential privacy guarantees. To accommodate system heterogeneity in FL, [29] proposes a KD-based aggregation mechanism whereby clients upload soft labels instead of model parameters, thereby mitigating the challenges arising from variations in client architectures. In contrast, both [30] and our work focus on a different facet of heterogeneity-the disparity in clients' transmission capabilities. To address this, we adopt adaptive strategies to adjust the quantization precision accordingly. Although both [30] and our work employ adaptive quantization to handle system heterogeneity and leverage KD to preserve training accuracy, our approach goes beyond [30] by additionally accounting for the impact of data heterogeneity on model training and the privacy requirements of FL.

## III. ALGORITHM DESIGN

In this paper, we use normal letters to denote scalars while bold ones for vectors, and [ X ] denotes the array [ 1 , 2 , · · · , X ] for a positive integer X without specific notation. For ease of reference, Table I summarizes the key notations used throughout this paper.

## A. Problem Formulation

We consider a typical DP-FL training scenario with synchronous update, where the PS is semi-honest as the adversary, attempting to obtain privacy information of clients but complying with a training protocol. The objective function of the model parameter vector, denoted by w ∈ ❘ m , where m denotes the number of model parameters, for the training is shown as:

TABLE II SUMMARY OF NOTATIONS

| Symbol       | Description                                          |
|--------------|------------------------------------------------------|
| w            | Model parameters                                     |
| m            | Number of model parameters                           |
| n 0          | Quantization precision coefficient                   |
| P            | Number of sampled clients in each iteration          |
| C            | Gradient clip threshold                              |
| σ            | Standard deviation of Gaussian noise                 |
| α            | Learning rate sequences                              |
| T            | Total Number of iterations                           |
| S t          | Client subset participate in the training at round t |
| B            | Local batch size                                     |
| F ( · )      | Objective function                                   |
| ∆            | Quantization precision                               |
| v            | Gamma variable                                       |
| u            | Uniform diatribution variable                        |
| T loc        | Local delay of clients                               |
| T comp       | Local computation time of clients                    |
| b            | Number of communication bits                         |
| k            | proportional coefficient of straggler identification |
| h            | Search length of quantization precision              |
| n min        | The lower limit of quantization precision            |
| L            | Number of knowledge distillation layers              |
| C s l ,C T l | Number of channels of student model or teacher model |
| H l ,W l     | Dimensions of the feature map                        |

$$
\begin{aligned}
\max _ { \substack { \min _ { w \in \mathbb { R } ^ { m } } F ( w ) \colon = \frac { 1 } { P } \sum _ { i = 1 } ^ { P } F _ { i } ( w ) , \\ \text { } \text {
\end{aligned}
$$

where F i ( w ) is the local objective function of client i ∈ { 1 , . . . , P } , depending on the training task and local datasets. Here, P denotes the number of participating clients.

For privacy protection, we adopt a DP technique. It defends against privacy leakage by information masking, which is usually achieved by noise injection. Its formal definition is given as follows:

Definition 1. ( ( ϵ, δ ) -DP [31]): Given a privacy parameter ϵ ∈ (0 , + ∞ ) and a tiny error parameter δ ≥ 0 , a random mechanism M satisfies ( ϵ, δ ) -DP if for any two neighboring datasets D and D ′ that differ in only one single record, the following formula holds for any subset S ∈ range ( M ) :

$$
P r [ \mathcal { M } ( \mathcal { D } ) \in \mathcal { S } ] \leq e ^ { \epsilon } P r [ \mathcal { M } ( \mathcal { D } ^ { \prime } ) \in \mathcal { S } ] + \delta . \quad ( 2 ) \quad \stackrel { d i s t i l l a l l } { i t s i n t }
$$

Here, ϵ is referred to as the 'privacy budget', which measures the strictness of privacy protection. A smaller budget indicates a more rigorous guarantee but has a more severe impact on model training. Additionally, δ represents the error space, typically taking a very small value, or even zero.

Following the Gaussian mechanism in [31], to achieve ( ϵ, δ ) -DP, the standard deviation of the added Gaussian noise should satisfy:

$$
\begin{aligned}
\sigma \geq \frac { \Delta \sqrt { 2 l n ( 1 . 2 5 / \delta ) } } { \epsilon } . \quad & \quad ( 3 ) \quad & \stackrel { \text {repr} } { \lambda } \, \co \, &
\end{aligned}
$$

Our optimization goal (1) is to improve the training efficiency and model accuracy against data and client heterogeneity under privacy guarantee, achieving a comprehensive practicality for FL system. For model accuracy, we utilize KD to enhance the model's generalization ability. With the help of teacher model, the generalization ability of local models can be enhanced, thus alleviating the bad impact of DP noise and data heterogeneity in practice.

As for training efficiency, we first introduce DQ to reduce the model scale, as well as to provide noise simulation for DP guarantee. Then, we consider to further improve the training efficiency with high client heterogeneity existing by mitigating stragglers' influence. Additionally, we tackle this problem based on the local delay of each client. For a client i at round t , its local delay T loc i,t mainly consists of local computation time T comp i,t for SGD and communication time T comm i,t for uploading updates. Note that we omit the time for model download since all clients acknowledge that at nearly the same time. Owing to the synchronize update mode, our objective is actually to make the longest local delay as small as possible among all clients in each round.

Nevertheless, the above optimization problem involves massive factors, and becomes NP-hard. Given this situation, we propose a DP-FL algorithm named 'Adaptive DQ with KD' (AdaDQ-KD) to find a good feasible solution. Additionally, it has remarkably fine performance in terms of training efficiency indeed. Specifically, we adaptively adjust the communication load according to the local delay T loc i,t for clients at each round, through the reallocation of local quantization precision. In this way, our proposed AdaDQ-KD algorithm can significantly improve the FL training efficiency with theoretical privacy protection and enhancement of model generalization, avoid solving the complicated optimization problem alone. Generally, AdaDQ-KD algorithm includes two key procedures for efficiency, which we will respectively introduce then.

## B. Knowledge Distillation in Local Training

To improve the effectiveness of client-side optimization under limited local data and quantization constraints, we employ KD as a guidance mechanism for local training. At each round t , each client i is pre-distributed a teacher model , which provides auxiliary knowledge during local optimization.

Given a mini-batch B sampled from the client's local dataset, the client performs a forward propagation through both the student and teacher models. Intermediate representations from selected layers are extracted to construct a feature-based distillation signal, which encourages the student model to align its internal representations with those of the teacher.

The local training objective on each client consists of two components: (i) a classification loss computed using local data, and (ii) a feature-level distillation loss that transfers knowledge from the teacher to the student.

The overall local objective is defined as

$$
\mathcal { L } _ { \text {local} } = \mathcal { L } _ { \text {CE} } + \lambda \mathcal { L } _ { \text {KD} } ,
$$

where L CE denotes the standard cross-entropy loss, L KD represents the KD loss aggregated over selected layers, and λ controls the contribution of the distillation term.

During local training, each client minimizes L local with respect to the student model parameters using stochastic gradient-based optimization. After completing local optimization, the client obtains the updated model parameters or the corresponding gradients as follows:

$$
g _ { i } ^ { t } = \nabla _ { w } \mathcal { L } _ { l o c a l } ( w ) ,
$$

which are then transmitted to the server for global aggregation.

Through this process, KD serves as an implicit regularizer that guides the local training trajectory and enables clients to generate more informative and stable updates for federated learning.

## C. Dithering Quantization with Adjusted Precision

To decrease the latency across all clients by adjusting the communication burden, a simple and practical method is to resort to gradient quantization. The number of transmitted bits can be explicitly calculated based on the quantization operation. However, it is difficult to achieve the DP assurance by conventional random quantization. Thus we turn to DQ [8], whose quantization error can be viewed as Gaussian noise supporting the DP guarantee. Additionally, for further adjustment of number of encoded bits according to the local delay, we introduce a quantization precision coefficient to establish this connection. Here follows the process of DQ.

Suppose that client i has generated its local gradient g t i with the guidance of teacher model through SGD at round t . For simplicity's sake, we regard it as a scalar for the moment and omit the round index t for subsequent intermediate variables. Client i first samples a gamma variable v i ∼ Γ(3 / 2 , 1 / 2) , and decides the quantization precision (i.e. quantization step) as ∆ i = 2 n i σ √ v i , where σ denotes the standard deviation of Gaussian noise for DP guarantee, and n i is the designated quantization precision coefficient for client i . Then, it samples a uniform distribution variable u i ∼ U ( -∆ i 2 , ∆ i 2 ) , and quantizes original gradient g t i as:

$$
q _ { i } ^ { t } = Q ( g _ { i } ^ { t } + u _ { i } ) , \quad ( 6 ) \ \sin c
$$

where the quantization function Q ( x ) is defined as ⌈ x ∆ i -1 2 ⌋ ∆ i + ∆ i 2 and ⌈·⌋ is the function rounding to the nearest integer. Repeating this process for each dimension of gradient j ∈ [ m ] , the local gradient will be quantized completely. Finally, client i uploads quantized gradient q t i to the PS for model aggregation.

It is worth mentioning that, the received quantized gradient q t i needs to be decoded as the noisy estimation of original gradient g t i by the PS before aggregation. The actual uploaded update is just the ⌈ g t i + u i ∆ i -1 2 ⌋ , since the decoding can be completed by the PS in our design where PS and clients share the same randomness. In addition, the theoretical illustration will be given in the next section.

We aim to further optimize training efficiency in the presence of stragglers by adaptively and dynamically adjusting clients' communication loads. Additionally, the link between local DQ and load adjustment lies in the quantization precision, which directly determines the number of transmitted bits and, consequently, controls the communication load. We will now introduce our idea in detail.

At first, the local delay of client i in the next round needs to be estimated. Since a client's computational capacity is usually stable, we represent the expected local computation time as the average of historical values:

$$
\mathbb { E } ( T _ { i , t + 1 } ^ { c o m p } ) = \frac { 1 } { t + 1 } \sum _ { l = 0 } ^ { t } T _ { i , l } ^ { c o m p } . \quad \ \ ( 7 ) \quad \ \
$$

Fig. 1. Example of adaptive quantization precision adjustment

<!-- image -->

As for communication time, although a stable communication environment cannot be guaranteed, it usually changes smoothly. Since the round duration is relatively short, we simplify the analysis by assuming that the transmission rate remains the same as in the previous round. Thus, the expected communication time is represented as follows:

$$
\begin{aligned}
\text {t and} \quad \mathbb { E } ( T _ { i , t + 1 } ^ { c o m m } ) = \mathbb { E } ( \frac { b _ { i , t + 1 } } { R _ { i , t + 1 } } ) = \frac { b _ { i , t + 1 } } { b _ { i , t } } \frac { b _ { i , t } } { R _ { i , t } } = \frac { b _ { i , t + 1 } } { b _ { i , t } } T _ { i , t } ^ { c o m m } . \\ 1 / 2 ) ,
\end{aligned}
$$

From (8), we establish the connection of the communication time between adjacent rounds. Additionally, we can derive the concrete value for bits number b i,t through analysis from Section III-C as follows:

$$
b _ { i , t + 1 } = m \log _ { 2 } ( 2 \lceil \frac { C } { \Delta _ { i , t + 1 } } + 1 \rceil ) .
$$

Since ∆ i,t = 2 n i,t σ √ v i , the number of bits b i,t is related to the quantization precision coefficient n i,t . By substituting (9) into (8), we establish a connection between the estimated communication time ❊ ( T comp i, t +1) and n i across successive rounds. This result provides a guideline for determining n i,t +1 . Specifically, we can now calculate the expected local delay based on (8) and (9) as follows:

$$
\begin{aligned}
\text {agent} \quad & \mathbb { E } ( T _ { i , t + 1 } ^ { l o c } ) = \mathbb { E } ( T _ { i , t + 1 } ^ { c o m p } ) + \mathbb { E } ( T _ { i , t + 1 } ^ { c o m m } ) \\ \text {original} \quad & \\ & \quad = \mathbb { E } ( T _ { i , t + 1 } ^ { c o m p } ) + \frac { \log _ { 2 } ( 2 \lceil \frac { C } { \Delta _ { i , t + 1 } } + 1 \rceil ) } { \log _ { 2 } ( 2 \lceil \frac { C } { \Delta _ { i , t } } + 1 \rceil ) } T _ { i , t } ^ { c o m m } \ \ ( 1 0 ) \\ \text {share} \quad &
\end{aligned}
$$

Among these, ❊ ( T comp i,t +1 ) can be calculated by the PS using uploaded historical computation times and T comm i,t can be obtained by the PS with the help of timestamps from client i 's uploads in the last round. Recall that our goal is to minimize the local delay across clients. A natural approach is to identify the stragglers and allocate lower communication loads to them, ensuring that all delays are nearly the same and avoiding unnecessary waiting. Clearly, the PS can calculate the appropriate n i,t +1 for each client at each round based on (10), provided it receives T comp i,t and records T comm i,t .

We illustrate the above idea in Fig. 1, where the lengths of the rectangles with sharp edges represent the delay values. The light-colored sections above indicate computation delay, while the dark-colored sections below represent communication delay. At round t , the PS collects and records the local delays of the clients. Then, the PS estimates the local delays for the

## Algorithm 1 Designation for Quantization Precision

Input : number of sampled clients in each iteration P , proportional coefficient of straggler identification k , search length of quantization precision h , the lower limit of quantization precision n min

- 1: PS reorders clients in descending order in terms of expectation of local delay ❊ ( T loc i,t ) , i.e. ❊ ( T loc 1 ,t ) ≥ ❊ ( T loc 2 ,t ) ≥ · · · ≥ ❊ ( T loc P,t ) .
- 2: PS calculates K = ⌊ kP ⌋ as the threshold for straggler identification, where ⌊·⌋ is to take the maximum integer not greater than itself, and makes standard value S = ❊ ( T loc K +1 ,t ) .
- 3: for Cilent i ( 1 ≤ i ≤ K ) do
- 4: while ❊ ( T loc i,t ) &gt; S and n i,t ≥ n min do
- 5: Lessens n i,t = n i,t -h

6:

7:

Recalculates

end while

end for

return

n

i,t

+1

=

n

i,t

next round, t + 1 , and reallocates the quantization precision for the clients accordingly. As shown in Fig. 1, assuming the cellphone has the longest delay initially, its communication time will be significantly reduced with the newly designed quantization precision. In contrast, for the tablet PC and laptop, which already have relatively low local delays, such a reduction is unlikely. Maintaining higher precision for these devices will enhance the model utility without introducing additional waiting time.

Subsequently, the only remaining problem is how to identify stragglers. To address this, we introduce a proportional coefficient k that defines the threshold for straggler tolerance. For example, if we set k = 0 . 3 , the lowest-performing 30% of participating clients are considered stragglers and will be assigned a lower communication load by using reduced quantization precision in the next round.

## D. Workflow

Building on the previous discussion, we present the formal description of the AdaDQ-KD algorithm in Algorithm 2 and provide a detailed explanation of its complete procedure.

- 1) Parameter Issuance (Lines 1 and 2). At the start of round t , the PS selects participating clients S t , then broadcasts the model parameter w t , pretrained global teacher model w tm and heterogeneous quantization parameter n i,t . Note that n i,t is decided by Algorithm 2, which will be introduced later.
- 2) Local Training (Lines 3 to 6). By the guidance of w mt , client i in S t generates local gradient g t i through SGD, and clips it by g t i = g t i / max { 1 , ∥ g t i ∥ 2 C } .
- 3) Dithering Quantization (Lines 7 to 12). Local gradient of Client i , g t i is quantized as q t i according to the description in Section III-C. Subsequently client i uploads q t i and T comp i,t with time stamp. As explained earlier, client i only has to upload ⌈ g t i + u i ∆ i -1 2 ⌋ for q t i here in actual.

❊

(

T

loc

i,t

)

through (7) and (10)

Algorithm 2 Adaptive Dithering Quantization with Knowledge Distillation (AdaDQ-KD) algorithm

Input : initial global model parameter w 0 , pretrained global teacher model parameter w tm , default quantization precision coefficient n 0 , number of sampled clients in each iteration P , gradient clip threshold C , standard deviation of Gaussian noise σ , learning rate sequences { α t } t ≥ 0 , total number of iterations T

```
1: for 0 ≤ t ≤ T -1 do 2: PS selects S t , broadcast w t , w tm and issue n i,t to participating clients 3: for client i in S t in parallel do 4: Receives w t , w tm and n i,t 5: After feature alignment via w tm , generates local gradient g t i through SGD 6: Gradient clip: g t i = g t i / max { 1 , ∥ g t i ∥ 2 C } 7: for each coordinate j of g t i in dimension [ m ] do 8: Samples v i,j ∼ Γ(3 / 2 , 1 / 2) 9: Calculates ∆ i,j = 2 n i,t σ √ v i,j 10: Samples u i,j ∼ U ( -∆ i,j 2 , ∆ i,j 2 ) 11: Quantizes g t i,j as q t i,j through (6) end for 12: Sends q t i and T comp i,t to PS with time stamp end for 13: PS receives the q t i and T comp i,t from all selected clients and calculate ❊ ( T loc i,t +1 ) through (7) and (10) 14: PS decides n i,t +1 for each client by Algorithm 2 15: for i ∈ [ P ] do 16: for j ∈ [ m ] do 17: Obtains the same u i,j through shared randomness of client i 18: Makes noisy estimation: ˆ g t i,j = q t i,j -u i,j end for end for 19: PS aggregates estimated gradients: g t = 1 P ∑ P i =1 ˆ g t i 20: PS updates the model: w t +1 = w t -α t g t end for 21: return w T
```

- 4) Adaptive Quantization Precision Adjustment (Lines 13 and 14). PS acknowledges the q t i and T comp i,t from all selected clients, and calculates ❊ ( T loc i,t +1 ) through (7), (8) and (9). Meanwhile, PS obtains the T comm i,t by the time stamp as well. Specially, if a selected client was not involved in training at last round, PS reads its latest record instead. Additionally, for t = 0 , the initial time for computation and communication can be specified in advance based on the past experience. Subsequently, PS decides heterogeneous quantization parameter n t i based on the analysis in Section III-C. To make our Algorithm 1 clear and easy to understand, we organize this procedure separately as Algorithm 2.
- 5) Model Aggregation (Lines 15 to 20). PS decodes q t i as a noisy estimation of local gradient by ˆ g t i,j = q t i,j -u i,j , where the u i,j is the same as that of client side due to the shared random seed. Finally, PS aggregates all gradients and updates the global model, starting a new round.

Our proposed AdaDQ-KD algorithm employs feature extraction to enhance model accuracy in the presence of data heterogeneity, DQ to reduce transmitted bits while supporting privacy protection, and adaptive communication load adjustment based on quantization precision to reduce the local delay in the FL system.

This approach significantly enhances the overall practicality of the system while providing a DP guarantee.

## E. Complexity Analysis

In this section, we analyze the computational complexity of the proposed algorithm AdaDQ-KD, which consists of two main operations: adaptive dithering quantization and knowledge distillation.

First, the primary computational complexity of adaptive dithering quantization lies in the quantization procedure, which involves clipping and quantizing each model parameter, resulting in a time complexity of O ( N ) , where N denotes the number of parameters.

Second, the primary operation in the KD procedure is the computation of feature distillation loss, which measures the discrepancy between the features of the student and teacher models. The forward pass through the teacher model incurs a time complexity of O ( F T ) , constituting the main computational overhead introduced by knowledge distillation. The computation of feature distillation loss considers feature maps of size B × C l × H l × W l at the l -th layer. Accordingly, the time complexity of feature distillation across L layers is

$$
\mathcal { O } \left ( \sum _ { l = 1 } ^ { L } B \times H _ { l } \times W _ { l } \times ( C _ { l } ^ { S } + C _ { l } ^ { T } + C _ { l } ^ { S } C _ { l } ^ { T } ) \right ) . \quad ( 1 1 ) \quad \text {and} \ | \mathcal { D } _ { i } | _ { \alpha } \geq 2 ,
$$

Therefore, the time complexity of the KD procedure is

$$
\begin{aligned}
\mathcal { O } \left ( F _ { T } + \sum _ { l = 1 } ^ { L } B \times H _ { l } \times W _ { l } \times ( C _ { l } ^ { S } + C _ { l } ^ { T } + C _ { l } ^ { S } C _ { l } ^ { T } ) \right ) . \\ \quad \ \ ( 1 2 ) \quad \ \end{aligned}
$$

Overall, the time complexity of our proposed algorithm AdaDQ-KD is

$$
\mathcal { O } \left ( N + F _ { T } + \sum _ { l = 1 } ^ { L } B H _ { l } W _ { l } \left ( C _ { l } ^ { S } + C _ { l } ^ { T } + C _ { l } ^ { S } C _ { l } ^ { T } \right ) \right ) . \quad ( 1 3 ) \quad \text {stating} \quad T h e r
$$

## IV. THEORETICAL ANALYSIS

## A. Privacy Guarantee

To assess the privacy benefit of Algorithm 2, we have to introduce the statistical characteristics of DQ first, which are given in the following lemmas.

Lemma 1. ( [32]): For a gamma random variable V ∼ Γ(3 / 2 , 1 / 2) , if the dependent random variable ( X | V = v ) ∼ U ( µ -σ √ v, µ + σ √ v ) , then we have the random variable X ∼ N ( µ, σ 2 ) .

Lemma 2. ( [33], [34]): For a quantization function Q ( · ) with step length ∆ , a uniform random variable U ∼ U ( -∆ 2 , ∆ 2 ) and a scalar Y to be quantized with Q ( · ) , let ˆ Y = Q ( Y + U ) -U , we have ˆ Y = Y + U ′ where U ′ follows the same uniform distribution as U and also independent from U .

Based on the two lemmas, we present Theorem 1, which states the equivalence between quantization error and Gaussian noise.

Theorem 1. (Gaussian noise simulation): ˆ g t i in Algorithm 2 is a noisy estimation of local gradient g t i for client i , where the error follows Gaussian distribution N (0 , ■ m σ 2 n 2 i,t ) .

Proof. For simplicity, we omit the indices i for the client and t for the round in this proof. Additionally, we begin with a scalar analysis and then extend it to an m-dimensional case, due to the independence of each coordinate. From Algorithm 2, u ∼ U ( -∆ 2 , ∆ 2 ) , substitute with ∆ = 2 σ √ v n , we have u ∼ U ( -σ √ v n , σ √ v n ) . Since u is dependent on a gamma variable v ∼ Γ(3 / 2 , 1 / 2) , we obtain that u ∼ N (0 , σ 2 n 2 ) according to Lemma 1. In addition, we know ˆ g = q -u = Q ( g + u ) -u , where the quantization step of Q ( · ) is ∆ = 2 σ √ v n . Thus we have ˆ g ∼ g + N (0 , σ 2 n 2 ) according to Lemma 2, Theorem 1 is proved.

Due to the equivalence, DQ can be regarded as a special Gaussian noise injection. Note that if the amplitude of simulated noise is affected by the quantization precision of the current round, then the privacy budget consumption is not fixed. Therefore, we have to analyze the corresponding privacy loss for the AdaDQ-KD algorithm, which is based on the following lemmas.

Lemma 3. ( [23]): For a DP-FL algorithm, if the sampling rate of client i is p = |B i | |D i | and the noise standard deviation of current round t is σ t , where |B i | denotes the batch size and |D i | denotes the size of local dataset, then for any integer α ≥ 2 , the privacy loss of client i after T iterations will be:

$$
\begin{aligned}
\varepsilon _ { i , T } ^ { \prime } = \sum _ { t = 0 } ^ { T } \frac { 1 } { \alpha - 1 } \sum _ { k = 0 } ^ { \alpha } \left ( \begin{matrix} \alpha \\ k \end{matrix} \right ) ( 1 - p ) ^ { \alpha - k } p ^ { k } e ^ { \frac { k ^ { 2 } - k } { 2 \sigma _ { t } ^ { 2 } } } .
\end{aligned}
$$

Lemma 4. (Translation from Renyi DP (RDP) to DP [35]): If a random mechanism M satisfies ( α, ϵ ) -RDP , then M will satisfy ( ϵ + log(1 /δ ) α -1 , δ ) -DP , where 0 &lt; δ &lt; 1 .

Based on the lemmas above, we have the following theorem stating the privacy loss of proposed AdaDQ-KD algorithm.

Theorem 2. (Privacy loss of Algorithm 2): For Algorithm 2, the privacy loss of client i after T iterations is:

$$
\text {have to } \quad \varepsilon _ { i , T } ^ { \prime } = \sum _ { t = 0 } ^ { T } \frac { 1 } { \alpha - 1 } \sum _ { k = 0 } ^ { \alpha } \binom { \alpha } { k } ( 1 - p ) ^ { \alpha - k } p ^ { k } e ^ { \frac { k ^ { 2 } - k } { 2 \sigma _ { t } ^ { 2 } } } + \frac { \log ( 1 / \delta ) } { \alpha - 1 } .
$$

Proof. According to Lemma 3 and Lemma 4, the theorem can be directly proved.

## B. Efficiency Analysis

In order to evaluate the communication efficiency of Algorithm 1, we present Theorem 3, which provides a formal quantification of communication efficiency.

Theorem 3. (The communication efficiency of Algorithm1): For Algorithm 1, given a training model with parameter size m and a total number of training rounds T , the upper bound on the number of bits transmitted by client i throughout the entire training process is as follows:

$$
m T \log _ { 2 } ( \lceil \frac { C } { \Delta } + 1 \rceil )
$$

Proof. For simplicity, we first consider the case of a singleround scalar, then extend this result to the full dimensionality and the entire process, while omitting the client index. Let y = ⌈ g t + u ∆ -1 2 ⌋ , Based on the foregoing, y represents the actual encoded information transmitted by the client to the server in round t . Because u ∈ [ -∆ 2 , ∆ 2 ] , | g t | ≤ C , The range of values for y lies within the interval [ ⌈-C ∆ -1 ⌋ , ⌈ C ∆ ⌋ ] . Consequently, the transmission of a scalar in a single round requires log 2 (2 ⌈ C ∆ + 1 ⌋ ) bits. Extending this result to m dimensions over T rounds, the conclusion is established.

In practical scenarios, the value of C ∆ is typically small, resulting in the upper bound specified in Theorem 3 being significantly lower than the communication overhead of conventional 32-bit or 64-bit floating-point algorithms, which have encoding upper bounds of 32 mT and 64 mT , respectively. It is noteworthy that, as elucidated in the prior description of Algorithm 1's procedure, the variable ∆ = 2 n σ √ v in (10) is dependent on both the quantization precision n and the noise scale σ . To some extent, Theorem 3 also reflects the tradeoff between communication bits and the parameters n and σ : when the precision requirement is low (i.e., n is small) and the privacy demand is modest (i.e., σ is small), the upper bound on communication overhead achieves a smaller value.

## C. Convergence Analysis

We first introduce the universal assumptions in ML and SGD works below [36]-[38], and then give the Theorem 1 stating the convergence of Algorithm 2 based on them.

Assumption 1. The local objective function F i ( · ) is LLipschitz smooth: F i ( y ) -F i ( x ) ≤ ⟨∇ F i ( x ) , y -x ⟩ + L 2 ∥ y -x ∥ 2 .

Assumption 2. The gradient of F i ( · ) is bounded: ❊ [ ∥∇ F i ( w ) ∥ 2 ] ≤ G 2 , where G &lt; ∞ .

Assumption 3. The variance between local gradient and global gradient is bounded: ❊ [ ∥∇ F i ( w ) -∇ F ( w ) ∥ 2 ] ≤ M 2 , where M &lt; ∞ .

Assumption 4. The global objective function F ( w ) has a lower bound: F ( w ) ≥ F inf , ∀ w ∈ ❘ m .

Theorem 4. (Convergence of Algorithm 2): Under Assumptions 1-4, if the initial learning rate satisfies Lα 0 ≤ 1 , then the convergence bound for Algorithm 2 can be given as:

$$
\begin{aligned}
\sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \| \nabla F ( w ^ { t } ) \| ^ { 2 } ] \leq \frac { 1 } { P } ( M ^ { 2 } + \sigma ^ { 2 } ) + \frac { 2 [ F ( w ^ { 0 } ) - F _ { i n f } ] } { \alpha ^ { 0 } T } . \quad \text {under $d$} \quad
\end{aligned}
$$

Proof. Due to the simplicity of the deductive approach and space constraints, we will only show the critical steps involved. From Theorem 1, we have:

$$
( 1 6 ) \quad w ^ { t + 1 } - w ^ { t } = - \frac { 1 } { P } \sum _ { i = 1 } ^ { P } \alpha ^ { t } \hat { g } _ { i } ^ { t } = - \frac { 1 } { P } \sum _ { i = 1 } ^ { P } \alpha ^ { t } ( g _ { i } ^ { t } + U _ { i } ) , \ \ ( 1 8 )
$$

where U i ∼ N (0 , ■ m σ 2 ) . By Assumption 1, we have:

$$
\begin{aligned}
\text {single} - \text { where } U _ { i } \sim \mathcal { N } ( 0 , \mathbb { I } _ { m } \sigma ^ { 2 } ) . \text { By Assumption } 1 , \text { we have:} \\ \text {nonlinearity} \\ \text {ex. Let } \quad \mathbb { E } [ F ( w ^ { t + 1 } ) - F ( w ^ { t } ) ] \leq \underbrace { - \alpha ^ { t } \left \langle \nabla F ( w ^ { t } ) , \frac { 1 } { P } \sum _ { i = 1 } ^ { P } \mathbb { E } ( g _ { i } ^ { t } + U _ { i } ) \right \rangle } _ { A } \\ \text {to the} \\ \text {,} \, \text {The} \\ \text {, } [ \frac { C } { \Delta } ] \cdot \\ \text {round} \\ \text {to } \, m \\ \text {ed.} \\ \text {small} \quad \text {In consideration of } \mathbb { E } ( U _ { i } ) = 0 , \langle y - x \rangle = \frac { 1 } { 3 } [ \| x \| ^ { 2 } + \| y \| ^ { 2 } -
\end{aligned}
$$

In consideration of ❊ ( U i ) = 0 , ⟨ y -x ⟩ = 1 2 [ ∥ x ∥ 2 + ∥ y ∥ 2 -∥ x -y ∥ 2 ] , we can obtain the result below through the triangle inequality, Assumption 2 and Assumption 3:

$$
A \leq - \frac { \alpha ^ { t } } { 2 } \| \nabla F ( w ^ { t } ) \| ^ { 2 } + \frac { \alpha ^ { t } } { 2 P } ( M ^ { 2 } - G ^ { 2 } )
$$

and B can be bounded as

$$
B \leq \frac { L ( \alpha ^ { t } ) ^ { 2 } } { 2 P } ( G ^ { 2 } + \sigma ^ { 2 } ) .
$$

Since the learning rate sequence is in descending order, if choosing an appropriate initial learning rate α 0 satisfying Lα 0 ≤ 1 , we will have:

$$
\| \nabla F ( w ^ { t } ) \| ^ { 2 } \leq \frac { 1 } { P } ( M ^ { 2 } + \sigma ^ { 2 } ) + \frac { 2 \mathbb { E } [ \| \nabla F ( w ^ { t } ) \| ^ { 2 } ] } { \alpha ^ { 0 } } .
$$

Finally, sum up this result by rounds and take the average, Theorem 4 is proved.

## V. EXPERIMENTS

## A. Experiment Settings

We consider a FL system with P = 20 clients, and they participate in the training throughout all rounds. To simulate the client heterogeneity among devices, we set the transmission rate R i for each client as a random number varying from 5 Mbps to 20 Mbps, where R i ∼ U (5 , 20) . Additionally, for computation capacity discrimination, we utilize a coefficient β i to calibrate the wall-clock time of local training, where β i ∼ U (0 . 8 , 1 . 0) . In addition, we use Gaussian variables to simulate the changes of communication environment and the CPU frequency over global iterations. The code is available at https://github.com/wangf622/AdaDQ-KD.

We choose public datasets FashionMNIST (FMNIST) [39] and CIFAR-10 [40] to conduct simulation experiments. Nonindependent identical distribution (Non-IID) circumstances are simulated through random probability vectors generated by Dirichlet distribution [41]. By adjusting concentration parameter α which controls the degree of data distribution heterogeneity, we can observe variations in model performance under different levels of data heterogeneity.

For model training, we adopt ResNet-18 [42] with default weights. It has 11,689,512 parameters in total, which is regarded as the dimension of local gradient in previous discuss (i.e. m = 11 , 689 , 512 ). Local batch size is 64, and total global rounds are 30 for FMNIST and 50 for CIFAR-10 respectively. The proportional coefficient of straggler identification k is 0.5. To validate the effectiveness of the proposed AdaDQ-KD algorithm, we use the following methods for comparison.

Fig. 2. Test accuracy of different algorithms with respect to training time under different privacy budgets ϵ on non-IID FMNIST and CIFAR-10 datasets.

<!-- image -->

- FLDP [43]: Using the traditional Gaussian to achieve DP guarantee.
- FLDQ [8]: FL with only DQ and no group optimization, as the SOTA and baseline algorithm compared with our proposed AdaDQ-KD.
- FLDQ-KD : Building upon the aforementioned algorithm, KD is introduced to alleviate the issue of data heterogeneity.
- AdaDQ : Only DQ and group optimization are utilized for comparison with FLDQ.

## B. Overall Performance

We evaluate the performance of the proposed algorithm, AdaDQ-KD, in comparison with two classical FL algorithms on the FMNIST and CIFAR-10 datasets. Specifically, FLDP serves as a classical FL paradigm that employs Gaussian noise to ensure DP, while FLDQ is the fundamental baseline algorithm upon which our method is built. In our experiments, we set the Dirichlet distribution parameter to α = 0 . 3 , representing a moderate level of data heterogeneity. Furthermore, we select n 0 = 2 for all subsequent experiments, as this value demonstrates competitive performance while requiring relatively fewer communication bits, thereby achieving a favorable trade-off between model accuracy and communication efficiency. We assess the effectiveness of the proposed algorithm from two perspectives: training efficiency and model accuracy, which we aim to improve.

The experimental results are organized into eight subfigures, each illustrating the performance of various algorithms across different datasets under varying privacy budget constraints. We evaluated the algorithms across four discrete privacy protection levels (i.e., ϵ ∈ { 0 . 1 , 0 . 3 , 0 . 8 , 1 . 2 } ) encompassing a wide range of privacy requirements.

Based on the simulation results, we observe that compared with FLDP, FLDQ achieves effective quantization while satisfying DP constraints, reducing training time without degrading the final model accuracy. Moreover, the proposed method AdaDQ-KD accelerates model convergence and significantly lowers the time cost to reach the same accuracy level.

On the CIFAR-10 dataset, under stronger privacy constraints (e.g., ϵ = 0 . 1 and ϵ = 0 . 3 ), the training task becomes more challenging due to increased noise injection. When ϵ = 0 . 1 in Fig. 2(e), the proposed method exhibits a slight degradation in final accuracy; however, the reduction in training time remains substantial. It is worth noting that under such a strict privacy budget, although the model is still trainable, its performance is insufficient for practical deployment. Therefore, this setting is included to evaluate the robustness of the proposed method under extreme privacy conditions. As shown in Fig. 2 (b) and (f), for ϵ = 0 . 3 , the proposed method yields significantly higher accuracy than the baselines.

In summary, across varying privacy budgets, the proposed method demonstrates strong overall performance. When compared with two representative DP-FL baselines, the algorithm significantly improves training efficiency and effectively accelerates convergence, while maintaining competitive model accuracy.

## C. Evaluation of Cost

In Table III, we present the computational cost of our proposed algorithm. It can be observed that the quantization operations account for a large portion of the overall time overhead. However, quantization simultaneously provides privacy guarantees and reduces communication overhead. Conventional approaches that aim to achieve these two effects typically require performing quantization and adding noise separately, which demands more time. The errors introduced by quantization and noise injection can interact, leading to potential performance degradation. Furthermore, KD, as a technique to ensure model training effectiveness, is also indispensable. Subsequently, the interplay between adaptive DQ and KD will be illustrated more intuitively through simulation experiments.

TABLE III DETAILED TIMING ANALYSIS FOR MODEL INFERENCE OPERATIONS

| Category   |   Average/s |   Std Dev/s | Range/s                               | Percentage   |
|------------|-------------|-------------|---------------------------------------|--------------|
| Forward    |      0.0026 |      0.0008 | 0.001 - 0.004 (Min - Max) ±0.0008 var | 0.37%        |
| KD         |      0.0332 |      0.0065 | 0.024 - 0.048 (Min - Max) ±0.0065 var | 4.74%        |
| DQ         |      0.6622 |      0.0519 | 0.623 - 0.808 (Min - Max) ±0.0519 var | 94.52%       |
| Backward   |      0.0026 |      0.0008 | 0.001 - 0.005 (Min - Max) ±0.0008 var | 0.37%        |
| Total      |      0.7006 |      0.0525 | 0.649 - 0.865 (Min - Max) ±0.0525 var | 100%         |

TABLE IV REQUIRED BITS FOR FMNIST TRAINING

| Algorithm   |   Bits in total |
|-------------|-----------------|
| AdaDQ-KD    |  31,528,893,636 |
| FLDQ-KD     |  36,807,426,986 |

TABLE V REQUIRED BITS FOR CIFAR-10 TRAINING

| Algorithm   |   Bits in total |
|-------------|-----------------|
| AdaDQ-KD    |  54,254,529,859 |
| FLDQ-KD     |  61,345,711,643 |

As demonstrated in Table IV and Table V, we present the communication bits required during the training process for FMNIST and CIFAR-10 datasets, respectively. Based on the experimental results, it can be concluded that the implementation of the adaptive communication load allocation method leads to significant savings in communication overhead. Specifically, for the FMNIST dataset, training over 30 epochs results in a reduction of 5 billion bits, corresponding to a 14.34% savings. For the CIFAR-10 dataset, training over 50 epochs achieves a reduction of 7 billion bits, representing an 11.56% savings compared to the scenario without adaptive adjustment.

## D. Performance on Different Heterogeneity Degree

In this section, we verify whether the KD method or not can enhance the training efficacy of the model. We evaluate the effectiveness of KD on data with different levels of heterogeneity. As the value of α increases, the degree of data heterogeneity gradually decreases.

Fig. 3. The test accuracy of AdaDQ-KD over heterogenous degree of data heterogeneity

<!-- image -->

As we can see in Fig. 3(a), the introduction of KD will ultimately improve the accuracy for training on the FMNIST dataset. However, as the degree of heterogeneity decreases, the training accuracy improvement brought by KD also gradually diminishes. This phenomenon can be attributed to the simplicity of the FMNIST dataset; as a result, client data heterogeneity is no longer the primary factor influencing training when the heterogeneity level is low.

These results are different on the CIFAR-10 dataset, as shown in Fig. 3(b). Given the differences in the inherent difficulty of training across datasets, we observe that with the same value of α , the CIFAR-10 dataset suffers a steeper drop in training accuracy. When training on the CIFAR-10 dataset, the middle layers contain more knowledge beneficial for model learning. As a result, there is a more significant improvement on CIFAR-10, even under circumstances of lower heterogeneity.

In summary, the introduction of KD can significantly enhance the final training accuracy across varying levels of heterogeneity. This indicates that our proposed method effectively mitigates the impact of data heterogeneity, leading to improved model performance.

## E. Effect of KD on Heterogeneity

We evaluate the impact of KD on client heterogeneity from two perspectives: gradient similarity among clients and gradient variance across clients.

Fig. 5. The effect of KD on client's variance

<!-- image -->

As shown in Fig. 4(a) and Fig. 5(a), on the FMNIST dataset, incorporating KD increases inter-client gradient similarity and reduces gradient variance. This is expected, as KD provides a common supervisory signal that aligns the optimization directions of different clients, thereby alleviating gradient dispersion caused by data heterogeneity.

In contrast, on the CIFAR-10 dataset in Fig. 4(b) and Fig. 5(b), gradient similarity among clients decreases and variance increases after introducing KD. This outcome suggests that, for complex tasks, KD enables clients to focus on different fine-grained aspects of the data, which amplifies diversity in local gradient updates.

Importantly, KD does not aim to directly eliminate data heterogeneity; rather, it indirectly influences local training dynamics by enriching the learned representations. As demonstrated in prior experiments, despite the increased gradient diversity under complex datasets, the incorporation of KD ultimately leads to improved model performance.

## F. Effect of KD on Quantization

In this section, we further investigate the impact of KD on quantization.

Fig. 6. The effect of kd on gradients with quantization

<!-- image -->

Fig. 8. The effect of kd on gradients similarity

First, as shown in Fig. 6, we observe that the influence of KD on quantized gradients exhibits opposite trends across different datasets. For the FMNIST dataset in Fig. 6 (a), KD mitigates the perturbation of gradients induced by quantization, as reflected in reduced relative changes. In contrast, on the more challenging CIFAR-10 dataset, KD leads to increased gradient changes after quantization.

This phenomenon can be attributed to the intrinsic complexity of the learning tasks. For relatively simple datasets such as FMNIST, the optimization landscape is smoother and the learned representations are less sensitive to fine-grained gradient variations. In this case, KD provides additional supervisory signals from the teacher model, guiding the student model toward more stable optimization directions and thereby mitigating the adverse effects of adaptive DQ.

However, for more complex datasets such as CIFAR-10, model training relies more heavily on high-dimensional and fine-grained gradient information to capture intricate semantic patterns. KD encourages the student model to learn these subtle distinctions from the teacher model, resulting in gradients that contain richer and more detailed structures. When such fine-grained gradients are subjected to quantization, the limited resolution of quantization may distort or coarsen these subtle variations, leading to larger discrepancies between gradients before and after quantization. As a result, the relative gradient change increases and the signal-to-noise ratio (SNR) decreases, while the cosine similarity between gradients is reduced. Although this observation may appear counterintuitive, it does not imply that KD degrades training performance. Instead, it reflects the fact that gradient-level quantization metrics do not always directly correlate with final model accuracy, especially in complex learning scenarios.

Fig. 9. The effect of adaptive quantization under various heterogeneous condition

<!-- image -->

## G. Effect of Adaptive Dithering Quantization

In this section, we conduct ablation studies to examine the impact of adaptive DQ on training efficiency. Specifically, we evaluate model performance under three different levels of heterogeneity. As illustrated in Fig. 9(a), the performance of AdaDQ-KD improves as the degree of heterogeneity decreases. This behavior is expected, as client heterogeneity directly affects gradient aggregation, and increasing the parameter α alleviates this effect, albeit with diminishing marginal gains as α increases.

According to Fig. 9, the introduction of the quantization parameter n significantly reduces training time, often with negligible accuracy loss. As shown in Fig. 9(b), compared with the relatively simple FMNIST task, the training performance on CIFAR-10 exhibits more pronounced variations. This can be attributed to the higher complexity and non-linearity of the CIFAR-10 dataset, which makes the learning dynamics more sensitive to global data distribution shifts and inter-client heterogeneity. It is worth noting that setting α = 0 . 5 leads to a slight accuracy degradation on the CIFAR-10 dataset in Fig. 9(b), reflecting the inherent trade-off between efficiency and accuracy. To address this issue, our method is explicitly designed to mitigate accuracy loss induced by adaptive DQ.

Overall, these results validate the effectiveness of the proposed adaptive DQ scheme. By adaptively adjusting the quantization process, AdaDQ-KD achieves substantial efficiency gains while maintaining robust performance. Although minor accuracy compromises may arise in complex tasks, our approach effectively mitigates their impact, ultimately outperforming existing DP-FL schemes, as detailed in the prior section.

## H. Effect of Adaptive Quantization on Privacy Bugdet

In this section, we evaluate the impact of adaptive DQ on privacy budget. First, we examine the per-round consumption of the privacy budget. As shown in Fig. 10, AdaDQ-KD incurs lower privacy budget consumption compared with FLDQ-KD, while maintaining comparable training accuracy. When adaptive adjustment is not applied, the privacy budget consumption increases in a nearly linear manner across training rounds. In contrast, with adaptive adjustment enabled, the overall privacy budget consumption is reduced by approximately half relative to the non-adaptive setting. Moreover, the consumption pattern deviates from strict linearity and exhibits moderate fluctuations due to the dynamic adjustment mechanism.

Fig. 10. The effect of adaptive quantization on privacy budget

<!-- image -->

Fig. 11. The effect of dynamic privacy budget consume on accuracy

Furthermore, Fig. 11 illustrates the relationship between training accuracy and privacy budget consumption. Despite the reduced privacy budget, the final training accuracy remains largely unaffected. This demonstrates that our adaptive DQ strategy effectively maintains model utility while ensuring privacy.

## I. Effect of Quantization Precision

In this section, we validate the effect of DQ precision coefficient n 0 without adjustments. As we can see from Fig. 12(a), a larger n 0 brings an overall better global model in FMNIST training. As shown in Fig. 12(a), the model utility decreases with the reduction in quantization precision n 0 , despite some fluctuations in the later stages of training. Additionally, at the early training stage, the large precision n 0 = 2 and n 0 = 3 actually show an overwhelming advantage. When n 0 = 10 , a relatively high quantization precision, it results in a noticeable improvement during the early and midto-late stages of training. However, the final achieved accuracy is comparable to that obtained with other values.

Fig. 12. The test accuracy of AdaDQ-KD over heterogenous quantization precision

<!-- image -->

Fig. 13. The test accuracy of AdaDQ-KD over different privacy budgets

<!-- image -->

While for CIFAR-10 in Fig. 12(b), the situation becomes different. In the early stage of training, test accuracy across all different precision coefficients is near the same. But in the late training, the superiority of a larger one ( n 0 = 2 , n 0 = 3 and n 0 = 10 ) starts to become significant. On the whole, a higher quantization precision contributes to a higher test accuracy.

To sum up, the DQ precision coefficient n 0 affects both test accuracy and transmitted bits with a positive correlation. In other words, achieving higher accuracy simultaneously leads to a higher communication load. This result validates the conception of our work, and also demenstrates the trade-off effect of n 0 .

## J. Effect of Privacy Budget

We evaluate the performance of AdaDQ-KD under different privacy demands. Fig. 13(a) illustrates the performance on FMNIST under various privacy levels. As we can see that when ϵ = 0 . 1 , model training struggles to converge, leading to a precipitous decline in performance. Once the training precision exceeds a certain threshold, the model converges steadily, and with the change in the privacy budget, the fluctuation range of model performance has increased. In some cases, a higher privacy level may outperform a lower one (e.g., interation [25,30] where ϵ = 0 . 8 and ϵ = 1 . 0 ). In addition, we introduce the reference baseline for infinite privacy budget. Although the training performance during the training process is generally better than that with a finite privacy budget, we can conclude that introducing adaptive DQ to protect privacy while improving communication efficiency does not significantly affect the model's training performance at the final convergence stage.

For the CIFAR-10 dataset in Fig. 13(b), the impact of the privacy budget is similar to that with FMNIST. A highly restrictive privacy budget of ϵ = 0 . 1 severely degrades the model's performance. However, once the privacy budget surpasses a threshold, the model performs as expected. However, unlike the FMNIST dataset, the impact of the privacy budget in the early stages of training is negligible, except when ϵ = 0 . 6 . While in the later stages of training, it is evident that a higher privacy budget consistently outperforms a lower one. Similarly, we also consider the case where the privacy budget is set to infinity. Unlike the FMNIST dataset, for the CIFAR-10 dataset, there is a slight gain in the early stages when privacy protection is not applied, compared to when a high privacy budget is used. However, in the later stages of model training, even with a limited privacy budget, the model's performance does not significantly degrade.

Fig. 14. Composition of the local delay of each client over different algorithms in last global iteration, where bottom lays the computation latency and top lays the communication latency

<!-- image -->

## K. Alleviation for Influence of Stragglers

To demonstrate the effect for mitigating the heterogeneity of clients by straggler control of our proposed AdaDQ-KD algorithm, we separately give a detailed composition of local delay in last global iteration for each client in Fig. 14, where the total number of participating clients is 10 and the training dataset is CIFAR-10 with Non-IID setting.

As shown in Fig. 14, the computation time over two algorithms is pretty close. That a client becomes the straggler is mainly because of its bad communication latency, which is exactly what our scheme will dynamically adjust throughout the training process. Compared with baseline FLDQ-KD algorithm without quantization precision adjustment, our proposed AdaDQ-KD algorithm significantly reduces communication time of all clients to mitigate the straggler's influence. Specially, the C 1 's decrement is relatively small, indicating that C 1 is expected to have a short local delay, so there has been nearly no change for its quantization precision to keep a more accurate gradient.

The communication load of all clients except for C 1 has been significantly reduced. For stragglers including C 4 , C 6 and C 9 , the reduction in communication latency is more substantial. Our method tends to keep the quantization precision as high as possible for those clients with least (or nearly least) computation time, as long as the local delay does not exceed the straggler. In this way, we can improve the model's utility without increasing the training time, achieving a better tradeoff between model accuracy and training time.

## VI. CONCLUSION

In the proposed scheme, an adaptive mechanism is introduced to dynamically modify the DQ levels based on the clients' local delays, mitigating the issue of unnecessary waiting caused by disparities in clients' transmission and computation capabilities. At the same time, we have employed KD to achieve feature alignment, thereby mitigating the adverse effects of noise injection. We have also conducted theoretical analyses of the proposed AdaDQ-KD algorithm regarding its privacy, efficiency, and convergence properties, along with a variety of simulation experiments to validate its effectiveness. Admittedly, while the proposed algorithm improves the robustness of FL training process, its effectiveness depends on the quality of the teacher model, which must provide reliable guidance. In addition, the introduced operations increase computational complexity, although this overhead is acceptable given the reduction in communication cost. In future work, we intend to explore additional strategies, such as client scheduling and the design of specialized quantization mechanisms, to further enhance our proposed scheme. Additionally, we plan to consider the generalization of our algorithm to broader applications.

## REFERENCES

- [1] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y. Arcas, 'Communication-Efficient Learning of Deep Networks from Decentralized Data,' in Proceedings of the 20th International Conference on Artificial Intelligence and Statistics , Proceedings of Machine Learning Research, pp. 1273-1282, PMLR, 2017.
- [2] P. Kairouz, H. B. McMahan, B. Avent, A. Bellet, M. Bennis, A. N. Bhagoji, K. Bonawitz, Z. Charles, G. Cormode, R. Cummings, et al. , 'Advances and open problems in federated learning,' Foundations and trends® in machine learning , vol. 14, no. 1-2, pp. 1-210, 2021.
- [3] S. Niknam, H. S. Dhillon, and J. H. Reed, 'Federated learning for wireless communications: Motivation, opportunities, and challenges,' IEEE Communications Magazine , vol. 58, no. 6, pp. 46-51, 2020.
- [4] Z. Lu, H. J. Asghar, M. A. Kaafar, D. Webb, and P. Dickinson, 'A differentially private framework for deep learning with convexified loss functions,' IEEE Transactions on Information Forensics and Security , vol. 17, pp. 2151-2165, 2022.
- [5] L. Zhu, Z. Liu, and S. Han, 'Deep leakage from gradients,' in Advances in Neural Information Processing Systems (H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alch´ e-Buc, E. Fox, and R. Garnett, eds.), vol. 32, Curran Associates, Inc., 2019.
- [6] C. Dwork, 'Differential privacy,' in International colloquium on automata, languages, and programming , pp. 1-12, Springer, 2006.
- [7] I. Mironov, 'R´ enyi differential privacy,' in 2017 IEEE 30th computer security foundations symposium (CSF) , pp. 263-275, IEEE, 2017.
- [8] B. Hasırcıo˘ glu and D. G¨ und¨ uz, 'Communication efficient private federated learning using dithering,' in ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pp. 7575-7579, IEEE, 2024.
- [9] R. Mishra, H. P. Gupta, and T. Dutta, 'A survey on deep neural network compression: Challenges, overview, and solutions,' arXiv preprint arXiv:2010.03954 , 2020.
- [10] M. Aghapour, A. Ferdowsi, and W. Saad, 'Two-bit aggregation for communication efficient and differentially private federated learning,' arXiv preprint arXiv:2110.03017 , 2021.
- [11] K. Wei, J. Li, C. Ma, M. Ding, F. Shu, H. Zhao, W. Chen, and H. Zhu, 'Gradient sparsification for efficient wireless federated learning with differential privacy,' Science China Information Sciences , vol. 67, no. 4, p. 142303, 2024.
- [12] Y. Shi, K. Wei, L. Shen, J. Li, X. Wang, B. Yuan, and S. Guo, 'Efficient federated learning with enhanced privacy via lottery ticket pruning in edge computing,' IEEE Transactions on Mobile Computing , vol. 23, no. 10, pp. 9946-9958, 2024.
- [13] J. Sun, T. Chen, G. B. Giannakis, Q. Yang, and Z. Yang, 'Lazily aggregated quantized gradient innovation for communication-efficient federated learning,' IEEE transactions on pattern analysis and machine intelligence , vol. 44, no. 4, pp. 2031-2044, 2022.
- [14] Y. Youn, Z. Hu, J. Ziani, and J. Abernethy, 'Randomized quantization is all you need for differential privacy in federated learning,' arXiv preprint arXiv:2306.11913 , 2023.
- [15] A. Reisizadeh, I. Tziotis, H. Hassani, A. Mokhtari, and R. Pedarsani, 'Straggler-resilient federated learning: Leveraging the interplay between statistical accuracy and system heterogeneity,' IEEE Journal on Selected Areas in Information Theory , vol. 3, no. 2, pp. 197-205, 2022.
- [16] R. Schlegel, S. Kumar, E. Rosnes, and A. G. i. Amat, 'Codedpaddedfl and codedsecagg: Straggler mitigation and secure aggregation in federated learning,' IEEE Transactions on Communications , vol. 71, no. 4, pp. 2013-2027, 2023.
- [17] J. Park, D.-J. Han, M. Choi, and J. Moon, 'Sageflow: Robust federated learning against both stragglers and adversaries,' Advances in neural information processing systems , vol. 34, pp. 840-851, 2021.
- [18] H. Liu, F. He, and G. Cao, 'Communication-efficient federated learning for heterogeneous edge devices based on adaptive gradient quantization,' in IEEE INFOCOM 2023-IEEE Conference on Computer Communications , pp. 1-10, IEEE, 2023.
- [19] F. Zhu, J. Zhang, and X. Wang, 'Stsyn: Speeding up local sgd with straggler-tolerant synchronization,' IEEE Transactions on Signal Processing , vol. 72, pp. 4050-4064, 2024.
- [20] I. Tziotis, Z. Shen, R. Pedarsani, H. Hassani, and A. Mokhtari, 'Straggler-resilient personalized federated learning,' arXiv preprint arXiv:2206.02078 , 2022.
- [21] Z. Zhao, Y. Mao, Z. Shi, Y. Liu, T. Lan, W. Ding, and X.-P. Zhang, 'Aquila: Communication efficient federated learning with adaptive quantization in device selection strategy,' IEEE Transactions on Mobile Computing , vol. 23, no. 6, pp. 7363-7376, 2024.
- [22] J. Feng, L. Liu, Q. Pei, and K. Li, 'Min-max cost optimization for efficient hierarchical federated learning in wireless edge networks,' IEEE Transactions on Parallel and Distributed Systems , vol. 33, no. 11, pp. 2687-2700, 2021.
- [23] J. Fu, Z. Chen, and X. Han, 'Adap dp-fl: Differentially private federated learning with adaptive noise,' in 2022 IEEE International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom) , pp. 656-663, IEEE, 2022.
- [24] Z. Zhang, Y. Guo, Y. Fang, and Y. Gong, 'Communication and energy efficient wireless federated learning with intrinsic privacy,' IEEE Transactions on Dependable and Secure Computing , vol. 21, no. 4, pp. 40354047, 2024.
- [25] G. Hinton, O. Vinyals, and J. Dean, 'Distilling the knowledge in a neural network,' arXiv preprint arXiv:1503.02531 , 2015.
- [26] Y. Liu, Y. Kang, and T. Chen, 'A secure federated transfer learning framework,' IEEE Intelligent Systems , vol. 35, no. 4, pp. 70-82, 2020.
- [27] C. Wu, F. Wu, L. Lyu, Y. Huang, and X. Xie, 'Communication-efficient federated learning via knowledge distillation,' Nature Communications , vol. 13, no. 1, p. 2032, 2022.
- [28] A. Romero, N. Ballas, S. E. Kahou, A. Chassang, C. Gatta, and Y. Bengio, 'Fitnets: Hints for thin deep nets,' arXiv preprint arXiv:1412.6550 , 2015.
- [29] G. Gad, E. Gad, Z. M. Fadlullah, M. M. Fouda, and N. Kato, 'Communication-efficient and privacy-preserving federated learning via joint knowledge distillation and differential privacy in bandwidthconstrained networks,' IEEE Transactions on Vehicular Technology , vol. 73, no. 11, pp. 17586-17601, 2024.
- [30] X. Qu, J. Wang, and J. Xiao, 'Quantization and knowledge distillation for efficient federated learning on edge devices,' in 2020 IEEE 22nd International Conference on High Performance Computing and Communications; IEEE 18th International Conference on Smart City; IEEE 6th International Conference on Data Science and Systems (HPCC/SmartCity/DSS) , pp. 967-972, 2020.
- [31] C. Dwork, A. Roth, et al. , 'The algorithmic foundations of differential privacy,' Foundations and Trends® in Theoretical Computer Science , vol. 9, no. 3-4, pp. 211-407, 2014.
- [32] S. Walker, 'The uniform power distribution,' Journal of Applied Statistics , vol. 26, no. 4, pp. 509-517, 1999.
- [33] S. P. Lipshitz, R. A. Wannamaker, and J. Vanderkooy, 'Quantization and dither: A theoretical survey,' Journal of the audio engineering society , vol. 40, no. 5, pp. 355-375, 1992.
- [34] L. Roberts, 'Picture coding using pseudo-random noise,' IRE Transactions on Information Theory , vol. 8, no. 2, pp. 145-154, 1962.
- [35] X. Yuan, W. Ni, M. Ding, K. Wei, J. Li, and H. V. Poor, 'Amplitudevarying perturbation for balancing privacy and utility in federated learning,' IEEE Transactions on Information Forensics and Security , vol. 18, pp. 1884-1897, 2023.
- [36] K. Wei, J. Li, M. Ding, C. Ma, H. H. Yang, F. Farokhi, S. Jin, T. Q. Quek, and H. V. Poor, 'Federated learning with differential privacy: Algorithms and performance analysis,' IEEE transactions on information forensics and security , vol. 15, pp. 3454-3469, 2020.

- [37] M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, and L. Zhang, 'Deep learning with differential privacy,' in Proceedings of the 2016 ACM SIGSAC conference on computer and communications security , pp. 308-318, 2016.
- [38] L. Bottou, F. E. Curtis, and J. Nocedal, 'Optimization methods for largescale machine learning,' SIAM review , vol. 60, no. 2, pp. 223-311, 2018.
- [39] H. Xiao, K. Rasul, and R. Vollgraf, 'Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms,' arXiv preprint arXiv:1708.07747 , 2017.
- [40] A. Krizhevsky, G. Hinton, et al. , 'Learning multiple layers of features from tiny images,' 2009.
- [41] T.-M. H. Hsu, H. Qi, and M. Brown, 'Measuring the effects of nonidentical data distribution for federated visual classification,' arXiv preprint arXiv:1909.06335 , 2019.
- [42] K. He, X. Zhang, S. Ren, and J. Sun, 'Deep residual learning for image recognition,' in Proceedings of the IEEE conference on computer vision and pattern recognition , pp. 770-778, 2016.
- [43] Q. Zheng, S. Chen, Q. Long, and W. Su, 'Federated f-differential privacy,' in Proceedings of The 24th International Conference on Artificial Intelligence and Statistics (A. Banerjee and K. Fukumizu, eds.), vol. 130 of Proceedings of Machine Learning Research , pp. 2251-2259, PMLR, 13-15 Apr 2021.