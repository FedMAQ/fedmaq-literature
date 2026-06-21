## Quantization and Knowledge Distillation for Efficient Federated Learning on Edge Devices

Xiaoyang Qu, Jianzong Wang /Letter ∗ , Jing Xiao Ping An Technology (Shenzhen) Co., Ltd.

Shenzhen, China

quxiaoy@gmail.com, jzwang@188.com, xiaojing661@pingan.com.cn

Abstract -Federated learning enables distributed machine learning for decentralized data on edge devices. As communication is a critical bottleneck for federated learning, we utilize model compression techniques for efficient federated learning. First, we propose an adaptive quantized federated average algorithm to reduce the communication cost by dynamically quantizing neural networks' weights. Then, we design a federated knowledge distillation method to achieve high-quality small models with limited labeled data. Adaptive quantized federated learning can significantly speed up model training while retaining model accuracy. With a small fraction of data as labeled data, our federated knowledge distillation can reach a fixed accuracy achieved by supervised learning with the entire labeled data set. Index Terms -Federated Learning, Knowledge Distillation,

Quantization, Federated Edge Intelligence.

## I. INTRODUCTION

The conventional machine learning/deep learning paradigm fits models on centralized data, which is pooled together from multiple data sources. However, the privacy and security of user data attracted more and more attention. Mobile devices and IoT devices have become the primary computing resources. There has been a surge of interest in using a distributed machine learning paradigm from decentralized data on edge devices. Federated learning [24] is proposed for distributed model training without sharing the raw data of clients.

Federated learning enables distributed machine learning directly on edge devices. In typical federated learning settings, a centralized server is used to collect gradient from clients, average the aggregated gradients, and distribute the averaged gradients to clients. For each communication round, only a fraction of devices are chosen for training considering the communication cost and scalability.

To enable distributed training on edge devices, federated learning has to tackle some challenges: communication, heterogeneity, Non-IID data, privacy, and security. Communication is a critical bottleneck for the performance of federated learning due to considerable weight of deep neural networks and aggregated gradients. We focus on the communication problem in this paper.

The communication challenges can be tackled by gradient quantization and gradient sparsity. Heterogeneous edge devices in the federated setting have different communication bandwidth, so it is wise to determine a suitable quantization bit for each device. Thus, we propose an adaptive quantized federated learning to reduce the communication cost by dynamically quantizing the weights of neural networks. In detail, the quantization level of edge device is set depending on the communication bandwidth and parallelism.

*Corresponding author: Jianzong Wang, jzwang@188.com

With our adaptive quantized federated learning, we can achieve a high-quality and relatively large model. Using this relatively large model as a teacher model, we design a federated knowledge distillation to teach small student models by mimicking the behaviors of the large teacher model in the federated setting. As it is easier and cheaper to fetch unlabeled data in a federated environment, our federated knowledge distillation can leverage unlabeled data to reduce the reliance on a large labeled dataset. Besides, the small student models using federated knowledge distillation can generalize better to unseen data.

The contributions are shown as follows.

- We design an adaptive quantized federated average algorithm to reduce communication costs in the federated learning setting. With considering the heterogeneity, the compression rate is approximately proportional to the communication bandwidth of various edge devices.
- We propose a federated knowledge distillation method to achieve high-quality small models with limited labeled data. With a small fraction of labeled data, our federated knowledge distillation can reach a fixed accuracy achieved by supervised learning with the entire labeled data set.
- The experiment results demonstrate that our adaptive quantized federated average algorithm can obtain compression ratio of worker-to-server traffic by a factor of 4x-16x. Our method can cut the gradient size of ResNet18 from 44.67 MB to 5.6 MB, retaining the accuracy within 2% of the original model.

The outline of this paper is shown as follows. Section II reviews related works. Section III illustrates the preliminaries, such as federated learning, quantization, knowledge distillation. The design of our method is shown in SectionIV. Section V shows the our evaluation. Section VI concludes this paper.

## II. RELATED WORK

## A. Federated learning

Federated learning [24] is a distributed machine learning for decentralized data. The main challenges of federated learning include communication, heterogeneity, Non-IID data, privacy, and security problem.

Awide range of optimization paradigms have been proposed to tackle these problems. Communication is a critical bottleneck due to the considerable weight of deep neural networks. The challenge can be tackled by various model compression techniques, such as low-rank [17],sparsity [17], quantization [28], pruning [16]. Non-IID problem means the data distribution follows a independent but not identically distributed fashion. The problem can be alleviated by sharing data [33]. As a fundamental problem for federated learning, privacy and security can be preserved by homomorphic encryption [7] and differential private [1].

## B. Model Compression

Han et al. [12] propose a simple compression pipeline using pruning, weight sharing, and Huffman coding. SVD decomposition [11] is also used to compress model. As the irregular weight sparsity in [13] is not friendly for model inference acceleration, [20] and [14] propose to prune filters or channels of deep convolution neural network. Quantization has been widely studied for model compression. Matthieu et al. [10] build a binary network with binary weights. XNOR [26] speeds up the inference using binarization and shift operations. Teacher-student compression is proposed in [8]. Hinton develops this idea into knowledge distillation [15].

Model Compression in Distributed Deep Learning. Stochastic gradient descent [27] is a simple and effective optimizer for training neural networks. For distributed deep learning, the gradient aggregation results in communication bottleneck. Some works [4], [6], [31], [32] incorporate quantization techniques into distributed deep learning. Some works [3], [22] proposed gradient compression methods for distributed deep learning paradigms. Recently, many paradigms [2], [9], [19], [21], [23], [29], [30] are proposed for federated distillation or federated transfer learning.

## III. PRELIMINARIES

This section illustrates the basic concepts of federated learning, quantization, and knowledge distillation.

## A. Federated Learning

The standard distributed deep learning is trained on centralized data drawn from multiple sources. But federated learning is designed for decentralized data. As shown in Figure 1, the typical architecture of federated learning consists of a centralized server and multiple clients. Here, we assume there are K clients. At the beginning of each communication round, n k out of n clients are randomly selected. And the server synchronizes the current master model weights to local models in participating clients. Then local computation executes with global states on local data. Lastly, every participating device uploads updated weights to the centralized server. The federated optimization objective is formulated as follows.

$$
\begin{aligned}
\min _ { w \in \mathbb { R } ^ { d } } f ( w ) \, w h e r e \, f ( w ) \stackrel { d e f } { = } \sum _ { k = 1 } ^ { K } \frac { n _ { k } } { n } \times F _ { k } ( w ) \quad ( 1 ) \\ \text {Here the local objective} \ F _ { t } ( w ) \text { is calculated as } F _ { t } ( w ) \, = \,
\end{aligned}
$$

Here, the local objective F k ( w ) is calculated as F k ( w ) = 1 n k ∑ n k i =1 f i ( w ) using mini-batch stochastic gradient descent.

Fig. 1. The federated learning procedure consists of three phases. (a) In the download phase, the master model in the server synchronizes the local models in participating clients. (b) In the local computation phase, clients compute a weight update independently based on global states and their local data. (c) In the upload phase, the server aggregates updated weights to update the master model with a gradient averaging operation.

<!-- image -->

## B. Quantization

As a branch of model compression, quantization can effectively reduce the model size and communication costs in distributing setting. Because high precision parameters are not important for model performance, the weights of the neural network can be quantized into low-precision weights. For example, a representative binary quantization function is formulated as follows.

$$
\begin{aligned}
x ^ { b } & = s i g n ( x ) = \begin{cases} + 1 & \text {if } x \geq 0 \\ - 1 & \text {otherwise} \end{cases} \\ \cdot \quad ( ) \quad & \cdot \quad ( ) \quad \cdot \quad \cdot \quad f
\end{aligned}
$$

where sign ( · ) returns the sign of x.

## C. Knowledge Distillation

Knowledge distillation, also called teacher-student compression, is a compression scheme using a large teacher model to teach small student models. In other words, the small student model is trained by mimicking the behaviors of large teacher models. Ba et al. [5] formulate teacher-student compression as a multitask regression problem with L 2 loss, L = ‖ g ( x ; θ ) -z ‖ 2 2 . Here, g ( x ; θ ) is the vector of logits predicted by the student model with weights θ for the input feature vector x . And z is the vector of logits predicted by the teacher model.

Hinton et al. [15] parametrize the knowledge distillation using a temperature coefficient T. For each class, the multitasking regression problem is defined as a cross entropy loss:

$$
\begin{aligned}
L _ { T } ( \theta ) = - \sum _ { i } q _ { i } ( z / T ) \log ( q _ { i } ( g ( x ; \theta ) / T ) ) \quad ( 3 )
\end{aligned}
$$

w h e r e

$$
\text {where } \theta \, \underset { \ } r e p r e s e n t s \, \text {student model parameters to be learned}
$$

where θ represents student model parameters to be learned, and q i ( · ) is calculated as q i = exp ( z i /T ) ∑ j exp ( z i /T ) .

## A. main idea

As communication is a critical bottleneck for federated learning, we proposed an adaptive quantized method to reduce communication costs by dynamically quantizing the weights of neural networks. While it is much more convenient or cheaper to obtain unlabeled data, we also incorporate the knowledge distillation scheme into a federated learning framework. Figure 2 presents a diagram of our main idea. Firstly, we use an adaptive quantized method to train a high-quality and relatively large teacher model. Then, using the relatively large teacher model, we propose a federated knowledge distillation method with unlabeled data to teach small networks. In this way, we achieve high-quality small models with limited communication costs and limited labeled data.

Adaptive Quantized Federated Average Algorithm stochastically quantizes the gradients in the worker-to-server direction in an unbiased way. As edge devices in the federated setting have different communication bandwidth, it is wise to determine a suitable quantization level for each device.

Federated Knowledge Distillation (FKD) transfers knowledge from the teacher model to student models using unlabeled data labeled by the teacher model. It can reduce the reliance on labeled data and ensure privacy and security.

Fig. 2. A diagram of our main idea. First, we design an adaptive quantized federated average algorithm to train a high-quality, relatively large model. Then, using the relatively large model as the teacher model, we propose a federated knowledge distillation with unlabeled data to teach small student models. In this way, we achieve high-quality small models with limited communication costs and limited labeled data.

<!-- image -->

## B. Adaptive Quantized Federated Average Algorithm

In this part, we present the detailed implementation of our adaptive quantized federated averaging algorithm, which applies different scale factors based on the communication bandwidth of heterogeneous devices. In other words, the number of quantization bits is based on the communication bandwidth and parallelism.

Our adaptive quantized federated average algorithm develops from QSGD [4] and FedAvg [25]. In the federated learning, the centralized server is in charge of collecting gradients from clients, averaging collected gradients, and distributing the averaged gradients.

## IV. PROPOSED METHODS

̸

In federated average algorithm [25], only a fraction of devices are chosen for training. The federated learning problem is formulated in Formulation 1. Here, we formulate the stochastic quantization used to reduce the communication cost of SGD [27]. For any v ∈ R n with v = 0 , the quantization function Q s ( v ) is defined as

$$
\begin{aligned}
Q _ { s } ( v _ { i } ) = \| v \| _ { 2 } \cdot s g n ( v _ { i } ) \cdot \xi _ { i } ( v , s ) \\ \\ \ u s ^ { \prime \prime } ( v _ { i } ) = \| v \| _ { 2 } \cdot s g n ( v _ { i } ) \cdot \xi _ { i } ( v , s )
\end{aligned}
$$

where ‖·‖ represent the norm function and sgn ( · ) returns the sign of v i . ξ i ( v, s ) is independent stochastic rounding function, shown as follows.

$$
\begin{aligned}
\stackrel { \text {large} } { \cdot } \text {in} \quad \xi _ { i } ( v , s ) = \begin{cases} l / s & \text {with probability} 1 - p ( \frac { | v _ { i } | } { | v | | _ { 2 } } , s ) \\ ( l + 1 ) / s & \text {otherwise} \end{cases} \quad ( 5 )
\end{aligned}
$$

Here, p ( a, s ) = as -l for any a ∈ [0 , 1] . Compared to deterministic rounding, stochastic rounding has unbiased expectation.

Our adaptive quantized federated average algorithm use two auto-tuning parameters s and K . The s is corresponding to the number of quantization levels. And K means the number of clients chosen for overall devices.

The first auto-tuning parameter s considers bandwidth constraint of various devices. We expected the quantization scale factor is linear to the commutation bandwidth of devices. Considering the utilization ratio of bits, the parameter s i is calculated based on

$$
s _ { i } = 2 ^ { \lceil l o g _ { 2 } \left ( \varphi _ { m a x } / \varphi i \right ) \rceil }
$$

where ϕ i means the bandwidth of i th devices and ϕ max means the maximum bandwidth of all devices.

The second auto-tuning parameter K considers the parallelism constrain of gradient aggregation. It ensures that the aggregated communication is not larger than the limited threshold T . The tuning parameter K should meet the condition as follows.

$$
\begin{aligned}
\sum _ { i = 0 } ^ { K } \varphi _ { i } \leq T \leq \sum _ { i = 0 } ^ { K + 1 } \varphi _ { i } \\ \text {attention is based on a greedy policy} .
\end{aligned}
$$

The implementation is based on a greedy policy.

Our adaptive quantized method is used for client-to-server traffic. If we use 1-bit quantization for all edge devices, our approach can reduce the upstream traffic by a factor of up to 32x. For server-to-client traffic, we can use the low-precision method.

## C. Federated Knowledge Distillation

While the training data for the given deep network are often unavailable due to privacy problems and legal issues, federated learning can avoid this problem. Unlabeled data can be more convenient to fetch and enable the model to generalize better to unseen data. Thus, we propose a federated knowledge distillation to reduce the reliance on labeled data.

For our federated knowledge distillation, the small student network mimics the behaviors of the large student network in the federated setting. The distillation objective of federated knowledge distillation is to minimize the L 2 loss between teacher model output logits and student output logits. At training time, the distilling objective operates in conjunction with supervised learning with labeled data. The objective function is formulated as follows.

$$
\begin{aligned}
L & = \alpha L _ { s } + ( 1 - \alpha ) \cdot L _ { K D } \\ & = - \alpha t _ { i } l o g \, y _ { i } ^ { S } - ( 1 - \alpha ) \cdot \| z ^ { T } - z ^ { S } \| ^ { 2 } _ { 2 } \\ \text {re} \, L _ { s } \, \text { means the supervised loss function on labeled} \, \alpha \, \text {} \\ \text {n} \, \text { means the distillation loss function and } \, \alpha \, \text {means}
\end{aligned}
$$

## ∥ ∥ Algorithm 1 Federated Knowledge Distillation

∥ ∥ where L s means the supervised loss function on labeled data, L KD means the distillation loss function, and α means the ratio of labeled data. While L s is defined as a cross-entropy loss function αt i log y S i , L KD is defined as a L 2 loss function ∥ z T -z S ∥ 2 2 .

- 1: Input: the total number of clients K , the fraction of chosen clients C , the number of local epochs E , the minibatch size B , the ratio of labeled data α .
- 2: Sever executes:
- 3: Distribute the teacher model state W into unlabeled clients 4: for each round t ← 1 , 2 , ...T do 5: Random choose α ∗ C labeled clients S m 6: Random choose (1 -α ) ∗ C unlabeled clients S n 7: for each client m ∈ S m in prallel do 8: w t m +1 ← LabeledClientUpdate ( m,w t ) 9: end for 10: for each client n ∈ S n in prallel do 11: w n t +1 ← UnlabeledClientUpdate ( n, w t ) 12: end for 13: w t +1 ← ∑ K k =1 n k n w k t +1 14: end for 15: 16: LabledClientUpdate(m,w): //Run on client k 17: β ← (split P k into batches of size B) 18: for i ← 1 , 2 , ..E do 19: for batch b ∈ β do 20: w ←-η ∆ l ( w ; b ) 21: end for 22: end for 23: return w to server 24: 25: UnlabledClientUpdate(n,w): //Run on client k 26: β ← (split P k into batches of size B) 27: for i ← 1 , 2 , ..E do 28: for batch b ∈ β do 29: Z T = f ( W ; b ) 30: Z S = f ( w ; b ) 31: w ←-η ∆ l ( w,Z T , Z S ; b ) 32: end for 33: end for 34: return w to server

Algorithm 1 illustrates the procedure of our federated knowledge distillation. (1) For the input of the algorithm, we follow the same notations in the naive FedAvg [25]. There are five key parameters: the total number of clients K , the fraction of chosen clients C , the number of local epochs E , the minibatch size B , the ratio of labeled data α . (2) As shown in Line 3 in Algorithm 1, the teacher model is distributed into unlabeled clients. There is no need for the synchronization of the teacher model. (3) For every communication round, α ∗ C labeled clients and (1 -α ) ∗ C unlabeled clients are chosen. (4) In the server, a global model is constructed to combine the weights collected from clients. After collected weights from labeled clients and unlabeled clients, the global averaging step is implemented using w t +1 ← ∑ K k =1 n k n w k t +1 , where w t +1 means the global weights. (5) For labeled clients, the labeled clients execute in E local epochs over batches of size B in every communication round. (6) For unlabeled clients, it uses the federated distilled loss function L distill = ∥ ∥ z T -z S ∥ ∥ 2 2 to update the weights of student models.

## V. EVALUATIONS

## A. Experiments setups

We conducted experiments to evaluate adaptive quantized federated average algorithm and federated knowledge distillation on two kinds of tasks: CV and NLP application. The processing of the benchmark datasets is shown in Table I. Here, IID means 'independent identical distribution', and Non-IID means 'independent but not identical distribution'.

CIFAR-10 (IID) : The image dataset is shuffled and partitioned into 20 clients, each with 2500 samples. Thus, it is almost an identical independent distribution in every devices.

CIFAR-10 (Non-IID) : The image dataset is sorted by the label and then divided it into 80 shards of size 625. Each client is randomly assigned 4 shards.

IMDb (IID) : The text classification dataset is shuffled and then partitioned into 20 clients, each with 1250 samples.

IMDb (Non-IID) : The dataset is sorted by the label and then divided it into 100 shards of size 250. Each client is randomly assigned 5 shards.

TABLE I THE CHARACTERISTICS OF DATA-SETS

|                          | CIFAR   | CIFAR   | IMDb   | IMDb     |
|--------------------------|---------|---------|--------|----------|
|                          | IID     | Non-IID | IID    | None-IID |
| shuffle or not           | ✓       | ×       | ✓      | ×        |
| # of total samples       | 50000   | 50000   | 25000  | 25000    |
| # of edge devices        | 20      | 20      | 20     | 20       |
| # of samples per shard   | 2500    | 625     | 1250   | 250      |
| # of shard per devices   | 1       | 4       | 1      | 5        |
| # of samples per devices | 2500    | 2500    | 1250   | 1250     |

## B. The impact of quantized federated learning

FedAvg is the basic baseline introduced in [25]. Here we use two federated settings. The number of local is different, E = 5 and E = 1 , respectively.

Adaptive Quantized FedAvg is short for our adaptive federated average algorithm. The quantization bit is approximately proportional to the bandwidth of devices. The quantization bit is calculated as s i = 2 ⌈ log 2 ( ϕ max /ϕi ) ⌉ . Here, s i and ϕ denotes the quantization bits and bandwidth of i th devices.

In overall, quantized federated learning achieved a compression ratio from 4x to 16x with retaining the accuracy within 2% of the original pre-train large model. The compression rate depends on two key factors: the number of chosen client and the bandwidth of various devices. Here we use three quantization levels: 4bits, 8bits, and 16bits. Adaptive Quantized FedAvg retains the accuracy within 2% of the original model, cutting the gradient size of ResNet-18 from 44.67 MB to 5.6 MB at most.

Figure 3 shows the convergence speed when using different federated learning methods during the training of ResNet-18 on CIFAR-10 and TextCNN on IMDb in a distributed setting with 20 clients for IID and Non-IID data. The critical factors of federated configurations are C , E , and B . Here C is set as 0.6, meaning 6 out of a total of 20 clients participate in every communication round. And the B is set as 100. The E is different in compared methods. Firstly, the experiment results prove the convergence guarantees of our Adaptive Quantized FedAvg on IID data and Non-IID data. Our method quantizes every gradient update to reduce the communication cost, but it does not have a significant influence on the convergence behaviors. Secondly, the convergence speed is faster for IID data than Non-IID data. In early communication round, only a small fraction of classes are used for training, so the test accuracy is low. However, the test accuracy of these methods can reach target performance. The local dataset in every client is imbalanced, but the entire dataset in the cluster is balanced.

Fig. 3. The convergence speed when using different federated learning methods during the training of ResNet-18 on CIFAR-10 and TextCNN on IMDb in a distributed setting with 20 clients for IID and Non-IID data.

<!-- image -->

## C. The impact of federated knowledge distillation

Table II evaluated the impact of federated knowledge distillation (FKD). In this setting, we choose 30% of the total amount of data as the labeled dataset, and the rest are unlabeled datasets.

For the computer vision application, we choose Resnet-18 as the teacher model, 2NN and LeNet as student models. As shown in Table II, using federated knowledge distillation, LeNet with FKD can get 9.7% gains in the IID CIFAR-10

dataset and 15.5% gains in the Non-IID dataset. The federated knowledge distillation also works in Non-IID fashion.

For the NLP application, we choose the ALBERT [18] as the teacher model, TextCNN and BiLSTM as student models. In federated settings( C = 0 . 6 , E = 5 ), using federated knowledge distillation, TextCNN with 30% labeled data can reach the target performance.

TABLE II THE IMPACT OF FEDERATED KNOWLEDGE DISTILLATION(FKD). THE RESULT PROVES THE METHOD WORKS IN IID AND NON-IID FASHION.

|         | IID Data         | IID Data          | Non-IID Data     | Non-IID Data      |
|---------|------------------|-------------------|------------------|-------------------|
|         | Acc(%) (w/o FKD) | Acc(%) (with FKD) | Acc(%) (w/o FKD) | Acc(%) (with FKD) |
| 2NN     | 43.1             | 55.7              | 22.1             | 30.2              |
| LeNet   | 75.9             | 83.3              | 35.7             | 41.2              |
| TextCNN | 76.4             | 83.2              | 72.4             | 79.1              |
| BiLSTM  | 73.1             | 82.2              | 71.3             | 78.6              |

TABLE III THE IMPACT OF FEDERATED KNOWLEDGE DISTILLATION(FKD). THE RESULT PROVES THE METHOD WORKS IN IID AND NON-IID FASHION.

|         | CIFAR-10(IID)    | CIFAR-10(IID)     | CIFAR-10(Non-IID)   | CIFAR-10(Non-IID)   |
|---------|------------------|-------------------|---------------------|---------------------|
|         | Acc(%) (w/o FKD) | Acc(%) (with FKD) | Acc(%) (w/o FKD)    | Acc(%) (with FKD)   |
| Device1 | 75.9             | 82.4              | 35.6                | 40.5                |
| Device2 | 75.9             | 83.3              | 35.7                | 41.9                |
| Device3 | 76.3             | 83.6              | 35.3                | 43.5                |
| Device4 | 76.6             | 82.1              | 35.1                | 41.8                |
| Device5 | 76.3             | 81.9              | 35.5                | 41.3                |
|         | IMDb(IID)        | IMDb(IID)         | IMDb(Non-IID)       | IMDb(Non-IID)       |
|         | Acc(%) (w/o FKD) | Acc(%) (with FKD) | Acc(%) (w/o FKD)    | Acc(%) (with FKD)   |
| Device1 | 69.5             | 78.2              | 74.7                | 75.7                |
| Device2 | 71.7             | 81.7              | 70.2                | 78.7                |
| Device3 | 69.5             | 81.2              | 74.3                | 77.3                |
| Device4 | 73.2             | 75.7              | 72.2                | 76.3                |
| Device5 | 70.5             | 81.9              | 72.5                | 81.3                |

To future discuss the impact of federated knowledge distillation(FKD), we randomly select five devices and use the local data in these devices as their validation data. The validation accuracy is shown in Table III. The top half of Table III shows IID and Non-IID experimental results on the CIFAR-10 image classification dataset. Using federated knowledge distillation, the accuracy can reach target performance with little labeled data. The bottom half of Table III presents ID and Non-IID experimental results on the IMDb text classification dataset. The federated knowledge distillation works well both in IID and Non-IID fashion.

## D. The impact of labeled data ratio

As shown in Figure 4, the x-axis measures the proportion of labeled data to total data consisted of labeled data and unlabeled data. In Figure4(a) and (b), with a small fraction of labeled data, the accuracy of LeNet with KFD can reach about 85.35%. Along with the decrease in the ratio of labeled data, the accuracy of LeNet without FKD decreases dramatically. In Figure4(c) and (d), with a small fraction of labeled data, models with KFD can reach a target accuracy of 82.2% achieved by supervised learning with the entire labeled dataset.

Fig. 4. The impact of labeled data ratio on federated knowledge distillation

<!-- image -->

## VI. CONCLUSION

This paper integrates model compression techniques into federated learning. For efficient federated learning, we proposed two novel methods: an adaptive quantized federated average algorithm and a federated knowledge distillation method. The former algorithm achieved a worker-to-server communication compression ratio from 4x to 16x with retaining the accuracy within 2% of the original pre-train large model. Only using 30% labeled data, federated knowledge distillation can reach a target accuracy of 83.35% achieved by supervised learning with the entire labeled data set. The overall experimental results demonstrate that our adaptive quantized federated average algorithm and our federated knowledge distillation achieves high-quality small models with limited communication costs and limited labeled data.

## VII. ACKNOWLEDGMENT

This paper is supported by National Key Research and Development Program of China under grant No. 2017YFB1401202, No. 2018YFB1003500, and No. 2018YFB0204400. Corresponding author is Jianzong Wang from Ping An Technology (Shenzhen) Co., Ltd.

## REFERENCES

- [1] M. Abadi, A. Chu et al. , 'Deep learning with differential privacy,' in Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security . ACM, 2016, pp. 308-318.
- [2] J.-H. Ahn, O. Simeone, and J. Kang, 'Wireless federated distillation for distributed edge learning with heterogeneous data,' in 2019 IEEE 30th Annual International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC) . IEEE, 2019, pp. 1-6.
- [3] A. F. Aji and K. Heafield, 'Sparse communication for distributed gradient descent,' arXiv preprint arXiv:1704.05021 , 2017.
- [4] D. Alistarh, D. Grubic et al. , 'Qsgd: Communication-efficient sgd via gradient quantization and encoding,' in Advances in Neural Information Processing Systems , 2017, pp. 1709-1720.
- [5] J. Ba and R. Caruana, 'Do deep nets really need to be deep?' in Advances in neural information processing systems , 2014, pp. 26542662.
- [6] J. Bernstein, Y.-X. Wang et al. , 'signsgd: Compressed optimisation for non-convex problems,' International Conference on Machine Learning , 2018.
- [7] K. Bonawitz, V. Ivanov et al. , 'Practical secure aggregation for privacypreserving machine learning,' in Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security . ACM, 2017, pp. 1175-1191.
- [8] C. Buciluˇ a, R. Caruana, and A. Niculescu-Mizil, 'Model compression,' in Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining . ACM, 2006, pp. 535-541.
- [9] H. Chang, V. Shejwalkar et al. , 'Cronus: Robust and heterogeneous collaborative learning with black-box knowledge transfer,' arXiv preprint arXiv:1912.11279 , 2019.
- [10] M. Courbariaux, I. Hubara et al. , 'Binarized neural networks: Training deep neural networks with weights and activations constrained to+ 1 or-1,' arXiv preprint arXiv:1602.02830 , 2016.
- [11] E. L. Denton, W. Zaremba et al. , 'Exploiting linear structure within convolutional networks for efficient evaluation,' in Advances in neural information processing systems(NIPS) , 2014, pp. 1269-1277.
- [12] S. Han, H. Mao, and W. J. Dally, 'Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding,' International Conference on Learning Representations (ICLR) , 2016.
- [13] S. Han, J. Pool et al. , 'Learning both weights and connections for efficient neural network,' in Advances in neural information processing systems(NIPS) , 2015, pp. 1135-1143.
- [14] Y. He, X. Zhang, and J. Sun, 'Channel pruning for accelerating very deep neural networks,' in Proceedings of the IEEE International Conference on Computer Vision(ICCV) , 2017, pp. 1389-1397.
- [15] G. Hinton, O. Vinyals, and J. Dean, 'Distilling the knowledge in a neural network,' arXiv preprint arXiv:1503.02531 , 2015.
- [16] Y. Jiang, S. Wang et al. , 'Model pruning enables efficient federated learning on edge devices,' arXiv preprint arXiv:1909.12326 , 2019.
- [17] J. Koneˇ cn` y, H. B. McMahan et al. , 'Federated learning: Strategies for improving communication efficiency,' Neural Information Processing System , 2016.
- [18] Z. Lan, M. Chen et al. , 'Albert: A lite bert for self-supervised learning of language representations,' arXiv preprint arXiv:1909.11942 , 2019.
- [19] D. Li and J. Wang, 'Fedmd: Heterogenous federated learning via model distillation,' arXiv preprint arXiv:1910.03581 , 2019.
- [20] H. Li, A. Kadav et al. , 'Pruning filters for efficient convnets,' International Conference on Learning Representations (ICLR) , 2017.
- [21] H. Li, D. Meng et al. , 'Knowledge federation: A unified and hierarchical privacy-preserving ai framework,' in 2020 IEEE International Conference on Knowledge Graph (ICKG) . IEEE, 2020, pp. 84-91.
- [22] Y. Lin, S. Han et al. , 'Deep gradient compression: Reducing the communication bandwidth for distributed training,' arXiv preprint arXiv:1712.01887 , 2017.
- [23] Y. Liu, Y. Kang et al. , 'A secure federated transfer learning framework,' IEEE Intelligent Systems , 2020.
- [24] B. McMahan and D. Ramage, 'Federated learning: Collaborative machine learning without centralized training data,' Google Research Blog , vol. 3, 2017.
- [25] H. B. McMahan, E. Moore et al. , 'Communication-efficient learning of deep networks from decentralized data,' arXiv preprint arXiv:1602.05629 , 2016.
- [26] M. Rastegari, V. Ordonez et al. , 'Xnor-net: Imagenet classification using binary convolutional neural networks,' in European Conference on Computer Vision . Springer, 2016, pp. 525-542.
- [27] H. Robbins and S. Monro, 'A stochastic approximation method,' The annals of mathematical statistics , pp. 400-407, 1951.
- [28] F. Sattler, S. Wiedemann et al. , 'Robust and communication-efficient federated learning from non-iid data,' arXiv preprint arXiv:1903.02891 , 2019.
- [29] S. Sharma, C. Xing et al. , 'Secure and efficient federated transfer learning,' in 2019 IEEE International Conference on Big Data (Big Data) . IEEE, 2019, pp. 2569-2576.
- [30] H. Wang, M. Yurochkin et al. , 'Federated learning with matched averaging,' arXiv preprint arXiv:2002.06440 , 2020.
- [31] W. Wen, C. Xu et al. , 'Terngrad: Ternary gradients to reduce communication in distributed deep learning,' in Advances in neural information processing systems , 2017, pp. 1509-1519.
- [32] M. Yu, Z. Lin et al. , 'Gradiveq: Vector quantization for bandwidthefficient gradient aggregation in distributed cnn training,' in Advances in Neural Information Processing Systems , 2018.
- [33] Y. Zhao, M. Li et al. , 'Federated learning with non-iid data,' arXiv preprint arXiv:1806.00582 , 2018.