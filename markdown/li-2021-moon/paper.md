<!-- image -->

## Model-Contrastive Federated Learning

Qinbin Li National University of Singapore Bingsheng He National University of Singapore

qinbin@comp.nus.edu.sg hebs@comp.nus.edu.sg

## Abstract

Federated learning enables multiple parties to collaboratively train a machine learning model without communicating their local data. A key challenge in federated learning is to handle the heterogeneity of local data distribution across parties. Although many studies have been proposed to address this challenge, we find that they fail to achieve high performance in image datasets with deep learning models. In this paper, we propose MOON: modelcontrastive federated learning. MOON is a simple and effective federated learning framework. The key idea of MOON is to utilize the similarity between model representations to correct the local training of individual parties, i.e., conducting contrastive learning in model-level. Our extensive experiments show that MOON significantly outperforms the other state-of-the-art federated learning algorithms on various image classification tasks.

## 1. Introduction

Deep learning is data hungry. Model training can benefit a lot from a large and representative dataset ( e.g ., ImageNet [6] and COCO [31]). However, data are usually dispersed among different parties in practice ( e.g ., mobile devices and companies). Due to the increasing privacy concerns and data protection regulations [40], parties cannot send their private data to a centralized server to train a model.

To address the above challenge, federated learning [20, 44, 27, 26] enables multiple parties to jointly learn a machine learning model without exchanging their local data. A popular federated learning algorithm is FedAvg [34]. In each round of FedAvg, the updated local models of the parties are transferred to the server, which further aggregates the local models to update the global model. The raw data is not exchanged during the learning process. Federated learning has emerged as an important machine learning area and attracted many research interests [34, 28, 22, 25, 41, 5, 16, 2, 11]. Moreover, it has been applied in many applications such as medical imaging [21, 23], object detection [32], and landmark classification [15].

Dawn Song UC Berkeley

dawnsong@berkeley.edu

A key challenge in federated learning is the heterogeneity of data distribution on different parties [20]. The data can be non-identically distributed among the parties in many real-world applications, which can degrade the performance of federated learning [22, 29, 24]. When each party updates its local model, its local objective may be far from the global objective. Thus, the averaged global model is away from the global optima. There have been some studies trying to address the non-IID issue in the local training phase [28, 22]. FedProx [28] directly limits the local updates by ℓ 2 -norm distance, while SCAFFOLD [22] corrects the local updates via variance reduction [19]. However, as we show in the experiments (see Section 4), these approaches fail to achieve good performance on image datasets with deep learning models, which can be as bad as FedAvg.

In this work, we address the non-IID issue from a novel perspective based on an intuitive observation: the global model trained on a whole dataset is able to learn a better representation than the local model trained on a skewed subset . Specifically, we propose mo del-c on trastive learning (MOON), which corrects the local updates by maximizing the agreement of representation learned by the current local model and the representation learned by the global model. Unlike the traditional contrastive learning approaches [3, 4, 12, 35], which achieve state-of-the-art results on learning visual representations by comparing the representations of different images, MOON conducts contrastive learning in model-level by comparing the representations learned by different models. Overall, MOON is a simple and effective federated learning framework, and addresses the non-IID data issue with the novel design of model-based contrastive learning.

We conduct extensive experiments to evaluate the effectiveness of MOON. MOON significantly outperforms the other state-of-the-art federated learning algorithms [34, 28, 22] on various image classification datasets including CIFAR-10, CIFAR-100, and Tiny-Imagenet. With only lightweight modifications to FedAvg, MOON outperforms existing approaches by at least 2% accuracy in most cases. Moreover, the improvement of MOON is very significant on some settings. For example, on CIFAR-100 dataset with 100 parties, MOON achieves 61.8% top-1 accuracy, while the best top-1 accuracy of existing studies is 55%.

Figure 1. The FedAvg framework. In this paper, we focus on the second step, i.e., the local training phase.

<!-- image -->

## 2. Background and Related Work

## 2.1. Federated Learning

FedAvg [34] has been a de facto approach for federated learning. The framework of FedAvg is shown in Figure 1. There are four steps in each round of FedAvg. First, the server sends a global model to the parties. Second, the parties perform stochastic gradient descent (SGD) to update their models locally. Third, the local models are sent to a central server. Last, the server averages the model weights to produce a global model for the training of the next round.

There have been quite some studies trying to improve FedAvg on non-IID data. Those studies can be divided into two categories: improvement on local training (i.e., step 2 of Figure 1) and on aggregation (i.e., step 4 of Figure 1). This study belongs to the first category.

As for studies on improving local training, FedProx [28] introduces a proximal term into the objective during local training. The proximal term is computed based on the ℓ 2 -norm distance between the current global model and the local model. Thus, the local model update is limited by the proximal term during the local training. SCAFFOLD [22] corrects the local updates by introducing control variates. Like the training model, the control variates are also updated by each party during local training. The difference between the local control variate and the global control variate is used to correct the gradients in local training. However, FedProx shows experiments on MNIST and EMNIST only with multinomial logistic regression, while SCAFFOLD only shows experiments on EMNIST with logistic regression and 2-layer fully connected layer. The effectiveness of FedProx and SCAFFOLD on image datasets with deep learning models has not been well explored. As shown in our experiments, those studies have little or even no advantage over FedAvg, which motivates this study for a new approach of handling non-IID image datasets with deep learning models. We also notice that there are other related contemporary work [1, 30, 43] when preparing this paper. We leave the comparison between MOON and these contemporary work as future studies.

As for studies on improving the aggregation phase, FedMA [41] utilizes Bayesian non-parametric methods to match and average weights in a layer-wise manner. FedAvgM [14] applies momentum when updating the global model on the server. Another recent study, FedNova [42], normalizes the local updates before averaging. Our study is orthogonal to them and potentially can be combined with these techniques as we work on the local training phase.

Another research direction is personalized federated learning [8, 7, 10, 47, 17], which tries to learn personalized local models for each party. In this paper, we study the typical federated learning, which tries to learn a single global model for all parties.

## 2.2. Contrastive Learning

Self-supervised learning [18, 9, 3, 4, 12, 35] is a recent hot research direction, which tries to learn good data representations from unlabeled data. Among those studies, contrastive learning approaches [3, 4, 12, 35] achieve state-ofthe-art results on learning visual representations. The key idea of contrastive learning is to reduce the distance between the representations of different augmented views of the same image (i.e., positive pairs ), and increase the distance between the representations of augmented views of different images (i.e., negative pairs ).

Atypical contrastive learning framework is SimCLR [3]. Given an image x , SimCLR first creates two correlated views of this image using different data augmentation operators, denoted x i and x j . A base encoder f ( · ) and a projection head g ( · ) are trained to extract the representation vectors and map the representations to a latent space, respectively. Then, a contrastive loss (i.e., NT-Xent [38]) is applied on the projected vector g ( f ( · )) , which tries to maximize agreement between differently augmented views of the same image. Specifically, given 2 N augmented views and a pair of view x i and x j of same image, the contrastive loss for this pair is defined as

̸

$$
l _ { i , j } = - \log \frac { \exp ( \sin ( x _ { i } , x _ { j } ) / \tau ) } { \sum _ { k = 1 } ^ { 2 N } \mathbb { I } _ { [ k \neq i ] } \exp ( \sin ( x _ { i } , x _ { k } ) / \tau ) } \quad ( 1 )
$$

where sim ( · , · ) is a cosine similarity function and τ de- notes a temperature parameter. The final loss is computed by summing the contrastive loss of all pairs of the same image in a mini-batch.

Besides SimCLR, there are also other contrastive learning frameworks such as CPC [36], CMC [39] and MoCo [12]. We choose SimCLR for its simplicity and effectiveness in many computer vision tasks. Still, the basic idea of contrastive learning is similar among these studies: the representations obtained from different images should be far from each other and the representations obtained from the same image should be related to each other. The idea is intuitive and has been shown to be effective.

There is one recent study [46] that combines federated learning with contrastive learning. They focus on the unsupervised learning setting. Like SimCLR, they use contrastive loss to compare the representations of different images. In this paper, we focus on the supervised learning setting and propose model-contrastive learning to compare representations learned by different models.

## 3. Model-Contrastive Federated Learning

## 3.1. Problem Statement

Suppose there are N parties, denoted P 1 , ..., P N . Party P i has a local dataset D i . Our goal is to learn a machine learning model w over the dataset D ≜ ⋃ i ∈ [ N ] D i with the help of a central server, while the raw data are not exchanged. The objective is to solve

$$
\arg \min _ { w } \mathcal { L } ( w ) = \sum _ { i = 1 } ^ { N } \frac { | { \mathcal { D } } ^ { i } | } { | { \mathcal { D } } | } L _ { i } ( w ) , \quad ( 2 ) \quad \mod ( 2 )
$$

where L i ( w ) = E ( x,y ) ∼D i [ ℓ i ( w ; ( x, y ))] is the empirical loss of P i .

## 3.2. Motivation

MOON is based on an intuitive idea: the model trained on the whole dataset is able to extract a better feature representation than the model trained on a skewed subset. For example, given a model trained on dog and cat images, we cannot expect the features learned by the model to distinguish birds and frogs which never exist during training.

To further verify this intuition, we conduct a simple experiment on CIFAR-10. Specifically, we first train a CNN model (see Section 4.1 for the detailed structure) on CIFAR10. We use t-SNE [33] to visualize the hidden vectors of images from the test dataset as shown in Figure 2a. Then, we partition the dataset into 10 subsets in an unbalanced way (see Section 4.1 for the partition strategy) and train a CNN model on each subset. Figure 2b shows the t-SNE visualization of a randomly selected model. Apparently, the model trained on the subset learns poor features. The feature representations of most classes are even mixed and cannot be distinguished. Then, we run FedAvg algorithm on 10 subsets and show the representation learned by the global model in Figure 2c and the representation learned by a selected local model (trained based on the global model) in Figure 2d. We can observe that the points with the same class are more divergent in Figure 2d compared with Figure 2c ( e.g ., see class 9). The local training phase even leads the model to learn a worse representation due to the skewed local data distribution. This further verifies that the global model should be able to learn a better feature representation than the local model, and there is a drift in the local updates. Therefore, under non-IID data scenarios, we should control the drift and bridge the gap between the representations learned by the local model and the global model.

<!-- image -->

## 3.3. Method

Based on the above intuition, we propose MOON. MOON is designed as a simple and effective approach based on FedAvg, only introducing lightweight but novel modifications in the local training phase. Since there is always drift in local training and the global model learns a better representation than the local model, MOON aims to decrease the distance between the representation learned by the local model and the representation learned by the global model, and increase the distance between the representation learned by the local model and the representation learned by the previous local model. We achieve this from the inspiration of contrastive learning, which is now mainly used to learn visual representations. In the following, we present the network architecture, the local learning objective and the learning procedure. At last, we discuss the relation to contrastive learning.

## 3.3.1 Network Architecture

The network has three components: a base encoder, a projection head, and an output layer. The base encoder is used to extract representation vectors from inputs. Like [3], an additional projection head is introduced to map the representation to a space with a fixed dimension. Last, as we study on the supervised setting, the output layer is used to produce predicted values for each class. For ease of presentation, with model weight w , we use F w ( · ) to denote the whole network and R w ( · ) to denote the network before the output layer (i.e., R w ( x ) is the mapped representation vector of input x ).

## 3.3.2 Local Objective

As shown in Figure 3, our local loss consists two parts. The first part is a typical loss term ( e.g ., cross-entropy loss) in supervised learning denoted as ℓ sup . The second part is our proposed model-contrastive loss term denoted as ℓ con .

Suppose party P i is conducting the local training. It receives the global model w t from the server and updates the model to w t i in the local training phase. For every input x , we extract the representation of x from the global model w t (i.e., z glob = R w t ( x ) ), the representation of x from the local model of last round w t -1 i (i.e., z prev = R w t -1 i ( x ) ), and the representation of x from the local model being updated w t i (i.e., z = R w t i ( x ) ). Since the global model should be able to extract better representations, our goal is to decrease the distance between z and z glob , and increase the distance between z and z prev . Similar to NT-Xent loss [38], we define model-contrastive loss as

$$
\begin{aligned}
\ell _ { c o n } = - \log \frac { \exp ( \sin ( z , z _ { g l o b } ) / \tau ) } { \exp ( \sin ( z , z _ { g l o b } ) / \tau ) + \exp ( \sin ( z , z _ { p e r v } ) / \tau ) } \quad \, \begin{array} { c c c } 9 & \text {Pa} \\ 9 & \text {Pa} \\ 10 & w _ { i } ^ { t } \\ 9 & \text {for} \end{array}
\end{aligned}
$$

where τ denotes a temperature parameter. The loss of an input ( x, y ) is computed by

$$
\ell = \ell _ { s u p } ( w _ { i } ^ { t } ; ( x , y ) ) + \mu \ell _ { c o n } ( w _ { i } ^ { t } ; w _ { i } ^ { t - 1 } ; w ^ { t } ; x ) , \ \ ( 4 )
$$

where µ is a hyper-parameter to control the weight of model-contrastive loss. The local objective is to minimize

$$
\begin{aligned}
\min _ { w _ { i } ^ { t } } \mathbb { E } _ { ( x , y ) \sim D ^ { i } } [ \ell _ { s u p } ( w _ { i } ^ { t } ; ( x , y ) ) + \mu \ell _ { c o n } ( w _ { i } ^ { t } ; w _ { i } ^ { t - 1 } ; w ^ { t } ; x ) ] . \\ \\ \quad \ \
\end{aligned}
$$

The overall federated learning algorithm is shown in Algorithm 1. In each round, the server sends the global model to the parties, receives the local model from the parties, and updates the global model using weighted averaging. In local training, each party uses stochastic gradient descent to update the global model with its local data, while the objective is defined in Eq. (5).

Figure 3. The local loss in MOON.

<!-- image -->

## Algorithm 1: The MOON framework

Input: number of communication rounds T , number of parties N , number of local epochs E , temperature τ , learning rate η , hyper-parameter µ

Output:

The final model w T

$$
\begin{array} { r l } { t } & { \quad \text {epochs } E , \, \text {temperature } \tau , \, \text {learning rate } \eta , } \\ { w ^ { t } } & { \quad \text {hyper-parameter } \mu } \\ { w \cdot } & { \quad \text {Output: The final model } w ^ { T } } \\ { l } & { \quad \text {Server executes:} } \\ { 2 } & { \quad \text {initialize } w ^ { 0 } } \\ { 3 } & { \quad \text {for } t = 0 , 1 , \dots , T - 1 \, \text { do } } \\ { 4 } & { \quad \text {for } i = 1 , 2 , \dots , N \, \text { in parallel } 0 } \\ { 5 } & { \quad \text {change } \left [ \begin{array} { c } \sum e d \, \text { the global model } w ^ { t } \, \text { to } P _ { i } \\ \sum _ { k } \left ( \begin{array} { c } w ^ { t } \, \left ( \begin{array} { c } - \, \text {PartyLocalTraining} ( i , w ^ { t } ) \\ w ^ { t } \, \left ( \begin{array} { c } - \, \text {N} & \left | \mathcal { D } ^ { 1 } \right | w ^ { t } \\ w ^ { t + 1 } \, \left ( \begin{array} { c } - \, \text {N} & \left | \mathcal { D } ^ { 1 } \right | w ^ { t } \\ 0 \end{array} \right ) \end{array} \right ] } \\ { 8 } & { \quad \text {return } w ^ { T } } \\ { 9 } & { \quad \text {PartyLocalTraining} ( i , w ^ { t } ) \colon } \\ { 1 / \tau } & { \quad w ^ { t } \, \leftarrow \, w ^ { t } } \\ { 0 } & { \quad w ^ { t } \, \leftarrow \, w ^ { t } } \\ { 1 } & { \quad \text {for } \text {epoch } i = 1 , 2 , \dots , E \, \text { do } } \\ { 1 } & { \quad \text {for } \text { each batch } b = \{ x , y \} \, \text { of } \mathcal { D } ^ { 1 } \, \text { do } } \\ { 1 } & { \quad \text {up } \, \left ( \begin{array} { c } \ell _ { \sup } \leftarrow \, C \, \text {crossEntropyLoss} ( F _ { w ^ { t } } ( x ) , y ) \\ z \leftarrow \, R _ { w ^ { t } } ( x ) \\ z _ { glob } \leftarrow \, R _ { w ^ { t } } ( x ) \\ z _ { prev } \leftarrow \, R _ { w ^ { t - 1 } } ( x ) \\ 1 \end{array} \right ) } \\ { 4 } & { \quad \text {for } \text {epoch } i = 1 , 2 , \dots , E \, \text { do } } \\ { 0 } & { \quad \text {for } \text { each batch } \ell \, \left ( \begin{array} { c } \exp ( \sin ( z , z _ { glob } ) / \tau ) \\ - \log \frac { \exp ( \sin ( z , z _ { glob } ) / \tau ) + \exp ( \sin ( z , z _ { prev } ) / \tau ) } { \exp ( \sin ( z , z _ { glob } ) / \tau ) } \end{array} \right ) } \\ { x ) } . \quad } \\ { 8 } & { \quad \left [ \begin{array} { c } \ell \leftarrow \, \ell _ { \sup } + \mu _ { \ell } \text {con} \\ w ^ { t } \leftarrow \, w ^ { t } _ { i } - \eta \nabla \ell \\ 0 \end{array} \right ] } \\ { A - } \\ { 2 } & { \quad \text {return } w ^ { t } _ { \text { to server} } } \\ { \text {and} } \\ { \text {and} } \\ { \text {local} } \\ { \text { up -} } & { \quad \text {For simplicity, we describe MOON without applying } } \\ { 1 } & { \quad \text {and } } \\ { 0 } & { \quad \text {using} } \\ { 1 } & { \quad \text {with} } \\ { 0 } & { \quad \text {using} } \\ { 0 } & { \quad \text {with} } \\ { 0 } & { \quad \text {using} } \\ {
$$

For simplicity, we describe MOON without applying sampling technique in Algorithm 1. MOON is still applicable when only a sample set of parties participate in federated learning each round. Like FedAvg, each party maintains its local model, which will be replaced by the global model and updated only if the party is selected to participate in a round. MOONonly needs the latest local model that the party has, even though it may not be updated in round ( t -1) (e.g., w t -1 i = w t -2 i ).

Figure 4. The comparison between SimCLR and MOON. Here x denotes an image, w denotes a model, and R denotes the function to compute representation. SimCLR maximizes the agreement between representations of different views of the same image, while MOON maximizes the agreement between representations of the local model and the global model on the mini-batches.

<!-- image -->

An notable thing is that considering an ideal case where the local model is good enough and learns (almost) the same representation as the global model (i.e., z glob = z prev ), the model-contrastive loss will be a constant (i.e., -log 1 2 ). Thus, MOON will produce the same result as FedAvg, since there is no heterogeneity issue. In this sense, our approach is robust regardless of different amount of drifts.

## 3.4. Comparisons with Contrastive Learning

A comparison between MOON and SimCLR is shown in Figure 4. The model-contrastive loss compares representations learned by different models, while the contrastive loss compares representations of different images. We also highlight the key difference between MOON and traditional contrastive learning: MOON is currently for supervised learning in a federated setting while contrastive learning is for unsupervised learning in a centralized setting. Drawing the inspirations from contrastive learning, MOON is a new learning methodology in handling non-IID data distributions among parties in federated learning.

## 4. Experiments

## 4.1. Experimental Setup

We compare MOON with three state-of-the-art approaches including (1) FedAvg [34], (2) FedProx [28], and (3) SCAFFOLD [22]. We also compare a baseline approach named SOLO, where each party trains a model with its local data without federated learning. We conduct experiments on three datasets including CIFAR-10, CIFAR-100, and Tiny-Imagenet 1 (100,000 images with 200 classes). Moreover, we try two different network architectures. For CIFAR-10, we use a CNN network as the base encoder, which has two 5x5 convolution layers followed by 2x2 max pooling (the first with 6 channels and the second with 16 channels) and two fully connected layers with ReLU activation (the first with 120 units and the second with 84 units). For CIFAR-100 and Tiny-Imagenet, we use ResNet-50 [13] as the base encoder. For all datasets, like [3], we use a 2layer MLP as the projection head. The output dimension of the projection head is set to 256 by default. Note that all baselines use the same network architecture as MOON (including the projection head) for fair comparison.

We use PyTorch [37] to implement MOON and the other baselines. The code is publicly available 2 . We use the SGD optimizer with a learning rate 0.01 for all approaches. The SGD weight decay is set to 0.00001 and the SGD momentum is set to 0.9. The batch size is set to 64. The number of local epochs is set to 300 for SOLO. The number of local epochs is set to 10 for all federated learning approaches unless explicitly specified. The number of communication rounds is set to 100 for CIFAR-10/100 and 20 for Tiny-ImageNet, where all federated learning approaches have little or no accuracy gain with more communications. For MOON, we set the temperature parameter to 0.5 by default like [3].

Like previous studies [45, 41], we use Dirichlet distribution to generate the non-IID data partition among parties. Specifically, we sample p k ∼ Dir N ( β ) and allocate a p k,j proportion of the instances of class k to party j , where Dir ( β ) is the Dirichlet distribution with a concentration parameter β (0.5 by default). With the above partitioning strategy, each party can have relatively few (even no) data samples in some classes. We set the number of parties to 10 by default. The data distributions among parties in default settings are shown in Figure 5. For more experimental results, please refer to Appendix.

## 4.2. Accuracy Comparison

For MOON, we tune µ from { 0 . 1 , 1 , 5 , 10 } and report the best result. The best µ of MOON for CIFAR-10, CIFAR100, and Tiny-Imagenet are 5, 1, and 1, respectively. Note that FedProx also has a hyper-parameter µ to control the weight of its proximal term (i.e., L FedProx = ℓ FedAvg + µℓ prox ). For FedProx, we tune µ from { 0 . 001 , 0 . 01 , 0 . 1 , 1 } (the range is also used in the previous paper [28]) and report the best result. The best µ of FedProx for CIFAR-10, CIFAR-100, and Tiny-Imagenet are 0 . 01 , 0 . 001 , and 0 . 001 , respectively. Unless explicitly specified, we use these µ settings for all the remaining experiments.

Table 1 shows the top-1 test accuracy of all approaches

1 https://www.kaggle.com/c/tiny-imagenet

2 https://github.com/QinbinLi/MOON

0

Figure 5. The data distribution of each party using non-IID data partition. The color bar denotes the number of data samples. Each rectangle represents the number of data samples of a specific class in a party.

<!-- image -->

Table 1. The top-1 accuracy of MOON and the other baselines on test datasets. For MOON, FedAvg, FedProx, and SCAFFOLD, we run three trials and report the mean and standard derivation. For SOLO, we report the mean and standard derivation among all parties.

| Method   | CIFAR-10     | CIFAR-100    | Tiny-Imagenet   |
|----------|--------------|--------------|-----------------|
| MOON     | 69.1% ± 0.4% | 67.5% ± 0.4% | 25.1% ± 0.1%    |
| FedAvg   | 66.3% ± 0.5% | 64.5% ± 0.4% | 23.0% ± 0.1%    |
| FedProx  | 66.9% ± 0.2% | 64.6% ± 0.2% | 23.2% ± 0.2%    |
| SCAFFOLD | 66.6% ± 0.2% | 52.5% ± 0.3% | 16.0% ± 0.2%    |
| SOLO     | 46.3% ± 5.1% | 22.3% ± 1.0% | 8.6% ± 0.4%     |

with the above default setting. Under non-IID settings, SOLO demonstrates much worse accuracy than other federated learning approaches. This demonstrates the benefits of federated learning. Comparing different federated learning approaches, we can observe that MOON is consistently the best approach among all tasks. It can outperform FedAvg by 2.6% accuracy on average of all tasks. For FedProx, its accuracy is very close to FedAvg. The proximal term in FedProx has little influence in the training since µ is small. However, when µ is not set to a very small value, the convergence of FedProx is quite slow (see Section 4.3) and the accuracy of FedProx is bad. For SCAFFOLD, it has much worse accuracy on CIFAR-100 and Tiny-Imagenet than other federated learning approaches.

## 4.3. Communication Efficiency

Figure 6 shows the accuracy in each round during training. As we can see, the model-contrastive loss term has little influence on the convergence rate with best µ . The speed of accuracy improvement in MOON is almost the same as FedAvg at the beginning, while it can achieve a better accuracy later benefit from the model-contrastive loss. Since the best µ values are generally small in FedProx, FedProx with best µ is very close to FedAvg, especially on CIFAR-10 and CIFAR-100. However, when setting µ = 1 , FedProx becomes very slow due to the additional proximal term. This implies that limiting the ℓ 2 -norm distance between the local model and the global model is not an effective solution. Our model-contrastive loss can effectively increase the accuracy without slowing down the convergence.

Table 2. The number of rounds of different approaches to achieve the same accuracy as running FedAvg for 100 rounds (CIFAR10/100) or 20 rounds (Tiny-Imagenet). The speedup of an approach is computed against FedAvg.

| Method   | CIFAR-10   | CIFAR-10   | CIFAR-100   | CIFAR-100   | Tiny-Imagenet   | Tiny-Imagenet   |
|----------|------------|------------|-------------|-------------|-----------------|-----------------|
| Method   | #rounds    | speedup    | #rounds     | speedup     | #rounds         | speedup         |
| FedAvg   | 100        | 1 ×        | 100         | 1 ×         | 20              | 1 ×             |
| FedProx  | 52         | 1.9 ×      | 75          | 1.3 ×       | 17              | 1.2 ×           |
| SCAFFOLD | 80         | 1.3 ×      |             | < 1 ×       |                 | < 1 ×           |
| MOON     | 27         | 3.7 ×      | 43          | 2.3 ×       | 11              | 1.8 ×           |

We show the number of communication rounds to achieve the same accuracy as running FedAvg for 100 rounds on CIFAR-10/100 or 20 rounds on Tiny-Imagenet in Table 2. We can observe that the number of communication rounds is significantly reduced in MOON. MOON needs about half the number of communication rounds on CIFAR-100 and Tiny-Imagenet compared with FedAvg. On CIFAR-10, the speedup of MOON is even close to 4. MOON is much more communication-efficient than the other approaches.

## 4.4. Number of Local Epochs

We study the effect of number of local epochs on the accuracy of final model. The results are shown in Figure 7. When the number of local epochs is 1, the local update is very small. Thus, the training is slow and the accuracy is relatively low given the same number of communication rounds. All approaches have a close accuracy (MOON is still the best). When the number of local epochs becomes too large, the accuracy of all approaches drops, which is due to the drift of local updates, i.e., the local optima are not consistent with the global optima. Nevertheless, MOON clearly outperforms the other approaches. This further verifies that MOON can effectively mitigate the negative effects

<!-- image -->

(a) CIFAR-10

(b) CIFAR-100

(c) Tiny-Imagenet

Figure 6. The top-1 test accuracy in different number of communication rounds. For FedProx, we report both the accuracy with best µ and the accuracy with µ = 1 .

<!-- image -->

Figure 7. The top-1 test accuracy with different number of local epochs. For MOON and FedProx, µ is set to the best µ from Section 4.2 for all numbers of local epochs. The accuracy of SCAFFOLD is quite bad when number of local epochs is set to 1 (45.3% on CIFAR10, 20.4% on CIFAR-100, 2.6% on Tiny-Imagenet). The accuracy of FedProx on Tiny-Imagenet with one local epoch is 1.2%.

of the drift by too many local updates.

## 4.5. Scalability

To show the scalability of MOON, we try a larger number of parties on CIFAR-100. Specifically, we try two settings: (1) We partition the dataset into 50 parties and all parties participate in federated learning in each round. (2) We partition the dataset into 100 parties and randomly sample 20 parties to participate in federated learning in each round (client sampling technique introduced in FedAvg [34]). The results are shown in Table 3 and Figure 8. For MOON, we show the results with µ = 1 (best µ from Section 4.2) and µ = 10 . For MOON ( µ = 1 ), it outperforms the FedAvg and FedProx over 2% accuracy at 200 rounds with 50 parties and 3% accuracy at 500 rounds with 100 parties. Moreover, for MOON ( µ = 10 ), although the large model-contrastive loss slows down the training at the beginning as shown in Figure 8, MOON can outperform the other approaches a lot with more communication rounds. Compared with FedAvg and FedProx, MOON achieves about about 7% higher accuracy at 200 rounds with 50 parties and at 500 rounds with 100 parties. SCAFFOLD has a low accuracy with a relatively large number of parties.

Table 3. The accuracy with 50 parties and 100 parties (sample fraction=0.2) on CIFAR-100.

| Method       | #parties=50   | #parties=50   | #parties=100   | #parties=100   |
|--------------|---------------|---------------|----------------|----------------|
|              | 100 rounds    | 200 rounds    | 250 rounds     | 500 rounds     |
| MOON( µ =1)  | 54.7%         | 58.8%         | 54.5%          | 58.2%          |
| MOON( µ =10) | 58.2%         | 63.2%         | 56.9%          | 61.8%          |
| FedAvg       | 51.9%         | 56.4%         | 51.0%          | 55.0%          |
| FedProx      | 52.7%         | 56.6%         | 51.3%          | 54.6%          |
| SCAFFOLD     | 35.8%         | 44.9%         | 37.4%          | 44.5%          |
| SOLO         | 10% ± 0.9%    | 10% ± 0.9%    | 7.3% ± 0.6%    | 7.3% ± 0.6%    |

## 4.6. Heterogeneity

We study the effect of data heterogeneity by varying the concentration parameter β of Dirichlet distribution on CIFAR-100. For a smaller β , the partition will be more unbalanced. The results are shown in Table 4. MOON always achieves the best accuracy among three unbalanced levels. When the unbalanced level decreases (i.e., β = 5 ), FedProx is worse than FedAvg, while MOON still outperforms FedAvg with more than 2% accuracy. The experiments demonstrate the effectiveness and robustness of MOON.

## 4.7. Loss Function

To maximize the agreement between the representation learned by the global model and the representation learned by the local model, our model-contrastive loss ℓ con is proposed inspired by NT-Xent loss [3]. Another intuitive option is to use ℓ 2 regularization, and the local loss is

<!-- image -->

Table 4. The test accuracy with β from { 0.1, 0.5, 5 } .

| Method   | β = 0 . 1    | β = 0 . 5   | β = 5        |
|----------|--------------|-------------|--------------|
| MOON     | 64.0%        | 67.5%       | 68.0%        |
| FedAvg   | 62.5%        | 64.5%       | 65.7%        |
| FedProx  | 62.9%        | 64.6%       | 64.9%        |
| SCAFFOLD | 47.3%        | 52.5%       | 55.0%        |
| SOLO     | 15.9% ± 1.5% | 22.3% ± 1%  | 26.6% ± 1.4% |

$$
\begin{aligned}
\ell = \ell _ { s u p } + \mu \left \| z - z _ { g l o b } \right \| _ { 2 }
\end{aligned}
$$

Here we compare the approaches using different kinds of loss functions to limit the representation: no additional term (i.e., FedAvg: L = ℓ sup ), ℓ 2 norm, and our modelcontrastive loss. The results are shown in Table 5. We can observe that simply using ℓ 2 norm even cannot improve the accuracy compared with FedAvg on CIFAR-10. While using ℓ 2 norm can improve the accuracy on CIFAR-100 and Tiny-Imagenet, the accuracy is still lower than MOON. Our model-contrastive loss is an effective way to constrain the representations.

Our model-contrastive loss influences the local model from two aspects. Firstly, the local model learns a close representation to the global model. Secondly, the local model also learns a better representation than the previous one until the local model is good enough (i.e., z = z glob and ℓ con becomes a constant).

## 5. Conclusion

Federated learning has become a promising approach to resolve the pain of data silos in many domains such as medical imaging, object detection, and landmark classification. Non-IID is a key challenge for the effectiveness of federated learning. To improve the performance of federated deep learning models on non-IID datasets, we propose model-contrastive learning (MOON), a simple and effective approach for federated learning. MOON introduces a new learning concept, i.e., contrastive learning in model-level. Our extensive experiments show that MOON achieves significant improvement over state-of-the-art approaches on various image classification tasks. As MOON does not require the inputs to be images, it potentially can be applied to non-vision problems.

Table 5. The top-1 accuracy with different kinds of loss for the second term of local objective. We tune µ from { 0.001, 0.01 , 0.1, 1, 5, 10 } for the ℓ 2 norm approach and report the best accuracy.

| second term   | CIFAR-10   | CIFAR-100   | Tiny-Imagenet   |
|---------------|------------|-------------|-----------------|
| none (FedAvg) | 66.3%      | 64.5%       | 23.0%           |
| ℓ 2 norm      | 65.8%      | 66.9%       | 24.0%           |
| MOON          | 69.1%      | 67.5%       | 25.1%           |

## Acknowledgements

This research is supported by the National Research Foundation, Singapore under its AI Singapore Programme (AISG Award No: AISG2-RP-2020-018). Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not reflect the views of National Research Foundation, Singapore. The authors thank Jianxin Wu, Chaoyang He, Shixuan Sun, Yaqi Xie and Yuhang Chen for their feedback. The authors also thank Yuzhi Zhao, Wei Wang, and Mo Sha for their supports of computing resources.

## References

- [1] Durmus Alp Emre Acar, Yue Zhao, Ramon Matas, Matthew Mattina, Paul Whatmough, and Venkatesh Saligrama. Federated learning based on dynamic regularization. In International Conference on Learning Representations , 2021.
- [2] Sebastian Caldas, Sai Meher Karthik Duddu, Peter Wu, Tian Li, Jakub Koneˇ cn` y, H Brendan McMahan, Virginia Smith, and Ameet Talwalkar. Leaf: A benchmark for federated settings. arXiv preprint arXiv:1812.01097 , 2018.
- [3] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. arXiv preprint arXiv:2002.05709 , 2020.
- [4] Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey Hinton. Big self-supervised models are strong semi-supervised learners. arXiv preprint arXiv:2006.10029 , 2020.
- [5] Zhongxiang Dai, Bryan Kian Hsiang Low, and Patrick Jaillet. Federated bayesian optimization via thompson sampling. Advances in Neural Information Processing Systems , 33, 2020.
- [6] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition , pages 248-255. Ieee, 2009.
- [7] Canh T Dinh, Nguyen H Tran, and Tuan Dung Nguyen. Personalized federated learning with moreau envelopes. arXiv preprint arXiv:2006.08848 , 2020.
- [8] Alireza Fallah, Aryan Mokhtari, and Asuman Ozdaglar. Personalized federated learning with theoretical guarantees: A model-agnostic meta-learning approach. Advances in Neural Information Processing Systems , 33, 2020.
- [9] Jean-Bastien Grill, Florian Strub, Florent Altch´ e, Corentin Tallec, Pierre H Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent: A new approach to self-supervised learning. arXiv preprint arXiv:2006.07733 , 2020.
- [10] Filip Hanzely, Slavom´ ır Hanzely, Samuel Horv´ ath, and Peter Richt´ arik. Lower bounds and optimal algorithms for personalized federated learning. arXiv preprint arXiv:2010.02372 , 2020.
- [11] Chaoyang He, Songze Li, Jinhyun So, Mi Zhang, Hongyi Wang, Xiaoyang Wang, Praneeth Vepakomma, Abhishek Singh, Hang Qiu, Li Shen, Peilin Zhao, Yan Kang, Yang Liu, Ramesh Raskar, Qiang Yang, Murali Annavaram, and Salman Avestimehr. Fedml: A research library and benchmark for federated machine learning. arXiv preprint arXiv:2007.13518 , 2020.
- [12] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 9729-9738, 2020.
- [13] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceed-
14. ings of the IEEE conference on computer vision and pattern recognition , pages 770-778, 2016.
- [14] Tzu-Ming Harry Hsu, Hang Qi, and Matthew Brown. Measuring the effects of non-identical data distribution for federated visual classification. arXiv preprint arXiv:1909.06335 , 2019.
- [15] Tzu-Ming Harry Hsu, Hang Qi, and Matthew Brown. Federated visual classification with real-world data distribution. arXiv preprint arXiv:2003.08082 , 2020.
- [16] Sixu Hu, Yuan Li, Xu Liu, Qinbin Li, Zhaomin Wu, and Bingsheng He. The oarf benchmark suite: Characterization and implications for federated learning systems. arXiv preprint arXiv:2006.07856 , 2020.
- [17] Yutao Huang, Lingyang Chu, Zirui Zhou, Lanjun Wang, Jiangchuan Liu, Jian Pei, and Yong Zhang. Personalized cross-silo federated learning on non-iid data. In Proceedings of the AAAI Conference on Artificial Intelligence , 2021.
- [18] Longlong Jing and Yingli Tian. Self-supervised visual feature learning with deep neural networks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence , 2020.
- [19] Rie Johnson and Tong Zhang. Accelerating stochastic gradient descent using predictive variance reduction. In Advances in neural information processing systems , pages 315-323, 2013.
- [20] Peter Kairouz, H Brendan McMahan, Brendan Avent, Aur´ elien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Keith Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. Advances and open problems in federated learning. arXiv preprint arXiv:1912.04977 , 2019.
- [21] Georgios A Kaissis, Marcus R Makowski, Daniel R¨ uckert, and Rickmer F Braren. Secure, privacy-preserving and federated machine learning in medical imaging. Nature Machine Intelligence , pages 1-7, 2020.
- [22] Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank J Reddi, Sebastian U Stich, and Ananda Theertha Suresh. Scaffold: Stochastic controlled averaging for ondevice federated learning. In Proceedings of the 37th International Conference on Machine Learning . PMLR, 2020.
- [23] Rajesh Kumar, Abdullah Aman Khan, Sinmin Zhang, WenYong Wang, Yousif Abuidris, Waqas Amin, and Jay Kumar. Blockchain-federated-learning and deep learning models for covid-19 detection using ct imaging. arXiv preprint arXiv:2007.06537 , 2020.
- [24] Qinbin Li, Yiqun Diao, Quan Chen, and Bingsheng He. Federated learning on non-iid data silos: An experimental study. arXiv preprint arXiv:2102.02079 , 2021.
- [25] Qinbin Li, Zeyi Wen, and Bingsheng He. Practical federated gradient boosting decision trees. In Proceedings of the AAAI Conference on Artificial Intelligence , pages 46424649, 2020.
- [26] Qinbin Li, Zeyi Wen, Zhaomin Wu, Sixu Hu, Naibo Wang, and Bingsheng He. A survey on federated learning systems: Vision, hype and reality for data privacy and protection. arXiv preprint arXiv:1907.09693 , 2019.
- [27] Tian Li, Anit Kumar Sahu, Ameet Talwalkar, and Virginia Smith. Federated learning: Challenges, methods, and future directions. arXiv preprint arXiv:1908.07873 , 2019.

- [28] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated optimization in heterogeneous networks. In Third Conference on Machine Learning and Systems (MLSys) , 2020.
- [29] Xiang Li, Kaixuan Huang, Wenhao Yang, Shusen Wang, and Zhihua Zhang. On the convergence of fedavg on non-iid data. In International Conference on Learning Representations , 2020.
- [30] Xiaoxiao Li, Meirui JIANG, Xiaofei Zhang, Michael Kamp, and Qi Dou. Fed { bn } : Federated learning on non-{ iid } features via local batch normalization. In International Conference on Learning Representations , 2021.
- [31] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision , pages 740-755. Springer, 2014.
- [32] Yang Liu, Anbu Huang, Yun Luo, He Huang, Youzhi Liu, Yuanyuan Chen, Lican Feng, Tianjian Chen, Han Yu, and Qiang Yang. Fedvision: An online visual object detection platform powered by federated learning. In Proceedings of the AAAI Conference on Artificial Intelligence , pages 1317213179, 2020.
- [33] Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research , 9(Nov):2579-2605, 2008.
- [34] H Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, et al. Communication-efficient learning of deep networks from decentralized data. arXiv preprint arXiv:1602.05629 , 2016.
- [35] Ishan Misra and Laurens van der Maaten. Self-supervised learning of pretext-invariant representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 6707-6717, 2020.
- [36] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 , 2018.
- [37] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. In Advances in neural information processing systems , pages 8026-8037, 2019.
- [38] Kihyuk Sohn. Improved deep metric learning with multiclass n-pair loss objective. In Advances in neural information processing systems , pages 1857-1865, 2016.
- [39] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. arXiv preprint arXiv:1906.05849 , 2019.
- [40] Paul Voigt and Axel Von dem Bussche. The eu general data protection regulation (gdpr). A Practical Guide, 1st Ed., Cham: Springer International Publishing , 2017.
- [41] Hongyi Wang, Mikhail Yurochkin, Yuekai Sun, Dimitris Papailiopoulos, and Yasaman Khazaeni. Federated learning with matched averaging. In International Conference on Learning Representations , 2020.
- [42] Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, and H Vincent Poor. Tackling the objective inconsistency problem in heterogeneous federated optimization. Advances in Neural Information Processing Systems , 33, 2020.
- [43] Lixu Wang, Shichao Xu, Xiao Wang, and Qi Zhu. Addressing class imbalance in federated learning. In Proceedings of the AAAI Conference on Artificial Intelligence , 2021.
- [44] Qiang Yang, Yang Liu, Tianjian Chen, and Yongxin Tong. Federated machine learning: Concept and applications. ACM Transactions on Intelligent Systems and Technology (TIST) , 10(2):1-19, 2019.
- [45] Mikhail Yurochkin, Mayank Agarwal, Soumya Ghosh, Kristjan Greenewald, Nghia Hoang, and Yasaman Khazaeni. Bayesian nonparametric federated learning of neural networks. In Proceedings of the 36th International Conference on Machine Learning . PMLR, 2019.
- [46] Fengda Zhang, Kun Kuang, Zhaoyang You, Tao Shen, Jun Xiao, Yin Zhang, Chao Wu, Yueting Zhuang, and Xiaolin Li. Federated unsupervised representation learning. arXiv preprint arXiv:2010.08982 , 2020.
- [47] Michael Zhang, Karan Sapra, Sanja Fidler, Serena Yeung, and Jose M. Alvarez. Personalized federated learning with first order model optimization. In International Conference on Learning Representations , 2021.