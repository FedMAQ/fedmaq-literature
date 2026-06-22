## DAdaQuant: Doubly-adaptive quantization for communication-efficient Federated Learning

Robert H¨ onig 1 Yiren Zhao 2 Robert Mullins 2

## Abstract

Federated Learning (FL) is a powerful technique to train a model on a server with data from several clients in a privacy-preserving manner. FL incurs significant communication costs because it repeatedly transmits the model between the server and clients. Recently proposed algorithms quantize the model parameters to efficiently compress FL communication. We find that dynamic adaptations of the quantization level can boost compression without sacrificing model quality. We introduce DAdaQuant as a doublyadaptive quantization algorithm that dynamically changes the quantization level across time and different clients. Our experiments show that DAdaQuant consistently improves client → server compression, outperforming the strongest nonadaptive baselines by up to 2 . 8 × .

## 1. Introduction

Edge devices such as smartphones, remote sensors and smart home appliances generate massive amounts of data (Wang et al., 2018b; Cao et al., 2017; Shi &amp; Dustdar, 2016). In recent years, Federated Learning (FL) has emerged as a technique to train models on this data while preserving privacy (McMahan et al., 2017; Li et al., 2018).

In FL, we have a single server that is connected to many clients. Each client stores a local dataset that it does not want to share with the server because of privacy concerns or law enforcement (Voigt &amp; Von dem Bussche, 2017). The server wants to train a model on all local datasets. To this end, it initializes the model and sends it to a random subset of clients. Each client trains the model on its local dataset and sends the trained model back to the server. The server

1 Department of Computer Science, ETH Zurich, Zurich, Switzerland 2 Department of Computer Science and Technology, University of Cambridge, Cambridge, United Kingdom. Correspondence to: Robert H¨ onig &lt; rhoenig@ethz.ch &gt; .

Proceedings of the 39 th International Conference on Machine Learning , Baltimore, Maryland, USA, PMLR 162, 2022. Copyright 2022 by the author(s).

accumulates all trained models into an updated model for the next iteration and repeats the process for several rounds until some termination criterion is met. This procedure enables the server to train a model without accessing any local datasets.

Today's neural network models often have millions or even billions (Brown et al., 2020) of parameters, which makes high communication costs a concern in FL. In fact, Qiu et al. (2020) suggest that communication between clients and server may account for over 70% of energy consumption in FL. Reducing communication in FL is an attractive area of research because it lowers bandwidth requirements, energy consumption and training time.

Communication in FL occurs in two phases: Sending parameters from the server to clients ( downlink ) and sending updated parameters from clients to the server ( uplink ). Uplink bandwidth usually imposes a tighter bottleneck than downlink bandwidth. This has several reasons. For one, the average global mobile upload bandwidth is currently less than one fourth of the download bandwidth (Speedtest). For another, FL downlink communication sends the same parameters to each client. Broadcasting parameters is usually more efficient than the accumulation of parameters from different clients that is required for uplink communication (Amiri et al., 2020; Reisizadeh et al., 2019). For these reasons, we seek to compress uplink communication.

Alarge class of compression algorithms for FL apply some lossy quantizer Q , optionally followed by a lossless compression stage. Q usually provides a 'quantization level' hyperparameter q to control the coarseness of quantization (e.g. the number of bins for fixed-point quantization). When q is kept constant during training, we speak of static quantization . When q changes, we speak of adaptive quantization . Adaptive quantization can exploit asymmetries in the FL framework to minimize communication. One such asymmetry lies in FL's training time, where Jhunjhunwala et al. (2021) observed that early training rounds can use a lower q without affecting convergence. Figure 2 illustrates how time-adaptive quantization leverages this phenomenon to minimize communication. Another asymmetry lies in FL's client space, because most FL algorithms weight client contributions to the global model proportional to their local dataset sizes. Figure 1 illustrates how clientadaptive quantization can minimize the quantization error. Intuitively, FL clients with greater weighting should have a greater communication budget and our proposed clientadaptive quantization achieves this in a principled way. To this end, we introduce the expected variance of an accumulation of quantized parameters, E [Var( ∑ Q ( p ))] , as a measure of the quantization error. Our client-adaptive quantization algorithm then assigns clients minimal quantization levels, subject to a fixed E [Var( ∑ Q ( p ))] . This lowers the amount of data communicated from clients to the server, without increasing the quantization error.

Figure 1. Static quantization vs. client-adaptive quantization when accumulating parameters p A and p B . (a): Static quantization uses the same quantization level for p A and p B . (b) Clientadaptive quantization uses a slightly higher quantization level for p B because p B is weighted more heavily. This allows us to use a significantly lower quantization level q A for p A while keeping the quantization error measure E p A ,p B [Var ( Q ( p ))] roughly constant. Since communication is approximately proportional to q A + q B , client-adaptive quantization communicates less data.

<!-- image -->

DAdaQuant (Doubly Adaptive Quantization) combines time- and client-adaptive quantization with an adaptation of the QSGD fixed-point quantization algorithm to achieve state-of-the-art FL uplink compression. In this paper, we make the following contributions:

- We introduce the concept of client-adaptive quantization and develop algorithms for time- and client-adaptive quantization that are computationally efficient, empirically superior to existing algorithms, and compatible

<!-- image -->

Communication

Figure 2. Time-adaptive quantization. A small quantization level (q) decreases the loss with less communication than a large q, but converges to a higher loss. This motivates an adaptive quantization strategy that uses a small q as long as it is beneficial and then switches over to a large q. We generalize this idea into an algorithm that monotonically increases q based on the training loss.

- with arbitrary FL quantizers. Our client-adaptive quantization is provably optimal for stochastic fixed-point quantizers.
- We create Federated QSGD as an adaptation of the stochastic fixed-point quantizer QSGD that works with FL. Federated QSGD outperforms all other quantizers, establishing a strong baseline for FL compression with static quantization.
- We combine time- and client-adaptive quantization into DAdaQuant. We demonstrate DAdaQuant's state-of-theart compression by empirically comparing it against several competitive FL compression algorithms.

## 2. Related Work

FL research has explored several approaches to reduce communication. We identify three general directions.

First, there is a growing interest of investigating FL algorithms that can converge in fewer rounds. FedAvg (McMahan et al., 2017) achieves this with prolonged local training, while FOLB (Nguyen et al., 2020) speeds up convergence through a more principled client sampling. Since communication is proportional to the number of training rounds, these algorithms effectively reduce communication.

Secondly, communication can be reduced by reducing the model size because the model size is proportional to the amount of training communication. PruneFL (Jiang et al., 2019) progressively prunes the model over the course of training, while AFD (Bouacida et al., 2021) only trains submodels on clients.

Thirdly, it is possible to directly compress FL training communication. FL compression algorithms typically apply techniques like top-k sparsification (Malekijoo et al., 2021; Rothchild et al., 2020) or quantization (Reisizadeh et al., 2019; Shlezinger et al., 2020) to parameter updates, optionally followed by lossless compression. Our work applies to quantization-based compression algorithms. It is partially based on QSGD (Alistarh et al., 2017), which combines lossy fixed-point quantization with a lossless compression algorithm to compress gradients communicated in distributed training. DAdaQuant adapts QSGD into Federated QSGD, which works with Federated Learning. DAdaQuant also draws inspiration from FedPAQ (Reisizadeh et al., 2019), the first FL framework to use lossy compression based on model parameter update quantization. However, FedPAQ does not explore the advantages of additional lossless compression or adaptive quantization. UVeQFed (Shlezinger et al., 2020) is an FL compression algorithm that generalizes scalar quantization to vector quantization and subsequently employs lossless compression with arithmetic coding. Like FedPAQ, UVeQFed also limits itself to a single static quantization level.

Faster convergence, model size reduction and communication compression are orthogonal techniques, so they can be combined for further communication savings. For this paper, we limit the scope of empirical comparisons to quantization-based FL compression algorithms.

For quantization-based compression for model training, prior works have demonstrated that DNNs can be successfully trained in low-precision (Banner et al., 2018; Gupta et al., 2015; Sun et al., 2019). There are also several adaptive quantization algorithms for training neural networks in a non-distributed setting. Shen et al. (2020) use different quantization levels for different parameters of a neural network. FracTrain (Fu et al., 2020) introduced multidimensional adaptive quantization by developing timeadaptive quantization and combining it with parameteradaptive quantization. However, FracTrain uses the current loss to decide on the quantization level. FL generally can only compute local client losses that are too noisy to be practical for FracTrain. AdaQuantFL introduces timeadaptive quantization to FL, but requires the global loss (Jhunjhunwala et al., 2021). To compute the global loss, AdaQuantFL has to communicate with every client each round. We show in Section 4.2 that this quickly becomes impractical as the number of clients grows. DAdaQuant's time-adaptive quantization overcomes this issue without compromising on the underlying FL communication. In addition, to the best of our knowledge, DAdaQuant is the first algorithm to use client-adaptive quantization.

## 3. The DAdaQuant method

## 3.1. Federated Learning

Federated Learning assumes a client-server topology with a set C = { c i | i ∈ { 1 , 2 ...N }} of N clients that are connected to a single server. Each client c k has a local dataset D k from the local data distribution D k . Given a model M with parameters p , a loss function f p ( d ∈ D k ) and the local loss F k ( p ) = 1 | D k | ∑ d ∈ D k f p ( d ) , FL seeks to minimize the global loss G ( p ) = ∑ N k =1 | D k | ∑ l | D l | F k ( p ) .

## 3.2. Federated Averaging (FedAvg)

DAdaQuant makes only minimal assumptions about the FL algorithm. Crucially, DAdaquant can complement FedAvg (McMahan et al., 2017), which is representative of a large class of FL algorithms.

FedAvg trains the model M over several rounds. In each round t , FedAvg sends the model parameters p t to a random subset S t of K clients who then optimize their local objectives F k ( p t ) and send the updated model parameters p k t +1 back to the server. The server accumulates all parameters into the new global model p t +1 = ∑ k ∈ S t | D k | ∑ j | D j | p k t +1 and starts the next round. Algorithm 1 lists FedAvg in detail. For our experiments, we use the FedProx (Li et al., 2018) adaptation of FedAvg. FedProx improves the convergence of FedAvg by adding the proximal term µ 2 ‖ p k t +1 -p t ‖ 2 to the local objective F k ( p k t +1 ) in line 19 of Algorithm 1.

## 3.3. Quantization with Federated QSGD

While DAdaQuant can be applied to any quantizer with a configurable quantization level, it is optimized for fixedpoint quantization. We introduce Federated QSGD as a competitive stochastic fixed-point quantizer on top of which DAdaQuant is applied.

In general, stochastic fixed-point quantization uses a quantizer Q q with quantization level q that splits R ≥ 0 and R ≤ 0 into q intervals each. Q q ( p ) then returns the sign of p and | p | stochastically rounded to one of the endpoints of its encompassing interval. Q q ( p ) quantizes the vector p elementwise.

We design DAdaQuant's quantization stage based on QSGD, an efficient fixed-point quantizer for state-of-theart gradient compression. QSGD quantizes a vector p in three steps:

1. Quantize p as Q q ( p || p || 2 ) into q bins in [0 , 1] , storing signs and || p || 2 separately. ( lossy )
2. Encode the resulting integers with 0 run-length encoding. ( lossless )
3. Encode the resulting integers with Elias ω coding. ( lossless )

QSGD has been designed specifically for quantizing gradients. This makes it not directly applicable to parameter compression. To overcome this limitation, we apply difference coding to uplink compression, first introduced to FL by FedPAQ. Each client c k applies Q q to the parameter updates p k t +1 -p t (cf. line 7 of Algorithm 1) and sends them to the server. The server keeps track of the previous parameters p t and accumulates the quantized parameter updates into the new parameters as p t +1 = p t + ∑ k ∈ S t | D k | ∑ l | D l | Q q ( p k t +1 -p t ) (cf. line 10 of Algorithm 1). We find that QSGD works well with parameter updates, which can be regarded as an accumulation of gradients over several training steps. We call this adaptation of QSGD Federated QSGD .

## 3.4. Time-adaptive quantization

Time-adaptive quantization uses a different quantization level q t for each round t of FL training. DAdaQuant chooses q t to minimize communication costs without sacrificing accuracy. To this end, we find that lower quantization levels suffice to initially reduce the loss, while partly trained models require higher quantization levels to further improve (as illustrated in Figure 2). FracTrain is built on similar observations for non-distributed training. Therefore, we design DAdaQuant to mimic FracTrain in monotonically increasing q t as a function of t and using the training loss to inform increases in q t .

When q is too low, FL converges prematurely. Like FracTrain, DAdaQuant monitors the FL loss and increases q when it converges. Unlike FracTrain, there is no single centralized loss function to evaluate and unlike AdaQuantFL, we do not assume availability of global training loss G ( p t ) . Instead, we estimate G ( p t ) as the average local loss ˆ G t = ∑ k ∈ S t | D k | ∑ l | D l | F k ( p t ) where S t is the set of clients sampled at round t . Since S t typically consists of only a small fraction of all clients, ˆ G t is a very noisy estimate of G ( p t ) . This makes it unsuitable for convergence detection. Instead, DAdaQuant tracks a running average loss ˆ ˆ G t = ψ ˆ ˆ G t -1 +(1 -ψ ) ˆ G t . Figure 3 visualizes ˆ ˆ G t on a real training example.

We initialize q 1 = q min for some q min ∈ N . DAdaQuant determines training to converge whenever ˆ ˆ G t ≥ ˆ ˆ G t +1 -φ for some φ ∈ N that specifies the number of rounds across which we compare ˆ ˆ G . On convergence, DAdaQuant sets q t = 2 q t -1 and keeps the quantization level fixed for at least φ rounds to enable reductions in G to manifest in ˆ ˆ G . Eventually, the training loss converges regardless of the quantization level. To avoid unconstrained quantization increases on convergence, we limit the quantization level to q max.

The following equation summarizes DAdaQuant's time- adaptive quantization:

Figure 3. Client losses ˆ G 0 t and ˆ G 1 t , global loss estimate ˆ G t and moving average loss ˆ ˆ G t when training on the Synthetic dataset with two clients per round, assuming equal client weightings.

<!-- image -->

$$
\begin{array} { r l } & { \tt }
$$

Doubling the quantization level proves empirically successful and ensures that experiments sensitive to noise quickly reach a sufficiently high quantization level. However, we note that other strategies, such as increments by one, could in principle match or even outperform doubling.

## 3.5. Client-adaptive quantization

FL algorithms typically accumulate each parameter p i over all clients into a weighted average p = ∑ K i =1 w i p i (see Algorithm 1). Quantized FL accumulates quantized parameters Q q ( p ) = ∑ K i =1 w i Q q ( p i ) where q is the quantization level. We define the quantization error e q p = | p -Q q ( p ) | .

We observe in our experiments that communication cost per client is roughly a linear function of Federated QSGD's quantization level q . This means that the communication cost per round is proportional to Q = Kq . We call Q the communication budget and use it as a proxy measure of communication cost.

Client-adaptive quantization dynamically adjusts the quantization level of each client. This means that even within a single round, each client c k can be assigned a different quantization level q k . The previous definitions then generalize to Q = ∑ K k =1 q k and Q q 1 ...q K ( p ) = ∑ K i =1 w i Q q i ( p i ) and e q 1 ...q K p = | p -Q q 1 ...q K ( p ) | .

Prior convergence results for distributed training and FL rely on an upper bound b on Var( Q q 1 ...q K ( p )) that determines the convergence speed (Li et al., 2017; Horv´ ath et al., 2019; Reisizadeh et al., 2019). This makes V( Q q 1 ...q K ( p )) a natural measure to optimize for when

<!-- image -->

- (d) Time-adaptive and client-adaptive quantization.

Figure 4. Exemplary quantization level assignment for 4 FL clients that train over 5 rounds. Each round, two clients get sampled for training.

choosing q k . We optimize for the closely related measure E p 1 ...p K [Var( Q q 1 ...q K ( p ))] that replaces the upper bound with an expectation over parameters p 1 . . . p K . Heuristically, we expect an this averaged measure to provide a better estimate of practically observed quantization errors than an upper bound. For a stochastic, unbiased fixed-point compressor like Federated QSGD, E p 1 ... -p K [Var( Q q 1 ...q K ( p ))] equals E p 1 ...p K [Var( e q p )] and can be evaluated analytically.

Wedevise an algorithm that chooses q k to minimize Q subject to E p 1 ...p K [Var( e q 1 ...q K p )] = E p 1 ...p K [Var( e q p )] for a given q . Thus, our algorithm effectively minimizes communication costs while maintaining a quantization error similar to static quantization. Theorem 1 provides us with an analytical formula for quantization levels q 1 . . . q K .

$$
\begin{aligned}
\text {an analyical formula for quantization levels} & q _ { 1 } \dots q _ { K } . & \text {use class} \\ \text {Theorem} \ 1 . \text { Given parameters } p _ { 1 } \dots p _ { k } \ & \sim \ \mathcal { U } [ - t , t ] \\ \text {and quantization level} \ q , \ \min _ { q _ { 1 } \dots q _ { K } } \sum _ { i = 1 } ^ { K } q _ { i } \ s u b j e c t \ t o \\ \mathbb { E } _ { p _ { 1 } \dots p _ { K } } [ \text {Var} ( e _ { p } ^ { q _ { 1 } \dots q _ { K } } ) ] & = \mathbb { E } _ { p _ { 1 } \dots p _ { K } } [ \text {Var} ( e _ { p } ^ { q } ) ] \text { is minimized} \\ b y \ q _ { i } & = \sqrt { \frac { a } { b } } \times w _ { i } ^ { 2 / 3 } \text { where } a \, = \, \sum _ { j = 1 } ^ { K } w _ { j } ^ { 2 / 3 } \text { and } b \, = \, \begin{matrix} \text {Adaptive} \\ \text {Adaptive} \end{matrix} \\ \sum _ { j = 1 } ^ { K } \frac { w _ { j } ^ { 2 } } { q ^ { 2 } } . & \text {as} \\ \text {a} \, \Delta \, a _ { 1 } \, \Delta \, a _ { 2 } \, \Delta \, a _ { 3 } \, \Delta \, a _ { 4 } \, \Delta \, a _ { 5 } \, \Delta \, a _ { 6 } \, \Delta \, a _ { 7 } \, \Delta \, a _ { 8 } \, \Delta \, a _ { 9 } \, \Delta \, a _ { 10 } \, \Delta \, a _ { 11 } \, \Delta \, a _ { 12 } \, \Delta \, a _ { 13 } \, \Delta \, a _ { 14 } \, \Delta \, a _ { 15 } \, \Delta \, a _ { 16 } \, \Delta \, a _ { 17 } \, \Delta \, a _ { 18 } \, \Delta \, a _ { 19 } \, \Delta \, a _ { 20 } \, \Delta \, a _ { 21 } \, \Delta \, a _ { 22 } \, \Delta \, a _ { 23 } \, \Delta \, a _ { 24 } \, \Delta \, a _ { 25 } \, \Delta \, a _ { 26 } \, \Delta \, a _ { 27 } \, \Delta \, a _ { 28 } \, \Delta \, a _ { 29 } \, \Delta \, a _ { 30 } \, \Delta \, a _ { 31 } \, \Delta \, a _ { 32 } \, \Delta \, a _ { 33 } \, \Delta \, a _ { 34 } \, \Delta \, a _ { 35 } \, \Delta \, a _ { 36 } \, \Delta \, a _ { 37 } \, \Delta \, a _ { 38 } \, \Delta \, a _ { 39 } \, \Delta \, a _ { 40 } \, \Delta \, a _ { 41 } \, \Delta \, a _ { 42 } \, \Delta \, a _ { 43 } \, \Delta \, a _ { 44 } \, \Delta \, a _ { 45 } \, \Delta \, a _ { 46 } \, \Delta \, a _ { 47 } \, \Delta \, a _ { 48 } \, \Delta \, a _ { 49 } \, \Delta \, a _ { 50 } \, \Delta \, a _ { 51 } \, \Delta \, a _ { 52 } \, \Delta \, a _ { 53 } \, \Delta \, a _ { 54 } \, \Delta \, a _ { 55 } \, \Delta \, a _ { 56 } \, \Delta \, a _ { 57 } \, \Delta \, a _ { 58 } \, \Delta \, a _ { 59 } \, \Delta \, a _ { 60 } \, \Delta \, a _ { 61 } \, \Delta \, a _ { 62 } \, \Delta \, a _ { 63 } \, \Delta \, a _ { 64 } \, \Delta \, a _ { 65 } \, \Delta \, a _ { 66 } \, \Delta \, a _ { 67 } \, \Delta \, a _ { 68 } \, \Delta \, a _ { 69 } \, \Delta \, a _ { 70 } \, \Delta \, a _ { 71 } \, \Delta \, a _ { 72 } \, \Delta \, a _ { 73 } \, \Delta \, a _ { 74 } \, \Delta \, a _ { 75 } \, \Delta \, a _ { 76 } \, \Delta \, a _ { 77 } \, \Delta \, a _ { 78 } \, \Delta \, a _ { 79 } \, \Delta \, a _ { 80 } \, \Delta \, a _ { 81 } \, \Delta \, a _ { 82 } \, \Delta \, a _ { 83 } \, \Delta \, a _ { 84 } \, \Delta \, a _ { 85 } \, \Delta \, a _ { 86 } \, \Delta \, a _ { 87 } \, \Delta \, a _ { 88 } \, \Delta \, a _ { 89 } \, \Delta \, a _ { 90 } \, \Delta \, a _ { 91 } \, \Delta \, a _ { 92 } \, \Delta \, a _ { 93 } \, \Delta \, a _ { 94 } \, \Delta \, a _ { 95 } \, \Delta \, a _ { 96 } \, \Delta \, a _ { 97 } \, \Delta \, a _ { 98 } \, \Delta \, a _ { 99 } \, \Delta \, a _ { 100 } \, \Delta \, a _ { 101 } \, \Delta \, a _ { 102 } \, \Delta \, a _ { 103 } \, \Delta \, a _ { 104 } \, \Delta \, a _ { 105 } \, \Delta \, a _ { 106 } \, \Delta \, a _ { 107 } \, \Delta \, a _ { 108 } \, \Delta \, a _ { 109 } \, \Delta \, a _ { 110 } \, \Delta \, a _ { 111 } \, \Delta \, a _ { 112 } \, \Delta \, a _ { 113 } \, \Delta \, a _ { 114 } \, \Delta \, a _ { 115 } \, \Delta \, a _ { 116 } \, \Delta \, a _ { 117 } \, \Delta \, a _ { 118 } \, \Delta \, a _ { 119 } \, \Delta \, a _ { 120 } \, \Delta \, a _ { 121 } \, \Delta \, a _ { 122 } \, \Delta \, a _ { 123 } \, \Delta \, a _ { 124 } \, \Delta \, a _ { 125 } \, \Delta \, a _ { 126 } \, \Delta \, a _ { 127 } \, \Delta \, a _ { 128 } \, \Delta \, a _ { 129 } \, \Delta \, a _ { 130 } \, \Delta \, a _ { 131 } \, \Delta \, a _ { 132 } \, \Delta \, a _ { 133 } \, \Delta \, a _ { 134 } \, \Delta \, a _ { 135 } \, \Delta \, a _ { 136 } \, \Delt
\end{aligned}
$$

DAdaQuant applies Theorem 1 to lower communication costs while maintaining the same loss as static quantization does with a fixed q . To ensure that quantization levels are natural numbers, DAdaQuant approximates the optimal real-valued solution as q i = max(1 , round ( √ a b × w 2 / 3 i )) . Appendix B gives a detailed proof of Theorem 1. To the best of our knowledge, DAdaQuant is the first algorithm to use client-adaptive quantization.

<!-- image -->

## 3.6. Doubly-adaptive quantization (DAdaQuant)

DAdaQuant combines the time-adaptive and clientadaptive quantization algorithms described in the previous sections. At each round t , time-adaptive quantization determines a preliminary quantization level q t . Clientadaptive quantization then finds the client quantization lev- Shakespeare

| Dataset     | Model        | Parameters   |   Clients |   Samples | Samples per client   | Samples per client   | Samples per client   | Samples per client   |
|-------------|--------------|--------------|-----------|-----------|----------------------|----------------------|----------------------|----------------------|
|             |              |              |           |           | mean                 | min                  | max                  | stddev               |
| Synthetic   | MLR          | 610          |        30 |     9,600 | 320.0                | 45                   | 5,953                | 1051.6               |
| FEMNIST     | 2-layer CNN  | 6 . 6 × 10 6 |     3,500 |   785,582 | 224.1                | 19                   | 584                  | 87.8                 |
| CelebA      | 4-layer CNN  | 6 . 3 × 10 5 |     9,343 |   200,288 | 21.4                 | 5                    | 35                   | 7.6                  |
| Sent140     | 2-layer LSTM | 1 . 1 × 10 6 |    21,876 |   430,707 | 51.1                 | 10                   | 549                  | 17.1                 |
| Shakespeare | 2-layer LSTM | 1 . 3 × 10 5 |     1,129 | 4,226,158 | 3743                 | 3                    | 66,903               | 6212                 |

Table 1. Statistics of the models and datasets used for evaluation. MLR stands for 'Multinomial Logistic Regression'.

|                   | Synthetic       | Synthetic              | FEMNIST         | FEMNIST                | Sent140         | Sent140                |
|-------------------|-----------------|------------------------|-----------------|------------------------|-----------------|------------------------|
| Uncompressed      | 78 . 3 ± 0 . 3  | 12 . 2 MB              | 77 . 7 ± 0 . 4  | 132 . 1 GB             | 69 . 7 ± 0 . 5  | 43 . 9 GB              |
| Federated QSGD    | - 0 . 1 ± 0 . 1 | 17 ×                   | +0 . 7 ± 0 . 5  | 2809 ×                 | - 0 . 0 ± 0 . 5 | 90 ×                   |
| FP8               | +0 . 1 ± 0 . 4  | 4 . 0 × ( 0 . 23 × × ) | - 0 . 1 ± 0 . 4 | 4 . 0 × ( 0 . 00 × × ) | - 0 . 2 ± 0 . 5 | 4 . 0 × ( 0 . 04 × × ) |
| FedPAQ (FxPQ)     | - 0 . 1 ± 0 . 1 | 6 . 4 × ( 0 . 37 × × ) | +0 . 7 ± 0 . 5  | 11 × ( 0 . 00 × × )    | - 0 . 0 ± 0 . 5 | 4 . 0 × ( 0 . 04 × × ) |
| FxPQ + GZip       | - 0 . 1 ± 0 . 1 | 14 × ( 0 . 82 × × )    | +0 . 6 ± 0 . 2  | 1557 × ( 0 . 55 × × )  | - 0 . 0 ± 0 . 6 | 71 × ( 0 . 79 × × )    |
| UVeQFed           | - 0 . 5 ± 0 . 2 | 0 . 6 × ( 0 . 03 × × ) | - 2 . 8 ± 0 . 5 | 12 × ( 0 . 00 × × )    | +0 . 0 ± 0 . 2  | 15 × ( 0 . 16 × × )    |
| DAdaQuant         | - 0 . 2 ± 0 . 4 | 48 × ( 2 . 81 × × )    | +0 . 7 ± 0 . 1  | 4772 × ( 1 . 70 × × )  | - 0 . 1 ± 0 . 4 | 108 × ( 1 . 19 × × )   |
| DAdaQuant time    | - 0 . 1 ± 0 . 5 | 37 × ( 2 . 16 × × )    | +0 . 8 ± 0 . 2  | 4518 × ( 1 . 61 × × )  | - 0 . 1 ± 0 . 6 | 93 × ( 1 . 03 × × )    |
| DAdaQuant clients | +0 . 0 ± 0 . 3  | 26 × ( 1 . 51 × × )    | +0 . 7 ± 0 . 4  | 3017 × ( 1 . 07 × × )  | +0 . 1 ± 0 . 6  | 105 × ( 1 . 16 × × )   |

0

267

0

.

3

.

0

.

6

0

.

4

0

.

6

0

.

6

0

.

4

0

.

5

0

5

.

0

MB

0

42

.

(

0

(

0

(

.

34

.

97

0

(

.

83

2

5

9

.

4

×

.

0

3

×

.

2

9

×

.

3

×

7

.

9

×

21

(

×

×

12

×

)

× ×

× ×

)

× ×

)

× ×

.

)

21

.

(

1

29

1

× ×

× ×

67

)

)

× ×

)

0

±

4

.

90

-

1

.

0

+0

±

.

-

0

-

0

-

0

-

0

-

0

-

0

.

0

0

0

.

±

1

.

1

.

4

.

1

.

1

.

.

(

1

.

1

0

±

0

±

±

±

±

±

.

1

.

1

0

.

2

0

.

3

0

.

1

0

.

2

0

.

0

Celeba

12

6

.

648

GB

×

4

.

0

×

6

4

.

×

×

494

31

×

(

0

.

01

0

01

(

.

0

(

0

(

775

× ×

)

× ×

.

)

76

.

×

716

×

05

(

× ×

× ×

)

)

1

.

20

1

(

×

700

.

10

1

(

.

× ×

× ×

08

)

)

× ×

)

Uncompressed

Federated QSGD

FP8

FedPAQ (FxPQ)

FxPQ + GZip

UVeQFed

DAdaQuant

DAdaQuanttime

DAdaQuantclients

16

5

.

±

±

±

±

±

±

±

±

±

.

49

-

0

-

9

.

0

-

0

-

0

-

0

-

0

-

0

-

5

.

2

.

5

.

5

.

0

.

6

.

5

.

4

0

Table 2. Top-1 test accuracies and total client → server communication of all baselines, DAdaQuant, DAdaQuanttime and DAdaQuantclients. Entry x ± y p × ( q × × ) denotes an accuracy difference of x% w.r.t. the uncompressed accuracy with a standard deviation of y%, a compression factor of p w.r.t. the uncompressed communication and a compression factor of q w.r.t. Federated QSGD.

els q k t , k ∈ { 1 , . . . , K } that minimize ∑ K i =1 q i subject to E p 1 ...p K [Var( e q 1 ...q K p )] = E p 1 ...p K [Var( e q p )] . Algorithm 1 lists DAdaQuant in detail. Figure 4 gives an example of how our time-adaptive, client-adaptive and doublyadaptive quantization algorithms set quantization levels.

Reisizadeh et al. (2019) prove the convergence of FL with quantization for convex and non-convex cases as long as the quantizer Q is (1) unbiased and (2) has a bounded variance. These convergence results extend to DAdaQuant when combined with any quantizer that satisfies (1) and (2) for DAdaQuant's minimum quantization level q = 1 . Crucially, this includes Federated QSGD.

We highlight DAdaQuant's low overhead and general applicability. The computational overhead is dominated by an additional evaluation epoch per round per client to compute ˆ ˆ G t , which is negligible when training for many epochs per round. In our experiments, we observe computational overheads of ≈ 1% (see Appendix A.3). DAdaQuant can compliment any FL algorithm that trains models over several rounds and accumulates a weighted average of client parameters. Most FL algorithms, including FedAvg, follow this design.

## 4. Experiments

## 4.1. Experimental details

Evaluation We use DAdaQuant with Federated QSGD to train different models with FedProx on different datasets for a fixed number of rounds. We monitor the test loss and accuracy at fixed intervals and measure uplink communication at every round across all devices.

Models &amp; datasets We select a broad and diverse set of five models and datasets to demonstrate the general applicability of DAdaQuant. To this end, we use DAdaQuant to train a linear model, CNNs and LSTMs of varying complexity on a federated synthetic dataset ( Synthetic ), as well as two federated image datasets ( FEMNIST and CelebA ) and two federated natural language datasets ( Sent140 and Shakespeare ) from the LEAF (Caldas et al., 2018) project for standardized FL research. Table 1 provides statistics of our models and datasets. We refer to Appendix A.1 for more information on the training objectives and implementation.

System heterogeneity In practice, FL has to cope with clients that have different compute capabilities. We follow Li et al. (2018) and simulate this system heterogeneity by randomly reducing the number of epochs to E ′ for a random subset S ′ t ⊂ S t of clients at each round t , where E ′ is sampled from [1 , . . . , E ] and | S ′ t | = 0 . 9 K .

Baselines We compare DAdaQuant against competing quantization-based algorithms for FL parameter compression, namely Federated QSGD, FedPAQ (Reisizadeh et al., 2019), GZip with fixed-point quantization (FxPQ + GZip), UVeQFed (Shlezinger et al., 2020) and FP8. Federated QSGD (see section 3.3) is our most important baseline because it outperforms the other algorithms. FedPAQ only applies fixed-point quantization, which is equivalent to Federated QSGD without lossless compression. Similarly, FxPQ + GZip is equivalent to Federated QSGD with Gzip for its lossless compression stages. UVeQFed generalizes scalar quantization to vector quantization, followed by arithmetic coding. We apply UVeQFed with the optimal hyperparameters reported by its authors. FP8 (Wang et al., 2018a) is a floating-point quantizer that uses an 8-bit floating-point format designed for storing neural network gradients. We also evaluate all experiments without compression to establish an accuracy benchmark.

Hyperparameters With the exception of CelebA , all our datasets and models are also used by Li et al.. We therefore adopt most of the hyperparameters from Li et al. and use LEAF's hyperparameters for CelebA (Caldas et al., 2018). For all experiments, we sample 10 clients each round. We train Synthetic , FEMNIST and CelebA for 500 rounds each. We train Sent140 for 1000 rounds due to slow convergence and Shakespeare for 50 rounds due to rapid convergence. We use batch size 10, learning rates 0.01, 0.003, 0.3, 0.8, 0.1 and µ s (FedProx's proximal term coefficient) 1, 1, 1, 0.001, 0 for Synthetic , FEMNIST , Sent140 , Shakespeare , CelebA respectively. We randomly split the local datasets into 80% training set and 20% test set.

To select the quantization level q for static quantization with Federated QSGD, FedPAQ and FxPQ + GZip, we run a gridsearch over q = 1 , 2 , 4 , 8 , . . . and choose for each dataset the lowest q for which Federated QSGD exceeds uncompressed training in accuracy. We set UVeQFed's 'coding rate' hyperparameter R = 4 , which is the lowest value for which UVeQFed achieves negligible accuracy differences compared to uncompressed training. We set the remaining hyperparameters of UVeQFed to the optimal values reported by its authors. Appendix A.4 shows further experiments that compare against UVeQFed with R chosen to maximize its compression factor.

For DAdaQuant's time-adaptive quantization, we set ψ to 0.9, φ to 1 / 10 th of the number of rounds and q max to the quantization level q for each experiment. For Synthetic and FEMNIST , we set q min to 1. We find that Sent140 , Shakespeare and CelebA require a high quantization level to achieve top accuracies and/or converge in few rounds. This prevents time-adaptive quantization from increasing the quantization level quickly enough, resulting in prolonged low-precision training that hurts model performance. To counter this effect, we set q min to q max / 2 . This effectively results in binary time-adaptive quantization with an initial low-precision phase with q = q max / 2 , followed by a high-precision phase with q = q max.

## 4.2. Results

We repeat the main experiments three times and report average results and their standard deviation (where applicable). Table 2 shows the highest accuracy and total communication for each experiment. Figure 5 plots the maximum accuracy achieved for any given amount of communication.

Baselines Table 2 shows that the accuracy of most experiments lies within the margin of error of the uncompressed experiments. This reiterates the viability of quantizationbased compression algorithms for communication reduction in FL. For all experiments, Federated QSGD achieves a significantly higher compression factor than the other baselines. The authors of FedPAQ and UVeQFed also compare their methods against QSGD and report them as superior. However, FedPAQ is compared against 'unfederated' QSGD that communicates gradients after each local training step and UVeQFed is compared against QSGD without its lossless compression stages.

Time-adaptive quantization The purely time-adaptive version of DAdaQuant, DAdaQuanttime, universally outperforms Federated QSGD and the other baselines in Table 2, achieving comparable accuracies while lowering communication costs. DAdaQuanttime performs particularly well on Synthetic and FEMNIST , where it starts from the lowest possible quantization level q = 1 . However, binary time-adaptive quantization still measurably improves over QSGD for Sent140 , Shakespeare and Celeba .

Figure 8 in Appendix A.5 provides empirical evidence that AdaQuantFL's communication scales linearly with the number of clients. As a result, AdaQuantFL is prohibitively expensive for datasets with thousands of clients such as Celeba and Sent140 . DAdaQuant does not face this problem because its communication is unaffected by the number of clients.

Client-adaptive quantization The purely time-adaptive version of DAdaQuant, DAdaQuantclients, also universally outperforms Federated QSGD and the other baselines in Table 2, achieving similar accuracies while lowering communication costs. Unsurprisingly, the performance of DAdaQuantclients is correlated with the coefficient of variation c v = σ µ of the numbers of samples in the local datasets with mean µ and standard deviation σ : Synthetic ( c v = 3 . 3 ) and Shakespeare ( c v = 1 . 7 ) achieve signifi- cantly higher compression factors than Sent140 ( c v = 0 . 3 ), FEMNIST ( c v = 0 . 4 ) and Celeba ( c v = 0 . 3 ).

Figure 5. Communication-accuracy trade-off curves for training with Federated QSGD and DAdaQuant. For each dataset, we plot the average highest accuracies achieved up to any given amount of client → server communication. Appendix A.2 shows curves for DAdaQuanttime and DAdaQuantclients, with similar results.

<!-- image -->

DAdaQuant DAdaQuant outperforms DAdaQuanttime and DAdaQuantclients in communication while achieving similar accuracies. The compression factors of DAdaQuant are roughly multiplicative in those of DAdaQuantclients and DAdaQuanttime. This demonstrates that we can effectively combine time- and client-adaptive quantization for maximal communication savings. Figure 5 shows that DAdaQuant achieves a higher accuracy than the strongest baseline, Federated QSGD, for any fixed amount of client → server communication.

## 5. Conclusion

We introduced DAdaQuant as a computationally efficient and robust algorithm to boost the performance of quantization-based FL compression algorithms. We showed intuitively and mathematically how DAdaQuant's dynamic adjustment of the quantization level across time and clients minimize client → server communication while maintaining convergence speed. Our experiments establish DAdaQuant as nearly universally superior over static quantizers, achieving state-of-the-art compression factors when applied to Federated QSGD. The communication savings of DAdaQuant effectively lower FL bandwidth usage, energy consumption and training time. Future work may apply and adapt DAdaQuant to new quantizers, further pushing the state of the art in FL uplink compression.

## 6. Reproducibility Statement

Our submission includes a repository with the source code for DAdaQuant and for the experiments presented in this paper. All the datasets used in our experiments are publicly available. Any post-processing steps of the datasets are described in Appendix A.1. To facilitate the reproduction of our results, we have bundled all our source code, dependencies and datasets into a Docker image. The repository submitted with this paper contains instructions on how to use this Docker image and reproduce all plots and tables in this paper.

## References

- Alistarh, D., Grubic, D., Li, J., Tomioka, R., and Vojnovic, M. QSGD: Communication-efficient SGD via gradient quantization and encoding. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R. (eds.), Advances in Neural Information Processing Systems 30 , pp. 1709-1720. Curran Associates, Inc., 2017. URL http://papers . nips . cc/paper/ 6768-qsgd-communication-efficientsgd-via-gradient-quantization-andencoding . pdf .
- Amiri, M. M., Gunduz, D., Kulkarni, S. R., and Poor, H. V. Federated learning with quantized global model updates. arXiv:2006.10672, 2020.
- Banner, R., Hubara, I., Hoffer, E., and Soudry, D. Scalable methods for 8-bit training of neural networks. In Proceedings of the 32nd International Conference on Neural Information Processing Systems , pp. 5151-5159, 2018.
- Beutel, D. J., Topal, T., Mathur, A., Qiu, X., Parcollet, T., and Lane, N. D. Flower: A friendly Federated Learning research framework. arXiv:2007.14390, 2020.
- Bouacida, N., Hou, J., Zang, H., and Liu, X. Adaptive Federated Dropout: Improving communication efficiency and generalization for federated learning. In IEEE INFOCOM 2021-IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS) , pp. 1-6. IEEE, 2021.
- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M. F., and Lin, H. (eds.), Advances in Neural Information Processing Systems , volume 33, pp. 1877-1901. Curran Associates, Inc., 2020. URL https://proceedings . neurips . cc/ paper/2020/file/

[1457c0d6bfcb4967418bfb8ac142f64a-](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)

[Paper](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)

[.](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)

[pdf](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)

.

- Caldas, S., Duddu, S. M. K., Wu, P., Li, T., Koneˇ cn´ y, J., McMahan, H. B., Smith, V., and Talwalkar, A. LEAF: A benchmark for federated settings. arXiv:1812.01097, 2018.
- Cao, B., Zheng, L., Zhang, C., Yu, P. S., Piscitello, A., Zulueta, J., Ajilore, O., Ryan, K., and Leow, A. D. DeepMood: modeling mobile phone typing dynamics for mood detection. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining , pp. 747-755, 2017.
- Fu, Y., You, H., Zhao, Y., Wang, Y., Li, C., Gopalakrishnan, K., Wang, Z., and Lin, Y. FracTrain: Fractionally squeezing bit savings both temporally and spatially for efficient DNN training. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M. F., and Lin, H. (eds.), Advances in Neural Information Processing Systems , volume 33, pp. 12127-12139. Curran Associates, Inc., 2020. URL https://proceedings . neurips . cc/ paper/2020/file/ 8dc5983b8c4ef1d8fcd5f325f9a65511Paper . pdf .
- Gupta, S., Agrawal, A., Gopalakrishnan, K., and Narayanan, P. Deep learning with limited numerical precision. In International conference on machine learning , pp. 1737-1746. PMLR, 2015.
- Horv´ ath, S., Kovalev, D., Mishchenko, K., Stich, S., and Richt´ arik, P. Stochastic distributed learning with gradient quantization and variance reduction. arXiv preprint arXiv:1904.05115 , 2019.
- Jhunjhunwala, D., Gadhikar, A., Joshi, G., and Eldar, Y. C. Adaptive quantization of model updates for communication-efficient federated learning. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pp. 31103114. IEEE, 2021.
- Jiang, Y., Wang, S., Ko, B. J., Lee, W.-H., and Tassiulas, L. Model pruning enables efficient Federated Learning on edge devices. arXiv:1909.12326, 2019.
- Li, H., De, S., Xu, Z., Studer, C., Samet, H., and Goldstein, T. Training quantized nets: A deeper understanding. In Proceedings of the 31st International Conference on Neural Information Processing Systems , pp. 58135823, 2017.
- Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., and Smith, V. Federated optimization in heterogeneous networks. arXiv:1812.06127, 2018.
- Malekijoo, A., Fadaeieslam, M. J., Malekijou, H., Homayounfar, M., Alizadeh-Shabdiz, F., and Rawassizadeh, R. FEDZIP: A compression framework for communicationefficient Federated Learning. arXiv:2102.01593, 2021.
- McMahan, B., Moore, E., Ramage, D., Hampson, S., and y Arcas, B. A. Communication-efficient learning of deep

- networks from decentralized data. In Artificial intelligence and statistics , pp. 1273-1282. PMLR, 2017.
- Nguyen, H. T., Sehwag, V., Hosseinalipour, S., Brinton, C. G., Chiang, M., and Poor, H. V. Fast-convergent federated learning. IEEE Journal on Selected Areas in Communications , 39(1):201-218, 2020.
- Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. PyTorch: An imperative style, high-performance deep learning library. In Wallach, H., Larochelle, H., Beygelzimer, A., d'Alch´ e-Buc, F., Fox, E., and Garnett, R. (eds.), Advances in Neural Information Processing Systems 32 , pp. 8024-8035. Curran Associates, Inc., 2019. URL http://papers . neurips . cc/paper/9015pytorch-an-imperative-style-highperformance-deep-learning-library . pdf .
- Qiu, X., Parcolle, T., Beutel, D. J., Topa, T., Mathur, A., and Lane, N. D. A first look into the carbon footprint of federated learning. arXiv preprint arXiv:2010.06537 , 2020.
- Reisizadeh, A., Mokhtari, A., Hassani, H., Jadbabaie, A., and Pedarsani, R. FedPAQ: A communication-efficient Federated Learning method with periodic averaging and quantization. arXiv:1909.13014, 2019.
- Rothchild, D., Panda, A., Ullah, E., Ivkin, N., Stoica, I., Braverman, V., Gonzalez, J., and Arora, R. FetchSGD: Communication-efficient federated learning with sketching. In International Conference on Machine Learning , pp. 8253-8265. PMLR, 2020.
- Shen, J., Wang, Y., Xu, P., Fu, Y., Wang, Z., and Lin, Y. Fractional skipping: Towards finer-grained dynamic CNN inference. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 34, pp. 5700-5708, 2020.
- Shi, W. and Dustdar, S. The promise of edge computing. Computer , 49(5):78-81, 2016.
- Shlezinger, N., Chen, M., Eldar, Y. C., Poor, H. V., and Cui, S. UVeQFed: Universal vector quantization for federated learning. IEEE Transactions on Signal Processing , 69:500-514, 2020.
- Speedtest. Speedtest global index. https:// www . speedtest . net/global-index . Accessed: 2021-05-12.
- Sun, X., Choi, J., Chen, C.-Y., Wang, N., Venkataramani, S., Srinivasan, V., Cui, X., Zhang, W., and Gopalakrishnan, K. Hybrid 8-bit floating point (HFP8) training and inference for deep neural networks. In Proceedings of the 33rd International Conference on Neural Information Processing Systems , pp. 4900-4909, 2019.
- Voigt, P. and Von dem Bussche, A. The EU general data protection regulation (GDPR). A Practical Guide, 1st Ed., Cham: Springer International Publishing , 10: 3152676, 2017.
- Wang, N., Choi, J., Brand, D., Chen, C.-Y., and Gopalakrishnan, K. Training deep neural networks with 8-bit floating point numbers. In Bengio, S., Wallach, H., Larochelle, H., Grauman, K., Cesa-Bianchi, N., and Garnett, R. (eds.), Advances in Neural Information Processing Systems 31 , pp. 7675-7684. Curran Associates, Inc., 2018a. URL http://papers . nips . cc/paper/ 7994-training-deep-neural-networkswith-8-bit-floating-point-numbers . pdf .
- Wang, P., Ye, F., and Chen, X. A smart home gateway platform for data collection and awareness. IEEE Communications magazine , 56(9):87-93, 2018b.

## A. Additional simulation details and experiments

## A.1. Additional simulation details

Here, we give detailed information on the models, datasets, training objectives and implementation that we use for our experiments. We set the five following FL tasks:

- Multinomial logistic regression (MLR) on a synthetic dataset called Synthetic that contains vectors in R 60 with a label of one out of 10 classes. We use the synthetic dataset generator in Li et al. (2018) to generate synthetic datasets. The generator samples Synthetic 's local datasets and labels from MLR models with randomly initialized parameters. For this purpose, parameters α and β control different kinds of data heterogeneity. α controls the variation in the local models from which the local dataset labels are generated. β controls the variation in the local dataset samples. We set α = 1 and β = 1 to simulate an FL setting with both kinds of data heterogeneity. This makes Synthetic a useful testbed for FL.
- Character classification into 62 classes of handwritten characters from the FEMNIST dataset using a CNN. FEMNIST groups samples from the same author into the same local dataset.
- Smile detection in facial images from the CelebA dataset using a CNN. CelebA groups samples of the same person into the same local dataset. We note that LEAF's CNN for CelebA uses BatchNorm layers. We replace them with LayerNorm layers because they are more amenable to quantization. This change does not affect the final accuracy.
- Binary sentiment analysis of tweets from the Sent140 dataset using an LSTM. Sent140 groups tweets from the same user into the same local dataset. The majority of local datasets in the raw Sent140 dataset only have a single sample. This impedes FL convergence. Therefore, we filter Sent140 to clients with at least 10 samples (i.e. one complete batch). Caldas et al. (2018); Li et al. (2018) similarly filter Sent140 for their FL experiments.
- Next character prediction on text snippets from the Shakespeare dataset of Shakespeare's collected plays using an LSTM. Shakespeare groups lines from the same character into the same local dataset.

For our experiments in Figure 8, AdaQuantFL requires a hyperparameter s that determines the initial quantization level. We set s to 2, the optimal value reported by the authors of AdaQuantFL. The remaining hyperparameters are identical to those used for the Synthetic dataset experiments in Table 2.

We implement the models with PyTorch (Paszke et al., 2019) and use Flower (Beutel et al., 2020) to simulate the FL server and clients.

## A.2. Additional communication-accuracy trade-off curves

Figure 6. Communication-accuracy trade-off curves for Federated QSGD and DAdaQuant time . We plot the average highest accuracies achieved up to any given amount of communication.

<!-- image -->

Figure 7. Communication-accuracy trade-off curves for Federated QSGD and DAdaQuant clients . We plot the average highest accuracies achieved up to any given amount of communication.

<!-- image -->

## A.3. Computational overhead of DAdaQuant

Table 3. Execution time measurements for different stages of a FL training round on FEMNIST with DAdaQuant. Each entry contains the execution time in seconds and as a fraction of the normal training time. The total overhead of DAdaQuant, including Federated QSGD, is ≈ 1% .

| Training   | DAdaQuant time   | DAdaQuant clients   | Federated QSGD   | Total overhead   |
|------------|------------------|---------------------|------------------|------------------|
| 36 s       | < 1 ms (0.00%)   | 0.17 s (0.47%)      | 0.24 s (0.67%)   | 0.41 s (1.14%)   |

## A.4. Additional UVeQFed experiments

To demonstrate that the choice of UVeQFed's 'coding rate' hyperparameter R does not affect our findings on the superior compression factors of DAdaQuant, we re-evaluate UVeQFed with R = 1 , which maximizes UVeQFed's compression factor. Our results in Table 4 show that with the exception of Shakespeare , DAdaQuant still achieves considerably higher compression factors than UVeQFed.

Table 4. Comparison of the compression factors of DAdaQuant, UVeQFed with R = 4 (default value used for our experiments in Table 2) and UVeQFed with R = 1 (maximizes UVeQFed's compression factor). Entry p × ( q × × ) denotes a compression factor of p w.r.t. the uncompressed communication and a compression factor of q w.r.t. Federated QSGD.

|               | Synthetic              | FEMNIST               | Sent140              | Shakespeare            | Celeba               |
|---------------|------------------------|-----------------------|----------------------|------------------------|----------------------|
| Uncompressed  | 12 . 2 MB              | 132 . 1 GB            | 43 . 9 GB            | 267 . 0 MB             | 12 . 6 GB            |
| QSGD          | 17 ×                   | 2809 ×                | 90 ×                 | 9 . 5 ×                | 648 ×                |
| UVeQFed (R=4) | 0 . 6 × ( 0 . 03 × × ) | 12 × ( 0 . 00 × × )   | 15 × ( 0 . 16 × × )  | 7 . 9 × ( 0 . 83 × × ) | 31 × ( 0 . 05 × × )  |
| UVeQFed (R=1) | 13 × ( 0 . 77 × × )    | 34 × ( 0 . 01 × × )   | 41 × ( 0 . 45 × × )  | 21 × ( 2 . 22 × × )    | 93 × ( 0 . 14 × × )  |
| DAdaQuant     | 48 × ( 2 . 81 × × )    | 4772 × ( 1 . 70 × × ) | 108 × ( 1 . 19 × × ) | 21 × ( 2 . 21 × × )    | 775 × ( 1 . 20 × × ) |

## A.5. Additional AdaQuantFL experiments

<!-- image -->

(a) Comparison of the per-round-communication for AdaQuantFL and DAdaQuant. We plot the average client → server communication per round that is required to train an MLR model on synthetic datasets with 10, 100, 200 and 400 clients. AdaQuantFL's communication increases linearly with the number of clients because it trains the model on all clients at each round. In contrast, DAdaQuant's communication does not change with the number of clients.

(b) Comparison of the convergence speed for AdaQuantFL and DAdaQuant. We plot the test accuracy while training on a synthetic dataset with 100 clients. Although AdaQuantFL has full client participation each round, it converges only slightly faster than DAdaQuant and achieves a similar top accuracy. This means that AdaQuantFL's linear increase in communication is not offset by a proportional reduction in training rounds.

<!-- image -->

Figure 8. Scalability of AdaQuantFL vs. DAdaQuant.

In principle, AdaQuantFL could be adapted to work with partial client participation by computing an estimate of the global loss from the sampled subset of clients. While a full evaluation of this approach is out of the scope of this paper, we conduct a brief feasibility study on FEMNIST. Concretely, we find that a single run of AdaQuantFL with partial client participation on FEMNIST achieved an accuracy of 78.7%, with a total client → server communication of 50.5 MB. In contrast, the same run with DAdaQuant time similarly achieved an accuracy of 78.4%, while lowering the total client → server communication to 27.5 MB.

## B. Proofs

Lemma 1. Take arbitrary quantization level q i ∈ N and parameter p i ∈ [ -t, t ] . Then, Q q i ( p i ) is an unbiased estimator of p i .

Proof. Let s i = t q i , b i = rem( p i , s i ) and u i = s i -b i . Then, we have

$$
\begin{aligned}
& \text {E} \left [ \mathfrak { Q } _ { q _ { i } } ( p _ { i } ) - p _ { i } \right ] \\ & = \frac { u _ { i } } { s _ { i } } ( p _ { i } - b _ { i } ) + \frac { b _ { i } } { s _ { i } } ( p _ { i } + u _ { i } ) \\ & = p _ { i }
\end{aligned}
$$

Lemma 2. For arbitrary t &gt; 0 and parameter p i ∈ [ -t, t ] , let s i = t q i , b i = rem( p i , s i ) and u i = s i -b i . Then, Var ( Q q i ( p i ) ) = u i b i .

Proof.

$$
\begin{aligned}
& \text {Var} \left ( \mathbb { Q } _ { q _ { i } } ( p _ { i } ) \right ) \\ & = \text {E} \left [ \left ( \mathbb { Q } _ { q _ { i } } ( p _ { i } ) - \mathbb { E } \left [ \mathbb { Q } _ { q _ { i } } ( p _ { i } ) \right ] \right ) ^ { 2 } \right ] \\ & = \text {E} \left [ \left ( \mathbb { Q } _ { q _ { i } } ( p _ { i } ) - p _ { i } \right ) ^ { 2 } \right ] & \text {see lemma 1} \\ & = \frac { b _ { i } } { s _ { i } } u _ { i } ^ { 2 } + \frac { u _ { i } } { s _ { i } } b _ { i } ^ { 2 } & \text {see Figure 9}
\end{aligned}
$$

$$
\begin{aligned}
D \lambda a \text {Quant: Doubly-adaptive quantization for communication-efficient Federate} \\ = \frac { u _ { i } b _ { i } } { s _ { i } } \left ( u _ { i } + b _ { i } \right ) \\ = u _ { i } b _ { i } & & \Box \\ & & P ( c s _ { i } ) = \frac { u _ { i } } { s _ { i } } & P ( ( c + 1 ) s _ { i } ) = \frac { b _ { i } } { s _ { i } } \\ & & \stackrel { \longleftrightarrow } { c s _ { i } } & \stackrel { \longleftrightarrow } { p _ { i } } & \cdots \\ & & & & & &
\end{aligned}
$$

Figure 9. Illustration of the Bernoulli random variable Q q i ( p i ) . s i is the length of the quantization interval. p i is rounded up to ( c +1) s i with a probability proportional to its distance from cs i .

Lemma 3. Assume that parameters p 1 . . . p K are sampled from U [ -t, t ] for arbitrary t &gt; 0 . Then, E p 1 ...p K [Var( e q 1 ...q K p )] = t 2 6 ∑ K i =1 w 2 i q 2 i .

## Proof.

$$
\begin{aligned}
\text {Lemma 3. Assume that parameters $p_{1} \dots p_{K}$ are sampled from $\mathbb{U}^{-t,t} $ for arbitrary $t> 0$. Then, } \\ \mathbb{E} _ { p _ { 1 } \dots p _ { K } } [Var(e _ { p } ^ { q } ) ] & = \frac { t ^ { 2 } } { 6 } \sum _ { i = 1 } ^ { K } \frac { w _ { i } ^ { 2 } } { q _ { i } } . \\ \\ \text {Proof.} \\ \\ \mathbb{E} _ { p _ { 1 } \dots p _ { K } } [Var(e _ { p } ) ] \\ & = \frac { 1 } { 2 t } \int _ { - t } ^ { t } \frac { 1 } { 2 t } \int _ { - t } ^ { t } \text {Var} \left ( \sum _ { i = 1 } ^ { K } w _ { i } q _ { i } ( p _ { i } ) - p \right ) d p _ { 1 } d p _ { 2 } \dots d p _ { K } \\ & = \frac { 1 } { t } \int _ { 0 } ^ { t } \int _ { t } ^ { t } \frac { 1 } { t } \int _ { 0 } ^ { t } \text {Var} \left ( \sum _ { i = 1 } ^ { K } w _ { i } q _ { i } ( p _ { i } ) - p \right ) d p _ { 1 } d p _ { 2 } \dots d p _ { K } \\ & = \frac { 1 } { t ^ { n } } \int _ { 0 } ^ { t } \int _ { 0 } ^ { t } \sum _ { i = 1 } ^ { K } w _ { i } ^ { 2 } \text {Var} \left ( Q _ { q _ { i } } ( p _ { i } ) \right ) d p _ { 1 } d p _ { 2 } \dots d p _ { K } \\ & = \frac { 1 } { t ^ { n } } \sum _ { i = 1 } ^ { K } \int _ { 0 } ^ { t } \int _ { 0 } ^ { t } \dots \int _ { 0 } ^ { t } w _ { i } ^ { 2 } \text {Var} \left ( Q _ { q _ { i } } ( p _ { i } ) \right ) d p _ { 1 } d p _ { 2 } \dots d p _ { K } \\ & = \frac { 1 } { t ^ { n } } \sum _ { i = 1 } ^ { K } w _ { i } ^ { 2 } \int _ { 0 } ^ { t } \text {Var} \left ( Q _ { q _ { i } } ( p _ { i } ) \right ) d p _ { i } \\ & = \frac { 1 } { t } \sum _ { i = 1 } ^ { K } w _ { i } ^ { 2 } \int _ { 0 } ^ { t } u _ { i } b _ { i } d p _ { i } \\ & = \frac { 1 } { t } \sum _ { i = 1 } ^ { K } w _ { i } ^ { 2 } q _ { i } \int _ { 0 } ^ { t } u _ { i } b _ { i } d p _ { i } \\ & = \frac { 1 } { t } \sum _ { i = 1 } ^ { K } w _ { i } ^ { 2 } q _ { i } \int _ { 0 } ^ { t } ( s _ { i } - p _ { i } ) p _ { i } d p _ { i } \\ & = \frac { 1 } { 6 } \sum _ { i = 1 } ^ { K } w _ { i } ^ { 2 } q _ { i } s _ { i } \\ & = \frac { t ^ { 2 } } { 6 } \sum _ { i = 1 } ^ { K } \frac { w _ { i } ^ { 2 } } { q _ { i } ^ { 2 } }
\end{aligned}
$$

Lemma 4. Let Q be a fixed-point quantizer. Assume that parameters p 1 . . . p K are sampled from U [ -t, t ] for arbitrary t &gt; 0 . Then, min q 1 ...q K E p 1 ...p K [Var( e q 1 ...q K p )] subject to Q = ∑ K i =1 q i is minimized by q i = Q w 2 / 3 i ∑ K k =1 w 2 / 3 k .

Proof. Define

Any (local) minimum ˆ q satisfies

$$
\begin{aligned}
\text {local} \, \min \, \hat { q } \, \text {satisfies} \, & & \nabla \mathcal { L } \left ( \hat { q } \right ) = 0 \\ & & \quad \Longleftrightarrow \, \nabla \frac { t ^ { 2 } } { 6 } \sum _ { i = 1 } ^ { K } \frac { w _ { i } ^ { 2 } } { q _ { i } ^ { 2 } } - \lambda \nabla \sum _ { i = 1 } ^ { K } q _ { i } = 0 \wedge \sum _ { i = 1 } ^ { K } q _ { i } = Q \quad \text {Lemma 3} \\ & & \quad \Longleftrightarrow \, \forall i = 1 \dots n . \, \frac { t ^ { 2 } } { - 3 } \frac { w _ { i } ^ { 2 } } { q _ { i } ^ { 3 } } = \lambda \wedge \sum _ { i = 1 } ^ { K } q _ { i } = Q \\ & & \quad \Longleftrightarrow \, \forall i = 1 \dots n . \, q _ { i } = \sqrt { \frac { t ^ { 2 } } { - 3 \lambda } } w _ { i } ^ { 2 } \wedge \sum _ { i = 1 } ^ { K } q _ { i } = Q \\ & & \quad \Longrightarrow \, \forall i = 1 \dots n . \, q _ { i } = Q \frac { w _ { i } ^ { 2 / 3 } } { \sum _ { j = 1 } ^ { K } w _ { j } ^ { 2 / 3 } }
\end{aligned}
$$

## B.1. Proof of Theorem 1

Proof. Using Lemma 4, it is straightforward to show that for any V , min q 1 ...q K ∑ K i =1 q i subject to E p 1 ...p K [Var( e q 1 ...q K p )] = V is minimized by q i = Cw 2 / 3 i for the unique C ∈ R &gt; 0 that satisfies E p 1 ...p K [Var( e q 1 ...q K p )] = V .

Then, taking V = E p 1 ...p K [Var( e q p )] and C = √ a b (see Theorem 1), we do indeed get

$$
\begin{aligned}
= \mathbb { E } _ { p _ { 1 } \dots p _ { K } } [ \text {Var} ( e _ { p } ^ { q } ) ] \text { and } C = \sqrt { \frac { \overline { a } } { b } } \left ( \text {see Theorem 1} , \text { we do indeed get} \\ \\ \mathbb { E } _ { p _ { 1 } \dots p _ { K } } [ \text {Var} ( e _ { p } ^ { q _ { 1 } \dots q _ { K } } ) ] \\ = \frac { t ^ { 2 } } { 6 } \sum _ { i = 1 } ^ { K } \frac { w _ { i } ^ { 2 } } { ( C w _ { i } ^ { 2 / 3 } ) ^ { 2 } } \\ = \frac { 1 } { C ^ { 2 } } \frac { t ^ { 2 } } { 6 } \sum _ { i = 1 } ^ { K } w _ { i } ^ { 2 / 3 } \\ = \frac { \sum _ { j = 1 } ^ { K } \frac { w _ { j } ^ { 2 } } { q ^ { \frac { j } { 2 } } } t ^ { 2 } } { \sum _ { j = 1 } ^ { K } w _ { j } ^ { 2 / 3 } } \frac { K } { 6 } \sum _ { i = 1 } ^ { K } w _ { i } ^ { 2 / 3 } \\ = \frac { t ^ { 2 } } { 6 } \sum _ { j = 1 } ^ { K } \frac { w _ { j } ^ { 2 } } { q ^ { 2 } } \\ = \mathbb { E } _ { p _ { 1 } \dots p _ { K } } [ \text {Var} ( e _ { p } ^ { q } ) ]
\end{aligned}
$$

$$
\begin{aligned}
f ( q ) & = \mathbb { E } _ { p _ { 1 } \dots p _ { K } } [ \text {Var} ( e _ { p } ^ { q _ { 1 } \dots q _ { K } } ) ] \\ g ( q ) & = \left ( \sum _ { i = 1 } ^ { n } q _ { i } \right ) \\ \mathcal { L } \left ( q \right ) & = f ( q ) - \lambda g ( q ) \left ( \text {Lagrangian} \right )
\end{aligned}
$$