## Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization

## Jianyu Wang

Carnegie Mellon University Pittsburgh, PA 15213 jianyuw1@andrew.cmu.edu

## Hao Liang

Carnegie Mellon University Pittsburgh, PA 15213 hliang2@andrew.cmu.edu

## Qinghua Liu

Princeton University Princeton, NJ 08544 qinghual@princeton.edu

## Gauri Joshi

Carnegie Mellon University Pittsburgh, PA 15213 gaurij@andrew.cmu.edu

## Abstract

In federated learning, heterogeneity in the clients' local datasets and computation speeds results in large variations in the number of local updates performed by each client in each communication round. Naive weighted aggregation of such models causes objective inconsistency, that is, the global model converges to a stationary point of a mismatched objective function which can be arbitrarily different from the true objective. This paper provides a general framework to analyze the convergence of heterogeneous federated optimization algorithms. It subsumes previously proposed methods such as FedAvg and FedProx, and provides the first principled understanding of the solution bias and the convergence slowdown due to objective inconsistency. Using insights from this analysis, we propose FedNova, a normalized averaging method that eliminates objective inconsistency while preserving fast error convergence.

## 1 Introduction

Federated learning [1-5] is an emerging sub-area of distributed optimization where both data collection and model training is pushed to a large number of edge clients that have limited communication and computation capabilities. Unlike traditional distributed optimization [6, 7] where consensus (either through a central server or peer-to-peer communication) is performed after every local gradient computation, in federated learning, the subset of clients selected in each communication round perform multiple local updates before these models are aggregated in order to update a global model.

Heterogeneity in the Number of Local Updates in Federated Learning. The clients participating in federated learning are typically highly heterogeneous, both in the size of their local datasets as well as their computation speeds. The original paper on federated learning [1] proposed that each client performs E epochs (traversals of their local dataset) of local-update stochastic gradient descent (SGD) with a mini-batch size B . Thus, if a client has n i local data samples, the number of local SGD iterations is τ i = ⌊ En i /B ⌋ , which can vary widely across clients. The heterogeneity in the number of local SGD iterations is exacerbated by relative variations in the clients' computing speeds. Within a given wall-clock time interval, faster clients can perform more local updates than slower clients. The number of local updates made by a client can also vary across communication rounds due to unpredictable straggling or slowdown caused by background processes, outages, memory limitations etc. Finally, clients may use different learning rates and local solvers (instead of vanilla

H. Vincent Poor Princeton University Princeton, NJ 08544 poor@princeton.edu SGD, they may use proximal gradient methods or adaptive learning rate schedules) which may result in heterogeneity in the model progress at each client.

Heterogeneity in Local Updates Causes Objective Inconsistency. Most recent works that analyze the convergence of federated optimization algorithms [8-37] assume that number of local updates is the same across all clients (that is, τ i = τ for all clients i ). These works show that periodic consensus between the locally trained client models attains a stationary point of the global objective function F ( x ) = ∑ m i =1 n i F i ( x ) /n , which is a sum of local objectives weighted by the dataset size n i . However, no current analysis provides insight into the convergence of local-update or federated optimization algorithms in the practical setting when the number of local updates τ i varies across clients 1 , . . . , m . In fact, as we show in Section 3, standard averaging of client models after heterogeneous local updates results in convergence to a stationary point - not of the original objective function F ( x ) , but of an inconsistent objective ˜ F ( x ) , which can be arbitrarily different from F ( x ) depending upon the relative values of τ i . To gain intuition into this phenomenon, observe in Figure 1 that if client 1 performs more local updates, then the updated x ( t +1 , 0) strays towards the local minimum x ∗ 1 , away from the true global minimum x ∗ .

The Need for a General Analysis Framework. A naive approach to overcome heterogeneity is to fix a target number of local updates τ that each client must finish within a communication round and keep fast nodes idle while the slow clients finish their updates. This method will ensure objective consistency (that is, the surrogate objective ˜ F ( x ) equals to the true objective F ( x ) ), nonetheless, waiting for the slowest one can significantly increase the total training time. More sophisticated approaches such as FedProx [38], VRLSGD [21] and SCAFFOLD [20], designed to handle non-IID local datasets, can be used to reduce (not eliminate) objective inconsistency to some extent, but these methods either result in slower convergence or require additional communication and memory. So far, there is no rigorous understanding of the objective inconsistency and the speed of convergence for this challenging setting of federated learning with heterogeneous local updates. It is also unclear how to best combine models trained with heterogeneous levels of local progress.

Contributions of this Paper. To the best of our knowledge, this work provides the first fundamental understanding of the bias in the solution (caused by objective inconsistency) and how the convergence rate is influenced by heterogeneity in clients' local progress. In Section 4 we propose a general theoretical framework that allows heterogeneous number of local updates, non-IID local datasets as

Figure 1: Model updates in the parameter space. Green squares and blue triangles denote the minima of global and local objectives, respectively.

<!-- image -->

well as different local solvers such as GD, SGD, SGD with proximal gradients, gradient tracking, adaptive learning rates, momentum, etc. It subsumes existing methods such as FedAvg and FedProx and provides novel insights on their convergence behaviors. In Section 5 we propose FedNova , a method that correctly weigh local models when averaging. It ensures objective consistency while preserving fast error convergence and outperforms existing methods as shown in Section 6. FedNova works with any local solver and server optimizer and is therefore complementary to existing approaches such as [38, 39, 20, 40].

## 2 System Model and Prior Work

The Federated Heterogeneous Optimization Setting. In federated learning, a total of m clients aim to jointly solve the following optimization problem:

$$
\begin{aligned}
\text {following optimization problem} \colon & \\ & \min _ { x \in \mathbb { R } ^ { d } } \left [ F ( x ) \colon = \sum _ { i = 1 } ^ { m } p _ { i } F _ { i } ( x ) \right ] \\ \text {notes the relative sample size and } & F ( x ) = \frac { 1 } { 1 } \sum _ { i = 1 } ^ { m } \ f ( x ) \colon \text {is the local}
\end{aligned}
$$

where p i = n i /n denotes the relative sample size, and F i ( x ) = 1 n i ∑ ξ ∈D i f i ( x ; ξ ) is the local objective function at the i -th client. Here, f i is the loss function (possibly non-convex) defined by the learning model and ξ represents a data sample from local dataset D i . In the t -th communication round, each client independently runs τ i iterations of local solver ( e.g. , SGD) starting from the current global model x ( t, 0) to optimize its own local objective.

In our theoretical framework, we treat τ i as an arbitrary scalar which can also vary across rounds. In practice, if clients run for the same local epochs E , then τ i = ⌊ En i /B ⌋ , where B is the mini-batch size. Alternately, if each communication round has a fixed length in terms of wall-clock time, then τ i represents the local iterations completed by client i within the time window and may change across clients (depending on their computation speeds and availability) and across communication rounds.

The FedAvg Baseline Algorithm. Federated Averaging ( FedAvg ) [1] is the first and most common algorithm used to aggregate these locally trained models at the central server at the end of each communication round. The shared global model is updated as follows:

$$
\text {communication round.} \ \
$$

where x ( t,k ) i denotes client i 's model after the k -th local update in the t -th communication round and ∆ ( t ) i = x ( t,τ i ) i -x ( t, 0) i denotes the cumulative local progress made by client i at round t . Also, η is the client learning rate and g i represents the stochastic gradient over a mini-batch of B samples. When the number of clients m is large, then the central server may only randomly select a subset of clients to perform computation at each round.

Convergence Analysis of FedAvg. [8-10] first analyze FedAvg by assuming the local objectives are identical and show that FedAvg is guaranteed to converge to a stationary point of F ( x ) . This analysis was further expanded to the non-IID data partition and client sampling cases by [11-18, 23, 24]. However, in all these works, they assume that the number of local steps and the client optimizer are the same across all clients. Besides, asynchronous federated optimization algorithms proposed in [41, 9] take a different approach of allowing clients make updates to stale versions of the global model, and their analyses are limited to IID local datasets and convex local functions.

FedProx: Improving FedAvg by Adding a Proximal Term. To alleviate inconsistency due to non-IID data and heterogeneous local updates, [38] proposes adding a proximal term µ 2 ‖ x -x ( t, 0) ‖ 2 to each local objective, where µ ≥ 0 is a tunable parameter. This proximal term pulls each local model backward closer to the global model x ( t, 0) . Although [38] empirically shows that FedProx improves FedAvg , its convergence analysis is limited by assumptions that are stronger than previous FedAvg analysis and only works for sufficiently large µ . Since FedProx is a special case of our general framework, our convergence analysis provides sharp insights into the effect of µ . We show that a larger µ mitigates (but does not eliminate) objective inconsistency, albeit at an expense of slower convergence. Our proposed FedNova method can improve FedProx by guaranteeing consistency without slowing down convergence.

Improving FedAvg via Momentum and Cross-client Variance Reduction. The performance of FedAvg has been improved in recent literature by applying momentum on the server side [25, 42, 40], or using cross-client variance reduction such as VRLSGD and SCAFFOLD [21, 20]. Again, these works do not consider heterogeneous local progress. Our proposed normalized averaging method FedNova is orthogonal to and can be easily combined with these acceleration or variance-reduction techniques. Moreover, FedNova is also compatible with and complementary to gradient compression/quantization [43-48] and fair aggregation techniques [49, 50].

## 3 A Case Study to Demonstrate the Objective Inconsistency Problem

In this section, we use a simple quadratic model to illustrate the convergence problem. Suppose that the local objective functions are F i ( x ) = 1 2 ‖ x -e i ‖ 2 , where e i ∈ R d is an arbitrary vector and it is the minimum x ∗ i of the local objective. Consider that the global objective function is defined as

Lemma1 ( Objective Inconsistency in FedAvg ) . For the objective function in (3) , if client i performs τ i local steps per round, then FedAvg (with sufficiently small learning rate η , deterministic gradients and full client participation) will converge to

$$
\begin{aligned}
& \quad \text {the minimum } x ^ { \prime } \text { of the local objective. Consider that the global objective function is defined as} \\ & \quad F ( x ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } F _ { i } ( x ) = \sum _ { i = 1 } ^ { m } \frac { 1 } { 2 } \| x - e _ { i } \| ^ { 2 } , \quad \text {which is minimized by } x ^ { \prime } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } e _ { i } . \text { (3)} \\ & \quad \text {Below, we show that the convergence point of FedAvg can be arbitrarily away from } x ^ { * } . \\ & \quad \text {Lemma 1 (Objective Inconsistency in FedAvg). For the objective function in (3) if client i performs}
\end{aligned}
$$

$$
\begin{aligned}
a n d \, \text {full client participation} \, \text {will converge to} \\ \tilde { x } _ { F e d A v g } = \lim _ { T \to \infty } x ^ { ( T , 0 ) } = \frac { \sum _ { i = 1 } ^ { m } \tau _ { i } } { \sum _ { i = 1 } ^ { m } \tau _ { i } } , \text {which minimizes the surrogate obj.} \tilde { F } ( x ) = \frac { \sum _ { i = 1 } ^ { m } \tau _ { i } F _ { i } ( x ) } { \sum _ { i = 1 } ^ { m } \tau _ { i } } .
\end{aligned}
$$

〈]√√∐{̂˜(√}(√[˜(}√√]⌉√⌉

〈]√√∐{̂˜(√}(√[˜(}√√]⌉√⌉

}⌉⌉√{]̂∐√]}{(√}√{̂√ }⌉⌉√{]̂∐√]}{(√}√{̂√ Figure 2: Simulations comparing the FedAvg , FedProx ( µ = 1 ), VRLSGD and our proposed FedNova algorithms for 30 clients with the quadratic objectives defined in (3), where e i ∼ N (0 , 0 . 01 I ) , i ∈ [1 , 30] . Clients perform GD with η = 0 . 05 , which is decayed by a factor of 5 at rounds 600 and 900 . Left : Clients perform the same number of local steps τ i = 30 -FedNova is equivalent to FedAvg in this case; Middle : Clients take different local steps τ i ∈ [1 , 96] with mean 30 but fixed across rounds; Right : local steps are IID, and time-varying Gaussians with mean 30 , i.e. , τ i ( t ) ∈ [1 , 96] . FedNova significantly outperforms others in the heterogeneous τ i setting.



 



 

<!-- image -->



 



 

The proof (of a more general version of Lemma 1) is deferred to the Appendix. While FedAvg aims at optimizing F ( x ) , it actually converges to the optimum of a surrogate objective ˜ F ( x ) . As illustrated in Figure 2, there can be an arbitrarily large gap between ˜ x ∗ FedAvg and x ∗ depending on the relative values of τ i and F i ( x ) . This non-vanishing gap also occurs when the local steps τ i are IID random variables across clients and communication rounds (see the right panel in Figure 2).

Convergence Problem in Other Federated Algorithms. We can generalize Lemma 1 to the case of FedProx to demonstrate its convergence gap, as given in the Appendix. From the simulations shown in Figure 2, observe that FedProx can slightly improve on the optimality gap of FedAvg , but it converges slower. Besides, previous cross-client variance reduction methods such as variancereduced local SGD ( VRLSGD ) [21] and SCAFFOLD [20] are only designed for homogeneous local steps case. In the considered heterogeneous setting, if we replace the same local steps τ in VRLSGD by different τ i 's, then we observe that it has drastically different convergence under different settings and even diverge when clients perform random local steps (see the right panel in Figure 2). These observations emphasize the critical need for a deeper understanding of objective inconsistency and new heterogeneous federated optimization algorithms.

## 4 New Theoretical Framework For Heterogeneous Federated Optimization

We now present a general theoretical framework that subsumes a suite of federated optimization algorithms and helps analyze the effect of objective inconsistency on their error convergence. Although the results are presented for the full client participation setting, it is fairly easy to extend them to the case where a subset of clients are randomly sampled in each round 1 .

## 4.1 A Generalized Update Rule for Heterogeneous Federated Optimization

Recall from (2) that the update rule of federated optimization algorithms can be written as x ( t +1 , 0) -x ( t, 0) = ∑ m i =1 p i ∆ ( t ) i , where ∆ ( t ) i := x ( t,τ i ) -x ( t, 0) denote the local parameter changes of client i at round t and p i = n i /n , the fraction of data at client i . We re-write this update rule in a more general form as follows:

$$
\begin{aligned}
x ^ { ( t + 1 , 0 ) } - x ^ { ( t , 0 ) } = - \tau _ { \text {eff} } \sum _ { i = 1 } ^ { m } w _ { i } \cdot \eta d _ { i } ^ { ( t ) } , \quad \text {which optimizes $\widetilde{F}(x)=\sum_{i=1}^{m}$} w _ { i } F _ { i } ( x ) . \quad (4 )
\end{aligned}
$$

The following three key elements of this update rule take different forms for different algorithms:

The following three key elements of this update rule take different forms for different algorithms:

1 In the case of client sampling, the update rule of FedAvg (2) should hold in expectation in order to guarantee convergence [12, 13, 38, 40]. One can achieve this by either (i) sampling q clients with replacement with respect to probability p i , and then averaging the cumulative local changes with equal weights, or (ii) sampling q clients without replacement uniformly at random, and then weighted averaging local changes, where the weight of client i is re-scaled to p i m/q . Our convergence analysis can be easily extended to these two cases.

1. Locally averaged gradient d ( t ) i : Without loss of generality, we can rewrite the cumulative local changes as ∆ ( t ) i = -η G ( t ) i a i , where G ( t ) i = [ g i ( x ( t, 0) i ) , g i ( x ( t, 1) i ) , . . . , g i ( x ( t,τ i ) i )] ∈ R d × τ i stacks all stochastic gradients in the t -th round, and a i ∈ R τ i is a non-negative vector and defines how stochastic gradients are locally accumulated. Then, by normalizing the gradient weights a i , the locally averaged gradient is defined as d ( t ) i = G ( t ) i a i / ‖ a i ‖ 1 . The normalizing factor ‖ a i ‖ 1 in the denominator is the ℓ 1 norm of the vector a i . By setting different a i , (4) works for most common client optimizers such as SGD with proximal updates, local momentum, and variable learning rate, and more generally, any solver whose cumulative changes ∆ ( t ) i = -η G ( t ) i a i , a linear combination of local gradients.

Specifically, if the client optimizer is vanilla SGD ( i.e. , the case of FedAvg ), then a i = [1 , 1 , . . . , 1] ∈ R τ i and ‖ a i ‖ 1 = τ i . As a result, the normalized gradient is just a simple average of all stochastic gradients within current round: d ( t ) i = G ( t ) i a i /τ i = ∑ τ i -1 k =0 g i ( x ( t,k ) i ) /τ i . Later in this section, we will present more specific examples on how to set a i in other algorithms.

2. Aggregation weights w i : Each client's locally averaged gradient d i is multiplied with weight w i when computing the aggregated gradient ∑ m i =1 w i d i . By definition, these weights satisfy ∑ m i =1 w i = 1 . Observe that these weights determine the surrogate objective ˜ F ( x ) = ∑ m i =1 w i F i ( x ) , which is optimized by the general algorithm in (4) instead of the true global objective F ( x ) = ∑ m i =1 p i F i ( x ) - we will prove this formally in Theorem 1.

<!-- image -->

Figure 3: Comparison between the novel framework and FedAvg in the model parameter space. Solid black arrows denote local updates at clients. Green and blue dots denote the global updates made by the novel generalized update rule and FedAvg respectively. While w i controls the direction of the solid green arrow, effective steps τ eff determines how far the global model moves along with this direction. FedAvg implicitly assigns too higher weights for clients with more local steps, resulting in a biased global direction.

local updates, the average number of local SGD steps per communication round is ¯ τ = ∑ m i =1 τ i /m . However, the server can scale up or scale down the effect of the aggregated updates by setting the parameter τ eff larger or smaller than ¯ τ (analogous to choosing a global learning rate [25, 40]). We refer to the ratio ¯ τ/τ eff as the slowdown , and it features prominently in the convergence analysis presented in Section 4.2.

## 3. Effective number of steps τ eff: Since client i makes τ i

The general rule (4) enables us to freely choose τ eff and w i for a given local solver a i , which helps design fast and consistent algorithms such as FedNova , the normalized averaging method proposed in Section 5. In Figure 3, we further illustrate how the above key elements influence the algorithm and compare the novel generalized update rule and FedAvg in the model parameter space. Besides, in terms of the implementation, the server is not necessary to know the specific form of local accumulation vector a i . Each client can send the normalized update -η d ( t ) i to the central server, which is just a re-scaled version of cumulative local changes ∆ ( t ) i .

Previous Algorithms as Special Cases. Any previous algorithms whose cumulative changes ∆ ( t ) i = -η G ( t ) i a i , a linear combination of local gradients can be subsumed by the above formulation. One can validate this as follows:

$$
\begin{aligned}
One \, \text {can validate this as follows:} \\ x ^ { ( t + 1 , 0 ) } - x ^ { ( t , 0 ) } = & \sum _ { i = 1 } ^ { m } p _ { i } \Delta _ { i } ^ { ( t ) } = - \sum _ { i = 1 } ^ { m } p _ { i } \| a _ { i } \| _ { 1 } \cdot \frac { \eta G _ { i } ^ { ( t ) } a _ { i } } { \| a _ { i } \| _ { 1 } } \\ = & - \left ( \sum _ { i = 1 } ^ { m } p _ { i } \| a _ { i } \| _ { 1 } \right ) \sum _ { i = 1 } ^ { m } \left ( \frac { p _ { i } \| a _ { i } \| _ { 1 } } { \sum _ { i = 1 } ^ { m } p _ { i } \| a _ { i } \| _ { 1 } } \right ) \, \underbrace { \left ( \frac { G _ { i } ^ { ( t ) } a _ { i } } { \| a _ { i } \| _ { 1 } } \right ) } _ { w _ { i } \colon \text {weight} } \, . \\ & 5
\end{aligned}
$$

˜

Unlike the more general form (4), in (6), which subsumes the following previous methods, τ eff and w i are implicitly fixed by the choice of the local solver ( i.e. , the choice of a i ). Due to space limitations, the derivations of following examples are relegated to the Appendix.

- Vanilla SGD as Local Solver (FedAvg). In FedAvg , the local solver is SGD such that a i = [1 , 1 , . . . , 1] ∈ R τ i and ‖ a i ‖ 1 = τ i . As a consequence, the locally averaged gradient d i is a simple average over τ i iterations, τ eff = ∑ m i =1 p i τ i , and w i = p i τ i / ∑ m i =1 p i τ i . That is, the normalized gradients with more local steps will be implicitly assigned higher weights.

τ eff = α -1 ∑ m i =1 p i [1 -(1 -α ) τ i ] , w i = p i [1 -(1 -α ) τ i ] / ∑ m i =1 p i [1 -(1 -α ) τ i ] . (7) When α = 0 , FedProx is equivalent to FedAvg . As α = ηµ increases, the w i in FedProx is more similar to p i , thus making the surrogate objective ˜ F ( x ) more consistent. However, a larger α corresponds to smaller τ eff, which slows down convergence, as we discuss more in the next subsection.

- Proximal SGD as Local Solver (FedProx). In FedProx , local SGD steps are corrected by a proximal term. It can be shown that a i = [(1 -α ) τ i -1 , (1 -α ) τ i -2 , . . . , (1 -α ) , 1] ∈ R τ i , where α = ηµ and µ is a tunable parameter. In this case, we have ‖ a i ‖ 1 = [1 -(1 -α ) τ i ] /α and hence,
- SGD with Decayed Learning Rate as Local Solver. Suppose the clients' local learning rates are exponentially decayed, then we have a i = [1 , γ i , . . . , γ τ i -1 i ] where γ i ≥ 0 can vary across clients. As a result, we have ‖ a i ‖ 1 = (1 -γ τ i i ) / (1 -γ i ) and w i ∝ p i (1 -γ τ i i ) / (1 -γ i ) . Comparing with the case of FedProx (7), changing the values of γ i has a similar effect as changing (1 -α ) .
- Momentum SGD as Local Solver. If we use momentum SGD where the local momentum buffers of active clients are reset to zero at the beginning of each round [25] due to the stateless nature of cross-device FL [2], then we have a i = [1 -ρ τ i , 1 -ρ τ i -1 , . . . , 1 -ρ ] / (1 -ρ ) , where ρ is the momentum factor, and ‖ a i ‖ 1 = [ τ i -ρ (1 -ρ τ i ) / (1 -ρ )] / (1 -ρ ) .

̸

More generally, the new formulation (6) suggests that w i = p i whenever clients have different ‖ a i ‖ 1 , which may be caused by imbalanced local updates ( i.e. , a i 's have different dimensions), or various local learning rate/momentum schedules ( i.e. , a i 's have different scales).

## 4.2 Convergence Analysis for Smooth Non-Convex Functions

In Theorem 1 and Theorem 2 below we provide a convergence analysis for the general update rule (4) and quantify the solution bias due to objective inconsistency. The analysis relies on Assumptions 1 and 2 used in the standard analysis of SGD [51] and Assumption 3 commonly used in the federated optimization literature [38, 12, 13, 20, 40, 2] to capture the dissimilarities of local objectives.

Assumption 1 (Smoothness) . Each local objective function is Lipschitz smooth, that is, ‖∇ F i ( x ) -∇ F i ( y ) ‖ ≤ L ‖ x -y ‖ , ∀ i ∈ { 1 , 2 , . . . , m } .

$$
\begin{aligned}
\mathbb { E } _ { \xi } [ \| g _ { i } ( x | \xi ) - \nabla F _ { i } ( x ) \| ^ { 2 } ] & \leq \sigma ^ { 2 } , \forall i \in \{ 1 , 2 , \dots , m \} , \sigma ^ { 2 } \geq 0 . \\ \text {Assumption} \, 3 \, ( \text {Bounded Disimilarity} ) \, \text {, For any sets of } w _ { \ } e q i g h t s \ \{ w _ { i } >
\end{aligned}
$$

Assumption 2 (Unbiased Gradient and Bounded Variance) . The stochastic gradient at each client is an unbiased estimator of the local gradient: E ξ [ g i ( x | ξ )] = ∇ F i ( x ) , and has bounded variance E 2 2 2 .

Theorem 1 ( Convergence to the Surrogate Objective ˜ F ( x ) 's Stationary Point ) . Under Assumptions 1 to 3, any federated optimization algorithm that follows the update rule (4) , will converge to a stationary point of a surrogate objective ˜ F ( x ) = ∑ m i =1 w i F i ( x ) . More specifically, if the total communication rounds T is pre-determined and the learning rate η is small enough η = √ m / τT where τ = 1 m ∑ m i =1 τ i , then the optimization error will be bounded as follows:

Assumption 3 (Bounded Dissimilarity) . For any sets of weights { w i ≥ 0 } i m =1 , ∑ m i =1 w i = 1 , there exist constants β 2 ≥ 1 , κ 2 ≥ 0 such that ∑ m i =1 w i ‖∇ F i ( x ) ‖ 2 ≤ β 2 ‖ ∑ m i =1 w i ∇ F i ( x ) ‖ 2 + κ 2 . If local functions are identical to each other, then we have β 2 = 1 , κ 2 = 0 .

$$
\begin{aligned}
w h e r \, \overline { \tau } & = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \tau _ { i } , \, \text {then the optimization error will be bounded as follows:} \\ & \min _ { t \in [ T ] } \mathbb { E } \| \nabla \widetilde { F } ( x ^ { t , 0 } ) \| ^ { 2 } \leq \underbrace { \mathcal { O } \left ( \frac { \overline { \tau } / \tau _ { e f f } } { \sqrt { m \tau T } } \right ) + \mathcal { O } \left ( \frac { A \sigma ^ { 2 } } { \sqrt { m \tau T } } \right ) + \mathcal { O } \left ( \frac { m C k ^ { 2 } } { \overline { \tau } T } \right ) } _ { t \in [ T ] } \quad ( 8 ) \\ & w h e r \, \mathcal { O } \text { swallows all constants (including L), and quantities A, B, C are defined as follows:} \\ & A = m \tau _ { \mathcal { T } } + m ^ { \frac { w | | A | | _ { 2 } } { 2 } } \quad B = \sum _ { i = 1 } ^ { m } w _ { i } ( \| \alpha _ { i } \| ^ { 2 } _ { 2 } - a _ { i } ^ { 2 } ) \quad C = \max _ { i } \{ \| \alpha _ { i } \| ^ { 2 } _ { 2 } - \| \alpha _ { i } \| _ { i } - a _ { i } \} _ { 2 } \quad ( 9 )
\end{aligned}
$$

where O swallows all constants (including L ), and quantities A,B,C are defined as follows:

where a i, -1 is the last element in the vector a i .

$$
\begin{aligned}
& \text {where } O \text { swatlows all constants} \left ( m \text {atanginals} \ L \right ) , \text { and } \text {quadratures} \ A , D , C \text { are} \ J \text {mea} \ a \text { join} \ W . \\ & \quad A = m \tau _ { e f f } \sum _ { i = 1 } ^ { m } \frac { w _ { i } ^ { 2 } \| a _ { i } \| _ { 2 } ^ { 2 } } { \| a _ { i } \| _ { 1 } ^ { 2 } } , \ B = \sum _ { i = 1 } ^ { m } w _ { i } ( \| a _ { i } \| _ { 2 } ^ { 2 } - a _ { i , - 1 } ^ { 2 } ) , \ C = \max _ { i } \{ \| a _ { i } \| _ { 1 } ^ { 2 } - \| a _ { i } \| _ { 1 } ^ { 2 } \| a _ { i , - 1 } \} \ \ ( 9 ) \\ & \quad \text {where } a _ { i , - 1 } \text { is the last element in the vector} a _ { i } .
\end{aligned}
$$

In the Appendix, we also provide another version of this theorem that explicitly contains the local learning rate η . Moreover, since the surrogate objective ˜ F ( x ) and the original objective F ( x ) are just different linear combinations of the local functions, once the algorithm converges to a stationary point of F ( x ) , one can also obtain some guarantees in terms of F ( x ) , as given by Theorem 2 below.

$$
\begin{aligned}
w i l l & \text { be bounded as follows:} \\ & \min _ { t \in [ T ] } \| \nabla F ( x ^ { ( t , 0 ) } ) \| ^ { 2 } \leq \underbrace { 2 \left [ \chi _ { p | | w } ^ { 2 } ( \beta ^ { 2 } - 1 ) + 1 \right ] } _ { \text { vanishing error term} } + \underbrace { 2 \chi _ { p | | w } ^ { 2 } \kappa ^ { 2 } } _ { \text { non-vanishing error due to obj. in consistency} } \\ & \text { where } \epsilon _ { \text {opt} } \detenotes the vanishing optimization error given by ( 8 ) \text { and } \chi _ { p | | w } ^ { 2 } = \sum _ { i = 1 } ^ { m } ( p _ { i } - w _ { i } ) ^ { 2 } / w _ { i } \\ & \text { represents the chi-square divergence between vectors } P = [ p _ { i } , \dots , p _ { m } ] \text { and } w = [ w _ { 1 } , w _ { m } , w _ { m } ] ,
\end{aligned}
$$

˜ Theorem 2 ( Convergence in Terms of the True Objective F ( x ) ) . Under the same conditions as Theorem 1, the minimal gradient norm of the true global objective function F ( x ) = ∑ m i =1 p i F i ( x ) will be bounded as follows:

where ϵ opt denotes the vanishing optimization error given by (8) and χ 2 p ‖ w = ∑ m i =1 ( p i -w i ) 2 /w i represents the chi-square divergence between vectors p = [ p 1 , . . . , p m ] and w = [ w 1 , . . . , w m ] .

Discussion: Theorems 1 and 2 describe the convergence behavior of a broad class of federated heterogeneous optimization algorithms. Observe that when all clients take the same number of local steps using the same local solver, we have p = w such that χ 2 = 0 . Also, when all local functions are identical to each other, we have β 2 = 1 , κ 2 = 0 . Only in these two special cases, is there no objective inconsistency. For most other algorithms subsumed by the general update rule in (4), both w i and τ eff are influenced by the choice of a i . When clients have different local progress ( i.e. , different a i vectors), previous algorithms will end up with a non-zero error floor χ 2 κ 2 , which does not vanish to 0 even with sufficiently small learning rate. In Appendix, we further construct a lower bound and show that lim T →∞ min t ∈ [ T ] ‖∇ F ( x ( t, 0) ) ‖ 2 = Ω( χ 2 p ‖ w κ 2 ) , suggesting (10) is tight.

Novel Insights Into the Convergence of FedProx and the Effect of µ . Recall that in FedProx a i =

̸

Figure 4: Illustration on how the parameter α = ηµ influences the convergence of FedProx . We set m = 30 , p i = 1 /m,τ i ∼ N (20 , 20) . ' Weight bias ' denotes the chisquare distance between p and w . ' Slowdown ' and ' Relative Variance ' quantify how the first and the second terms in (8) change.

<!-- image -->

[(1 -α ) τ i -1 , . . . , (1 -α ) , 1] , where α = ηµ . Accordingly, substituting the effective steps and aggregated weight, given by (7), into (8) and (10), we get the convergence guarantee for FedProx . Again, it has objective inconsistency because w i = p i . As we increase α , the weights w i come closer to p i and thus, the non-vanishing error χ 2 κ 2 in (10) decreases (see blue curve in Figure 4). However increasing α worsens the slowdown τ/τ eff, which appears in the first error term in (8) (see the red curve in Figure 4). In the extreme case when α = 1 , although FedProx achieves objective consistency, it has a significantly slower convergence because τ eff = 1 and the first term in (8) is τ times larger than that with FedAvg (eq. to α = 0 ).

Theorem 1 also reveals that, in FedProx , there should exist a best value of α that balances all terms in (8). In Appendix, we provide a corollary showing that α = O ( m 1 2 / τ 1 2 T 1 6 ) optimizes the error bound (8) of FedProx and yields a convergence rate of O ( 1 / √ mτT + 1 / T 2 3 ) on the surrogate objective. This can serve as a guideline on setting α in practice.

Linear Speedup Analysis. Another implication of Theorem 1 is that when the communication rounds T is sufficiently large, then the convergence of the surrogate objective will be dominated by the first two terms in (8), which is 1 / √ mτT . This suggests that the algorithm only uses T/γ total rounds when using γ times more clients ( i.e. , achieving linear speedup) to reach the same error level.

## 5 FedNova: Proposed Federated Normalized Averaging Algorithm

Theorems 1 and 2 suggest an extremely simple solution to overcome the problem of objective inconsistency. When we set w i = p i in (4), then the second non-vanishing term χ 2 p ‖ w κ 2 in (10) will just become zero. This simple intuition yields the following new algorithm:

$$
\begin{aligned}
j \text { become zero. This simple intuition yields the following new algorithm} \\ \text { } \
\end{aligned}
$$

Comparing to previous algorithm x ( t +1 , 0) -x ( t, 0) = ∑ m i =1 p i ∆ ( t ) i , each local change in FedNova is re-scaled by ( ∑ m i =1 p i τ ( t ) i ) /τ ( t ) i . This simple tweak in the aggregation weights eliminates inconsistency in the solution.

$$
\begin{aligned}
\Delta _ { t - 1 } x ^ { ( t + 1 , 0 ) } - x ^ { ( t , 0 ) } & = ( \sum _ { i = 1 } ^ { m } p _ { i } \tau _ { i } ^ { ( t ) } ) \sum _ { i = 1 } ^ { m } p _ { i } \frac { \Delta _ { i } ^ { ( t ) } } { \tau _ { i } ^ { ( t ) } } .
\end{aligned}
$$

x ^ { ( t + 1 , 0 )

$$
- x ^ { ( t , 0 ) } = x ^ { ( t , 0 ) } - \sum _ { i = 1 } ^ { m } p _ { i } \Delta ^ { ( t ) } \ e a c h l o c h a n g i n F e d N o v a }
$$

Flexibility in Choosing Hyper-parameters and Local Solvers. Besides vanilla SGD, the new formulation of FedNova naturally allows clients to choose various local solvers ( i.e. , client-side optimizer). As discussed in Section 4.1, the local solver can also be GD/SGD with decayed local learning rate, GD/SGD with proximal updates, GD/SGD with local momentum, etc. Furthermore, the value of τ eff is not necessarily to be controlled by the local solver as previous algorithms. For example, when using SGD with proximal updates, one can simply set τ eff = ∑ m i =1 p i τ i instead of its default value ∑ m i =1 p i [1 -(1 -α ) τ i ] /α . This can help alleviate the slowdown problem discussed in Section 4.2.

Convergence Analysis. The local solvers at clients do not necessarily need to be the same or fixed across rounds. In the following theorem, we obtain strong convergence guarantee for FedNova , even with arbitrarily time-varying local updates and client optimizers.

Combination with Acceleration Techniques. If clients are stateful and have additional communication bandwidth, they can use cross-client variance reduction techniques to further accelerate the training [21, 20, 39]. In this case, the local gradient at the k -th local step becomes g i ( x ( t,k ) ) + ∑ m i =1 p i d ( t -1) i -d ( t -1) i . Besides, on the server side, one can also implement server momentum or adaptive server optimizers [25, 42, 40], in which the aggregated normalized gradient -τ eff ∑ m i =1 ηp i d i is used to update the server momentum buffer instead of directly updating the server model.

Theorem 3 ( Convergence of FedNova to a Consistent Solution ) . Suppose that each client performs arbitrary number of local updates τ i ( t ) using arbitrary gradient accumulation method a i ( t ) , t ∈ [ T ] per round. Under Assumptions 1 to 3, if local learning rate is set as η = √ m 2 /K , where K = m ∑ T -1 t =0 τ i ( t ) denotes the number of processed mini-batches across all clients after T rounds, then FedNova converges to a stationary point of F ( x ) . The detailed bound is the same as the right hand side of (8) , except that τ, A, B, C are replaced by their average values over all rounds.

Using the techniques developed in [12, 20, 13], Theorem 3 can be further generalized to incorporate client sampling schemes. We provide corresponding corollaries in the Appendix. Moreover, forcing all clients to perform τ = min i τ i local steps (let us call this algorithm FedAvg-min ) can also ensure objective consistency. However, in each round, FedAvg-min will go over less data samples than FedNova ( mbτ min versus b ∑ m i =1 τ i where b is the mini-batch size), resulting in worse performance. Another drawback of a fixed τ algorithm like FedAvg-min is that faster nodes would remain idle in each round while waiting for slower nodes. FedNova avoids such straggling delays by allowing nodes to make different numbers of local updates.

## 6 Experimental Results

Experimental Setup. We evaluate all algorithms on two setups with non-IID data partitioning: (1) Logistic Regression on a Synthetic Federated Dataset : The dataset Synthetic (1 , 1) is originally constructed in [38]. The local dataset sizes n i , i ∈ [1 , 30] follows a power law. (2) DNN trained on a Non-IID partitioned CIFAR-10 dataset : We train a VGG-11 [52] network on the CIFAR10 dataset [53], which is partitioned across 16 clients using a Dirichlet distribution Dir 16 (0 . 1) , as done in [54]. The original CIFAR-10 test set (without partitioning) is used to evaluate the generalization performance of the trained global model. The local learning rate η is decayed by a constant factor after finishing 50% and 75% of the communication rounds. The initial value of η is tuned separately for FedAvg with different local solvers. When using the same solver, FedNova uses the same η as FedAvg to guarantee a fair comparison. On CIFAR-10, we run each experiment with 3 random seeds and report the average and standard deviation. More details are in Appendix 2 .

Figure 5: Results on the synthetic dataset under three different settings. In FedProx , we set µ = 1 , the best value reported in [38]. Left : All clients perform E i = 5 local epochs; Middle : Only C = 0 . 3 fraction of clients are randomly selected per round to perform E i = 5 local epochs; Right : Only C = 0 . 3 fraction of clients are randomly selected per round to perform random and time-varying local epochs E i ( t ) ∼ U (1 , 5) .

<!-- image -->

## Synthetic Dataset Simulations. In

Figure 5, we observe that by simply changing w i to p i , FedNova not only converges faster than FedAvg but also achieves consistently the best performance under three different settings. Note that the only difference between FedNova and FedAvg is the aggregated weights when averaging the normalized gradients.

Non-IID CIFAR-10 Experiments. In Table 1 we compare the performance of FedNova and FedAvg on non-IID CIFAR-10 with various client optimizers run for 100 communication rounds. When the client optimizer is SGD or SGD with momentum, simply changing the weights yields a 6 -9% improvement on the test

Table 1: Results comparing FedAvg and FedNova with various client optimizers ( i.e. , local solvers) trained on nonIID CIFAR-10 dataset. FedProx and SCAFFOLD correspond to FedAvg with proximal SGD updates and cross-client variance-reduction (VR), respectively.

| Local Epochs                                  | Client Opt.                                      | Test Accuracy %                                                                 | Test Accuracy %                                                            |
|-----------------------------------------------|--------------------------------------------------|---------------------------------------------------------------------------------|----------------------------------------------------------------------------|
|                                               |                                                  | FedAvg                                                                          | FedNova                                                                    |
| E i = 2 (16 ≤ τ i ≤ 408)                      | Vanilla Momentum Proximal [38]                   | 60 . 68 ± 1 . 05 65 . 26 ± 2 . 42 60 . 44 ± 1 . 21                              | 66.31 ± 0 . 86 73.32 ± 0 . 29 69.92 ± 0 . 34                               |
| E ( t ) i ∼ U (2 , 5) (16 ≤ τ ( t ) i ≤ 1020) | Vanilla Momentum Proximal [38] VR [20] Momen.+VR | 64 . 22 ± 1 . 06 70 . 44 ± 2 . 99 63 . 74 ± 1 . 44 74 . 72 ± 0 . 34 Not Defined | 73.22 ± 0 . 32 77.07 ± 0 . 12 73.41 ± 0 . 45 74.72 ± 0 . 19 79.19 ± 0 . 17 |

accuracy; When the client optimizer is proximal SGD, FedAvg is equivalent to FedProx . We manually tune the value of µ from { 0 . 0005 , 0 . 001 , 0 . 005 , 0 . 01 } . By setting τ eff = ∑ m i =1 p i τ i and correcting the weights w i = p i while keeping a i same as FedProx, FedNova-Prox achieves about 10% higher test accuracy than FedProx . When using variance-reduction methods such as SCAFFOLD (that requires doubled communication), FedNova -based method preserves the same test accuracy. Furthermore, combining local momentum and variance-reduction in FedNova achieves the highest test accuracy among all other solvers. This kind of combination is non-trivial and has not appeared yet in the literature. We provide its pseudo-code in the Appendix.

Effectiveness of Local Momentum. From Table 1, it is worth noting that using momentum SGD as the local solver is an effective way to improve the performance. It generally achieves 3 -7% higher test accuracy than vanilla SGD. This local momentum scheme can be further combined with server momentum [25, 42, 40]. When E i ( t ) ∼ U (2 , 5) , the hybrid momentum scheme achieves test accuracy 81 . 15 ± 0 . 38% As a reference, using server momentum alone achieves 77 . 49 ± 0 . 25% .

[2 Our code is available at: https://github.com/JYWa/FedNova .](https://github.com/JYWa/FedNova)

## Broader Impact

The future of machine learning lies in moving both data collection as well as model training to the edge. This nascent research field called federated learning considers a large number of resourceconstrained devices such as cellphones or IoT sensors that collect training data from their environment. Due to limited communication capabilities as well as privacy concerns, these data cannot be directly sent over to the cloud. Instead, the nodes locally perform a few iterations of training and only send the resulting model to the cloud. In this paper, we develop a federated training algorithm that is system-aware (robust and adaptable to communication and computation variabilities by allowing heterogeneous local progress) and data-aware (can handle skews in the size and distribution of local training data by correcting model aggregation scheme). This research has the potential to democratize machine learning by transcending the current centralized machine learning framework. It will enable lightweight mobile devices to cooperatively train a common machine learning model while maintaining control of their training data.

## Acknowledgments and Disclosure of Funding

This research was generously supported in part by NSF grants CCF-1850029, the 2018 IBM Faculty Research Award, and the Qualcomm Innovation fellowship (Jianyu Wang). We thank Anit Kumar Sahu, Tian Li, Zachary Charles, Zachary Garrett, and Virginia Smith for helpful discussions.

## References

- [1] H Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, et al. Communicationefficient learning of deep networks from decentralized data. In International Conference on Artificial Intelligence and Statistics (AISTATS) , 2017.
- [2] Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Keith Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. Advances and open problems in federated learning. arXiv preprint arXiv:1912.04977 , 2019.
- [3] Jakub Koneˇ cn` y, H Brendan McMahan, Felix X Yu, Peter Richtárik, Ananda Theertha Suresh, and Dave Bacon. Federated learning: Strategies for improving communication efficiency. arXiv preprint arXiv:1610.05492 , 2016.
- [4] Jakub Koneˇ cn` y, Brendan McMahan, and Daniel Ramage. Federated optimization: Distributed optimization beyond the datacenter. arXiv preprint arXiv:1511.03575 , 2015.
- [5] Wei Yang Bryan Lim, Nguyen Cong Luong, Dinh Thai Hoang, Yutao Jiao, Ying-Chang Liang, Qiang Yang, Dusit Niyato, and Chunyan Miao. Federated learning in mobile edge networks: A comprehensive survey. IEEE Communications Surveys &amp; Tutorials , 2020.
- [6] Mu Li, David G Andersen, Jun Woo Park, Alexander J Smola, Amr Ahmed, Vanja Josifovski, James Long, Eugene J Shekita, and Bor-Yiing Su. Scaling distributed machine learning with the parameter server. In OSDI , volume 14, pages 583-598, 2014.
- [7] Angelia Nedi´ c, Alex Olshevsky, and Michael G Rabbat. Network topology and communicationcomputation tradeoffs in decentralized optimization. Proceedings of the IEEE , 106(5):953-976, 2018.
- [8] Jianyu Wang and Gauri Joshi. Cooperative SGD: A unified framework for the design and analysis of communication-efficient SGD algorithms. arXiv preprint arXiv:1808.07576 , 2018.
- [9] Sebastian U Stich. Local SGD converges fast and communicates little. In International Conference on Learning Representations (ICLR) , 2019.
- [10] Fan Zhou and Guojing Cong. On the convergence properties of a k-step averaging stochastic gradient descent algorithm for nonconvex optimization. In Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI) , pages 3219-3227, 2018.

- [11] Hao Yu, Sen Yang, and Shenghuo Zhu. Parallel restarted SGD for non-convex optimization with faster convergence and less communication. arXiv preprint arXiv:1807.06629 , 2018.
- [12] Xiang Li, Kaixuan Huang, Wenhao Yang, Shusen Wang, and Zhihua Zhang. On the convergence of FedAvg on non-IID data. In International Conference on Learning Representations , 2020.
- [13] Farzin Haddadpour and Mehrdad Mahdavi. On the convergence of local descent methods in federated learning. arXiv preprint arXiv:1910.14425 , 2019.
- [14] Farzin Haddadpour, Mohammad Mahdi Kamani, Mehrdad Mahdavi, and Viveck Cadambe. Trading redundancy for communication: Speeding up distributed SGD for non-convex optimization. In International Conference on Machine Learning , pages 2545-2554, 2019.
- [15] Farzin Haddadpour, Mohammad Mahdi Kamani, Mehrdad Mahdavi, and Viveck Cadambe. Local SGD with periodic averaging: Tighter analysis and adaptive synchronization. In Advances in Neural Information Processing Systems , pages 11080-11092, 2019.
- [16] A Khaled, K Mishchenko, and P Richtárik. Tighter theory for local SGD on identical and heterogeneous data. In The 23rd International Conference on Artificial Intelligence and Statistics (AISTATS 2020) , 2020.
- [17] Sebastian U Stich and Sai Praneeth Karimireddy. The error-feedback framework: Better rates for SGD with delayed gradients and compressed communication. arXiv preprint arXiv:1909.05350 , 2019.
- [18] Shiqiang Wang, Tiffany Tuor, Theodoros Salonidis, Kin K Leung, Christian Makaya, Ting He, and Kevin Chan. Adaptive federated learning in resource constrained edge computing systems. IEEE Journal on Selected Areas in Communications , 37(6):1205-1221, 2019.
- [19] Jianyu Wang and Gauri Joshi. Adaptive communication strategies to achieve the best errorruntime trade-off in local-update SGD. arXiv preprint arXiv:1810.08313 , 2018.
- [20] Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank J Reddi, Sebastian U Stich, and Ananda Theertha Suresh. SCAFFOLD: Stochastic controlled averaging for on-device federated learning. arXiv preprint arXiv:1910.06378 , 2019.
- [21] Xianfeng Liang, Shuheng Shen, Jingchang Liu, Zhen Pan, Enhong Chen, and Yifei Cheng. Variance reduced local SGD with lower communication complexity. arXiv preprint arXiv:1912.12844 , 2019.
- [22] Blake Woodworth, Kumar Kshitij Patel, Sebastian U Stich, Zhen Dai, Brian Bullins, H Brendan McMahan, Ohad Shamir, and Nathan Srebro. Is local SGD better than minibatch SGD? arXiv preprint arXiv:2002.07839 , 2020.
- [23] Anastasia Koloskova, Nicolas Loizou, Sadra Boreiri, Martin Jaggi, and Sebastian U Stich. A unified theory of decentralized SGD with changing topology and local updates. arXiv preprint arXiv:2003.10422 , 2020.
- [24] Hao Yu, Rong Jin, and Sen Yang. On the linear speedup analysis of communication efficient momentum SGD for distributed non-convex optimization. In International Conference on Machine Learning , 2019.
- [25] Jianyu Wang, Vinayak Tantia, Nicolas Ballas, and Michael Rabbat. SlowMo: Improving communication-efficient distributed SGD with slow momentum. In International Conference on Learning Representations , 2020.
- [26] Zhouyuan Huo, Qian Yang, Bin Gu, Lawrence Carin Huang, et al. Faster on-device training using new federated momentum algorithm. arXiv preprint arXiv:2002.02090 , 2020.
- [27] Fan Zhou and Guojing Cong. A distributed hierarchical SGD algorithm with sparse global reduction. arXiv preprint arXiv:1903.05133 , 2019.
- [28] Xinwei Zhang, Mingyi Hong, Sairaj Dhople, Wotao Yin, and Yang Liu. FedPD: A federated learning framework with optimal rates and adaptivity to non-IID data. arXiv preprint arXiv:2005.11418 , 2020.

- [29] Reese Pathak and Martin J Wainwright. FedSplit: An algorithmic framework for fast federated optimization. arXiv preprint arXiv:2005.05238 , 2020.
- [30] Ahmed Khaled, Konstantin Mishchenko, and Peter Richtárik. First analysis of local gd on heterogeneous data. arXiv preprint arXiv:1909.04715 , 2019.
- [31] Blake E Woodworth, Jialei Wang, Adam Smith, Brendan McMahan, and Nati Srebro. Graph oracle models, lower bounds, and gaps for parallel stochastic optimization. In Advances in neural information processing systems , pages 8496-8506, 2018.
- [32] Yue Zhao, Meng Li, Liangzhen Lai, Naveen Suda, Damon Civin, and Vikas Chandra. Federated learning with non-IID data. arXiv preprint arXiv:1806.00582 , 2018.
- [33] Cong Xie, Oluwasanmi Koyejo, Indranil Gupta, and Haibin Lin. Local AdaAlter: Communication-efficient stochastic gradient descent with adaptive learning rates. arXiv preprint arXiv:1911.09030 , 2019.
- [34] Tao Lin, Sebastian U Stich, and Martin Jaggi. Don't use large mini-batches, use local SGD. In International Conference on Learning Representations (ICLR) , 2020.
- [35] Grigory Malinovsky, Dmitry Kovalev, Elnur Gasanov, Laurent Condat, and Peter Richtarik. From local sgd to local fixed point methods for federated learning. arXiv preprint arXiv:2004.01442 , 2020.
- [36] Jianyu Wang, Hao Liang, and Gauri Joshi. Overlap local-SGD: An algorithmic approach to hide communication delays in distributed SGD. arXiv preprint arXiv:2002.09539 , 2020.
- [37] Aymeric Dieuleveut and Kumar Kshitij Patel. Communication trade-offs for local-sgd with large step size. In Advances in Neural Information Processing Systems , pages 13579-13590, 2019.
- [38] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated optimization in heterogeneous networks. In Conference on Machine Learning and Systems , 2020.
- [39] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smithy. Feddane: A federated newton-type method. In 2019 53rd Asilomar Conference on Signals, Systems, and Computers , pages 1227-1231. IEEE, 2019.
- [40] Sashank Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Koneˇ cn` y, Sanjiv Kumar, and H Brendan McMahan. Adaptive federated optimization. arXiv preprint arXiv:2003.00295 , 2020.
- [41] Cong Xie, Sanmi Koyejo, and Indranil Gupta. Asynchronous federated optimization. arXiv preprint arXiv:1903.03934 , 2019.
- [42] Tzu-Ming Harry Hsu, Hang Qi, and Matthew Brown. Measuring the effects of non-identical data distribution for federated visual classification. arXiv preprint arXiv:1909.06335 , 2019.
- [43] Amirhossein Reisizadeh, Aryan Mokhtari, Hamed Hassani, Ali Jadbabaie, and Ramtin Pedarsani. Fedpaq: A communication-efficient federated learning method with periodic averaging and quantization. arXiv preprint arXiv:1909.13014 , 2019.
- [44] Debraj Basu, Deepesh Data, Can Karakus, and Suhas Diggavi. Qsparse-local-SGD: Distributed SGD with quantization, sparsification and local computations. In Advances in Neural Information Processing Systems , pages 14668-14679, 2019.
- [45] Hongyi Wang, Scott Sievert, Shengchao Liu, Zachary Charles, Dimitris Papailiopoulos, and Stephen Wright. Atomo: Communication-efficient learning via atomic sparsification. In Advances in Neural Information Processing Systems , pages 9850-9861, 2018.
- [46] Felix Sattler, Simon Wiedemann, Klaus-Robert Müller, and Wojciech Samek. Robust and communication-efficient federated learning from non-IID data. IEEE transactions on neural networks and learning systems , 2019.

- [47] Zhize Li, Dmitry Kovalev, Xun Qian, and Peter Richtárik. Acceleration for compressed gradient descent in distributed and federated optimization. arXiv preprint arXiv:2002.11364 , 2020.
- [48] Feijie Wu, Shiqi He, Yutong Yang, Haozhao Wang, Zhihao Qu, and Song Guo. On the convergence of quantized parallel restarted sgd for serverless learning. arXiv preprint arXiv:2004.09125 , 2020.
- [49] Tian Li, Maziar Sanjabi, and Virginia Smith. Fair resource allocation in federated learning. In International Conference on Learning Representations (ICLR) , 2020.
- [50] Mehryar Mohri, Gary Sivek, and Ananda Theertha Suresh. Agnostic federated learning. arXiv preprint arXiv:1902.00146 , 2019.
- [51] Léon Bottou, Frank E Curtis, and Jorge Nocedal. Optimization methods for large-scale machine learning. SIAM Review , 60(2):223-311, 2018.
- [52] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556 , 2014.
- [53] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, Citeseer, 2009.
- [54] Hongyi Wang, Mikhail Yurochkin, Yuekai Sun, Dimitris Papailiopoulos, and Yasaman Khazaeni. Federated learning with matched averaging. In International Conference on Learning Representations (ICLR) , 2020.