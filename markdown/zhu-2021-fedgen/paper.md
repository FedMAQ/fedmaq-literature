## Data-Free Knowledge Distillation for Heterogeneous Federated Learning

Zhuangdi Zhu 1 Junyuan Hong 1 Jiayu Zhou 1

## Abstract

Federated Learning (FL) is a decentralized machine-learning paradigm in which a global server iteratively aggregates the model parameters of local users without accessing their data. User heterogeneity has imposed significant challenges to FL, which can incur drifted global models that are slow to converge. Knowledge Distillation has recently emerged to tackle this issue, by refining the server model using aggregated knowledge from heterogeneous users, other than directly aggregating their model parameters. This approach, however, depends on a proxy dataset, making it impractical unless such prerequisite is satisfied. Moreover, the ensemble knowledge is not fully utilized to guide local model learning, which may in turn affect the quality of the aggregated model. In this work, we propose a data-free knowledge distillation approach to address heterogeneous FL, where the server learns a lightweight generator to ensemble user information in a data-free manner, which is then broadcasted to users, regulating local training using the learned knowledge as an inductive bias. Empirical studies powered by theoretical implications show that, our approach facilitates FL with better generalization performance using fewer communication rounds, compared with the state-of-the-art 1 .

## 1. Introduction

Federated Learning (FL) is an effective machine learning approach that enables the decentralization of computing and data resources. Classical FL, represented by FEDAVG (McMahan et al., 2017), obtains an aggregated model by iteratively averaging the parameters of distributed local user models, therefore omits the need of accessing their data. Serving as a communication-efficient and

1 Department of Computer Science and Engineering, Michigan State University, Michigan, USA. Correspondence to: Zhuangdi Zhu &lt; zhuzhuan@msu.edu &gt; , Jiayu Zhou &lt; jiayuz@msu.edu &gt; .

Proceedings of the 38 th International Conference on Machine Learning , PMLR 139, 2021. Copyright 2021 by the author(s).

privacy-preserving learning scheme, FL has shown its potential to facilitate real-world applications, including healthcare (Sheller et al., 2020), biometrics (Aggarwal et al., 2021), and natural language processing (Hard et al., 2018; Ammad-Ud-Din et al., 2019), to name just a few.

Along with its promising prospect, FL faces practical challenges from data heterogeneity (Li et al., 2020b), in that user data from real-world is usually non-iid distributed, which inherently induces deflected local optimum (Karimireddy et al., 2020). Moreover, the permutation-invariant property of deep neural networks has further increased the heterogeneity among user models (Yurochkin et al., 2019; Wang et al., 2020b). Thus, performing element-wise averaging of local models, as adopted by most existing FL approaches, may not induce an ideal global model (Li et al., 2020c;b).

A variety of efforts have been made to tackle user heterogeneity, mainly from two complementary perspectives: one focuses on stabilizing local training, by regulating the deviation of local models from a global model over the parameter space (Li et al., 2020b; Dinh et al., 2020; Karimireddy et al., 2020). This approach may not fully leverage the underlying knowledge across user models, whose diversity suggests informative structural differences of their local data and thus deserves more investigation. Another aims to improve the efficacy of model aggregation (Yurochkin et al., 2019; Chen &amp; Chao, 2021), among which knowledge distillation has emerged as an effective solution (Lin et al., 2020; Li &amp;Wang, 2019). Provided with an unlabeled dataset as the proxy, knowledge distillation alleviates the model drift issue induced by heterogeneity, by enriching the global model with the ensemble knowledge from local models, which is shown to be more effective than simple parameter-averaging. However, the prerequisite of a proxy data can leave such an approach infeasible for many applications, where a carefully engineered dataset may not always be available on the server. Moreover, by only refining the global model, the inherent heterogeneity among user models is not fully addressed, which may in turn affect the quality of the knowledge ensemble, especially if they are biased due to limited local data (Khoussainov et al., 2005), which is a typical case for FL.

Observing the challenge in the presence of user heterogeneity and the limitations of prior art, in this work, we propose a data-free knowledge distillation approach for FL, dubbed as FEDGEN ( Fe derated D istillation via Gen erative Learning). Specifically, FEDGEN learns a generative model derived solely from the prediction rules of user models, which, given a target label, can yield feature representations that are consistent with the ensemble of user predictions. This generator is later broadcasted to users, escorting their model training with augmented samples over the latent space, which embodies the distilled knowledge from other peer users. Given a latent space with a dimension much smaller than the input space, the generator learned by FEDGEN can be lightweight, introducing minimal overhead to the current FL framework.

The proposed FEDGEN enjoys multifold benefits: i) It extracts the knowledge out of users which was otherwise mitigated after model averaging, without depending on any external data. ii) Contrary to certain prior work that only refines the global model, our approach directly regulates local model updating using the extracted knowledge. We show that such knowledge imposes an inductive bias to local models, leading to better generalization performance under non-iid data distributions. iii) Furthermore, the proposed approach is ready to address more challenging FL scenarios, where sharing entire model parameters is impractical due to privacy or communication constraints, since the proposed approach only requires the prediction layer of local models for knowledge extraction.

Extensive empirical studies echoed by theoretical elaborations show that, our proposed approach yields a global model with better generalization performance using fewer communication rounds, compared with the state-of-the-art.

## 2. Notations and Preliminaries

Without ambiguity, in this work, we discuss a typical FL setting for supervised learning , i.e., the general problem of multi-class classification. Let X ⊂ R p be an instance space, Z ⊂ R d be a latent feature space with d &lt; p , and Y ⊂ R be an output space. T denotes a domain which consists of a data distribution D over X and a ground-truth labeling function c ∗ : X → Y , i.e. T := 〈 D , c ∗ 〉 . Note that we will use the term domain and task equivalently. A model parameterized by θ := [ θ f ; θ p ] consists of two components: a feature extractor f : X → Z parametrized by θ f , and a predictor h : Z →△ Y parameterized by θ p , where △ Y is the simplex over Y . Given a non-negative, convex loss function l : △ Y × Y → R , the risk of a model parameterized by θ on domain T is defined as L T ( θ ) := E x ∼ D [ l ( h ( f ( x ; θ f ); θ p ) , c ∗ ( x ) )] .

Federated Learning aims to learn a global model parameterized by θ that minimizes its risk on each of the user tasks T k (McMahan et al., 2017):

$$
\min _ { \theta } \, \mathbb { E } _ { \mathcal { T } _ { k } \in \mathcal { T } } \left [ \mathcal { L } _ { k } ( \theta ) \right ] , \quad \ ( 1 )
$$

where T = {T k } K k =1 is the collection of user tasks. We consider all tasks sharing the same labeling rules c ∗ and loss function l , i.e. , T k = 〈 D k , c ∗ 〉 . In practice, Equation 1 is empirically optimized by min θ 1 K ∑ K k =1 ˆ L k ( θ ) , where ˆ L k ( θ ) := 1 | ˆ D k | ∑ x i ∈ ˆ D k [ l ( h ( f ( x i ; θ f ); θ p ) , c ∗ ( x i )) ] is the empirical risk over an observable dataset ˆ D k . An implied assumption for FL is that the global data ˆ D is distributed to each of the local domains, with ˆ D = ∪ { ˆ D k } K k =1 .

Knowledge Distillation (KD) is also referred as a teacherstudent paradigm, with the goal of learning a lightweight student model using knowledge distilled from one or more powerful teachers (Buciluˇ a et al., 2006; Ba &amp; Caruana, 2014). Typical KD leverages a proxy dataset ˆ D P to minimize the discrepancy between the logits outputs from the teacher model θ T and the student model θ S , respectively. A representative choice is to use Kullback-Leibler divergence to measure such discrepancy (Hinton et al., 2015):

$$
\begin{array} { l l } \text {gulates} & \quad \\ \text {edge. We } & \min _ { \theta _ { S } } \, \mathbb { E } _ { x \sim \hat { D } _ { P } } \left [ D _ { K L } \left [ \sigma ( g ( f ( x ; \theta _ { T } ^ { f } ) ; \theta _ { T } ^ { p } ) \| \sigma ( g ( f ( x ; \theta _ { S } ^ { f } ) ; \theta _ { S } ^ { p } ) ) \right ] \right ] , \\ \text {to local} & \\ \text {to local} & \end{array} ,
$$

where g ( · ) is the logits output of an predictor h , and σ ( · ) is the non-linear activation applied to such logits, i.e. h ( z ; θ p ) = σ ( g ( z ; θ p )) .

The idea of KD has been extended to FL to tackle user heterogeneity (Lin et al., 2020; Chen &amp; Chao, 2021), by treating each user model θ k as the teacher , whose information is aggregated into the student (global) model θ to improve its generalization performance:

$$
\begin{aligned}
a \text { global} \quad \\ \text {ing fewer} \quad \min _ { \theta } \, \mathbb { E } \left [ D _ { K L } [ \sigma ( \frac { 1 } { K } \sum _ { k = 1 } ^ { K } g ( f ( x ; \theta _ { k } ^ { f } ) ; \theta _ { k } ^ { p } ) ) \| \sigma ( g ( f ( x ; \theta ^ { f } ) ; \theta ^ { p } ) ] \right ] .
\end{aligned}
$$

One primary limitation of the above approach resides in its dependence on a proxy dataset ˆ D P, the choice of which needs delicate consideration and plays a key role in the distillation performance (Lin et al., 2020). Next, we show how we make KD feasible for FL in a data-free manner.

Figure 1. Overview of FEDGEN: a generator G w ( ·| y ) is learned by the server to aggregate information from different local clients without observing their data. The generator is then sent to local users, whose knowledge is distilled to user models to adjust their interpretations of a good feature distribution.

<!-- image -->

## Algorithm 1 FEDGEN

- 1: Require: Tasks K

Global parameters θ , local parameters { θ k } K k =1 ; Generator parameter w ; ˆ p ( y ) uniformly initialized; Learning rate α , β , local steps T , batch size B , local c

- {T k } k =1 ; label counter k .

## 2: repeat

- 3: Server selects active users A uniformly at random, then broadcast w , θ , ˆ p ( y ) to A .
- 4: for all user k ∈ A in parallel do
- 5: θ k ← θ ,
- 6: for t = 1 , . . . , T do
- 7: { x i , y i } B i =1 ∼ T k , { ˆ z i ∼ G w ( ·| ˆ y i ) , ˆ y i ∼ ˆ p ( y ) } B i =1 . 8: Update label counter c k .
- 9: θ k ← θ k -β ∇ θ k J ( θ k ) . ▷ Optimize Equation 5 10: end for
- 11: User sends θ k , c k back to server.
- 12: end for
- 13: Server updates θ ← 1 |A| ∑ k ∈ A θ k , and ˆ p ( y ) based on { c k } k ∈ A .
- 14: w ← w -α ∇ w J ( w ) . ▷ Optimize Equation 4 15: until training stop

## 3. FEDGEN: Data-Free Federated Distillation via Generative Learning

In this section, we elaborate our proposed approach with a summary shown in Algorithm 1. An overview of its learning procedure in illustrated in Figure 1.

## 3.1. Knowledge Extraction

Our core idea is to extract knowledge about the global view of data distribution, which is otherwise non-observable by conventional FL, and distill such knowledge to local models to guide their learning. We first consider learning a conditional distribution Q ∗ : Y → X to characterize such knowledge, which is consistent with the ground-truth data distributions:

$$
\begin{aligned}
Q ^ { * } = \arg \max _ { Q \cdot y \to \chi } \, \mathbb { E } _ { y \sim p ( y ) } \mathbb { E } _ { x \sim Q ( x | y ) } [ \log p ( y | x ) ] , \quad ( 2 )
\end{aligned}
$$

where p ( y ) and p ( y | x ) are the ground-truth prior and posterior distributions of the target labels, respectively, both of which are unknown. To make Equation 2 optimizable w.r.t Q , we replace p ( y ) and p ( x | y ) with their empirical approximations. First, we estimate p ( y ) as:

$$
\begin{aligned}
\hat { p } ( y ) \, \infty \sum _ { k } \mathbb { E } _ { x \sim \hat { \mathcal { D } } _ { k } } [ I ( c ^ { * } ( x ) = y ) ] , \\ U ( \, ) \, \text {in} \, y \, \infty \, \text {function} \, f \, x \sim \hat { \mathcal { D } } _ { k } \, \hat { U } \, \text { in} \, \hat { \mathcal { D } } \, y \, \infty \,
\end{aligned}
$$

where I ( · ) is an indicator function, and ˆ D k is the observable data for domain T k . In practice, ˆ p ( y ) can be obtained by requiring the training label counts from users during the model uploading phase. Next, we approximate p ( y | x ) using the ensemble wisdom from user models:

$$
\begin{aligned}
\log \hat { p } ( y | x ) \, \infty \, \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \log p ( y | x ; \theta _ { k } ) .
\end{aligned}
$$

Equipped with the above approximations, directly optimizing Equation 2 over the input space X can still be prohibitive: it brings computation overloads when X is of high dimension, and may also leak information about the user data profile. A more approachable idea is hence to recover an induced distribution G ∗ : Y → Z over a latent space, which is more compact than the raw data space and can alleviate certain privacy-related concerns:

$$
\begin{aligned}
G ^ { * } = \arg \max _ { G \colon y \to z } \, \mathbb { E } _ { y \sim \hat { p } ( y ) } \mathbb { E } _ { z \sim G ( z | y ) } \left [ \sum _ { k = 1 } ^ { K } \log p ( y | z ; \theta _ { k } ^ { p } ) \right ] . \ ( 3 )
\end{aligned}
$$

Following the above reasoning, we aim to perform knowledge extraction by learning a conditional generator G , parameterized by w to optimize the following objective:

$$
\begin{aligned}
\min _ { w } J ( w ) \colon = \mathbb { E } _ { y \sim \hat { p } ( y ) } \mathbb { E } _ { z \sim G _ { w } ( z | y ) } \left [ l ( \sigma ( \frac { 1 } { K } \sum _ { k = 1 } ^ { K } g ( z ; \theta _ { k } ^ { p } ) ) , y ) \right ] ,
\end{aligned}
$$

where g and σ are the logit-output and the activation function as defined in Section 2. Given an arbitrary sample y , optimizing Equation 4 only requires access to the predictor modules θ p k of user models. Specifically, to enable diversified outputs from G ( ·| y ) , we introduce a noise vector ϵ ∼ N (0 , I ) to the generator, which is resemblant to the reparameterization technique proposed by prior art (Kingma &amp; Welling, 2014), so that z ∼ G w ( ·| y ) ≡ G w ( y, ϵ | ϵ ∼ N (0 , I )) . We discuss more implementation details in the supplementary.

Given arbitrary target labels y , the proposed generator can yield feature representations z ∼ G w ( ·| y ) that induce ideal predictions from the ensemble of user models. In other words, the generator approximates an induced image of a consensual distribution, which is consistent with the user data from a global view.

## 3.2. Knowledge Distillation

The learned generator G w is then broadcasted to local users, so that each user model can sample from G w to obtain augmented representations z ∼ G w ( ·| y ) over the feature space. As a result, the objective of a local model θ k is altered to maximize the probability that it yields ideal predictions for the augmented samples:

$$
\min _ { \theta _ { k } } \, J ( \theta _ { k } ) \colon = \hat { \mathcal { L } } _ { k } ( \theta _ { k } ) + \hat { \mathbb { E } } _ { y \sim \hat { p } ( y ) , z \sim G _ { w } ( z | y ) } \left [ l ( h ( z ; \theta _ { k } ^ { p } ) ; y ) \right ] , \ ( 5 )
$$

$$
\begin{array} { r l } & { w h e r e \ \hat { \mathcal { L } } _ { k } ( \theta _ { k } ) \colon = \frac { 1 } { | \mathcal { D } _ { k } | } \sum _ { x _ { i } \in \hat { \mathcal { D } } _ { k } } \left [ l ( h ( f ( x _ { i } ; \theta _ { k } ^ { f } ) ; \theta _ { k } ^ { p } ) , c ^ { * } ( x _ { i } ) ) \right ] } \\ & { v i s t h e e p i r i c a l r i s k g i v e n l o c a l d a t a \ \hat { \mathcal { D } } _ { k } . W e s h o w l a t e r } \end{array}
$$

is the empirical risk given local data ˆ D k . We show later that the augmented samples can introduce inductive bias to local users, reinforcing their model learning with a better generalization performance.

Up to this end, our proposed approach has realized datafree knowledge distillation, by interactively learning a lightweight generator that primarily depends on the prediction rule of local models, and leveraging the generator to convey consensual knowledge to local users. We justify in Section 6.2 that our approach can effectively handle user heterogeneity in FL, which also enjoys theoretical advantages as analyzed in Section 4.

## 3.3. Extensions for Flexible Parameter Sharing

In addition to tackling data heterogeneity, FEDGEN can also handle a challenging FL scenario where sharing the entire model is against communication or privacy prerequisites. On one hand, advanced networks with deep feature extraction layers typically contain millions of parameters (He et al., 2016; Brown et al., 2020), which bring significant burdens to communication. On the other hand, it has been shown feasible to backdoor regular FL approaches (Wang et al., 2020a). For practical FL applications such as healthcare or finance, sharing entire model parameters may be associated with considerable privacy risks, as discussed in prior work (He et al., 2020).

FEDGEN is ready to alleviate those problems, by sharing only the prediction layer θ p k of local models, which is the primary information needed to optimizing Equation 4, while keeping the feature extractor θ f k localized. This partial sharing paradigm is more efficient, and at the same time less vulnerable to data leakage, as compared with a strategy that shares the entire model. Empirical study in Section 6.4 shows that, FEDGEN significantly benefits local users, even without sharing feature extraction modules. We defer the algorithmic summary of this variant approach to the supplementary.

## 4. FEDGEN Analysis

In this section, we provide multiple perspectives to understand our proposed approach. We first visualize what knowledge is learned and distilled by FEDGEN, then analyze why the distilled knowledge is favorable, from the viewpoint of distribution matching and domain adaptation , respectively. Weprimarily focus on interpreting the rationale behind FEDGEN and leave detailed discussion and derivations to the supplementary.

## 4.1. Knowledge Distillation for Inductive Bias

We illustrate the KD process in FEDGEN on a FL prototype, which contains three users, each assigned with a disjoint dataset ˆ D k , k ∈ { 1 , 2 , 3 } . When trained using only the local data, a user model is prone to learn biased decision boundaries (See Figure 2a).

Next, a generator G w ( ·| y ) is learned based on the prediction rule of user models. For clear visualizations, we learn G w ( ·| y ) on the raw feature space Y → X ⊂ R 2 instead of a latent space. As shown in Figure 3, r ( x | y ) , which denotes the distribution derived from G w ( x | y ) , gradually coincides with the ground-truth p ( x | y ) (Figure 2d), even when the individual local models are biased. In other words, G w ( x | y ) can fuse the aggregated information from user models to approximate a global data distribution.

We then let users sample from G w ( x | y ) , which serves as an inductive bias for users with limited data. As a result, each user can observe beyond its own training data and adjust their decision boundaries to approach to the ensemble wisdom (Figure 2b).

## 4.2. Knowledge Distillation for Distribution Matching

A notable difference between FEDGEN and prior work is that the knowledge is distilled to user models instead of the global model. As a result, the distilled knowledge, which conveys inductive bias to users, can directly regulate their learning by performing distribution matching over the latent space Z :

Remark 1. Let p ( y ) be the prior distribution of labels, and r ( z | y ) : Y → Z be the conditional distribution derived from generator G w . Then regulating a user model θ k using samples from r ( z | y ) can minimize the conditional KL-divergence between two distributions, derived from the generator and the user, respectively:

$$
\begin{aligned}
& \max _ { \theta _ { k } } \mathbb { E } _ { y \sim p ( y ) , z \sim r ( z | y ) } \left [ \log p ( y | z ; \theta _ { k } ) \right ] \\ & \equiv \min _ { \theta _ { k } } \ D _ { K L } [ r ( z | y ) | p ( z | y ; \theta _ { k } ) ] ,
\end{aligned}
$$

where we define p ( z | y ; θ k ) as the probability that the input feature to the predictor θ k is z given that it yields a label y . In practice, Equation 6 is optimized by using empirical samples from the generator: { ( z, y ) | y ∼ ˆ p ( y ) , z ∼ G w ( z | y ) } , which is consistent with the second term of the local model objective (Equation 5), in that ∀ y ∈ Y :

$$
\max _ { \theta _ { k } } \, \mathbb { E } _ { z \sim r ( z | y ) } \left [ \log p ( y | z ; \theta _ { k } ) \right ] \approx \min _ { \theta _ { k } } \, \mathbb { E } _ { z \sim G _ { w } ( z | y ) } \left [ l ( h ( z ; \theta _ { k } ^ { p } ) ; y ) \right ] .
$$

Distinguished from prior work that applies weight regularization to local models (Li et al., 2020b; Dinh et al., 2020), FEDGEN can serve as an alternative and compatible solution to address user heterogeneity, which inherently bridges the gap among user models w.r.t their interpretations of an ideal feature distribution.

## 4.3. Knowledge Distillation for Improved Generalization

One can also draw a theoretical connection from the knowledge learned by FEDGEN to an improved generalization bound. To see this, we first present a performance bound for the aggregated model in FL, which is built upon prior arts from domain adaptation (Ben-David et al., 2007; 2010):

Theorem 1. ( Generalization Bounds for FL ) Consider an FL system with K users. Let T k = 〈 D k , c ∗ 〉 and T = 〈 D , c ∗ 〉 be the k -th local domain and the global domain, respectively. Let

## Data-Free Knowledge Distillation for Heterogeneous Federated Learning

(a) User decision boundaries before KD.

<!-- image -->

Userdecisionboundary[after]distillation GlobalData&amp;OracleDecisionBoundary

(b) User decision boundaries after KD.

<!-- image -->

(c) Global model decision boundaries.

<!-- image -->

(d) Oracle decision boundaries learned over all data.

<!-- image -->

Figure 2. After KD, accuracy has improved from 81 . 2% to 98 . 4% for one user ( Fig 2a - Fig 2b), while a global model obtained by parameter-averaging ( without KD) has 93 . 2% accuracy (Fig 2c), compared with an oracle model with 98 . 6% accuracy (Fig 2d).

<!-- image -->

(a) Randomly initialized r ( x | y ) .

<!-- image -->

(b) r ( x | y ) learned after 50 training steps.

(c) r ( x | y ) learned after 150 training steps.

<!-- image -->

(d) r ( x | y ) learned after 250 training steps.

<!-- image -->

Figure 3. Samples from the generator gradually approaches to ground-truth distribution, where each user model (teacher) sees limited, disjoint local data. Background color indicates oracle decision boundaries learned over the global data.

R : X → Z be a feature extraction function that is simultaneously shared among users. Denote h k the hypothesis learned on domain T k , and h = 1 K ∑ K k =1 h k the global ensemble of user hypotheses. Then with probability at least 1 -δ :

$$
\begin{aligned}
T h \text { with probability at least } 1 - \delta \colon & & \text {of } \mathcal { D } _ { \mathcal { I } } \\ \mathcal { L } _ { \tau } ( h ) \equiv \mathcal { L } _ { \tau } \left ( \frac { 1 } { K } \sum _ { k } h _ { k } \right ) & & \text {d} \mathcal { H } \triangle q \\ \leq \frac { 1 } { K } \sum _ { k } \hat { \mathcal { L } } _ { \tau _ { k } } ( h _ { k } ) + \frac { 1 } { K } \sum _ { k } \left ( d \mathcal { H } \triangle \mathcal { H } ( \tilde { D } _ { k } , \tilde { D } ) + \lambda _ { k } \right ) & & \mathcal { L } _ { \tau } \\ & + \sqrt { \frac { 4 } { m } \left ( d \log \frac { 2 e m } { d } + \log \frac { 4 K } { \delta } \right ) } , \\ \text {where } \hat { \mathcal { L } } _ { \tau } ( h _ { k } ) \text { is the empirical risk on } \mathcal { T } _ { k } , \lambda _ { k } \coloneqq \min _ { k } ( \mathcal { L } _ { \tau } ( h ) + \mathcal { L } _ { \tau } ( h _ { k } ) )
\end{aligned}
$$

where ˆ L T k ( h k ) is the empirical risk on T k , λ k := min h ( L T k ( h )+ L T ( h )) denotes an oracle performance. d H △ H ( ˜ D k , ˜ D ) denotes the divergence measured over a symmetric-difference hypothesis space. ˜ D k and ˜ D is the induced image of D k and D over R , respectively, s.t. E z ∼ ˜ D k [ B ( z )] = E x ∼ D k [ B ( R ( x ))] given a probability event B , and so for ˜ D .

Specifically, L T ( h ) is usually considered as an ideal upperbound for the global model in FL (Peng et al., 2019; Lin et al., 2020). Two key implications can be derived from Theorem 1: i) Large user heterogeneity leads to high distribution divergence ( d H △ H ( ˜ D k , ˜ D ) ), which undermines the quality of the global model; ii) More empirical samples ( m ) are favorable to the generalization performance, which softens the numerical constraints.

In other words, the generalization performance can be improved by enriching local users with augmented data that aligns with the global distribution:

Corollary 1. Let T , T k , R defined as in Theorem 1. D A denotes an augmented distribution, and D ′ k = 1 2 ( D k + D A ) is a mixture of distributions. Accordingly, ˜ D A , ˜ D ′ k denotes the induced image of D A , D ′ k over R , respectively. Let ˆ D ′ k = ˆ D k ∪ ˆ D A be an empirical dataset of D ′ k , with | ˆ D k | = m , | ˆ D ′ k | = m ′ &gt; m . If d H △ H ( ˜ D A , ˜ D ) is bounded, s.t ∃ ϵ &gt; 0 , d H △ H ( ˜ D A , ˜ D ) ≤ ϵ , then with probability 1 -δ :

$$
\begin{aligned}
\Lambda _ { k } ) & & \mathcal { L } _ { \tau } ( h ) \leq \frac { 1 } { K } \sum _ { k } \mathcal { L } _ { \tau _ { k } ^ { \prime } } ( h _ { k } ) + \frac { 1 } { K } \sum _ { k } ( d _ { \mathcal { H } \triangle \mathcal { H } } ( \tilde { \mathcal { D } } _ { k } ^ { \prime } , \tilde { \mathcal { D } } ) + \lambda _ { k } ^ { \prime } ) \\ & & + \sqrt { \frac { 4 } { m ^ { \prime } } \left ( d \log \frac { 2 e m ^ { \prime } } { d } + \log \frac { 4 K } { \delta } \right ) } , \\ ( h ) + & &
\end{aligned}
$$

where T ′ k = {D ′ k , c ∗ } is the updated local domain, d H △ H ( ˜ D ′ k , ˜ D ) ≤ d H △ H ( ˜ D k , ˜ D ) when ϵ is small, and √ 4 m ′ ( d log 2 em ′ d +log 4 K δ ) &lt; √ 4 m ( d log 2 em d +log 4 K δ ) .

Such an augmented distribution D A can facilitate FL from multiple aspects: not only does it relax the numerical constraints with more empirical samples ( m ′ &gt; m ), but it also reduces the discrepancy between the local and global feature distributions ( d H △ H ( ˜ D ′ k , ˜ D ) ). This finding coincides the merits of FEDGEN: since the generator G w ( z | y ) is learned to recover an aggregated distribution over the feature space, one can treat samples from the generator { z | y ∼ ˆ p ( y ) , z ∼ G w ( z | y ) } as the augmented data from ˜ D A, which naturally has a small deviation from the global induced distribution ˜ D . More rigorous analysis along this line is left to our future work. We elaborate the role of such an augmentation distribution D A in the supplementary.

## 5. Related Work

Federated Learning (FL) is first proposed by (McMahan et al., 2017) as a decentralized machine learning paradigm. Subsequent work along this line tackles different challenges faced by FL, including heterogeneity (Karimireddy et al., 2020; Li et al., 2020b; Mansour et al., 2020), privacy (Duchi et al., 2014; Agarwal et al., 2018), communication efficiency (Guha et al., 2019; Koneˇ cn` y et al., 2016), and convergence analysis (Kairouz et al., 2019; Qu et al., 2020; Yuan &amp; Li, 2019). Specifically, a wealth of work has been proposed to handle user heterogeneity , by regularizing model weight updates (Li et al., 2020b), allowing personalized user models (Fallah et al., 2020; Dinh et al., 2020), or introducing new model aggregation schemes (Yurochkin et al., 2019; Mansour et al., 2020). We refer readers to (Li et al., 2020a) for an organized discussion of recent progress on FL.

Knowledge Distillation (KD) is a technique to compress knowledge from one or more teacher models into an empty student (Hinton et al., 2015; Buciluˇ a et al., 2006; Ba &amp; Caruana, 2014; Jacobs et al., 1991). Conventional KD hinges on a proxy dataset (Hinton et al., 2015). More recent work enables KD with fewer data involved, such as dataset distillation (Wang et al., 2018), or core-data selection (Tsang et al., 2005; Sener &amp; Savarese, 2018). Later there emerges data-free KD approaches which aim to reconstruct samples used for training the teacher (Yoo et al., 2019; Micaelli &amp; Storkey, 2019). Particularly, (Lopes et al., 2017) extracts the meta-data from the teacher's activation layers. (Yoo et al., 2019) learns a conditional generator which yield samples that maximizes the teacher's prediction probability of a target label. Along the same spirit, (Micaelli &amp; Storkey, 2019) learns a generator by adversarial training. Different from prior work, we learn a generative model that is tailored for FL, by ensembling the knowledge of multiple user models over the latent space, which is more lightweight for learning and communication.

Knowledge Distillation in Federated Learning has recently emerged as an effective approach to tackle user heterogeneity. Most existing work is data-dependent (Lin et al., 2020; Sun &amp; Lyu, 2020; Guha et al., 2019; Chen &amp; Chao, 2021). Particularly, (Lin et al., 2020) proposed FEDDFUSION , which performs KD to refine the global model, assuming that an unlabeled dataset is available with samples from the same or similar domains. Complementary KD efforts have been made to confront data heterogeneity (Li &amp; Wang, 2019; Sattler, 2021). Specifically, (Li &amp; Wang, 2019) transmits the proxy dataset instead of the model parameters. FEDAUX (Sattler, 2021) performs data-dependent distillation by leveraging an auxiliary dataset to initialize the server model and to weighted-ensemble user models, while FEDGEN performs knowledge distillation in a data-free manner. FEDMIX (Yoon, 2021) is a data-augmented FL framework, where users share their batch-averaged data among others to assist local training. On the country, FEDGEN extracts knowledge from the existing user model parameters, which faces less privacy risks. FEDDISTILL (Federated Distillation) is proposed by (Seo et al., 2020) which extracts from user models the statistics of the logit-vector outputs, and shares this meta-data to users for KD. We provide detailed comparisons with work along this line in Section 6.

## 6. Experiments

In this section, we compare the performance of our proposed approach with other key related work. We leave implementation details and extended experimental results to the supplementary.

## 6.1. Setup

Baselines: In addition to FEDAVG (McMahan et al., 2017), FEDPROX regularizes the local model training with a proximal term in the model objective (Li et al., 2020b). FEDENSEMBLE extends FEDAVG to ensemble the prediction output of all user models. FEDDFUSION is a data-based KD approach (Lin et al., 2020), for which we provide unlabeled training samples as the proxy dataset. FEDDISTILL (Jeong et al., 2018) is a data-free KD approach which shares labelwise average of logit-vectors among users. It does not share network parameters and therefore experience non-negligible performance drops compared with other baselines. For a fair comparison, we derive a baseline from FEDDISTILL, which shares both model parameters and the label-wise logit vectors. We name this stronger baseline as FEDDISTILL + .

Dataset: We conduct experiments on three image datasets: MNIST (LeCun &amp; Cortes, 2010), EMNIST (Cohen et al., 2017), and CELEBA (Liu et al., 2015), as suggested by the LEAF FL benchmark (Caldas et al., 2018). Among them, MNIST and EMNIST dataset is for digit and character image classifications, and CELEBA is a celebrity-face dataset which is used to learn a binary-classification task, i.e. to predict whether the celebrity in the picture is smiling.

Configurations: Unless otherwise mentioned, we run 200 global communication rounds, with 20 user models in total and an active-user ratio r = 50% . Weadopt a local updating step T = 20 , and each step uses a mini batch with size B = 32 . We use at most 50% of the total training dataset and distribute it to user models, and use all testing dataset for performance evaluation. For the classifier, we follow the network architecture of (McMahan et al., 2017), and treat the last MLP layer as the predictor θ p k and all previous layers as the feature extractor θ f k . The generator G w is MLP based. It takes a noise vector ϵ and an one-hot label vector y as the input, which, after a hidden layer with dimension d h , outputs a feature representation with dimension d . To further increase the diversity of the generator output, we also leverage the idea of diversity loss from prior work (Mao et al., 2019) to train the generator model.

<!-- image -->

Figure 4. Visualization of statistical heterogeneity among users on MNIST dataset, where the x -axis indicates user IDs, the y -axis indicates class labels, and the size of scattered points indicates the number of training samples for a label available to that user.

Table 1. Performance overview given different data settings. For MNIST and EMNIST, a smaller α indicates higher heterogeneity. For CELEBA, r denotes the ratio between active users and total users. T denotes the local training steps (communication delay).

|               | Top-1 Test Accuracy.   | Top-1 Test Accuracy.   | Top-1 Test Accuracy.   | Top-1 Test Accuracy.   | Top-1 Test Accuracy.   | Top-1 Test Accuracy.   | Top-1 Test Accuracy.   | Top-1 Test Accuracy.   |
|---------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| Dataset       | Setting                | FEDAVG                 | FEDPROX                | FEDENSEMBLE            | FEDDISTILL             | FEDDISTILL +           | FEDDFUSION             | FEDGEN                 |
| MNIST, T =20  | α = 0.05               | 87.70 ± 2.07           | 87.49 ± 2.05           | 88.85 ± 0.68           | 70.56 ± 1.24           | 86.70 ± 2.27           | 90.02 ± 0.96           | 91.30 ± 0.74           |
| MNIST, T =20  | α = 0.1                | 90.16 ± 0.59           | 90.10 ± 0.39           | 90.78 ± 0.39           | 64.11 ± 1.36           | 90.28 ± 0.89           | 91.11 ± 0.43           | 93.03 ± 0.32           |
| MNIST, T =20  | α = 1                  | 93.84 ± 0.25           | 93.83 ± 0.29           | 93.91 ± 0.28           | 79.88 ± 0.66           | 94.73 ± 0.15           | 93.37 ± 0.40           | 95.52 ± 0.07           |
| CELEBA, T =20 | r = 5 / 10             | 87.48 ± 0.39           | 87.67 ± 0.39           | 88.48 ± 0.23           | 76.68 ± 1.23           | 86.37 ± 0.41           | 87.01 ± 1.00           | 89.70 ± 0.32           |
| CELEBA, T =20 | r = 5 / 25             | 89.13 ± 0.25           | 88.84 ± 0.19           | 90.22 ± 0.31           | 74.99 ± 1.57           | 88.05 ± 0.43           | 88.93 ± 0.79           | 89.62 ± 0.34           |
| CELEBA, T =20 | r = 10 / 25            | 89.12 ± 0.20           | 89.01 ± 0.33           | 90.08 ± 0.24           | 75.88 ± 1.17           | 88.14 ± 0.37           | 89.25 ± 0.56           | 90.29 ± 0.47           |
| EMNIST, T =20 | α = 0.05               | 62.25 ± 2.82           | 61 . 93 ± 2.31         | 64.99 ± 0.35           | 60.49 ± 1.27           | 61.56 ± 2.15           | 70.40 ± 0.79           | 68.53 ± 1.17           |
|               | α = 0.1                | 66.21 ± 2.43           | 65.29 ± 2.94           | 67.53 ± 1.19           | 50.32 ± 1.39           | 66.06 ± 3.18           | 70.94 ± 0.76           | 72.15 ± 0.21           |
|               | α = 10                 | 74.83 ± 0.69           | 74.24 ± 0.81           | 74.90 ± 0.80           | 54.77 ± 0.33           | 75.55 ± 0.94           | 74.36 ± 0.40           | 78.43 ± 0.74           |
| EMNIST, α =1  | T = 20                 | 74.83 ± 0.99           | 74.12 ± 0.88           | 75.12 ± 1.07           | 46.19 ± 0.70           | 75.41 ± 1.05           | 75.43 ± 0.37           | 78.48 ± 1.04           |
| EMNIST, α =1  | T = 40                 | 77.02 ± 1.09           | 75.93 ± 0.95           | 77.68 ± 0.98           | 46.72 ± 0.73           | 78.12 ± 0.90           | 77.58 ± 0.37           | 78.92 ± 0.73           |

Figure 6. Selected learning curves, averaged over 3 random seeds.

<!-- image -->

User heterogeneity : for MNIST and EMNIST dataset, we follow prior arts (Lin et al., 2020; Hsu et al., 2019) to model non-iid data distributions using a Dirichlet distribution Dir ( α ) , in which a smaller α indicates higher data heterogeneity, as it makes the distribution of p k ( y ) more biased for a user k . We visualize the effects of adopting different α on the statistical heterogeneity for the MNIST dataset in Figure 4. For CELEBA, the raw data is naturally non-iid distributed. We further increase the data heterogeneity by aggregating pictures belonging to different celebrities into disjoint groups, with each group assigned to one user.

## 6.2. Performance Overview:

From Table 1, we can observe that FEDGEN outperforms other baselines with a considerable margin.

Impacts of data heterogeneity: FEDGEN is the only algorithm that is robust against different levels of user heterogeneity while consistently performs well. As shown in Figure 5, the gain of FEDGEN is more notable when the data distributions are highly heterogeneous (with a small α ). This result verifies our motivations, since the advantage of FEDGEN is induced from the knowledge distilled to local users, which mitigates the discrepancy of latent distributions across users. This knowledge is otherwise not accessible by baselines such as FEDAVG or FEDPROX.

As one of most competitive baselines, the advantage of FEDDFUSION vanishes as data heterogeneity becomes mitigated, which gradually becomes comparable to FEDAVG, as shown in Figure 5a and Figure 5c. Unlike FEDDFUSION, the performance gain of our approach is consistently significant, which outperforms FEDDFUSION in most cases. This discrepancy implies that our proposed approach, which directly distills the knowledge to user models, can be more effective than fine-tuning the global model using a proxy dataset, especially when the distilled knowledge contains inductive bias to guide local model learning.

As a data-free KD baseline, FEDDISTILL experiences nonnegligible performance drops, which implies the importance of parameter sharing in FL. FEDDISTILL + , on the other hand, is vulnerable to data heterogeneities. As shown in Table 1, it can outperform FEDAVG when data distributions are near-iid ( e.g. when α ≥ 1 ), thanks to the shared logit statistics as the distilled knowledge, but performs worse than FEDAVG when α gets smaller, which indicates that sharing such meta-data alone may not be effective enough to confront user heterogeneity.

FEDENSEMBLE enjoys the benefit of ensemble predictions from all user models, although its gain is less significant compared with FEDGEN. We ascribe the leading performance of our approach to the better generalized performance of local models. Guided by the distilled knowledge, a user model in FEDGEN can quickly jump out of its local opti- mum, whose aggregation can be better than the ensemble of potentially biased models as in FEDENSEMBLE.

Learning efficiency: As shown in Figure 6, FEDGEN has the most rapid learning curves to reach a performance and outperforms other baselines. Although FEDDFUSION enjoys a learning efficiency higher than other baselines under certain data settings, due the advantages induced from a proxy data, our approach can directly benefit each local user with actively learned knowledge, whose effect is more explicit and consistent (More illustrations in supplementary).

Comments on sharing generative model : Given a compact latent space, the generative model can be lightweight for learning or downloading. In practice, we use a generator network with 2 compact MLP layers, whose parameter size is small compared with the user classification model. The above empirical results also indicates that the leading performance gain combined with a faster convergence rate can trade off the communication load brought by sharing a generative model.

## 6.3. Sensitivity Analysis

Impacts of straggler users: We explore different numbers of total users versus active users on the CELEBA dataset, with the active ratios r ranging from 0 . 2 to 0 . 9 . Figure 5b shows that our approach is next to FEDENSEMBLE when the number of straggler users are high ( r = 0 . 2 , with 5 out 25 active users per learning round), and is consistently better than all baselines w.r.t to the asymptotic performance given a moderate number of active users. Combined with Figure 6a and Figure 6b, one can observe that our approach requires much less communication rounds to reach high performance, regardless of the setting of straggler users.

Effects of different network architectures: we conduct analysis on the MNIST dataset, using both CNN and MLP network architectures. As shown in Figure 5d and Figure 5c, the outstanding performance of FEDGEN is consistent across two different network settings, although the overall performance trained with CNN networks is noticeably higher than those with MLP networks.

Effects of communication frequency: We explore different local updating steps T on the EMNIST, so that a higher T means longer communication delays before the global communication. Results in Table 1 indicates that our approach is robust against different levels of communication delays (See supplementary for more results).

Effects of the generator's network architecture and sampling size: Extended analysis has verified that FEDGEN is robust across different generator network architectures (Table 2). Moreover, sampling synthetic data from the generator only adds minor training workload to local users (Table 2). The gain of FEDGEN over FEDAVG is consistently remarkable given different synthetic sample sizes, whereas a sufficient number of synthetic samples brings even better performance (Figure 7). Especially, in Table 2, we explored different dimensions for the input noise ( d ϵ ) and the hidden layer ( d h ) of the generator while keeping its output layer dimension fixed ( i.e. the dimension of the feature space Z ). Table 3 shows the training time for one local update, averaged across users and the communication rounds. B G denotes the number of synthetic samples used for each minibatch optimization. By default, we set B G = B , and B is the number of real samples drawn from the local dataset (see Algorithm 1).

<!-- image -->

Figure 7. Effects of synthetic samples.

Figure 8. Learning curves on MNIST with limited param. sharing.

<!-- image -->

Table 4. Performance overview on MNIST, by only sharing the last prediction layer.

| Top 1 Accuracy ( % )   | Top 1 Accuracy ( % )   | Top 1 Accuracy ( % )   | Top 1 Accuracy ( % )   | Top 1 Accuracy ( % )   |
|------------------------|------------------------|------------------------|------------------------|------------------------|
| Algorithms             | FEDAVG                 | FEDDISTILL +           | FEDDFUSION             | FEDGEN                 |
| α = 0.05               | 59.67 ± 0.76           | 58.83 ± 0.62           | 59.62 ± 0.84           | 63.60 ± 0.63           |
| α = 0.1                | 58.39 ± 0.74           | 56.25 ± 0.98           | 58.38 ± 0.81           | 65.42 ± 0.29           |
| α = 1                  | 74.49 ± 0.57           | 74.24 ± 0.60           | 74.51 ± 0.55           | 79.72 ± 0.52           |
| α = 10                 | 86.35 ± 0.60           | 86.89 ± 0.26           | 86.28 ± 0.69           | 87.92 ± 0.46           |

Table 3. Effects of the number of synthetic samples, using EMNIST dataset with α = 0 . 1 .

|                                                                                              | Effects of the Generator Network Structure.                                                  | Effects of the Generator Network Structure.                                                  | Effects of the Generator Network Structure.                                                  | Effects of the Generator Network Structure.                                                  | Effects of the Generator Network Structure.                                                  | Effects of the Generator Network Structure.                                                  |
|----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| [ d ϵ , d h ]                                                                                | [64 , 256]                                                                                   | [32 , 256]                                                                                   | [32 , 128]                                                                                   | [16 , 128]                                                                                   | [16 , 128]                                                                                   | [32 , 64]                                                                                    |
| Accuracy(%)                                                                                  | FEDAVG=66.22 ± 2.58                                                                          | FEDAVG=66.22 ± 2.58                                                                          | FEDAVG=66.22 ± 2.58                                                                          | FEDAVG=66.22 ± 2.58                                                                          | FEDAVG=66.22 ± 2.58                                                                          | FEDAVG=66.22 ± 2.58                                                                          |
| FEDGEN                                                                                       | 71.61 ± 0.25 72.09 ±                                                                         | 71.61 ± 0.25 72.09 ±                                                                         | 0.46 72.43                                                                                   | ± 0.57                                                                                       | 72.01 ± 0.76 70.98 ± 0.85                                                                    | 72.01 ± 0.76 70.98 ± 0.85                                                                    |
| Table 2. Effects of the generator's network structure, using EMNIST dataset with α = 0 . 1 . | Table 2. Effects of the generator's network structure, using EMNIST dataset with α = 0 . 1 . | Table 2. Effects of the generator's network structure, using EMNIST dataset with α = 0 . 1 . | Table 2. Effects of the generator's network structure, using EMNIST dataset with α = 0 . 1 . | Table 2. Effects of the generator's network structure, using EMNIST dataset with α = 0 . 1 . | Table 2. Effects of the generator's network structure, using EMNIST dataset with α = 0 . 1 . | Table 2. Effects of the generator's network structure, using EMNIST dataset with α = 0 . 1 . |
|                                                                                              | Performance w.r.t different synthetic sample sizes.                                          | Performance w.r.t different synthetic sample sizes.                                          | Performance w.r.t different synthetic sample sizes.                                          | Performance w.r.t different synthetic sample sizes.                                          | Performance w.r.t different synthetic sample sizes.                                          |                                                                                              |
| Generator sampling size                                                                      | B G = 8                                                                                      | B G = 16                                                                                     | B G =                                                                                        | 32                                                                                           | B G = 64                                                                                     | B G = 128                                                                                    |
| Local training time (ms)                                                                     | FEDAVG = 47 . 66 ± 1 . 68                                                                    | FEDAVG = 47 . 66 ± 1 . 68                                                                    | FEDAVG = 47 . 66 ± 1 . 68                                                                    | FEDAVG = 47 . 66 ± 1 . 68                                                                    | FEDAVG = 47 . 66 ± 1 . 68                                                                    | FEDAVG = 47 . 66 ± 1 . 68                                                                    |
| FEDGEN                                                                                       | 57.20 ± 2.22                                                                                 | 57.39 ± 2.21                                                                                 | 58.17                                                                                        | ± 2.24                                                                                       | 58.91 ± 2.29                                                                                 | 60.06 ± 2.32                                                                                 |

## 6.4. Extensions to Flexible Parameter Sharing

Motivated to alleviate privacy and communication concerns, FEDGEN is ready to benefit distributed learning without sharing entire model parameters. To explore this potential, we conduct a case study on FEDAVG, FEDDISTILL + , and FEDGEN, where user models share only the last prediction layer and keep their feature extraction layers localized. Note that FEDDFUSION is not designed to address FL with partial parameter sharing, which requires entire user models for KD. For a fair comparison, we modify FEDDFUSION to let it upload entire user models during the model aggregation phase, but disable the downloading of feature extractors, so that the server model can still be fine-tuned using the proxy data.

Results in Table 4 show that our approach consistently outperforms other baselines by a remarkable margin, the trend of which is more significant given high data heterogeneity (Figure 8). Its distinguished performance from FEDDFUSION verifies the efficacy of data-free distillation under this challenging scenario. This promising results show that FEDGEN has the potential to further reduce communication workload, not only by fast convergence but also by a flexible parameter sharing strategy.

## 7. Conclusions

In this paper, we propose an FL paradigm that enables efficient knowledge distillation to address user heterogeneity without requiring any external data. Extensive empirical experiments, guided by theoretical implications, have shown that our proposed approach can benefit federated learning with better generalization performance using less communication rounds, compared with the state-of-the-art.

## Acknowledgments

This research was jointly supported by the National Science Foundation IIS-1749940, the Office of Naval Research N00014-20-1-2382, and the National Institue on Aging RF1AG072449.

## References

- Agarwal, N., Suresh, A. T., Yu, F., Kumar, S., and Mcmahan, H. B. cpsgd: Communication-efficient and differentiallyprivate distributed sgd. Advances in Neural Information Processing Systems , 2018.
- Aggarwal, D., Zhou, J., and Jain, A. K. Fedface: Collaborative learning of face recognition model. Proceedings of International Joint Conference on Biometrics , 2021.
- Ammad-Ud-Din, M., Ivannikova, E., Khan, S. A., Oyomno, W., Fu, Q., Tan, K. E., and Flanagan, A. Federated collaborative filtering for privacy-preserving personalized recommendation system. arXiv preprint arXiv:1901.09888 , 2019.
- Ba, J. and Caruana, R. Do deep nets really need to be deep? Advances in neural information processing systems , 27: 2654-2662, 2014.
- Ben-David, S., Blitzer, J., Crammer, K., and Pereira, F. Analysis of representations for domain adaptation. In Advances in neural information processing systems , pp. 137-144, 2007.
- Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., and Vaughan, J. W. A theory of learning from different domains. Machine learning , 79(1-2):151-175, 2010.
- Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., and Wortman, J. Learning bounds for domain adaptation. In Advances in neural information processing systems , pp. 129-136, 2008.
- Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in Neural Information Processing Systems , 2020.
- Buciluˇ a, C., Caruana, R., and Niculescu-Mizil, A. Model compression. In Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining , pp. 535-541, 2006.
- Caldas, S., Duddu, S. M. K., Wu, P., Li, T., Koneˇ cn` y, J., McMahan, H. B., Smith, V., and Talwalkar, A. Leaf: A benchmark for federated settings. arXiv preprint arXiv:1812.01097 , 2018.
- Chen, H.-Y. and Chao, W.-L. Fedbe: Making bayesian model ensemble applicable to federated learning. ICLR , 2021.
- Cohen, G., Afshar, S., Tapson, J., and Van Schaik, A. Emnist: Extending mnist to handwritten letters. In 2017 International Joint Conference on Neural Networks (IJCNN) , pp. 2921-2926. IEEE, 2017.
- Dinh, C. T., Tran, N. H., and Nguyen, T. D. Personalized federated learning with moreau envelopes. 34th Conference on Neural Information Processing Systems (NeurIPS 2020) , 2020.
- Duchi, J. C., Jordan, M. I., and Wainwright, M. J. Privacy aware learning. Journal of the ACM (JACM) , 61(6):1-57, 2014.
- Fallah, A., Mokhtari, A., and Ozdaglar, A. Personalized federated learning with theoretical guarantees: A modelagnostic meta-learning approach. Advances in Neural Information Processing Systems , 33, 2020.
- Guha, N., Talwalkar, A., and Smith, V. One-shot federated learning. arXiv preprint arXiv:1902.11175 , 2019.
- Hard, A., Rao, K., Mathews, R., Ramaswamy, S., Beaufays, F., Augenstein, S., Eichner, H., Kiddon, C., and Ramage, D. Federated learning for mobile keyboard prediction. arXiv preprint arXiv:1811.03604 , 2018.
- He, C., Annavaram, M., and Avestimehr, S. Group knowledge transfer: Federated learning of large cnns at the edge. Advances in Neural Information Processing Systems , 33, 2020.
- He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , pp. 770-778, 2016.
- Hinton, G., Vinyals, O., and Dean, J. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531 , 2015.
- Hsu, T.-M. H., Qi, H., and Brown, M. Measuring the effects of non-identical data distribution for federated visual classification. arXiv preprint arXiv:1909.06335 , 2019.
- Jacobs, R. A., Jordan, M. I., Nowlan, S. J., and Hinton, G. E. Adaptive mixtures of local experts. Neural computation , 3(1):79-87, 1991.
- Jeong, E., Oh, S., Kim, H., Park, J., Bennis, M., and Kim, S.L. Communication-efficient on-device machine learning: Federated distillation and augmentation under non-iid private data. arXiv preprint arXiv:1811.11479 , 2018.
- Kairouz, P., McMahan, H. B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A. N., Bonawitz, K., Charles, Z., Cormode, G., Cummings, R., et al. Advances and open problems in federated learning. arXiv preprint arXiv:1912.04977 , 2019.
- Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S., Stich, S., and Suresh, A. T. Scaffold: Stochastic controlled averaging for federated learning. In International Conference on Machine Learning , pp. 5132-5143. PMLR, 2020.

- Khoussainov, R., Heß, A., and Kushmerick, N. Ensembles of biased classifiers. In Proceedings of the 22nd international conference on Machine learning , pp. 425-432, 2005.
- Kingma, D. P. and Welling, M. Auto-encoding variational bayes. ICLR , 2014.
- Koneˇ cn` y, J., McMahan, H. B., Yu, F. X., Richt´ arik, P., Suresh, A. T., and Bacon, D. Federated learning: Strategies for improving communication efficiency. arXiv preprint arXiv:1610.05492 , 2016.
- LeCun, Y. and Cortes, C. MNIST handwritten digit database. 2010. URL http://yann.lecun.com/ exdb/mnist/ .
- Li, D. and Wang, J. Fedmd: Heterogenous federated learning via model distillation. arXiv preprint arXiv:1910.03581 , 2019.
- Li, T., Sahu, A. K., Talwalkar, A., and Smith, V. Federated learning: Challenges, methods, and future directions. IEEE Signal Processing Magazine , 37(3):50-60, 2020a.
- Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., and Smith, V. Federated optimization in heterogeneous networks. Proceedings of Machine Learning and Systems , 2:429-450, 2020b.
- Li, X., Huang, K., Yang, W., Wang, S., and Zhang, Z. On the convergence of fedavg on non-iid data. ICLR , 2020c.
- Lin, T., Kong, L., Stich, S. U., and Jaggi, M. Ensemble distillation for robust model fusion in federated learning. arXiv preprint arXiv:2006.07242 , 2020.
- Liu, Z., Luo, P., Wang, X., and Tang, X. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV) , December 2015.
- Lopes, R. G., Fenu, S., and Starner, T. Data-free knowledge distillation for deep neural networks. arXiv preprint arXiv:1710.07535 , 2017.
- Mansour, Y., Mohri, M., Ro, J., and Suresh, A. T. Three approaches for personalization with applications to federated learning. arXiv preprint arXiv:2002.10619 , 2020.
- Mao, Q., Lee, H.-Y., Tseng, H.-Y., Ma, S., and Yang, M.H. Mode seeking generative adversarial networks for diverse image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 1429-1437, 2019.
- McMahan, B., Moore, E., Ramage, D., Hampson, S., and y Arcas, B. A. Communication-efficient learning of deep networks from decentralized data. In Artificial Intelligence and Statistics , pp. 1273-1282. PMLR, 2017.
- Micaelli, P. and Storkey, A. J. Zero-shot knowledge transfer via adversarial belief matching. In Advances in Neural Information Processing Systems , pp. 9551-9561, 2019.
- Peng, X., Huang, Z., Zhu, Y., and Saenko, K. Federated adversarial domain adaptation. arXiv preprint arXiv:1911.02054 , 2019.
- Qu, Z., Lin, K., Kalagnanam, J., Li, Z., Zhou, J., and Zhou, Z. Federated learning's blessing: Fedavg has linear speedup. arXiv preprint arXiv:2007.05690 , 2020.
- Sattler, F. e. a. Fedaux: Leveraging unlabeled auxiliary data in federated learning. arXiv preprint arXiv:2102.02514 , 2021.
- Sener, O. and Savarese, S. Active learning for convolutional neural networks: A core-set approach. ICLR , 2018.
- Seo, H., Park, J., Oh, S., Bennis, M., and Kim, S.L. Federated knowledge distillation. arXiv preprint arXiv:2011.02367 , 2020.
- Sheller, M. J., Edwards, B., Reina, G. A., Martin, J., Pati, S., Kotrotsou, A., Milchenko, M., Xu, W., Marcus, D., Colen, R. R., et al. Federated learning in medicine: facilitating multi-institutional collaborations without sharing patient data. Scientific reports , 10(1):1-12, 2020.
- Sun, L. and Lyu, L. Federated model distillation with noise-free differential privacy. arXiv preprint arXiv:2009.05537 , 2020.
- Tsang, I. W., Kwok, J. T., and Cheung, P.-M. Core vector machines: Fast svm training on very large data sets. Journal of Machine Learning Research , 6(Apr):363-392, 2005.
- Wang, H., Sreenivasan, K., Rajput, S., Vishwakarma, H., Agarwal, S., Sohn, J.-y., Lee, K., and Papailiopoulos, D. Attack of the tails: Yes, you really can backdoor federated learning. arXiv preprint arXiv:2007.05084 , 2020a.
- Wang, H., Yurochkin, M., Sun, Y., Papailiopoulos, D., and Khazaeni, Y. Federated learning with matched averaging. ICLR , 2020b.
- Wang, T., Zhu, J.-Y., Torralba, A., and Efros, A. A. Dataset distillation. arXiv preprint arXiv:1811.10959 , 2018.
- Yoo, J., Cho, M., Kim, T., and Kang, U. Knowledge extraction with no observable data. In Advances in Neural Information Processing Systems , pp. 2705-2714, 2019.
- Yoon, T. e. a. Fedmix: Approximation of mixup under mean augmented federated learning. ICLR , 2021.
- Yuan, X.-T. and Li, P. On convergence of distributed approximate newton methods: Globalization, sharper bounds and beyond. arXiv preprint arXiv:1908.02246 , 2019.

Yurochkin, M., Agarwal, M., Ghosh, S., Greenewald, K., Hoang, N., and Khazaeni, Y. Bayesian nonparametric federated learning of neural networks. In International Conference on Machine Learning , pp. 7252-7261. PMLR, 2019.