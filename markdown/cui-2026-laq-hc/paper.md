## Lightweight Adaptive Quantization Algorithms for Federated Learning With Heterogeneous Clients

Hengrui Cui , Zhihao Qu , Member, IEEE , Baoliu Ye , Member, IEEE , Bin Tang , Member, IEEE , Tao Zhuang , Member, IEEE , Xinyu Wang, and Yue Zeng , Member, IEEE

Abstract -Quantization is a common method to improve communication efficiency in federated learning (FL) by compressing the gradients that clients upload. Currently, most application scenarios involve cloud-edge collaboration, where edge clients exhibit significant heterogeneity, making previous methods with uniform quantizationlevels unsuitable. To address these issues, we introduce a novel algorithm named Lightweight Adaptive Quantization for Heterogeneous Clients (LAQ-HC), which enables each client to adaptively choose its quantization level based on its data quality and communication capabilities, without increasing computation costs. The core idea is that clients with lower communication capabilities should use higher quantization levels, whereas those with higher capabilities should use lower levels. This ensures that clients complete their uploads in a similar time. Furthermore, LAQ-HC models the relationship between quantization levels and the impact on client quality, which remains consistent between clients and adjacent training rounds. This allows for a lightweight estimation of the impact of quantization levels on training convergence, as demonstrated in our theoretical analysis. Under the constraints of limited wireless mobile communication bandwidth, LAQ-HC achieves faster convergence and higher accuracy compared to the latest adaptive quantization algorithms, while using only 56.74% of the computation time, 80.57% of the overall runtime, and 94.04% of the communication overhead.

IndexTerms -Federatedlearning,quantization, communication compression, wireless mobile communication.

## I. INTRODUCTION

A S THEimportance of privacy protection grows, traditional distributed machine learning methods that rely on uploading raw data to cloud servers are becoming inadequate for modern intelligent applications. In response, federated learning

Received 26 January 2025; revised 20 September 2025; accepted 16 October 2025. Date of publication 22 October 2025; date of current version 4 February 2026. This work was supported in part by the National Natural Science Foundation of China under Grant 62441225 and Grant 62402226, and in part by Jiangsu Science Foundation under Grant BK20253011. Recommended for acceptance by F. Wang. (Corresponding authors: Zhihao Qu; Baoliu Ye.)

Hengrui Cui, Baoliu Ye, Tao Zhuang, and Xinyu Wang are with the State Key Laboratory for Novel Software Technology, School of Computer Science, Nanjing University, Nanjing 210023, China (e-mail: hrcui@smail.nju.edu.cn; yebl@nju.edu.cn; tao\_zhuang@smail.nju.edu.cn; mf21330082@smail.nju. edu.cn).

Zhihao Qu and Bin Tang are with the Key Laboratory of Water Big Data Technology of Ministry of Water Resources, College of Computer Science and Software Engineering, Hohai University, Nanjing 211100, China (e-mail: quzhihao@hhu.edu.cn; cstb@hhu.edu.cn).

Yue Zeng is with the Department of Computer Science and Engineering, Nanjing University of Science and Technology, Nanjing 210094, China (e-mail: zengyue@njust.edu.cn).

Digital Object Identifier 10.1109/TMC.2025.3623984

Fig. 1. A basic framework and operational steps of FL.

<!-- image -->

(FL) [1] has emerged as a promising alternative, whose operational framework is shown in Fig. 1. FL operates by enabling individual clients to conduct local training and transmit only the updates of their local models to a central server for aggregation, thereby safeguarding the privacy of their raw data. This shift in approach has spurred the development of numerous FL algorithms, enhancing its applicability across various domains. The rapid advancement of FL make it essential in many aspects of daily life, including healthcare [2], education [3], finance [4], and transportation [5]. Its ability to maintain data privacy while leveraging distributed data has positioned FL as a pivotal technology in these fields.

Since FL clients are distributed at the edge and machine learning models are becoming increasingly complex, the dimensions of gradients (i.e., updates of local models) computed by clients are also increasing. This leads to significant communication overhead in FL. Recently, several works are dedicated to develop the communication-efficient FL framework. Common methods include quantization and sparsification. Quantization is hardware-friendly and features low computational and implementation complexity, making it easy to deploy. Even without considering compensation algorithms, quantization can still ensure relatively high model accuracy. Furthermore, the communication data volume involved is straightforward to calculate. In contrast, sparsification is not naturally suited to heterogeneous data and communication environments, and often requires additional compensation algorithms to guarantee accuracy. This leads to increased implementation and deployment complexity, as well as reduced flexibility. Therefore, we adopt quantization as the foundation upon which we build our method. Quantization [6], [7], [8], [9], [10] in FL has been extensively studied as

1536-1233 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

a key technique, where high-precision values are converted into lower-precision versions to reduce communication and storage costs. Typically, quantization levels such as 16/8/4/2/1-bit are used instead of 64/32-bit float numbers to represent gradients.

Mostcurrent quantization approaches apply a uniform quantization level throughout the training process, overlooking important factors such as training progress, differences in communication capabilities, and variations in data quality among clients. FL is increasingly being applied in wireless mobile scenarios [11], [12] nowadays, where issues such as limited resources (e.g., limited total bandwidth) and client heterogeneity exist. The original communication compression methods are not sufficiently suited to address the problems present in wireless mobile scenarios. This mismatch can negatively impact both model accuracy and convergence speed. Additionally, in wireless mobile environments, the clients are mostly heterogeneous which possess varying communication capabilities. As a result, the time required for data transmission at the same quantization level varies, leading to delays and extending the overall runtime. In this case, the time saved through quantification becomes less apparent. In recent work, Gong et al. [13] introduces LST-MMFL, a resource optimization framework designed to address client heterogeneity in satellite-ground networks using Lyapunov optimization to balance computational and communication overhead. These studies are valuable as they offer complementary perspectives on resource allocation and quantization in distributed learning environments. AdaGQ [14] introduces an adaptive quantization approach, allowing each client to adjust its quantization level in each round. While this method accounts for training progress and communication differences, it fails to consider the variability in data quality across clients. This oversight could lead to excessive compression of gradients from high-quality data, and the algorithm also incurs additional computational costs.

To address these issues, we propose LAQ-HC, Lightweight Adaptive Quantization Algorithms for Federated Learning with Heterogeneous Clients. Initially, we estimate a client's data quality by considering the volume of data used in training and the loss value per round without quantization. Our preliminary experiments reveal a distinct pattern in how different quantization levels affect the rate of loss reduction. By modeling this trend, we estimate the ratio of loss changes under various quantization levels compared to the full-precision scenario using a hyperbolic tangent function. This ratio serves as a metric to evaluate the impact of different quantization levels on training effectiveness. Further experiments indicate that this metric remains consistent across different clients, with the function curves of adjacent rounds being nearly identical. This allows us to compute the impact function with a single fitting, creating a lightweight model that describes the data quality of various clients under different quantization levels. Our approach comprehensively integrates data volume, data quality, and quantization impact into the overall client quality estimation. We consider not only the heterogeneity of client data but also the variability in communication capabilities. The core idea is for clients with limited communicationresourcestoutilize higher levels of quantization, whereas those with better communication capabilities utilize lower levels. This guarantees more consistent transmission times across clients, thereby reducing overall runtime. Finally, we propose algorithms for selecting appropriate clients and their corresponding quantization levels, both in scenarios with unlimited communication resources and under resource constraints, such as limited total bandwidth. The main contributions are summarized as follows:

- /a114 Based on observations, the impact of different quantization levels on the training process follows a hyperbolic tangent function trend, initially increasing rapidly with the number of quantization bits and then leveling off. Thus, we use hyperbolic tangent functions with different parameters to model this impact relationship. This allows us to estimate the client quality with adaptive quantization levels in a lightweight manner.
- /a114 We propose LAQ-HC, which utilizes the aforementioned metric to provide a lightweight method. This approach considers model training progress and client data quality heterogeneity, allowing each client to adaptively select its quantization level to enhance the efficiency of the FL system.
- /a114 Theoretical analysis demonstrates that the convergence rate of LAQ-HC is comparable to that of non-compression algorithms. The results also highlight how quantization error affects the convergence rate, with significant quantization error slowing down the training speed.
- /a114 We conduct extensive experiments and show that in contrast to the latest adaptive gradient quantization algorithms, LAQ-HC achieves faster convergence speed and higher convergence accuracy while consuming only 56 . 74% -72 . 32% of computation time, 80 . 57% -82 . 10% of overall runtime and 94 . 04% -98 . 11% of communication overhead with limited bandwidth.

## II. RELATED WORK

This section reviews existing research from two perspectives: quantization methods and heterogeneous clients in FL.

## A. Quantization Methods

Communicationisasignificantbottleneck in FL due to the frequent uploading of local models and downloading of the global model. To address this, various communication compression techniques have been proposed [15], [16], [17], [18], [19], with quantization being a popular method to reduce communication overhead. Though quantization is lossy by nature, existing research has demonstrated methods to mitigate its impact on FL accuracy. For instance, UVeQFed [9] (a vector quantization-based approach) enhances FL performance by reducing quantization distortion. Q-FL [20] is an insightful approach that employs varying quantization bit-widths tailored to different privacy protection scenarios, achieving adaptive quantization based on the degree of privacy requirement. It also provides an analytical upper bound on the quantization error. This work shares a common interest in adaptive quantization with ours, and it also focuses more extensively on the relationship between privacy protection and quantization bits. LAQ [21] employs a dynamic quantization method based on gradient variation saliency, where an adaptive threshold determines the significance of gradient updates. It selectively applies encoding and transmission only to gradient differences exceeding both the quantization error and historical update magnitudes, thereby reducing communication overhead while preserving convergence performance. Jhunjhunwala et al. [22] propose a method that adjusts the quantization level for each training round by leveraging variations in training loss value to reflect training progress. However, this approach assigns a uniform quantization level to all clients within the same round, failing to adaptively adjust based on client heterogeneity. Hashemi et al. [23] empirically show that 16-bit and 8-bit quantization cause minimal accuracy loss, which is compensated by energy and storage savings. Similarly, Seo et al. [24] find that although higher-bit quantization (such as 16-bit and 8-bit) minimally affects accuracy, it can slow convergence rates. This indicates that different quantization levels can be evaluated by examining changes in loss values. However, previous research often applies uniform quantization levels across all clients, lacking a detailed analysis of their impact on individual client performance. Moreover, these studies do not leverage varying quantization levels to account for client heterogeneity, which could lead to a more effective global quantization strategy.

## B. Heterogeneous Clients in FL

This work [11] points out that in FL under wireless mobile environments, how to manage and allocate limited resources and effectively address the heterogeneity among various clients are key issues. Some studies [25], [26], [27], [28], [29] concentrate on the diversity of client data. For instance, FAIR [27] is an incentive mechanism that estimates client contributions by considering data quantity and quality but overlooks communication capability heterogeneity, which can lead to the 'straggler' problem and increased runtime. Oort [28] addresses training and communicationtimedifferencesbyexcludingclientsthatexceed a time limit, reducing runtime but potentially affecting model performance. Yang et al. [30] measures the quantization sensitivity of data across different clients, where clients with higher sensitivity undergo stronger quantization, while those with lower sensitivity retain higher precision through milder compression. Other approaches, like asynchronous schemes [31], consider communication capability heterogeneity but may introduce errors due to delayed gradient uploads. Hou et al. [32] employ a reinforcement learning-based approach, which jointly and dynamically optimizes the user selection, gradient quantization, and resource allocation. However, this method introduces additional computation cost and suffers from high decision-making latency in resource-constrained edge devices. Alternative strategies [14], [17] leverage an anchor client as a reference node, dynamically determining quantization levels for other clients based on its parameters. For instance, AdaGQ [14] assigns different quantization levels based on clients' communication and computational abilities, optimizing computation and transmission time. However, it does not account for data quality variations and the impact of quantization on client performance. Moreover, when adaptively selecting the quantization levels at various clients, only the communication and computation time are considered, without paying attention to the training process. To address these limitations, we propose a lightweight adaptive quantization method. This approach focuses on the training process and estimates clients' communication capabilities and the effects of quantization levels on client quality, enabling clients to adopt strategies that maximize quality within constraints, ensuring model accuracy and precision without additional computation costs.

TABLE I NOTATIONS

## III. PRELIMINARY

In this section, we introduce the framework of FL with Gradient Quantization and discuss the estimation of client quality. Key notations used in this work and their descriptions are listed in Table I.

## A. Framework of FL With Gradient Quantization

We consider a FL system that comprising of a central server and a number of clients denoted by N = { 1 , 2 , . . . , n } . Each client i has a local dataset D i containing D i data samples. The overall number of data samples is denoted as D = ∑ n i =1 D i . The objective is to collaboratively train a global model w ∈ R d that minimizes the loss function F ( w ) , defined as follows:

$$
\begin{aligned}
\text {er} ^ { - } \quad F ( w ) = \sum _ { i = 1 } ^ { n } \frac { D _ { i } } { D } F _ { i } ( w ) , \ F _ { i } ( w ) = \frac { 1 } { D _ { i } } \sum _ { y \in \mathcal { D } _ { i } } f ( w , y ) , \ ( 1 ) \\ \text {and} \quad
\end{aligned}
$$

a r e

$$
\begin{aligned}
\text {er} \quad \ F ( w ) = \sum _ { i = 1 } ^ { n } \frac { D _ { i } } { D } F _ { i } ( w ) , \ F _ { i } ( w ) = \frac { 1 } { D _ { i } } \sum _ { y \in \mathcal { D } _ { i } } f ( w , y ) , \ ( 1 )
\end{aligned}
$$

where f ( w , y ) denotes the sample-wise loss function and F i ( w ) denotes the local objective of client i .

The training process consists of multiple rounds t = 1 , 2 , . . . . In each round, each client employs the stochastic gradient descent (SGD) algorithm to train the local model. The specific steps of each round are described as follows:

- /a114 Client Selection: At the beginning of round t , the server transmits the current global model weights w t to all clients. The client uploads the corresponding parameters (such as D i , F i ( w t ) , ℓ ( i ) , etc.), after which the server selects clients according to the requirements. The selected clients and their respective quantization levels are recorded in C t .

- /a114 Local Model Training: Each candidate client trains the local model based on the SGD algorithm with respect to the current global model w t . After the local training, the local model of client i is represented by w t i . Subsequently, each selected client participating in this round of FL calculates its local gradient according to:

$$
\begin{aligned}
w _ { i } ^ { t } \leftarrow w ^ { t } - \eta g _ { i } ( w ^ { t } ; \xi _ { i } ^ { t } ) , \quad & ( 2 ) \quad ^ { \text {grad} } _ { \text {value} }
\end{aligned}
$$

where η denotes learning rate and g i ( w t ; ξ t i ) denotes gradient with respect to global model w t of client i .

- /a114 Gradient Quantization: To reduce communication overhead, every client i ∈ C t uploads a quantized version of its local gradient g i ( w t ; ξ t i ) with level ℓ ( i ) for each ℓ ( i ) ∈ L ( i ) , denoted by Q i ( g i ( w t ; ξ t i ) , ℓ ( i )) , to the server. More specifically, we use the widely adopted quantization method proposed in [33] to obtain Q i ( g i ( w t ; ξ t i ) , ℓ ( i )) , which works as follows. Let g i ( w t ; ξ t i ) be [ g 1 , g 2 , . . . , g d ] . For each u = 1 , . . . , d , g u is quantized to an integer in the range [0 , 2 ℓ ( i ) -1] . Given x min = min { g 1 , g 2 , . . . , g d } and x max = max { g 1 , g 2 , . . . , g d } , the scale and zero point can be represented as R = ( x max -x min ) / (2 ℓ ( i ) -1) and z = round ( -x min / R ) , respectively. Then, the quantization process is formulated as follows:

$$
x _ { i n t } = r o u n d \left ( \frac { g _ { i } ( w ^ { t } ; \xi _ { i } ^ { t } ) } { \mathcal { R } } \right ) + z . \quad \text { (3)} \quad \text {Estim} \quad \text {espec}
$$

Additionally, we use the clamp function to confine the quantization results within the gradient boundaries:

$$
\begin{aligned}
Q _ { i , \ell ( i ) } ^ { t } & = Q _ { i } ( g _ { i } ( \mathbf w ^ { t } ; \xi _ { i } ^ { t } ) , \ell ( i ) ) \\ & = c l a m p ( 0 , 2 ^ { \ell ( i ) } - 1 , x _ { i n t } ) , \\ \text {where} & \quad \ \end{aligned}
$$

$$
\begin{aligned}
c l a m p ( a , b , x ) = \begin{cases} a & x < a \\ x & a \leq x \leq b \\ b & x > b \end{cases}
\end{aligned}
$$

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

d o w s

d i s c h t r

$$
\int
$$

where

-  /a114 Gradient De-quantization: After receiving the quantized gradients of all selected clients, the server performs the de-quantization operation to obtain the approximation of the original gradients ˆ g i ( w t ) :

$$
\begin{aligned}
\hat { g } _ { i } ( w ^ { t } ) & = ( Q _ { i , \ell ( i ) } ^ { t } - z ) \mathcal { R } & ( 5 ) & \quad \text {which} \\ \dot { g } _ { i } & \cdot \Gamma \colon \Gamma _ { i } \, \quad \Gamma _ { i } & \cdot \quad \text {Our}
\end{aligned}
$$

- /a114 Model Aggregation: Finally, the server use aggregation algorithm to update the global model.

$$
\begin{aligned}
w ^ { t + 1 } = w ^ { t } - \frac { \eta } { D _ { c } } \sum _ { i \in \mathcal { C } ^ { t } } D _ { i } \hat { g } _ { i } ( w ^ { t } ) , \quad \\ \text {pre}
\end{aligned}
$$

## B. Estimation of Client Quality

$$
\begin{aligned}
\text {where } D _ { c } = \sum _ { i \in \mathcal { C } ^ { t } } D _ { i } . \\ \text {.} \text { Estimation of Client Quality}
\end{aligned}
$$

In general, without accounting for quantization, the quality of a client per round is determined by the amount of data provided and its quality, as indicated by the change in the loss value each round. This can be expressed with the following mathematical formula:

$$
q _ { i , \ell ( i ) } ^ { t } = D _ { i } [ F _ { i } ( \mathbf w ^ { t } ) - F _ { i } ( \mathbf w _ { i } ^ { t } , m ) ] ,
$$

$$
n ) ] ,
$$

where m denotes the quantization levels under full precision (typically m = 2 ℓ ( i ) = 2 5 = 32 when ℓ ( i ) = 5 ). Nonetheless, acquiring F i ( w t i , m ) demands a full round of local training, rendering it inefficient for client selection. To estimate the quality of data involved in the training for each client, many previous works [34], [35], [36] consider the L 2 norm of the model update gradient. If the learning rate is fixed, the change in the loss value is typically positively correlated with the L 2 norm of the gradient. In [37], demonstrate that a larger loss value is typically associated with a larger L 2 norm of gradient, indicating a strong correlation between these variables. From these observations, we can infer that a higher loss value suggests a more significant change in the loss after model updates via the gradient. To eliminate the calculation of F i ( w t i , m ) , we can use the loss value respect to global model to replace [ F i ( w t ) -F i ( w t i , m )] and rewrite (7) as:

$$
q _ { i , \ell ( i ) } ^ { t } = D _ { i } F _ { i } ( \mathbf w ^ { t } ) ,
$$

However, when considering quantization, F i ( w t i , m ) will be expressed as F i ( w t i , 2 ℓ ( i ) ) and cannot be simply omitted, as it is also influenced by different quantization levels ℓ ( i ) .

## IV. DESIGN OF LAQ-HC

The primary difficulty in developing LAQ-HC is accurately estimating how different quantization levels affect client quality, especially in a way that minimizes extra overhead. This strategy allows clients to compress gradients using different quantization levels, while the server assigns these levels dynamically according to the clients' requirements. Typically, clients with higher bandwidth utilize lower degrees of quantization, whereas those with limited bandwidth tend to utilize higher degrees. This synchronization guarantees that all clients finish transmitting gradients simultaneously, thereby minimizing overall runtime. The estimation process remains efficient by avoiding intricate model training and large data transfers, like gradients, thus preserving the lightweight nature of the algorithm.

Our core idea is to select clients and their respective quantization levels such that client quality and bandwidth are optimized at a unit quantization level ℓ ( i ) . As ℓ ( i ) increases, client quality improves, but this also leads to an increase in gradient size, which raises communication costs and bandwidth requirements. Our goal is to is to identify an optimal balance in this trade-off, where the scenario at unit ℓ ( i ) serves as the ideal solution. The specific framework is illustrated as shown in Fig. 2. At the outset, the server distributes the global model w t . Subsequently, Clients proceed to upload their relevant parameters along with the corresponding quantization levels ℓ ( i ) . The server then calculates the client quality q t i,ℓ ( i ) across different quantization levels for each client. Finally, the server determines the sorting criteria flag i,ℓ ( i ) andexecutesthesorting process. This allows the server to prioritize clients based on their quality metrics, optimizing the overall model aggregation.

## A. Implementation of LAQ-HC

We consider two scenarios. The first scenario represents a general case without any constraints, where the server's objective is to maximize the overall client quality while ensuring that each client finishes data transmission in approximately the same duration. The second scenario introduces constraints, such as limited total bandwidth, aiming to achieve the previously stated objective while adhering to these restrictions.

Fig. 2. Overview of LAQ-HC.

<!-- image -->

1) LAQ-HC Without Constraint: Initially, we consider the LAQ-HC without any constraints scenario, representing the most general case. In this situation, the server does not need to consider any limitations and can include every candidate client in FL process, focusing solely on selecting the quantization strategy that maximizes client quality. The expression for flag i,ℓ ( i ) is:

$$
\begin{aligned}
\ f l a g _ { i , \ell ( i ) } = [ \alpha q _ { i , \ell ( i ) } ^ { t } + ( 1 - \alpha ) B _ { i } ] / \ell ( i ) ,
\end{aligned}
$$

where α is a parameter that adjusts the importance of client quality and bandwidth and B i is bandwidth of client i . The specific algorithm is detailed in Algorithm 1. The algorithm is crafted from both the client and server perspectives, encompassing the entire process of training and information transmission in one round. It ultimately outputs the set of quantization strategies chosen by each client C t and the updated global model w t + 1 . Initially, set up various parameters (line 1). The server first broadcasts the global model, awaiting clients to upload relevant parameter information (lines 13-14). Once the global model is received by the clients, they upload the relevant parameter information back to the server (lines 3-7). The server then calculates the flag for each quantization strategy for every client and returns the strategy with the highest flag to the clients (lines 15-21). After receiving feedback from the server, the clients train the model, quantize the gradients according to the selected strategy, and sends them to the server (lines 8-11). Finally, after the server receives all the gradients, it de-quantizes and aggregates them to update the global model (lines 22-25).

2) LAQ-HC With Constraint: In a wireless mobile environment, resources are often limited, especially for communications, where the total bandwidth is subject to constraints. Assuming our constraint is limited total bandwidth, the core idea is to aggregate the greater the weighted sum of client quality and bandwidth under unit ℓ ( i ) and imposed constraints, which can be reduced to a grouped knapsack problem (GKP). The size of the knapsack is equivalent to the quantization levels ℓ ( i ) and constraints s i , aiming to the weighted sum of client quality and bandwidth. Additionally, no more than one quantization strategy can be selected for each client. The expression for flag i,ℓ ( i ) is:

<!-- image -->

$$
\ f l a g _ { i , \ell ( i ) } = [ \alpha q _ { i , \ell ( i ) } ^ { t } + ( 1 - \alpha ) B _ { i } ] / ( \ell ( i ) \times s _ { i } ) ,
$$

where s i is the constraint parameter corresponding to client i (It refers to the bandwidth of client i here). The specific algorithm is detailed in Algorithm 2. The algorithm is also crafted from both the client and server perspectives, encompassing the entire process of training and information transmission in one round. It ultimately outputs the set of the selected clients with their corresponding quantization strategies C t and the updated global model w t + 1 . The difference from the scenario without constraints lies in that not all quantization strategies with the highest flag are selected. Instead, these strategies are arranged in descending order to select as many clients as possible along with their optimal quantization strategies within limited resources, which can be summarized as a grouped knapsack problem (lines 22-34). Furthermore, the grouped knapsack problem can be proven to be NP-hard, and this will be demonstrated subsequently.

Theorem 1. LAQ-HC with Constraint Algorithm is NP-hard.

## Algorithm 2: LAQ-HC With Constraint Algorithm.

Proof. The 0-1 knapsack problem is a known NP-hard problem [38]. We choose to reduce the 0-1 knapsack problem to the grouped knapsack problem in order to prove that the grouped knapsackproblemisNP-hard.Thedefinitionofthe0-1knapsack problem is: given a set of items, each with a weight and a value, the capacity of the knapsack is fixed. The problem is to choose which items to put into the knapsack so that the total value of the items in the knapsack is maximized, while not exceeding the capacity of the knapsack. Suppose we have an instance of the 0-1

knapsack problem, in which there are n items, each with its own value and weight, and a capacity of a knapsack. We can reduce this problem to a GKP instance, in which there are n groups, each with only one item. In this way, each solution to the 0-1 knapsack problem corresponds to a solution to the GKP. The 0-1 knapsack problem is a well-known NP-hard problem [38]. Grouped Knapsack Problem (GKP) can be reduced to the 0-1 knapsack problem, and it is also NP-hard. Each client can be regarded as a group, which consists of various quantization strategies as items. The value of each item corresponds to the client quality of the respective client under that strategy, while the weight represents the bandwidth required by that strategy. Each group (client) can select at most one item (quantization strategy), and the total bandwidth constraint can be viewed as the total capacity of the knapsack. The problem addressed by LAQ-HC can be reduced to an instance of GKP, thus indicating that the problem tackled by LAQ-HC is NP-hard. □

## B. Impact of Quantization

For LAQ-HC, the most crucial aspect is how to estimate the impact on client quality. In other words, how this significant factor of F i ( w t i , 2 ℓ ( i ) ) can be derived in the (11):

$$
\begin{aligned}
q _ { i , \ell ( i ) } ^ { t } = D _ { i } [ F _ { i } ( \mathbf w ^ { t } ) - F _ { i } ( \mathbf w _ { i } ^ { t } , 2 ^ { \ell ( i ) } ) ]
\end{aligned}
$$

Due to its dependency on the quantization level ℓ ( i ) , it cannot be easily disregarded. Thus, we convert the (11) into (12) as follows:

$$
\begin{aligned}
D _ { i } [ F _ { i } ( w ^ { t } ) - F _ { i } ( w _ { i } ^ { t } , m ) ] \frac { [ F _ { i } ( w ^ { t } ) - F _ { i } ( w _ { i } ^ { t } , 2 ^ { \ell ( i ) } ) ] } { [ F _ { i } ( w ^ { t } ) - F _ { i } ( w _ { i } ^ { t } , m ) ] } , \quad ( 1 2 )
\end{aligned}
$$

A o r d i n g t o w the p o r d i n g o w h o t e r

According to the preceding chapter, F i ( w t i , m ) can be disregarded, therefore the equation can be written as follows:

$$
\begin{aligned}
D _ { i } F _ { i } ( w ^ { t } ) \frac { [ F _ { i } ( w ^ { t } ) - F _ { i } ( w _ { i } ^ { t } , 2 ^ { \ell ( i ) } ) ] } { [ F _ { i } ( w ^ { t } ) - F _ { i } ( w _ { i } ^ { t } , m ) ] }
\end{aligned}
$$

o m t h e x i n t i n g w i s t h o s e r $ y $ t h o s e $ b y $ [ 1 4 ] $ a n d $ [ 2 4 ] $

From the existing works, such as those by [14] and [24], it is evident that different quantization levels, particularly when ℓ ( i ) is relatively large, have minimal impact on model accuracy, primarily affecting the model's convergence speed instead. The change in loss value across the training rounds is the key indicator of the training speed. Therefore, we can use the change in loss values under different quantization levels relative to the scenario when ℓ ( i ) = 5 that means 32-bit full precision to characterize this impact, and coincidentally, the right-hand side of the (13) represents this ratio, and we can formulate it into a functional relationship regarding the impact of quantization IQ ( t, i, ℓ ( i )) . However, accurately calculating this ratio under different rounds, different clients, and different quantization levels separately is unrealistic. Thus, we are dedicated to characterize the impact function via preliminary experiments. Our goal is to identify which parameters have minimal influence on the IQ , thereby facilitating its simplification. As the loss value under the the scenario without quantization should be obtained during the local training on the client side, we examine the ratio of changed loss value under different quantization levels relative to the scenario without quantization, which helps us to quantify the impact caused by quantization. We conduct experiments under different clients and different rounds, using the EMNIST [39] with LeNet5, CIFAR-10 [40] with ResNet18, CIFAR-100 with ResNet18 and TinyImageNet with GoogleNet(Inception\_v3). The following results is observed according to these experiments:

Fig. 3. Relationship between number of quantization bits and relative changed loss value.

<!-- image -->

- /a114 Observation 1: From Fig. 3(a), we notice that the variation of changed loss value under different quantization levels relative to the scenario without quantization is almost identical across different clients.
- /a114 Observation 2: From Fig. 3(b), we observe that although the variation of changed loss value under different quantization levels relative to the scenario without quantization at different rounds is not exactly the same, the curves of adjacent rounds almost coincide.
- /a114 Observation3: Comparingdifferent color lines in Fig. 3(b), different datasets and training model can affect the relationship.

Based on these observations, we conclude that the impact function is not sensitive to the clients. In addition, although there are slight discrepancies in this ratio curve under different rounds, the curves of adjacent rounds are nearly identical. Therefore, we use IQ ( t, ℓ ( i )) to denote this ratio for simplicity, which is defined as the impact under different quantization levels ℓ ( i ) ∈ J = { 0 , 1 , . . . , log 2 m } in round t . To fit the impact function IQ ( t, ℓ ( i )) , it is not necessary for all clients to complete the whole round of the local training. Based on the trends of this function via the preliminary experiments, we use hyperbolic tangent function with proper parameters to fit the relationship between the quantization levels on the x -axis and impact value on the y -axis:

$$
I Q ( t , \ell ( i ) ) = \left ( a \times \frac { e ^ { b \times 2 ^ { t ( i ) } } - 1 } { e ^ { c \times 2 ^ { t ( i ) } } + 1 } + d \right ) , \quad ( 1 4 )
$$

where a, b, c, d are different parameters for function fitting. Because the curves of Impact ( t, ℓ ( i )) are almost identical between two adjacent rounds, we can use the results of the previous round to approximate the results of current round. According to (13), three parameters F i ( w t -1 ) , F i ( w t -1 i , 2 ℓ ( i ) ) and F i ( w t -1 i , m ) of all selected clients in the previous round have been calculated. When there are some quantization strategies for the selected clients besides the case without quantization, i.e., the ration is 1 and the quantization level is the original bit width of the gradient, we can derive the values of a , b , c , and d in this round without additional computation.

Fig. 4. Function relationship fitted under different combinations of datasets and models.

<!-- image -->

Fig. 4(a) shows the function relationship fitted under six different quantization levels, using the EMNIST dataset, where a = 1 . 0660 , b = 0 . 2799 , c = 0 . 2797 , d = -0 . 0726 . Fig. 4(b) shows the function relationship fitted under six different quantization, using the CIFAR-10 dataset, where a = 1 . 0778 , b = 0 . 2761 , c = 0 . 2759 , d = -0 . 0844 . Fig. 4(c) shows the function relationship fitted under six different quantization, using the CIFAR-100 dataset, where a = 1 . 0491 , b = 0 . 2164 , c = 0 . 2153 , d = -0 . 0876 . Fig. 4(d) shows the function relationship fitted under six different quantization, using the Tiny-ImageNet dataset, where a = 1 . 0904 , b = 0 . 2578 , c = 0 . 2575 , d = -0 . 1012 . In these experiments, we select 10 clients to participant and consider the second round of the FL training. The results reveal that under different combinations of datasets and models, appropriate hyperbolic tangent functions can be fitted to describe the impact of quantization.

Therefore, with consideration of quantization, we can define the update quality of client i with quantization level ℓ ( i ) in round t as:

$$
\begin{aligned}
q _ { i , \ell ( i ) } ^ { t } & = D _ { i } F _ { i } ( w ^ { t } ) \frac { [ F _ { i } ( w ^ { t - 1 } ) - F _ { i } ( w _ { i } ^ { t - 1 } , 2 ^ { \ell ( i ) } ) ] } { [ F _ { i } ( w ^ { t - 1 } ) - F _ { i } ( w _ { i } ^ { t - 1 } , m ) ] } \\ & = D _ { i } F _ { i } ( w ^ { t } ) \times I Q ( t - 1 , \ell ( i ) )
\end{aligned}
$$

t h e c k u m p a t i o n a r v e d a f o r w h e d o w t i f t i g h e p h o r b l c t a n g t u n p f u c

Thecomputationoverheadoffitting the hyperbolic tangent function is very low, as the fitting process itself typically completes within seconds to tens of seconds. This approach allows us to efficiently approximate the loss reduction under different quantization levels. In contrast, obtaining ground-truth results would require clients to perform full model training separately at each quantization level. This process is computationally intensive and can take several hours to tens of hours, especially with complex models and datasets. An important consideration is that in the first round, there are no previous results to refer to for the fitting of the impact function. Therefore, a cold-start phase is necessary to establish the initial functional relationship. Similarly, the computation overhead during this stage remains manageable. Theoretically, the cold-start phase involves training on a single client across four different quantization levels to acquire the loss reductions needed for fitting the four-parameter function. Since typical FL training requires far more than four rounds to converge (often exceeding 100 rounds), the additional cost incurred in the cold-start phase is relatively negligible. This lightweight characteristic is made possible by our key findings. Based on extensive preliminary experiments, a properly parameterized hyperbolic tangent function can effectively characterize the effect of different quantization levels on the training loss reduction. Moreover, our architecture automatically handles intermittent client dropouts by setting update quality q t i,ℓ ( i ) to zero for inactive participants. This exclusion mechanism ensures current round decisions remain statistically valid without affecting: a) the client's future participation eligibility, b) its server selection probability in subsequent rounds, and c) model convergence properties.

## C. Convergence Analysis

The following three theorems establish the convergence rate of LAQ-HC. Our theoretical findings indicate that LAQ-HC achieves an asymptotic linear convergence rate of O (1 /T ) for strongly convex objectives (e.g., logistic regression models) and a sublinear convergence rate of O (1 / √ T ) for non-convex objectives (such as deep learning models like CNNs). First, we theoretically guarantee that the convergence rate of LAQ-HC is of the same order as non-compression algorithms, allowing each client to adaptively quantize the gradient. Second, we characterize the effect of quantization error on the convergence rate. Despite the order equality, a large quantization error (denoted by ϵ i in the following theorems) indeed slows down the training speed. This finding is consistent with our observations in the preliminary experiments shown in Fig. 3.

Basic Assumption: We make the following assumptions 1 :

1. ( unbiased gradient, bounded variance ) The stochastic gradient is unbiased for any parameter w , i.e., E ξ [ g ( w ; ξ )] = ∇ F ( w ) . The variance of stochastic gradient is bound by a constant σ 2 , i.e., E ξ [ ‖ g ( w ; ξ ) -∇ F ( w ) ‖ 2 ] ≤ σ 2 , where ξ denotes the random samples selected in SGD.
2. ( bounded gradient ) The magnitude of stochastic gradient is bounded by a constant G, i.e., ‖ g ( w ; ξ ) ‖ ≤ G .
3. ( unbiased and error-bounded quantization ) For any gradient g , we have E [ Q ( g )] = g (unbiasedness), and the the magnitude of the quantization error is bounded, i.e., E [ ‖ Q ( g i ( w ; ξ ) , ℓ ( i )) -g i ( w ; ξ ) ‖ ] ≤ ϵ i ‖ g ( w ; ξ ) ‖ . The error bound of client have been derived in QSGD [6], i.e., ϵ i = √ min { d/ℓ ( i ) 2 , √ d/ℓ ( i ) } , where d is the dimension of the gradient vector and ℓ ( i ) is the quantization level of client i .

Theorem2. ( stronglyconvexcase ) Under Basic Assumption, when F ( · ) is µ -strongly convex and LAQ-HC is running with fixed learning rate η for all iterations, and satisfying

1 Assumptions 1-3 are widely applied in the theoretical analysis for the convergence property of FL [41], [42], [43], [44].

$$
\eta \leq \frac { 1 } { 3 L }
$$

then, the expected optimality gap satisfies the following inequality for all t ∈ N

$$
\begin{aligned}
& \mathbb { E } F ( \mathbf w ^ { t + 1 } ) - F ( \mathbf w ^ { * } ) \\ & \leq ( 1 - \eta \mu ) ^ { t } [ F ( \mathbf w ^ { 1 } ) - F ( \mathbf w ^ { * } ) - A ] + A , \\ & \quad 3 \ln ( \mathcal { C } _ { 2 } ^ { 2 } + \mathcal { C } _ { 1 } ^ { 2 } N )
\end{aligned}
$$

where A = 3 Lη ( ϵG 2 + σ 2 N ) 2 N 2 µ and ϵ = N i =1 ϵ 2 i .

∑ Proof. Based on the update of global model and taking the expectation for both side with respect to ξ t , we have

$$
\begin{aligned}
\text { where } A = \frac { 2 \, N ^ { 2 } \mu } { 2 \, N ^ { 2 } \mu } \text { and } \epsilon = \sum _ { i = 1 } ^ { L } \epsilon _ { i } ^ { 2 } . \\ \text { } \
\end{aligned}
$$

$$
\begin{array} { r l } { o t a i n d e b a s e d o n } & { \geq n \sum _ { i = 1 } ^ { n } a _ { i } \| ^ { 2 } \leq n \sum _ { i = 1 } ^ { n } \| a _ { i } \| ^ { 2 } . T h e n , l e t \epsilon b e } \\ { the summations of the compression error \epsilon _ { i } ^ { 2 } f o r a l l c i s t a i s i , w e } \\ { d e r v e the b o u n d o f \mathbb { E } _ { \xi ^ { \prime } } t \| \sum _ { i = 1 } ^ { N } Q ( g ( w _ { i } ^ { t } ; \xi _ { i } ^ { t } ) , \ell ( i ) ) \| ^ { 2 } as f o l w s: } \\ { a . e . } & { \quad } \\ { \Gamma _ { \mathbb { E } _ { \xi ^ { \prime } } } } & { \left [ \| \sum _ { i = 1 } ^ { N } Q ( g ( w _ { i } ^ { t } ; \xi _ { i } ^ { t } ) , \ell ( i ) ) \| ^ { 2 } \right ] } \\ { i } & { = \mathbb { E } _ { \xi ^ { \prime } } \left [ \| \sum _ { i = 1 } ^ { N } Q ( g ( w _ { i } ^ { t } ; \xi _ { i } ^ { t } ) , \ell ( i ) ) - \sum _ { i = 1 } ^ { N } g ( w _ { i } ^ { t } ; \xi _ { i } ^ { t } ) } \\ { a n y } & { \quad } \\ { a n d } & { \quad + \sum _ { i = 1 } ^ { N } g ( w _ { i } ^ { t } ; \xi _ { i } ^ { t } ) \| ^ { 2 } \right ] } \\ { \xi } & { \right ] \| . } \\ { [ 6 ] } & { = \mathbb { E } _ { \xi ^ { \prime } } \left [ \| \sum _ { i = 1 } ^ { N } Q ( g ( w _ { i } ^ { t } ; \xi _ { i } ^ { t } ) , \ell ( i ) ) - \sum _ { i = 1 } ^ { N } g ( w _ { i } ^ { t } ; \xi _ { i } ^ { t } ) } \\ { t } & { the } \\ { \Gamma _ { \mathbb { E } _ { \xi ^ { \prime } } } } & { \left ( + \sum _ { i = 1 } ^ { N } g ( w _ { i } ^ { t } ; \xi _ { i } ^ { t } ) - \sum _ { i = 1 } ^ { N } \nabla F ( w _ { i } ^ { t } ) + \sum _ { i = 1 } ^ { N } \nabla F ( w _ { i } ^ { t } ) \right \| ^ { 2 } \right ] } \\ { \quad } \\ { \Gamma _ { \mathbb { E } _ { \xi ^ { \prime } } } } & { \left [ \| \sum _ { i = 1 } ^ { N } Q ( g ( w _ { i } ^ { t } ; \xi _ { i } ^ { t } ) , \ell ( i ) ) - \sum _ { i = 1 } ^ { N } g ( w _ { i } ^ { t } ; \xi _ { i } ^ { t } ) \| ^ { 2 } \right ] } \\ { \quad } \\ { a d e n o j u n e 1 2 , 2 0 2 a t 1 6 \colon 2 0 5 U T C f o r m e X p l o r . R e s t i o n a p l y . } \end{array}
$$

∥ ∥ where the first equality follows according to the unbiasedness of quantization and stochastic gradient, and the last inequality is obtained based on ‖ ∑ n i =1 a i ‖ 2 ≤ n ∑ n i =1 ‖ a i ‖ 2 . Then, let ϵ be the summations of the compression error ϵ 2 i for all clients i , we derive the bound of E ξ t ‖ ∑ N i =1 Q ( g ( w t ; ξ t i ) , ℓ ( i )) ‖ 2 as follows:

$$
\begin{aligned}
CUI et al . \, L i g W I E G T A P T I V E \, Q U N T I A G I O R T H M S \, F E D E R A D E \, L E A R S \, \\ & \quad + 3 \mathbb { E } _ { \xi } \left \| \sum _ { i = 1 } ^ { N } \left [ g ( w _ { i } ^ { t } ; \xi _ { i } ^ { t } ) - \nabla F ( w _ { i } ^ { t } ) \right ] \right \| ^ { 2 } \quad a n d \quad \\ & \quad + 3 \mathbb { E } _ { \xi } \left \| \sum _ { i = 1 } ^ { N } \nabla F ( w _ { i } ^ { t } ) \right \| ^ { 2 } \\ & \leq 3 \sum _ { i = 1 } ^ { N } \mathbb { E } _ { \xi ^ { t } } \left [ \| Q ( g ( w _ { i } ^ { t } ; \xi _ { i } ^ { t } ) , \ell ( i ) ) - g ( w _ { i } ^ { t } ; \xi _ { i } ^ { t } ) \| ^ { 2 } \right ] \quad j e c t i v e \, 1 \leq N \colon \\ & \quad + 3 \sum _ { i = 1 } ^ { N } \mathbb { E } _ { \xi } \left [ \| g ( w _ { i } ^ { t } ; \xi _ { i } ^ { t } ) - \nabla F ( w _ { i } ^ { t } ) \| ^ { 2 } \right ] \\ & \quad + 3 \mathbb { E } _ { \xi } \left [ \sum _ { i = 1 } ^ { N } \nabla F ( w _ { i } ^ { t } ) \right ] ^ { 2 } \\ & \leq 3 \epsilon \| g ( w _ { i } ^ { t } ; \xi _ { i } ^ { t } ) \| ^ { 2 } + 3 N \sigma ^ { 2 } + 3 \mathbb { E } _ { \xi } \left \| \sum _ { i = 1 } ^ { N } \nabla F ( w _ { i } ^ { t } ) \right \| ^ { 2 } \quad \text {Pro} \quad \mathbb { A } _ { \sigma } \quad \mathbb { A } _ { \sigma } ^ { 2 } \\ & \leq 3 G ^ { 2 } \epsilon + 3 N \sigma ^ { 2 } + 3 \mathbb { E } _ { \xi } \left \| \sum _ { i = 1 } ^ { N } \nabla F ( w _ { i } ^ { t } ) \right \| ^ { 2 } \quad \text {By sum} \quad \mathbb { S } _ { \widetilde { s } } , \quad \text {sides,} \, w _ { i } , \\ & \quad \text {Replacing } \mathbb { E } _ { \xi ^ { t } } [ F ( w ^ { t + 1 } ) ] - F ( w ^ { t } ) \\ & \quad \cdot \quad 2 \left \| \nabla F ( w ^ { t } ) \right \| ^ { 2 } - \frac { \eta } { } \left \| \sum _ { i = 1 } ^ { N } \nabla F ( w _ { i } ^ { t } ) \right \| ^ { 2 } \quad \cdot \quad -
\end{aligned}
$$

∥ ∥ Replacing E ξ t [ ‖ N i =1 Q ( g ( w t i ; ξ t i )) ‖ 2 ] in (18), we have

$$
\begin{aligned}
& \leq 3 G ^ { 2 } \epsilon + 3 N \sigma ^ { 2 } + 3 \mathbb { E } _ { \xi } \left \| \sum _ { i = 1 } ^ { N } \nabla F ( w _ { i } ^ { t } ) \right \| \\ & \quad \ \end{aligned}
$$

When 3 Lη 2 -η 2 N 2 ≤ 0 , i.e., η ≤ 1 3 L , we can eliminate the term ‖ ∑ N i =1 ∇ F ( w t ) ‖ 2 and obtain:

$$
\begin{aligned}
\mathbb { E } _ { \xi ^ { t } } [ F ( w ^ { t + 1 } ) ] \\ \leq F ( w ^ { t } ) - \frac { \eta } { 2 } \| \nabla F ( w ^ { t } ) \| ^ { 2 } + \frac { 3 L \eta ^ { 2 } ( \epsilon G ^ { 2 } + \sigma ^ { 2 } N ) } { 2 N ^ { 2 } } \quad \text {which}
\end{aligned}
$$

According to the µ -strongly convex property that ‖∇ F ( w t ) ‖ 2 ≥ 2 µ [ F ( w t ) -F ( w ∗ )] , and introducing notation A = 3 Lη ( ϵG 2 + σ 2 N ) 2 N 2 µ , we can derive:

$$
\begin{aligned}
\mathbb { E } [ F ( w ^ { t + 1 } ) - F ( w ^ { * } ) ] & \leq ( 1 - \eta \mu ) \mathbb { E } [ F ( w ^ { t } ) - F ( w ^ { * } ) ] + \mu \eta A .
\end{aligned}
$$

Thus, subtracting A from both sides and rearranging yield that:

$$
\begin{aligned}
\mathbb { E } F ( w ^ { t + 1 } ) - F ( w ^ { * } ) \\ \leq ( 1 - \eta \mu ) \mathbb { E } [ F ( w ^ { t } ) - F ( w ^ { * } ) - A ] + A \\ \leq ( 1 - \eta \mu ) ^ { t } [ F ( w ^ { 1 } ) - F ( w ^ { * } ) - A ] + A , \quad ( 2 2 ) \\ \leq \text {which completes the proof.}
\end{aligned}
$$

which completes the proof.

□

Theorem 3. ( non-convex case, bound of the expected average-squared gradient ) Under Assumption 1, suppose that LAQ-HCisrunning with fixed learning rate η for all rounds, and satisfying

$$
\begin{aligned}
\eta - 3 L ^ { 2 } \eta ^ { 2 } & \geq 0 , \\
\end{aligned}
$$

Then, the expected average-squared gradients of general objective function F ( · ) satisfies the following inequalities for all T ∈ N :

$$
\begin{aligned}
T \in \mathbb { N } & \colon \\ & \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| \nabla F ( w ^ { t } ) \| ^ { 2 } \\ & \leq \frac { 2 | F ( w ^ { 1 } ) - F ^ { ( w ^ { t } ) } | } { \eta T } + \frac { 3 L \eta ( \epsilon G ^ { 2 } + \sigma ^ { 2 } N ) } { N ^ { 2 } } \\
\end{aligned}
$$

r o f o r $ B s e d o n 20 $ the main difference in non convex case is

Proof. Based on 20, the main difference in non-convex case is that we have to take the expectation for both side with respect to ξ and the whole stochastic batch space in all iteration 1 , 2 , . . . , t . By summing this inequality for iteration 1 , 2 , . . . , T for both sides, we have

$$
\begin{aligned}
s d s , & \text { we have} \\ & \mathbb { E } _ { \xi ^ { t } } [ F ( w ^ { T + 1 } ) - F ( w ^ { 1 } ) ] \\ & \leq - \, \frac { 2 } { \eta } \sum _ { t = 1 } ^ { T } \| \nabla F ( w ^ { t } ) \| ^ { 2 } - \frac { \eta - 3 L \eta ^ { 2 } } { 2 N ^ { 2 } } \sum _ { t = 1 } ^ { T } \left \| \sum _ { i = 1 } ^ { N } \nabla F ( w _ { i } ^ { t } ) \right \| ^ { 2 } \\ & + \frac { 3 T L \eta ^ { 2 } ( \epsilon G ^ { 2 } + \sigma ^ { 2 } N ) } { 2 N ^ { 2 } } \\ & \text {Let } \eta - 3 L ^ { 2 } \eta ^ { 2 } \geq 0 , \text { we have}
\end{aligned}
$$

Let η -3 L 2 η 2 ≥ 0 , we have

$$
\begin{aligned}
\mathbb { E } _ { \xi } [ F ( w ^ { T + 1 } ) - F ( w ^ { 1 } ) ] \\ \leq - \ \frac { 2 } { \eta } \sum _ { t = 1 } ^ { T } \| \nabla F ( w ^ { t } ) \| ^ { 2 } + \frac { 3 T L \eta ^ { 2 } ( \epsilon G ^ { 2 } + \sigma ^ { 2 } N ) } { 2 N ^ { 2 } } \quad ( 2 6 ) \\ \text {So that we can get}
\end{aligned}
$$

So that we can get

$$
\begin{aligned}
& \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| \nabla F ( w ^ { t } ) \| ^ { 2 } \\ & \leq \frac { 2 | F ( w ^ { 1 } ) - F ( w ^ { * } ) | } { \eta T } + \frac { 3 L \eta ( \epsilon G ^ { 2 } + \sigma ^ { 2 } N ) } { N ^ { 2 } } \quad ( 2 7 ) \\ \text {which completes the proof}
\end{aligned}
$$

which completes the proof.

□

Theorem 4. ( non-convex case, convergence rate ) Under the Basic Assumptions, when we set learning rate η as follows:

$$
\eta = \sqrt { \frac { 2 | F \left ( w ^ { 1 } \right ) - F \left ( w ^ { * } \right ) | N ^ { 2 } } { 3 L ( \epsilon G ^ { 2 } + \sigma ^ { 2 } N ) T } }
$$

We have the following expected average-squared gradient:

$$
\begin{aligned}
& \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| \nabla F ( w ^ { t } ) \| ^ { 2 } \\ & \quad \leq 2 \sqrt { \frac { 6 [ F ( w ^ { 1 } ) - F ( w ^ { * } ) ] L ( \epsilon G ^ { 2 } + \sigma ^ { 2 } N ) } { N ^ { 2 } } } * \frac { 1 } { \sqrt { T } } \quad ( 2 9 )
\end{aligned}
$$

which implies that the convergence rate of LAQ-HC is O (1 / √ T ) .

Proof. By setting the learning rate as

$$
\eta = \sqrt { \frac { 2 | F \left ( w ^ { 1 } \right ) - F \left ( w ^ { * } \right ) | N ^ { 2 } } { 3 L ( \epsilon G ^ { 2 } + \sigma ^ { 2 } N ) T } } , \quad ( 3 0 ) \quad ^ { 3 2 \times } \quad t a i n s \quad \quad t h i n g r a l { C }
$$

we can achieve

$$
\begin{aligned}
\text {we can achieve} & & \text {different} & & \text {difference} & & \text {validatiti} \\ & \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| \nabla F ( w ^ { t } ) \| ^ { 2 } & & \text {validatiti} & & 1 & & \text {dataset} \\ & \leq 2 \sqrt { \frac { 6 [ F ( w ^ { 1 } ) - F ( w ^ { * } ) ] L ( \epsilon G ^ { 2 } + \sigma ^ { 2 } N ) } { N ^ { 2 } } } * \frac { 1 } { \sqrt { T } } & & \text {dataset} & & ( 3 1 ) & & \text {Benc} \\ & \text {Combining with the constraint of stepsize } \eta - 3 L ^ { 2 } \eta ^ { 2 } > 0 , \text { we } & & \text {considere} & & \text {to} \, & & \text {tuples} \, & & \text {to} \, & & \text {tuples}
\end{aligned}
$$

Combiningwiththe constraint of stepsize η -3 L 2 η 2 ≥ 0 , we can derive the condition of T

$$
\begin{aligned}
T & \geq \frac { 6 | F \left ( w ^ { 1 } \right ) - F \left ( w ^ { * } \right ) | N ^ { 2 } L ^ { 3 } } { \left ( \epsilon G ^ { 2 } + \sigma ^ { 2 } N \right ) } , & \quad \ \end{aligned}
$$

where η = 1 3 L 2 is the positive root of η -3 L 2 η 2 = 0 . □

Discussion. Obviously, η = 1 3 L 2 is definitely positive, thus we can get

$$
\begin{aligned}
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| \nabla F ( w ^ { t } ) \| ^ { 2 } \preceq \sqrt { \frac { \epsilon } { N ^ { 2 } T } } \\ \text { } \bullet \\ \text { } \quad .
\end{aligned}
$$

This result suggests LAQ-HC essentially admits the same convergence rate as non-compression SGD since it has the asymptotical convergence rate O (1 / √ T ) . More specifically, LAQ-HC runs with the convergence rate O (1 / √ N 2 T ) , which is in the same order of distributed SGD and uniform-bit compression approaches. This means it has the linear speedup property and high efficiency in large-scale distributed learning. For any adaptive quantization level of gradient pushing, Theorem 4 guarantees that LAQ-HC converges with a sub-linear speed.

## V. PERFORMANCE EVALUATION

In this section, we conduct experiments to compare LAQ-HC with four other FL algorithms (QSGD, AdaQuantFL, AdaGQ and R-algorithm) across four dimensions: accuracy, communication overhead, computation time and overall runtime. We conduct experiments under both unconstrained conditions and constrained conditions (limited total bandwidth) to demonstrate the advantages of LAQ-HC.

## A. Experiments Setup

Experiments are conducted based on the FL framework Plato [45]. We configure 20 clients, each possessing Non-IID data in varying amounts, with different bandwidth and CPU frequency. We then make them run LAQ-HC along with four other distinct algorithms for comparison and all experiments adhere to non-IID data distributions across clients.

Model and datasets. We consider four combinations of datasets and models, corresponding to a small dataset with a minor model and a large dataset with a complex model: training Lenet-5 on EMNIST [39], training ResNet-18 on CIFAR10 [40], training ResNet-152 on CIFAR-100 [40] and training GoogleNet (Inception v3) on Tiny-ImageNet. EMNIST dataset is an extension of the MNIST dataset, which contains 47 different character classes, including digits, uppercase letters, and some common symbols. CIFAR-10 contains 60 thousand 32 × 32 color images, divided into 10 classes. CIFAR-100 contains 60 thousand 32 × 32 color images across 100 categories, with 600 images per category. Tiny-ImageNet [46] contains 200 different classes and each class has 500 training images, 50 validation images, and 50 test images, totaling approximately 120,000 images. In the subsequent parts of the paper, the name of the dataset is used to stand for the combination of the entire dataset and the model.

Benchmarks. To evaluate the performance of LAQ-HC, we consider the following four benchmarks: QSGD, which maintains a consistent fixed quantization level across all training rounds; AdaQuantFL, which dynamically adjusts the quantization level per training round while ensuring all clients share the same quantization level within each round; and AdaGQ, which enables fully individualized quantization where each client can independently adapt its quantization level every round.

- /a114 QSGD [6]: To ensure a fair comparison, the clients performed local training for 5 epochs, then quantized the gradients and uploaded them to the server, with the number of quantization bits uniformly set to 8-bit.
- /a114 AdaQuantFL[22]: Wefollow the method described in the paper to adaptively adjust the quantization level for each subsequent training round based on the loss value from the first round of model training as the baseline. It's important to note that within the same training round, the quantization level remains consistent for all clients and the clients also performed local training for 5 epochs.
- /a114 AdaGQ[14]: Weemploythe Heterogeneous Quantization approach outlined in the paper to determine the different quantization levels used by each client in the respective round. Similarly, we set the number of local training epochs per round to 5.
- /a114 R-Algorithm [32]: Building upon the core concepts of the original work, we implement the reinforcement learning framework. To ensure a fair comparison, we refine the optimization function to focus solely on quantization error and transmission latency, excluding factors that were not addressed by baseline algorithms. Given that the original study does not assign an abbreviated name to this approach, we designate it as R-Algorithm. We also set the number of local training epochs per round to 5.

Hyperparameters. As the default configuration, we set the local batch size to 64 and each client is assigned a different amount of data, randomly distributed between 2,000 and 60,000 samples. For each client, the bandwidth B is uniformly distributed between 4 Mbps and 20 Mbps, with the variance adjusted according to the specific experimental requirements. TheCPUfrequency f n is uniformly distributed between 1.5 GHz and 5 GHz, while the CPU cycles per bit of data are uniformly distributed between 10 cycles/bit and 30 cycles/bit. For model training, the learning rate is fixed at 0.01 and the decay is set at 0.998. All data is non-IID, and the default value of the Dirichlet coefficient is 1 ( Dir = 1 ) unless otherwise specified. For ease of comparison, the hyperparameters of the other four algorithms are likewise configured in the same manner.

<!-- image -->

Accuracy under different Q with(b）Runtime under different Q wit FAR-100 CIFAR-100

<!-- image -->

<!-- image -->

Fig. 5. Performance under different α with CIFAR-100 and TinyImageNet.

<!-- image -->

## B. Selected Value of α

As a parameter balancing client quality and bandwidth, the value of α influences the accuracy of global model and overall runtime. We first normalize client quality and bandwidth to bring them to a common scale, aiming to find an optimal α that minimizes runtime while ensuring model accuracy for different bandwidth variance distributions. We first set α to 0.5, indicating equal importance for client quality and bandwidth, and record the results for different bandwidth variance distributions at this setting. Then, we assign various values to α and record the ratios of accuracy and runtime compared to when α is 0.5, as illustrated in Fig. 5. This facilitates the observation of performance variations across different values of α . We quantify client heterogeneity through the variance of bandwidth distributions σ 2 across clients.

First, we conduct experiments on CIFAR-100 to observe the reasonable values of α for relatively simple datasets and models. As can be seen from Fig. 5(a) and (b), the effective range for α is between 0.44 and 0.56. Adjusting α in increments of 0.02 can affect the model's accuracy and runtime. When it is less than 0.44, the accuracy will drop rapidly and may even fail to converge. When it is greater than 0.56, the accuracy hardly increases, but runtime increases significantly. In Fig. 5(b) and (d), the red infinity symbol indicates that the runtime cannot be calculated due to non-convergence during training. As the variance in bandwidth distribution decreases, α canbeincreased. In particular, when the variance is zero, meaning all clients are homogeneous in terms of bandwidth distribution, α can be set to 1. This maximizes accuracy without increasing runtime. When the variance increases, α can be moderately reduced (not lower than 0.52) to balance maintaining accuracy with reducing runtime. Setting α too low could lead to a significant drop in accuracy and an unintended increase in runtime due to more rounds needed for model convergence. In the following experimental results, α will always be set to the most optimal value. When conducting experiments on the more complex dataset, Tiny-ImageNet, generally consistent conclusions can be drawn. The minor difference is that the effective range of α narrows, approximately between 0.46 and 0.54, and the values are more sensitive. A slight change in α can lead to significant changes in both the accuracy and the runtime.

Fig. 6. Optimal α with CIFAR-100 and Tiny-ImageNet.

<!-- image -->

<!-- image -->

We define the optimal α as the value that minimizes runtime while maintaining accuracy within 1% of the maximum observedperformance.Additionally, the case of σ 2 = 100 is employed to model scenarios with extreme heterogeneity. Through further experimentation, we identify these optimal α values across diverse heterogeneous conditions. Fig. 6 demonstrates that as client heterogeneity intensifies, the optimal α initially decreases. Upon reaching a specific threshold, it subsequently begins to rise to ensure model convergence. Comparing Fig. 6(a) and (b), the inflection point occurs earlier in Tiny-ImageNet due to its higher category count and greater task complexity, where heterogeneity more readily deteriorates model convergence. This adaptive adjustment ensures efficient training convergence while preventing detrimental impacts on total runtime performance. In summary, the guideline for selecting optimal value of α is as follows. When client heterogeneity is low, α should be set around 0.54. As heterogeneity increases, the optimal value of α decreases. However, after a certain turning point, it gradually increases and eventually returns to approximately 0.54. Furthermore, the complexity of the dataset influences the position of this turning point, i.e., more complex datasets lead to an earlier occurrence of the transition.

## C. LAQ-HC Without Constraints

When constraints are not considered, the goal of LAQ-HC is to ensure that clients with higher client quality per unit bit and larger bandwidth have a lower degree of quantization, while those with lower client quality and lower bandwidth have a higher degree of quantization. This approach ensures that high-quality updates are compressed less, while also allowing all clients to complete their computation and data transmission in roughly the same amount of time.

WecanobservefromFig.7(a)thatAdaQuantFLandLAQ-HC initially converges more slowly because the clients compress the gradients more, whereas the reinforcement learning-based RAlgorithm demonstrates significantly faster convergence speed.

Fig. 7. Accuracy of different combinations of models and datasets without constraints.

<!-- image -->

Astraining progresses, LAQ-HC progressively narrows the convergence gap with QSGD and AdaQuantFL, eventually exceeding their performance. This is because LAQ-HC compresses the gradients of clients with low flag more and the gradients of clients with high flag less, resulting in a faster convergence speed compared to other quantization algorithms. Near-optimal quantization strategy allocation across clients is enabled in R-Algorithm via reinforcement learning, at the expense of increased resource consumption that is experimentally presented in subsequent subsection.

## D. LAQ-HC With Constraints

We consider scenarios with additional constraints, such as this subsection where there is a total bandwidth limit for clients uploading information, set in the experiments to 80% of the combined upload bandwidth of all clients. LAQ-HC selects as many clients with high flag as possible without exceeding the bandwidth limit, while other algorithms proportionally adjust each client's bandwidth to 80% of the original.

For this scenario, we also conduct experiments on four different dataset and model combinations. As shown in Fig. 8, with bandwidth constraints, LAQ-HC maintains consistent convergence accuracy relative to other algorithms. Despite a slower initial training phase, it ultimately can converge in fewer rounds and even achieve higher convergence accuracy, matching the performanceoftheR-Algorithm.Thisphenomenonisevenmore pronounced with bandwidth constraints, as LAQ-HC initially discards clients with low flag , resulting in a slower start. However, as the training progresses with clients having higher flag , the model's training speed gradually catches up and surpasses the others.

Additionally, we evaluate the cost of different algorithms from three dimensions: computation time, overall runtime and communication overhead. To facilitate presentation, we apply a logarithmic transformation to the y -axis of Fig. 9(a) and (b). For Fig. 9(c), since the data spans a wider range and to highlight the differences at each scale, we apply a logarithmic transformation to the y -axis for the left half (EMNIST and CIFAR10) while keeping the y -axis linear for the right half (CIFAR100 and Tiny ImageNet). As shown in Fig. 9, LAQ-HC reduces computation, overall runtime and communication overhead across different dataset and model combinations. From Fig. 9(a), we can see that LAQ-HC requires the least computation time. This is because it converges in fewer rounds compared to QSGD and AdaQuantFL. In contrast to AdaGQ and R-Algorithm, it doesn't introduce additional computation. From Fig. 9(b), it is observed that LAQ-HC needs the least overall runtime. This is due to the fact that it spends less on both communication and computation time compared to other algorithms and also alleviates the 'straggler issue'. Fig. 9(c) shows that LAQ-HC consumes the least communication overhead because it maximally compresses gradients while maintaining convergence accuracy and speed compared to the other four algorithms.

Fig. 8. Accuracy of different combinations of models and datasets with constraints.

## E. Impact of Dropout on LAQ-HC

Since LAQ-HC supports client selection, low dropout rates have negligible impact on the overall framework. To investigate the effects of higher dropout rates, we conducted additional experiments. Using the CIFAR-100 dataset with ResNet-152 modelarchitecture, with a client bandwidth distribution variance of 20, we increased the number of backup clients to 40. Fig. 10(a) reveals that low dropout rates have negligible impact on accuracy. However, when the dropout rate increases to the point where the budget can fully cover all non-dropout clients (experimentally above 20%), accuracy begins to decline. This occurs because the dropout of high-quality clients forces the server to select lower-quality alternatives, with insufficient low-quality clients available to compensate for the loss of high-performing ones. Analysis of Fig. 10(b) indicates negligible overall runtime impact at low dropout levels, though runtime exhibits greater sensitivity to dropout escalation compared to accuracy. Beyond the 20% experimental threshold, where resource allocation suffices for all active clients, overall runtime accelerates notably. This phenomenon stems from the preferential dropout of highefficiency clients (those delivering superior update quality per bandwidth unit), which consequently become unavailable for server selection.

Fig. 9. Cost under varying dataset and model combinations.

<!-- image -->

Fig. 10. Impact of varying dropout rates on accuracy and overall runtime.

<!-- image -->

## F. Impact of Data Heterogeneity on LAQ-HC

In this controlled experiment, we employ CIFAR-100 dataset to train ResNet-152 architecture while holding all hyperparameters constant. By modulating the Dirichlet coefficient ( Dir ), we induce varying levels of data heterogeneity to comprehensively evaluate the efficacy of LAQ-HC across diverse non-IID scenarios. It can be observed from Fig. 11 that under varying degrees of data heterogeneity (where smaller Dir values indicate greater heterogeneity), LAQ-HC consistently maintains minimal overhead across all kinds of cost. This holds true even in extreme heterogeneity scenarios ( Dir = 0 . 1 ) and near-IID conditions ( Dir = 1000 ). Although changes in data heterogeneity impact the algorithm's convergence rate, which consequently alters convergence rounds and overall costs, the relative magnitudes of costs among different algorithms remain unchanged.

## G. Impact of Bandwidth Heterogeneity on LAQ-HC

Keeping other client parameters and the mean of the client bandwidth distribution constant, we alter the variance of the client bandwidth distribution to change the heterogeneity of participating FL clients. This is done to explore the impact of client heterogeneity on the performance of LAQ-HC. All experiments are conducted under a total bandwidth constraint, limited to 80% of the aggregate client bandwidth. Dataset CIFAR-100 is used to train model ResNet-152.

We organize the experimental results of various algorithms across different variance distributions into a Table II for easier reading and observation. Additionally, a separate figure for each cost dimension is created for clearer presentation. From Fig. 12(a), it can be observed that as the variance of the bandwidth distribution increases, the clients' heterogeneity in bandwidth distribution also grows. The computation time of QSGD increases faster than that of AdaQuantFL and LAQ-HC because client heterogeneity has a more pronounced impact on its convergence. In contrast, AdaGQ and R-Algorithm incurs additional computational overhead, leading to rapid growth in its computation time. This increase is particularly amplified under extreme heterogeneity. When heterogeneity intensifies from a moderate level to an extreme level, the rate at which computation time increases gradually slows down. Meanwhile, Fig. 12(b) reveals that as the variance increases, the overall runtime of all five algorithms also rises. AdaGQ, R-Algorithm and LAQ-HC display relatively smaller increments, with LAQ-HC showing the least. This is attributed to AdaGQ, R-Algorithm and LAQHC's ability to alleviate the 'straggler issue', with LAQ-HC achievingfaster convergence and thus requiring less time. Unlike the previous two algorithms, R-Algorithm is less effective in mitigating the 'straggler issue', yet its accelerate convergence rate results in significantly lower overall runtime. Fig. 12(c) effectively illustrates the advantages of adaptive quantization. Fixed quantization level algorithms like QSGD face issues with increased communication rounds and communication overhead due to the inability to dynamically adjust quantization strategies. While R-Algorithm employs reinforcement learning and introduces additional per-round communication overhead, its accelerated convergence rate and reduced total training rounds collectively result in significantly lower overall communication overhead. Adaptive quantization algorithms, such as AdaGQ and LAQ-HC, can dynamically adjust the quantization strategy for eachclient in each round. LAQ-HC can compress or even discard updates from clients with low flag values, allowing it to reduce communication costs while minimally impacting convergence accuracy and speed.

Fig. 11. Cost under varying data heterogeneity.

<!-- image -->

Fig. 12. Cost under varying bandwidth distribution variance.

TABLE II COMPARISON OF RESOURCE COST UNDER VARYING BANDWIDTH DISTRIBUTION VARIANCE

<!-- image -->

## VI. CONCLUSION

We develop an innovative algorithm called LAQ-HC, designed to empower clients with the ability to adaptively select quantization levels. This strategic selection is crucial for minimizing overall runtime and communication overhead, while maintaining convergence accuracy without incurring additional computation costs. Our approach employs a lightweight method to estimate the impact of various quantization levels on client quality, allowing clients to choose the most effective quantization levels to enhance their performance. LAQ-HC excels in optimizing model performance under various constraints, ensuring efficient client operation even when bandwidth is limited. Its scalability allows it to adapt seamlessly to different scenarios, making it a versatile tool for diverse applications. Beyondaddressing bandwidth limitations, LAQ-HC proves beneficial in other resource-constrained environments, such as those with a limited number of clients or restricted computational resources. The adaptability and scalability of LAQ-HC make it a robust solution across numerous settings, ensuring optimal performance and efficiency. Notably, LAQ-HC holds significant promise for mobile applications, where resource constraints are common. Addtionally, our future work will explore integrating adaptive quantization with sparsification or low-rank approximation techniques. This synergy aims to mitigate the high error susceptibility and compatibility limitations of the latter two methodswhilefurther reducing overhead. Furthermore, building upon the adaptive quantization strategies for privacy concerns in related studies, future work could explore integrating blockchain and differential privacy technology into our framework to provide decentralized and auditable privacy mechanisms, thereby enhancing security in FL systems.

## REFERENCES

- [1] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, 'Communication-efficient learning of deep networks from decentralized data,' in Proc. Int. Conf. Artif. Intell. Statist. , 2017, pp. 1273-1282.
- [2] D. C. Nguyen et al., 'Federated learning for smart healthcare: A survey,' ACM Comput. Surv. , vol. 55, no. 3, pp. 1-37, 2022.
- [3] S. Guo, D. Zeng, and S. Dong, 'Pedagogical data analysis via federated learning toward education 4.0,' Amer. J. Educ. Inf. Technol. , vol. 4, no. 2, pp. 56-65, 2020.
- [4] G. Long, Y. Tan, J. Jiang, and C. Zhang, 'Federated learning for open banking,' in Proc. Federated Learn.: Privacy Incentive , 2020, pp. 240254.
- [5] D. M. Manias and A. Shami, 'Making a case for federated learning in the internet of vehicles and intelligent transportation systems,' IEEE Netw. , vol. 35, no. 3, pp. 88-945, May/Jun. 2021.
- [6] D. Alistarh, D. Grubic, J. Li, R. Tomioka, and M. Vojnovic, 'QSGD: Communication-efficient SGD via gradient quantization and encoding,' in Proc. Adv. Neural Inf. Process. Syst. , 2017, pp. 1707-1718.
- [7] F. Wu et al., 'Sign bit is enough: A learning synchronization framework for multi-hop all-reduce with ultimate compression,' in Proc. 59th ACM/IEEE Des. Automat. Conf. , 2022, pp. 193-198.
- [8] M. M. Amiri, D. Gunduz, S. R. Kulkarni, and H. V. Poor, 'Federated learning with quantized global model updates,' 2020, arXiv:2006.10672 .
- [9] N. Shlezinger, M. Chen, Y. C. Eldar, H. V. Poor, and S. Cui, 'UVeQFed: Universal vector quantization for federated learning,' IEEE Trans. Signal Process. , vol. 69, pp. 500-514, 2021.
- [10] A. Reisizadeh, A. Mokhtari, H. Hassani, A. Jadbabaie, and R. Pedarsani, 'FedPAQ: A communication-efficient federated learning method with periodic averaging and quantization,' in Proc. Int. Conf. Artif. Intell. Statist. , 2020, pp. 2021-2031.
- [11] M. N. Nguyen, N. H. Tran, Y. K. Tun, Z. Han, and C. S. Hong, 'Toward multiple federated learning services resource sharing in mobile edge networks,' IEEE Trans. Mobile Comput. , vol. 22, no. 1, pp. 541-555, Jan. 2023.
- [12] X. Jin, W. Hua, Z. Wang, and Y. Chen, 'A survey of research on computation offloading in mobile cloud computing,' Wireless Netw. , vol. 28, no. 4, pp. 1563-1585, 2022.
- [13] Y. Gong et al., 'Multi-modal federated learning based resources convergence for satellite-ground twin networks,' IEEE Trans. Mobile Comput. , vol. 24, no. 5, pp. 4104-4117, May 2025.
- [14] H. Liu, F. He, and G. Cao, 'Communication-efficient federated learning for heterogeneous edge devices based on adaptive gradient quantization,' in Proc. IEEE Conf. Comput. Commun. , 2023, pp. 1-10.
- [15] W.Liu,L.Chen,andW.Zhang,'Decentralizedfederatedlearning:Balancing communication and computing costs,' IEEETrans.Signal Inf. Process. Netw. , vol. 8, pp. 131-143, 2022.
- [16] C. Wu, F. Wu, L. Lyu, Y. Huang, and X. Xie, 'Communication-efficient federated learning via knowledge distillation,' Nature Commun. , vol. 13, no. 1, pp. 1-8, 2022.
- [17] Y. Mao et al., 'Communication-efficient federated learning with adaptive quantization,' ACM Trans. Intell. Syst. Technol. , vol. 13, no. 4, pp. 1-26, 2022.
- [18] H. Wang, Z. Qu, S. Guo, N. Wang, R. Li, and W. Zhuang, 'LOSP: Overlap synchronization parallel with local compensation for fast distributed training,' IEEEJ.Sel. Areas Commun. , vol. 39, no. 8, pp. 541-2557, Aug. 2021.
- [19] I. Markov, A. Vladu, Q. Guo, and D. Alistarh, 'Quantized distributed training of large models with convergence guarantees,' in Proc. Int. Conf. Mach. Learn. , 2023, pp. 24020-24044.
- [20] Y. Gong, D. Yu, H. Yao, X. Cheng, A. Nallanathan, and G. K. Karagiannidis, 'Multi-modal learning-based multi-task offloading schemes for satellite-ground integrated networks,' IEEE Trans. Wireless Commun. , vol. 24, no. 7, pp. 5635-5647, Jul. 2025.
- [21] J. Sun, T. Chen, G. B. Giannakis, Q. Yang, and Z. Yang, 'Lazily aggregated quantized gradient innovation for communication-efficient federated learning,' IEEETrans. Pattern Anal. Mach. Intell. , vol. 44, no. 4, pp. 20312044, Apr. 2022.
- [22] D. Jhunjhunwala, A. Gadhikar, G. Joshi, and Y. C. Eldar, 'Adaptive quantization of model updates for communication-efficient federated learning,' in Proc. Int. Conf. Acoust. Speech Signal Process. , 2021, pp. 3110-3114.
- [23] S. Hashemi, N. Anthony, H. Tann, R. I. Bahar, and S. Reda, 'Understanding the impact of precision quantization on the accuracy and energy of neural networks,' in Proc. Des. Autom. Test Europe Conf. Exhib. , 2017, pp. 1474-1479.
- [24] S. Seo, J. Lee, H. Ko, and S. Pack, 'Situation-aware cluster and quantization level selection algorithm for fast federated learning,' IEEE Internet Things J. , vol. 10, no. 15, pp. 13292-13302, Aug. 2023.
- [25] A. Li, J. Sun, P. Li, Y. Pu, H. Li, and Y. Chen, 'Hermes: An efficient federated learning framework for heterogeneous mobile clients,' in Proc. 27th Annu. Int. Conf. Mobile Comput. Netw. , 2021, pp. 420-437.
- [26] J. Wang, Q. Liu, H. Liang, G. Joshi, and H. V. Poor, 'A novel framework for the analysis and design of heterogeneous federated learning,' IEEE Trans. Signal Process. , vol. 69, pp. 5234-5249, 2021.
- [27] Y. Deng et al., 'FAIR: Quality-aware federated learning with precise user incentive and model aggregation,' in Proc. IEEE Conf. Comput. Commun. , 2021, pp. 1-10.
- [28] F. Lai, X. Zhu, H. V. Madhyastha, and M. Chowdhury, 'OORT: Efficient federated learning via guided participant selection,' in Proc. 15th USENIX Symp. Operating Syst. Des. Implementation , 2021, pp. 19-35.
- [29] L. Yi, W. Gang, and L. Xiaoguang, 'QSFL: A two-level uplink communication optimization framework for federated learning,' in Proc. Int. Conf. Mach. Learn. , 2022, pp. 25501-25513.
- [30] C. Yang et al., 'Communication-efficient satellite-ground federated learning through progressive weight quantization,' IEEE Trans. Mobile Comput. , vol. 23, no. 9, pp. 8999-9011, Sep. 2024.
- [31] Q. Ma, Y. Xu, H. Xu, Z. Jiang, L. Huang, and H. Huang, 'FedSA: A semi-asynchronous federated learning mechanism in heterogeneous edge computing,' IEEE J. Sel. Areas Commun. , vol. 39, no. 12, pp. 3654-3672, Dec. 2021.
- [32] X. Hou, J. Wang, C. Jiang, Z. Meng, J. Chen, and Y. Ren, 'Efficient federated learning for metaverse via dynamic user selection, gradient quantization and resource allocation,' IEEE J. Sel. Areas Commun. , vol. 42, no. 4, pp. 850-866, Apr. 2024.
- [33] B. Jacob et al., 'Quantization and training of neural networks for efficient integer-arithmetic-only inference,' in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. , 2018, pp. 2704-2713.

- [34] A. Katharopoulos and F. Fleuret, 'Not all samples are created equal: Deep learning with importance sampling,' in Proc. Int. Conf. Mach. Learn. , 2018, pp. 2525-2534.
- [35] A. Li, L. Zhang, J. Tan, Y. Qin, J. Wang, and X.-Y. Li, 'Sample-level data selection for federated learning,' in Proc. IEEE Conf. Comput. Commun. , 2021, pp. 1-10.
- [36] P. Zhao and T. Zhang, 'Stochastic optimization with importance sampling for regularized loss minimization,' in Proc. Int. Conf. Mach. Learn. , 2015, pp. 1-9.
- [37] T. B. Johnson and C. Guestrin, 'Training deep models faster with robust, approximate importance sampling,' in Proc. Adv. Neural Inf. Process. Syst. , 2018, pp. 7276-7286.
- [38] H. R. Garey and D. S. Johnson, 'Computers and intractability. A guide to the theory of NP-completeness,' J. Symbolic Log. , vol. 48, no. 2, pp. 498500, 1983.
- [39] G. Cohen, S. Afshar, J. Tapson, and A. Van Schaik, 'EMNIST: Extending MNIST to handwritten letters,' in Proc. Int. Joint Conf. Neural Netw. , 2017, pp. 2921-2926.
- [40] A. Krizhevsky et al., 'Learning multiple layers of features from tiny images,' Univ. Toronto, Toronto: Toronto, ON, Canada, 2009. [Online]. Available: https://www.cs.utoronto.ca/ ∼ kriz/learning-features-2009-TR. pdf
- [41] Z. Qu et al., 'Partial synchronization to accelerate federated learning over relay-assisted edge networks,' IEEE Trans. Mobile Comput. , vol. 21, no. 12, pp. 4502-4516, Dec. 2022.
- [42] F. Wu, S. Guo, Z. Qu, S. He, Z. Liu, and J. Gao, 'Anchor sampling for federated learning with partial client participation,' in Proc. Int. Conf. Mach. Learn. , 2023, pp. 37379-37416.
- [43] H. Tang, C. Yu, X. Lian, T. Zhang, and J. Liu, 'DoubleSqueeze: Parallel stochastic gradient descent with double-pass error-compensated compression,' in Proc. Int. Conf. Mach. Learn. , 2019, pp. 6155-6165.
- [44] R. Jin, Y. Liu, Y . Huang, X. He, T. Wu, and H. Dai, 'Sign-based gradient descent with heterogeneous data: Convergence and byzantine resilience,' IEEE Trans. Neural Netw. Learn. Syst. , vol. 36, no. 2, pp. 3834-3846, Feb. 2025.
- [45] B. Li, N. Ying, and F. Wang, TL-System, 'Plato: An open-source research framework for production federated learning,' in Proc. ACM Turing Award Celebrati Conf.-China , 2023, pp. 1-2. [Online]. Available: https://platodocs.netlify.app/
- [46] P. Chrabaszcz, I. Loshchilov, and F. Hutter, 'A downsampled variant of ImageNet as an alternative to the CIFAR datasets,' 2017, arXiv:1707.08819 .

<!-- image -->

Hengrui Cui received the MS degree from the Department of Computer Engineering, Stevens Institute of Technology, New Jersey, in 2020. He is currently working toward the PhD degree with the School of Computer Science, Nanjing University, China. His research interests include distributed computing, edge computing and federated learning.

<!-- image -->

Zhihao Qu (Member, IEEE) received the BS and PhD degrees in computer science and technology from Nanjing University, Nanjing, China, in 2009, and 2018, respectively. He is currently an associate professor with the College of Computer Science and SoftwareEngineering,HohaiUniversity.Hisresearch interests are mainly in the areas of edge intelligence and federated learning.

<!-- image -->

Baoliu Ye (Member, IEEE) received the PhD degree in computer science and technology from Nanjing University, China, in 2004. He is a full professor with the School of Computer Science, Nanjing University. He served as a visiting researcher of the University of Aizu, Japan, from March 2005 to July 2006, and the dean of the School of Computer and Information, Hohai University since January 2018. His current research interests mainly include distributed systems, cloud computing, wireless networks with more than 100 papers published in major conferences

and journals. He served as the TPC co-chair of HotPOST12, Hot-POST11, and P2PNet10. He is the regent of CCF, the secretary-general of CCF Technical Committee of Distributed Computing and Systems.

<!-- image -->

Bin Tang (Member, IEEE) received the BS and PhD degrees in computer science and technology from Nanjing University, Nanjing, China, in 2007, and 2014, respectively. He was an assistant researcher with Nanjing University from 2014 to 2020, and also a research fellow with The Hong Kong Polytechnic University, in 2019. He is currently a professor with Hohai University. His research interests lie in the area of communications, network coding, and distributed computing.

<!-- image -->

<!-- image -->

<!-- image -->

TaoZhuang (Member, IEEE) received the BS degree in computer science and technology from Xidian University, in 2024. He is currently working toward the MS degree with the School of Computer Science, Nanjing University, China. His research interests include federated learning and personalized federated learning.

Xinyu Wang received the MS degree in computer science and technology from Nanjing University, Nanjing, China, in 2024. He is currently a software engineer with Tencent Holdings Ltd.

Yue Zeng (Member, IEEE) received the PhD degree from Nanjing University. He is currently an associate professor with the Nanjing University of Science and Technology. He has published more than 20 papers in top journals and conferences, including IEEE Transactions on Computers , IEEE Transactions on Services Computing , IEEE Transactions on Mobile Computing , IEEE Transactions on Communications , IEEE Transactions on Cloud Computing , and IEEE CVPR. His research interests include edge intelligence, deep reinforcement learning, machine learning

training and inference, federated learning, and serverless computing.