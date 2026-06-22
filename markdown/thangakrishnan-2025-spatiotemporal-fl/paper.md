## Spatiotemporal Federated Learning for Privacy-Preserving Load Forecasting and Appliance Scheduling in Smart City Homes

M. Suresh Thangakrishnan , V. Mahesh Kumar Reddy, Marimuthu Karuppiah, and Sanjeev Thakur

Abstract -As smart cities evolve, optimizing energy use within residential environments via intelligent consumer electronics becomes crucial to ensure grid stability, enhance user comfort, and promote sustainability. This paper proposes a novel IoT-enabled demand response (DR) framework that integrates AI-based spatio-temporal load forecasting and intelligent appliance scheduling through federated learning (FL) across consumer electronics such as smart meters, home hubs, and smartphones. The system leverages household-level energy consumption data, which are inherently temporal and spatial in nature, to make localized, context-aware scheduling decisions while preserving data privacy. To address key challenges in home energy management, such as data heterogeneity, forecast uncertainty, and multi-objective optimization, the framework employs a Split Federated Learning (Split-FL) architecture with Gated Recurrent Units (GRU) for accurate, privacy-preserving load forecasting. Raw data remain on edge devices, ensuring compliance with privacy regulations while enabling collaborative model training. Additionally, a robust hybrid metaheuristic algorithm, combining the Improved Grey Wolf Optimizer (IGWO) and TeachingLearning-Based Optimization (TLBO), is deployed at the Fog layer to solve the complex multi-objective appliance scheduling problem, explicitly accounting for forecast uncertainties modeled via ellipsoidal sets. Comprehensive experimental evaluations in real-world datasets (REDD, UK-DALE and Pecan Street) demonstrate that the proposed framework achieves an average reduction of 32% in total energy cost, a decrease of 51% in user discomfort, and a reduction of 40% in the peak to average ratio, while maintaining a compliance rate of demand response 92%. The Split-FL GRU forecasting model also outperforms centralized approaches in prediction accuracy. By enabling decentralized, adaptive, and privacy-preserving intelligence across heterogeneous consumer devices, this framework supports scalable and resilient energy optimization in smart homes. It directly contributes to the operational efficiency and sustainability of smart city infrastructures and lays the foundation for future location-sensitive energy recommender systems.

Received 6 June 2025; revised 12 August 2025; accepted 7 September 2025. Date of publication 2 October 2025; date of current version 8 December 2025. (Corresponding author: M. Suresh Thangakrishnan.)

- M. Suresh Thangakrishnan is with the Department of Computer Science and Engineering, Einstein College of Engineering, Tirunelveli 627012, India (e-mail: suresh.nellai@gmail.com).
- V. Mahesh Kumar Reddy is with the Department of EEE, Chaitanya Bharathi Institute of Technology, Proddatur 516360, India (e-mail: maheshvmkr2020@gmail.com).

Marimuthu Karuppiah is with the Presidency School of Computer Science and Engineering, Presidency University, Bengaluru 560064, India (e-mail: marimuthu.k@presidencyuniversity.in; marimuthume@gmail.com).

Sanjeev Thakur is with the Department of Computer Science and Engineering, Amity School of Engineering and Technology, Amity University Uttar Pradesh, Noida 201313, India (e-mail: sthakur3@amity.edu).

Digital Object Identifier 10.1109/TCE.2025.3617029

Index Terms -Federated learning, spatiotemporal data analytics, consumer applications, smart homes, privacy preserving.

## I. INTRODUCTION

T HE ELECTRICITY is a fundamental pillar of modern digital ecosystems, serving as an indispensable resource that enables the functioning of industrial operations, public infrastructure, and residential environments [1]. With the emergence of smart cities, where ubiquitous sensing, intelligent automation, and real-time decision-making are deeply embedded, the efficient and sustainable use of electrical energy has become increasingly critical. These urban environments are characterized by a high density of interconnected systems and devices, including a rapidly growing array of smart consumer electronics, such as intelligent home appliances, electric vehicles, wearable health monitors, and distributed energy assets, which collectively contribute to a surge in energy demand, variability, and management complexity.

Unlike conventional commodities such as fossil fuels or water, electricity must be consumed at the time it is generated. This intrinsic limitation creates significant challenges for power system operators, particularly during peak demand intervals. According to a global energy assessment conducted in 2019, electricity demand is expanding at nearly double the rate of the previous decade, driven largely by urbanization, industrial growth, and the proliferation of smart technologies within consumer environments [2]. Traditional electric grids, originally designed for centralized and unidirectional energy flows, are increasingly incapable of meeting the evolving demands imposed by smart city infrastructures and heterogeneous smart consumer devices. The consequences are multiple, ranging from poor power quality and transformer overloads to severe short circuits and infrastructure failures, particularly during peak periods [3].

To address these challenges, the transition from legacy grids to intelligent Smart Grids (SGs) has become imperative. Smart Grids integrate state-of-the-art technologies -including the Internet of Things (IoT) [4], fifth-generation (5G) and beyond communication networks [5], big data analytics and machine learning (ML) techniques to support dynamic, bidirectional communication between utilities and consumers. This capability not only facilitates the granular monitoring and control of distributed assets but also enables real-time decision support and autonomous energy management, particularly in

1558-4127 c © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence

and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.

the landscape of consumer electronics embedded within smart homes and buildings [6].

Demand Response (DR) has emerged as a pivotal mechanism within the SG paradigm, incentivizing consumers to alter or change their patterns of electricity use in response to grid conditions or pricing signals. When integrated with advanced data-driven modeling, DR can be automated to align consumption profiles with availability, particularly in systems with high penetration of renewable energy [7]. Deep learning techniques such as recurring neural networks (RNNs), conditionally restricted Boltzmann machines, and deep reinforcement learning have been used to enhance forecast accuracy, uncover hidden consumption patterns, and enable predictive load control [8].

Despite these advancements, traditional centralized learning approaches face several critical limitations, particularly when applied to privacy-sensitive smart home environments. The continuous exchange of raw energy usage data between consumer electronic devices and cloud-based servers results in excessive communication overhead, increased latency, and increased privacy concerns [9]. This is especially problematic in smart cities, where the density and diversity of connected consumer electronics generate vast volumes of heterogeneous and often sensitive data. To overcome these challenges, Federated Learning (FL) has emerged as a promising decentralized learning paradigm that enables the training of collaborative models directly on the edge, without transferring raw data to centralized servers [10], [11].

To address these limitations, this research proposes a novel federated AI-driven energy management framework tailored for smart home environments within smart cities, with a specific focus on scalable deployment across distributed consumer electronics. The proposed system is based on a hierarchical Edge-Fog-Cloud architecture, which enables efficient distribution of computational tasks while maintaining real-time responsiveness and data confidentiality. At its core, the framework incorporates a Split Federated Learning (SplitFL) paradigm utilizing Gated Recurrent Unit (GRU) neural networks for short-term load forecasting. By executing partial model training on edge devices, such as smart thermostats, home energy gateways, and embedded controllers in consumer appliances, and completing refinement at intermediate fog nodes, the system balances forecast accuracy with stringent privacy requirements.

By unifying federated learning-based forecasting, robust hybrid optimization, and hierarchical system architecture, the proposed framework facilitates intelligent coordination of smart consumer electronics, improves load adaptability, and advances sustainable energy practices within smart cities. The system ensures interoperability, scalability, and security, making it highly applicable for large-scale urban deployments that involve heterogeneous consumer appliances, distributed energy resources, and intelligent control systems. The primary contributions of this research are as follows:

- Development of a novel Split-FL based GRU forecasting framework explicitly designed for smart home energy management applications in smart cities, which safeguards consumer data privacy through decentralized
- model training on consumer electronics and edge devices.
- Formulation and implementation of a hybrid IGWOTLBO optimization algorithm that integrates robust uncertainty modeling to produce reliable, cost-effective, and practically feasible appliance schedules for diverse categories of consumer electronics.
- Seamless integration of forecasting and optimization modules within a resilient and scalable Edge-Fog-Cloud system architecture, providing a holistic and adaptive energy management solution for smart cities.
- Application of a comprehensive uncertainty modeling paradigm to ensure schedule robustness in the presence of fluctuating energy demands and variable renewable energy availability.

The remainder of this paper is organized as follows. Section II provides an in-depth review of the related literature, including the evolution of smart grids, demand response techniques, advances in home energy management systems, and current limitations of centralized learning in energy forecasting. Section III presents the proposed federated learning-based framework, including system architecture, mathematical modeling, and operational mechanisms. Section IV discusses the experimental setup, simulation results, and comparative analysis with state-of-the-art methods. Finally, Section V concludes the paper by summarizing key findings and suggesting directions for future research in federated and AI-driven energy management for smart cities.

## II. RELATED WORKS

With the rapid integration of smart consumer electronics, the Internet of Things (IoT) devices, and renewable energy systems within urban infrastructures, there is a pressing need for adaptive, scalable, and privacy-preserving energy management solutions. This literature investigates contemporary advances in three key domains: foundational architectures for energy management, federated learning (FL) techniques for secure and distributed model training, and heuristic optimization strategies for dynamic load scheduling. The analysis delineates the core methodologies, evaluates their effectiveness in real-world deployments, and outlines existing challenges and limitations that must be addressed to facilitate the deployment of next-generation DSM frameworks in smart cities. Wang et al. [12] introduced a real-time solar energy management system, which integrates photovoltaic generation with battery storage for dynamic energy control. The architecture facilitated real-time scheduling based on PV production, showcasing the potential of microprocessor-driven systems in distributed renewable environments. However, the absence of user-driven customization and adaptive learning mechanisms limits its scalability for contemporary smart homes that rely on consumer electronics and AI-based control.

Wang et al. [13] proposed a Bluetooth-based smart home automation system using microcontrollers and central server communication via smartphones. This low-cost solution was effective in managing basic loads, but it lacked cloud connectivity, predictive intelligence, and support for dynamic pricing, all of which are crucial for real-time demand response in modern energy-sensitive smart cities. Lee et al. [14] designed an intelligent cloud-based home energy management system that prioritized appliances according to renewable energy forecast capabilities. Their work emphasized appliance scheduling to maximize the utility of available solar energy while reducing grid dependency. Despite its cloud intelligence, the solution did not consider latency issues, edge-level fail over, or data privacy controls-key elements in modern federated environments.

Soliman et al. [15] presented an IoT-integrated HEMS architecture that uses cloud services and Web APIs to manage ZigBee-enabled devices. The system was offered as a platform-as-a-service and provided fundamental capabilities in remote control and automation. However, its strong reliance on cloud computing could introduce bandwidth and latency bottlenecks in real-time scenarios, especially in bandwidth-constrained or privacy-sensitive environments. Zhao and Shen [16] advanced a cloud-integrated framework called CloudThings, designed to support IaaS, PaaS, and SaaS services for IoT-enabled HEMS applications. Enhances the flexibility and modularity of smart home implementations, allowing rapid deployment. The framework, while versatile, lacks support for decentralized learning, collaborative scheduling, or privacy-preserving mechanisms such as federated learning.

Bernabé-Sánchez et al. [17] emphasized an edge computing approach to energy management in smart homes, advocating localized decision making to reduce latency and improve responsiveness. The model demonstrates practical potential for decentralized control, though it does not address collaborative learning between households or the integration of predictive optimization for fluctuating renewable inputs. Al Faruque and Vatanparvar [18] introduced the concept of Energy Management-as-a-Service (EMaaS) over a fog computing platform. This approach improves computation efficiency by processing sensor data near the source before cloud aggregation. Although beneficial in reducing cloud dependency, the approach does not incorporate dynamic forecasting, robust uncertainty modeling, or privacy-preserving learning strategies that are vital in federated smart grid infrastructures.

## III. PROPOSED WORK

In smart grid integrated smart homes, energy management must be predictive, adaptive, and preserve privacy. The proposed work designs an Edge-Fog-Cloud architecture in which consumer electronics at the Edge collaboratively forecast energy demand, optimize appliance schedules, and adapt dynamically to grid signals without exposing private data. The technological backbone of this system features Split-FL, employing a Gated Recurrent Unit (GRU) for precise and private demand forecasting. For optimal appliance scheduling, a hybrid metaheuristic approach is implemented, combining the global search strengths of an Improved Grey Wolf Optimizer (IGWO) with the local refinement capabilities of TeachingLearning-Based Optimization (TLBO). This scheduling is performed under a robust uncertainty modeling paradigm to effectively handle the inherent unpredictability of forecasts and renewable energy generation. Figure 1 shows the system architecture of the proposed work.

Fig. 1. SPlit-FL GRU Framework.

<!-- image -->

## A. Smart Home at Edge Layer

In the Edge layer, each smart home i ∈ { 1 , 2 , . . . , N } contains a set of Mi smart appliances A i = { Ai , 1 , Ai , 2 , . . . , Ai , Mi } with controllable operation schedules. Each smart home operates autonomously, continuously collecting real-time operational data from its appliances, photovoltaic units (PV) that produce P PV , i ( t ) at time t , and a battery energy storage system (BESS) characterized by capacity E bat , i ( t ) and operational power P bat-dis , i ( t ) (discharge), P bat-chg , i ( t ) (charge).

Consumer electronics such as HVAC units, washing machines, refrigerators, and EV chargers are monitored for operational status, power consumption profiles, and flexibility parameters. Mathematically, the instantaneous net load Li ( t ) for each home at time t is formulated as follows:

$$
\begin{aligned}
L _ { i } ( t ) & = \sum _ { m = 1 } ^ { M _ { i } } s _ { i , m } ( t ) P _ { i , m } ^ { r a t e d } + P _ { g r i d , i } ( t ) \\ & \quad - P _ { P V , i } ( t ) - P _ { b a t - d i s , i } ( t ) + P _ { b a t - c h g , i } ( t ) \quad ( 1 ) \\ \text {where } s _ { i , m } ( t ) \in \{ 0 , 1 \} \text { indicates the on/off status of appliance}
\end{aligned}
$$

where si , m ( t ) ∈ { 0 , 1 } indicates the on/off status of appliance m at time t , and P grid , i ( t ) represents the bidirectional power interaction with the smart grid through the smart meter.

Each home locally pre-processes these data and trains a partial GRU model to forecast its near-future energy demand. Executes the client-side component of the GRU model training to enable real-time predictions. A crucial aspect of the privacy-preserving design is actualized here: instead of transmitting potentially revealing raw consumption data to upstream servers, only the intermediate activations are communicated to the Fog layer. This process forms the core of the Split-FL methodology used, which ensures user privacy while enabling collaborative learning.

## B. Privacy-Preserving Demand Forecasting via Split-FL GRU

The computation model governs the series of operations distributed across the Edge, Fog, and Cloud layers to perform real-time, robust, and coordinated smart home energy scheduling. It encapsulates data pre-processing, federated learning, demand forecasting, metaheuristic scheduling, and conflict resolution workflows in a logically connected sequence.

- 1) Smart Home Data Acquisition and Feature Encoding (Edge): At the Edge layer, each smart home collects highfrequency time series data from its appliance network, the battery energy storage system (BESS), the solar generation unit and the smart meter. At each time step t , smart home i gathers the following multivariate time series feature vector:

$$
\begin{aligned}
\mathbf x _ { t } = \left [ P _ { i , m } ( t ) , \ P _ { \mathbf P V , i } ( t ) , \ E _ { \mathbf b a t , i } ( t ) , \ \lambda ( t ) \right ] \quad ( 2 ) \quad \text {privacy} \quad \begin{matrix} 0 & 1 & 0 \\ & & 0 \\ & & & 0 \\ & & & 0 \end{matrix} \quad .
\end{aligned}
$$

where:

- Pi , m ( t ) is the power consumption of appliance m at time t ,
- E bat , i ( t ) is the current energy level in the BESS,
- P PV , i ( t ) is the solar power generated by the PV unit,
- λ( t ) represents the time-dependent dynamic electricity pricing signal.

This characteristic vector captures the energy environment and the operational context of the home i at time t and serves as input to the GRU (gated recurring unit) model for demand forecasting.

In alignment with the privacy-by-design principle and considering the computational limitations of Edge devices, only the initial layers of the GRU network are executed locally. This design choice reduces both the exposure to data and computational overhead. Intermediate activations from these Edge-side GRU layers are then securely transmitted to the Fog layer for completion of the model inference and training as part of the Split Federated Learning (Split-FL) architecture.

## C. Partial GRU Computation at Edge

The GRU network is selected as the core model for time series forecasting due to its proven effectiveness in capturing temporal dependencies and complex patterns inherent in energy consumption data. GRUs provide a computationally efficient alternative to Long Short-Term Memory (LSTM) networks, often achieving comparable performance with fewer parameters.

A GRU cell maintains a hidden state ht at each time step t , which is dynamically updated through gating mechanisms that control the flow of information. During the partial forward pass executed on the Edge device for smart home i , using local parameters θ Edge i , the GRU cell processes the input sequence x ( i ) t . The intermediate activation f ( i ) t , which corresponds to the hidden state ht in the designated split layer, is computed as:

$$
f _ { t } ^ { ( i ) } = F _ { E d g e } \left ( x _ { t } ^ { ( i ) } ; \theta _ { i } ^ { E d g e } \right )
$$

The partial forward pass typically involves one or two GRU layers and includes the following gating mechanisms:

$$
\begin{aligned}
z _ { t } = \sigma ( W _ { z } x _ { t } + U _ { z } h _ { t - 1 } ) \, ( \text {Update Gate} ) \\ r _ { t } = \sigma ( W _ { r } x _ { t } + U _ { r } h _ { t - 1 } ) \, ( \text {Reset Gate} ) \\ \text {operations} \quad \tilde { h } _ { t } = \tanh ( W x _ { t } + U ( r _ { t } \odot h _ { t - 1 } ) ) \, ( \text {Candidate Hidden State} ) \\ \text {perform} \quad f _ { t } = h _ { t } = ( 1 - z _ { t } ) \odot h _ { t - 1 } + z _ { t } \odot \tilde { h } _ { t } \, ( \text {Final Hidden State} ) \quad ( 4 ) \\ \text {schedules} \quad \\ \text {learning} \quad \text {where} \colon
\end{aligned}
$$

## where:

- W z , U z , W r , U r , W , U are learnable weight matrices.
- σ( · ) denotes the activation function of the sigmoid, producing output in [0 , 1], ideal for gating.
- tanh ( · ) represents the hyperbolic tangent function.

This partial computation is designed to reduce both the computational burden on Edge devices and the risk of privacy leakage, as only f ( i ) t is transmitted to the Fog layer for the rest of GRU processing. This approach exemplifies the principle of privacy preservation at the heart of the split-federated learning (Split-FL) paradigm.

- ⊙ denotes element-wise (Hadamard) multiplication.

## D. Split-FL Training Process: Backpropagation Across Edge and Fog

Training for the split GRU model involves a coordinated backpropagation mechanism that spans both the Fog and Edge layers to update the parameters θ Fog i and θ Edge i for each smart home i .

- 1) Fog-Side Gradient Computation: The Fog node first computes the loss gradient L ( i ) with respect to its local parameters:

$$
\nabla _ { \theta ^ { \text {Fog} } } = \frac { \partial \mathcal { L } ^ { ( i ) } } { \partial \theta _ { i } ^ { \text {Fog} } }
$$

It also computes the loss gradient with respect to intermediate activation f ( i ) t :

$$
\nabla _ { f _ { t } } = \frac { \partial \mathcal { L } ^ { ( i ) } } { \partial f _ { t } ^ { ( i ) } }
$$

- 2) Gradient Transmission: The Fog node securely transmits the activation gradient ∇ f t to the originating Edge node. This maintains the split learning principle by keeping the raw data locally, while enabling collaborative model training.
- 3) Edge-Side Gradient Computation: Upon receiving ∇ f t , the Edge node continues the backpropagation process through its sub-model F Edge. Using the chain rule, the Edge computes the gradient of the loss with respect to its local parameters:

$$
\frac { \partial \mathcal { L } ^ { ( i ) } } { \partial \theta _ { i } ^ { \text {Edge} } } = \nabla f _ { t } \cdot \frac { \partial f _ { t } ^ { ( i ) } } { \partial \theta _ { i } ^ { \text {Edge} } }
$$

- 4) Local Parameter Updates: With gradients computed, both the Edge and Fog nodes update their parameters using an optimization method such as SGD or Adam. For a learning rate η :

## At Edge:

At Fog:

$$
\theta _ { i } ^ { \text {Fog} } \leftarrow \theta _ { i } ^ { \text {Fog} } - \frac { \partial \mathcal { L } ^ { ( i ) } } { \partial \theta _ { i } ^ { \text {Fog} } }
$$

- 5) Local Model Formation and Aggregation: After updates, the Edge and Fog parameters are concatenated to form the complete local GRU model for smart home i :

$$
\theta _ { i } = \theta _ { i } ^ { \text {Edge} } \cup \theta _ { i } ^ { \text {Fog} }
$$

Each pair ( θ i , ni ) , where ni is the number of training samples at node i , is sent to the Cloud for federated aggregation. The global model is computed using the Federated Averaging (FedAvg) algorithm:

$$
\begin{aligned}
\theta _ { g \text {global} } = \sum _ { i = 1 } ^ { N } \frac { n _ { i } } { n } \theta _ { i } , \quad \text {where} \ n = \sum _ { i = 1 } ^ { N } n _ { i } \quad ( 1 1 ) \\ \text {The global model} \ \theta _ { i } + \dots + i \text { is radiubuted to all clients } ( \mathbb { E } \ \{ \theta _ { i } \} )
\end{aligned}
$$

The global model θ global is redistributed to all clients (Edge and Fog) for the next training round.

Note: Algorithm 1 in the appendix presents the complete pseudocode for Privacy-Preserving Demand Forecasting using the Split-FL GRU framework.

## E. Robust Optimization Uncertainty Modeling With Hybrid IGWO-TLBO Appliance Scheduling

In smart home energy management, uncertainties arise in load forecasting due to variability in user behavior and environmental factors. To ensure feasible and reliable appliance scheduling despite forecast errors, we incorporate robust optimization using an ellipsoidal uncertainty set.

The schedule objective for each home i minimizes the total cost Ji , balancing the electricity cost, user discomfort, and the peak load given in Eqn. (12).

$$
\begin{aligned}
\min _ { s _ { i } , m _ { i } } J _ { i } & = \omega _ { 1 } \sum _ { t = 1 } ^ { T } \lambda _ { t } P _ { \text {grid,} i t } + \omega _ { 2 } \sum _ { m \in A _ { i } } P _ { \text {penalty} } ^ { A _ { i , m } } - \omega _ { 3 } \max _ { i \in T } L _ { i t } \quad \text {volatility} \quad \text {to be
\end{aligned}
$$

subject to appliance operation, battery dynamics, and energy balance constraints.

Here, λ t is the price of electricity in real time t ; P grid , it is the power consumption of the home grid i ; P Ai , m penalty\_delay quantifies user discomfort as penalties for appliance m 's scheduling delays; and Lit is the total load, where minimizing peak load improves grid stability. The weights ω 1 , ω 2 , ω 3 sum to 1 and are adjusted to balance cost efficiency, comfort, and peak reduction. Robust optimization ensures that the schedules remain effective despite the uncertainties of the forecast.

$$
\begin{aligned}
\theta _ { i } ^ { \text {Edge} } \leftarrow \theta _ { i } ^ { \text {Edge} } - \eta \cdot \frac { \partial f _ { t } ^ { ( i ) } } { \partial \theta _ { i } ^ { \text {Edge} } } \quad ( 8 ) \quad \text {In} \quad
\end{aligned}
$$

## Algorithm 1 Privacy-Preserving Demand Forecasting via Split-FL GRU

Input: R (number of global training rounds), η Edge , η Fog (learning rates), H (set of Smart Homes), F ping), (local dataset at home h )

(Fog mapDh Output: Trained global model parameters θ global = ( θ Edge\_global , θ Fog\_global ) Initialize global parameters: θ Edge\_global , θ Fog\_global for each round r = 1 to R do Cloud broadcasts θ Edge\_global to Edges and θ Fog\_global to Fog nodes for each home h ∈ H do Set θ Edge h ← θ Edge\_global for each ( x h t , Lh ( t )) ∈ Dh do Compute GRU forward pass to obtain f h t Send ( f h t , Lh ( t )) to Fog node F ( h ) end for end for for each Fog node f do Set θ Fog h ← θ Fog\_global for each received ( f h t , Lh ( t )) do Compute prediction ˆ Lh ( t ) and loss L ( i ) Compute ∇ θ Fog and ∇ f t using Eqs. (5), (6) Update θ Fog h using Eq. (9) Send ∇ f t to Edge node h end for end for for each home h ∈ H do for each received ∇ f t do Compute ∇ θ Edge using Eq. (7) Update θ Edge h using Eq. (8) end for Concatenate model: θ h = θ Edge h ∪ θ Fog h (Eq. (10)) Send ( θ h , nh ) to Cloud end for Cloud aggregates all models using Eq. (11) to update θ global end for return θ global

## F. Robust Optimization Under Ellipsoidal Uncertainty

Since the predicted load ˆ Li ( t ) is subject to model errors and volatility (for example, solar PV variation), we model the true load Li ( t ) as being within an ellipsoidal uncertainty set. This formulation captures realistic deviations due to forecasting inaccuracies. We define the ellipsoidal uncertainty set as follows.

$$
\text {quantifies} \quad \mathcal { W } \left ( \hat { L } _ { i } ( t ) , \Delta _ { i } \right ) = \left \{ L \colon \left ( L - \hat { L } _ { i } ( t ) \right ) ^ { \top } \Delta _ { i } ^ { - 1 } \left ( L - \hat { L } _ { i } ( t ) \right ) \leq 1 \right \} ( 1 3 )
$$

This set of uncertainties U represents all load profiles L that are statistically consistent with the predicted mean ˆ Li ( t ) , confined within a confidence ellipsoid determined by the positive definite matrix /Delta1 i .

## G. Hypergraph-Based Conflict Resolution

To manage conflicting appliance scheduling requests in multiple households, we employ a hypergraph-based conflict resolution mechanism. In this model, each appliance request is represented as a node, and hyper-edges connect requests that compete for shared resources (e.g., power capacity at a given time slot). Conflict resolution is formulated as a hypergraph coloring problem, ensuring that no connected requests share the same execution slot while minimizing energy cost and user discomfort. The hypergraph is dynamically constructed at the Fog layer based on predicted loads and appliance priorities, and resolved using an iterative cost-aware assignment strategy integrated with our hybrid metaheuristic scheduler. This approach ensures fairness, scalability, and adaptability in heterogeneous smart home environments.

## H. Scalability and Communication Overhead

To ensure real-world applicability, the proposed Split-FL framework is designed to scale to thousands of homes while minimizing communication overhead. Communication efficiency is enhanced through compression of model parameters, selective gradient sharing, and asynchronous updates, thereby reducing uplink bandwidth requirements. A hierarchical aggregation strategy across the edge-fog-cloud layers further alleviates network congestion and improves parallelism. This architecture maintains convergence performance while significantly lowering latency and resource demands in large-scale deployments.

## IV. RESULTS AND DISCUSSION

This section presents a comprehensive evaluation of the proposed Federated AI-Enabled Demand-Response and Appliance Scheduling Framework. The framework's effectiveness and robustness are assessed through experimental simulations conducted on well-established public smart home datasets. We detail the experimental setup, evaluation metrics, and the strategy for the comparison of results, followed by an in-depth analysis of the results obtained and a concluding discussion.

## A. Dataset Description

To evaluate the effectiveness and robustness of the proposed framework, experimental simulations are performed on wellestablished public smart home datasets. Primarily, three datasets are utilized: REDD Dataset (Reference Energy Disaggregation Dataset): This dataset provides high-resolution appliance-level power consumption traces for multiple residential buildings. It includes critical appliances such as HVAC, water heaters, lighting, and kitchen appliances, providing granular insight into typical residential load behavior. UK-DALE Dataset (UK Domestic Appliance Level Electricity Dataset): UK-DALE contains long-duration energy consumption data at both aggregate and individual appliance levels for five U.K. homes. Capture a wide range of residential devices with varied operational patterns, making it particularly suitable for robust demand-side management experiments. Pecan Street Dataset:

Fig. 2. Performance on TEC for Economic Efficiency and Cost Reduction.

<!-- image -->

This dataset offers extensive records of residential solar PV generation, electric vehicle (EV) charging, battery storage, and detailed energy consumption. It enables testing the system's performance under renewable energy fluctuations and storage dynamics, an important aspect of future smart grids. From each dataset, daily profiles of total load, appliance-specific demand, PV generation, and battery behavior are extracted and preprocessed into 96 time slots of 15-minute intervals to simulate a realistic day-ahead scheduling horizon.

Aprimary outcome of the proposed framework is substantial economic savings for end-users. The robust IGWO-TLBO scheduler intelligently shifted appliance operations and optimized BESS usage, leading to significant reductions in Total Energy Cost (TEC) compared to all baselines, as shown in Figure 2. When benchmarked against the non-robust scheduling variant (D-IGWO-TLBO), the inclusion of robustness yielded average daily cost reductions of 17.3% (REDD), 18.9% (UK-DALE) and 16.4% (Pecan Street), highlighting the value of managing forecast uncertainty. In general, compared to rules-based controls, the system achieved an average reduction in TEC of 32%, translating to average daily savings of approximately $0.65 per home in the Pecan Street dataset under simulated RTP. These savings were consistently observed, with a low standard deviation between the test days. The analysis indicated that the savings were mainly driven by optimized load shifting to off-peak hours ( ˜ 60%) and enhanced BESS utilization for self-consumption and price arbitrage ( ˜ 40%). A grouped bar graph comparing the TEC between methods would visually emphasize the consistent and significant cost advantage of the proposed model across the three diverse datasets.

The proposed framework significantly improved grid stability and user comfort through robust and cost-effective appliance scheduling. As shown in Figure 3, it reduced the Peak-to-Average Ratio (PAR) by 22% (REDD), 19% (UKDALE) and 17% (Pecan Street) compared to non-robust methods, and over 40% relative to greedy or rule-based approaches, effectively smoothing aggregate demand and lowering peak household loads from 8-9 kWh to approximately 5.5-6 kWh. Furthermore, it decreased the duration above the maximum load 80% by 35%, while maintaining excellent Demand Response Compliance (over 92%) without compromising user comfort. Figure 4 highlights that the User Discomfort Index (UDI) averaged only 2.8, a 51% reduction from greedy cost-centric methods, with minimal delays (&lt;15 minutes) for priority tasks and acceptable deferral (2-3 hours) of flexible loads. This balance was achieved through the TLBO refinement stage, which ensured that comfort constraints were maintained along with cost savings, as illustrated by stacked bar charts showing minimal discomfort along with significant cost reductions.

Fig. 3. Performance on PAR for Grid Stability Enhancement and Demand Response.

<!-- image -->

Fig. 4. Performance on UDC.

<!-- image -->

Figure 5 represents the performance of the proposed work based on Demand Response Compliance (DRC), the proposed framework outperforms all benchmarks. During simulated DR event periods, where the system must reduce the load in response to price signals or grid stress, the proposed method achieves compliance over 92% without severely compromising end-user comfort or appliance operation deadlines.

Figure 6 represents the privacy-preserving forecast performance in different data sets for different benchmark methods. The Split-FL GRU architecture successfully delivered highly accurate load forecasts while inherently preserving user data privacy at the Edge. Quantitative analysis revealed consistently low forecasting errors; the normalized root mean square error (NRMSE) 1 hour earlier averaged 5.8% in REDD, 5.5% in UK-DALE and 5.9% on Pecan Street.

Figure 7 the line plot illustrates time-series energy load profiles over a 24-hour period for five different scheduling strategies. Rule-Based, Greedy, IGWO, TLBO and the proposed IGWO + TLBO hybrid model. The Greedy approach Datasets shows frequent and high-amplitude fluctuations, indicating poor load stability and high peak demands. The IGWO and TLBO methods demonstrate improved load smoothing compared to Greedy, but still show moderate oscillations. Figure 8 represents a heat map that illustrates transformer load percentages in a 24-hour period before and after scheduling. In the red heatmap, which represents the load before scheduling, darker shades indicate frequent occurrences of high transformer loads, especially during peak hours, suggesting significant load imbalances and potential overloading risks. In contrast, the lower green heatmap shows the load distribution after applying a scheduling algorithm, where the colors appear more TransformerLoadBeforeScheduling uniformly distributed and lighter, indicating that the peak loads have been effectively flattened. This uniformity demonstrates improved load balancing across all transformers and hours, which improves grid reliability, reduces infrastructure strain, and optimizes energy distribution efficiency.

<!-- image -->

Fig. 5. Analysis of Demand Response Compliance.

<!-- image -->

Fig. 6. Analysis for Privacy-Preserving Forecasting.

Fig. 7. Time-series Energy Profile over 24 hours.

<!-- image -->

Fig. 8. Transformer Load Before and After Scheduling.

<!-- image -->

## V. CONCLUSION

This paper presented a comprehensive Federated AI-Enabled Demand-Response and Appliance Scheduling Framework designed for smart homes within a smart grid ecosystem. Using a split-federated learning (SplitFL) architecture with GRU-based forecasting distributed between Edge and Fog layers, the framework successfully balances privacy preservation and high forecast accuracy. The robust two-stage hybrid optimization approach, combining the improved greywolf optimizer (IGWO) and the teachinglearning-based optimization (TLBO), effectively manages scheduling under uncertainty, while the hypergraph-based conflict resolution mechanism addresses appliance and infrastructure-level conflicts. Key findings highlight significant trade-offs: Enhancing privacy through Split-FL reduces communication overhead, but requires careful design to maintain model accuracy and responsiveness. Robust optimization using ellipsoidal uncertainty sets improves schedule feasibility under variable conditions, but may increase computational complexity. Evaluation in real-world datasets (REDD, UK-DALE and Pecan Street) demonstrated significant improvements -including a 32% reduction in energy costs, a 51% decrease in user discomfort, and greater than 92% compliance with demand response objectives -underscoring the practical viability of the integrated approach. This study advances federated learning applications for smart cities by illustrating how privacy-preserving distributed learning and robust scheduling can coexist to create resilient and scalable energy management solutions. Future research will focus on integrating transformer-based forecasting to better capture long-range dependencies in dynamic energy environments, and adaptive robust optimization to dynamically tune uncertainty modeling based on real-time data. In addition, communication-efficient federated training strategies, such as gradient sparsification and asynchronous updates, will be explored to further optimize the performance of the system. In general, these insights offer valuable guidance for the design of next-generation intelligent energy systems that are user-centric and robust to uncertainties, pushing the boundaries of privacy-aware smart grid technologies.

## REFERENCES

- [1] P. Moriarty and D. Honnery, 'What is the global potential for renewable energy?' Renew. Sustain. Energy Rev. , vol. 16, no. 1, pp. 244-252, 2012.
- [2] M. Tariq and M. Adnan, 'Stabilizing super smart grids using V2G: A probabilistic analysis,' in Proc. IEEE 89th Veh. Technol. Conf. (VTCSpring) , 2019, pp. 1-5.
- [3] J. Li et al., 'Resource orchestration of cloud-edge-based smart grid fault detection,' ACM Trans. Sens. Netw. , vol. 18, no. 3, pp. 1-26, 2022.
- [4] A. Al-Fuqaha, M. Guizani, M. Mohammadi, M. Aledhari, and M. Ayyash, 'Internet of Things: A survey on enabling technologies, protocols, and applications,' IEEE Commun. Surveys Tuts. , vol. 17, no. 4, pp. 2347-2376, 4th Quart., 2015.
- [5] A. Kumari, S. Tanwar, S. Tyagi, N. Kumar, M. S. Obaidat, and J. J. Rodrigues, 'Fog computing for smart grid systems in the 5G environment: Challenges and solutions,' IEEE Wireless Commun. , vol. 26, no. 3, pp. 47-53, Jun. 2019.
- [6] K. Wang et al., 'Wireless big data computing in smart grid,' IEEE Wireless Commun. , vol. 24, no. 2, pp. 58-64, Apr. 2017.
- [7] M. Kumar and N. Pal, 'Machine learning-based electric load forecasting for peak demand control in smart grid,' Comput. Mater. Continua , vol. 74, no. 3, pp. 4785-4799, 2023.
- [8] Y. Tang, K. Yang, S. Zhang, and Z. Zhang, 'Photovoltaic power forecasting: A hybrid deep learning model incorporating transfer learning strategy,' Renew. Sustain. Energy Rev. , vol. 162, Jul. 2022, Art. no. 112473.
- [9] J. Byun, I. Hong, B. Kang, and S. Park, 'A smart energy distribution and management system for renewable energy distribution and context-aware services based on user patterns and load forecasting,' IEEE Trans. Consum. Electron. , vol. 57, no. 2, pp. 436-444, May 2011.
- [10] D. Yang, X. Gao, L. Kong, Y. Pang, and B. Zhou, 'An event-driven convolutional neural architecture for non-intrusive load monitoring of residential appliance,' IEEE Trans. Consum. Electron. , vol. 66, no. 2, pp. 173-182, May 2020.
- [11] J. Wen, H. Dai, J. He, M. Xi, S. Xiao, and J. Yang, 'Federated offline reinforcement learning with multimodal data,' IEEE Trans. Consum. Electron. , vol. 70, no. 1, pp. 4266-4276, Feb. 2024.
- [12] R. Wang, S. Jiang, D. Ma, Q. Sun, H. Zhang, and P. Wang, 'The energy management of multiport energy router in smart home,' IEEE Trans. Consum. Electron. , vol. 68, no. 4, pp. 344-353, Nov. 2022.
- [13] B. Wang, Z. Zha, L. Zhang, L. Liu, and H. Fan, 'Deep reinforcement learning-based security-constrained battery scheduling in home energy system,' IEEE Trans. Consum. Electron. , vol. 70, no. 1, pp. 3548-3561, Feb. 2024.
- [14] Y.-T. Lee, W.-H. Hsiao, C.-M. Huang, and S.-C. T. Chou, 'An integrated cloud-based smart home management system with community hierarchy,' IEEE Trans. Consum. Electron. , vol. 62, no. 1, pp. 1-9, Feb. 2016.
- [15] M. Soliman, T. Abiodun, T. Hamouda, J. Zhou, and C.-H. Lung, 'Smart home: Integrating Internet of Things with Web services and cloud computing,' in Proc. IEEE 5th Int. Conf. Cloud Comput. Technol. Sci. , vol. 2, 2013, pp. 317-320.
- [16] C. Zhao and W. Shen, 'Federated domain generalization: A secure and robust framework for intelligent fault diagnosis,' IEEE Trans. Ind. Informat. , vol. 20, no. 2, pp. 2662-2670, Feb. 2024.
- [17] I. Bernabé-Sánchez, D. Díaz-Sánchez, and M. Muñoz-Organero, 'Specification and unattended deployment of home networks at the edge of the network,' IEEE Trans. Consum. Electron. , vol. 66, no. 4, pp. 279-288, Nov. 2020.
- [18] M. A. Al Faruque and K. Vatanparvar, 'Energy management-as-a-service over fog computing platform,' IEEE Internet Things J. , vol. 3, no. 2, pp. 161-169, Apr. 2016.