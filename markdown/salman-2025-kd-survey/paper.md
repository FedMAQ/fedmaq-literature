## REVIEW

## Open Access

## Knowledge distillation in federated learning: a comprehensive survey

<!-- image -->

Hassan Salman 1 , Chamseddine Zaki 2 , Nour Charara 3 , Sonia Guehis 4 , Jean-François Pradat-Peyre 1 and Abbass Nasser 5*

*Correspondence: Abbass Nasser

abbass.nasser@usek.edu.lb

Full list of author information is available at the end of the article

<!-- image -->

## Abstract

Federated Learning, often known as FL, is an approach that has recently emerged as a potentially helpful method for training machine learning models in a distributed manner without the requirement of central data storage. However, when attempting to aggregate information, the inherent variety and discrepancies in the data contributed by many FL contributors might be a substantial obstacle. In order to address this problem, researchers have offered various solutions, one of which is called knowledge distillation (KD). Such a solution seeks to transfer knowledge from a larger, more precise model to a smaller model, thus enhancing its performance. This study provides a detailed examination of the effectiveness of KD in responding to these challenges posed by FL. We comprehensively review existing research, emphasizing the benefits and limitations of using these techniques in FL and discussing the numerous challenges and research questions in this field.

Keywords Federated Learning, Knowledge distillation, Transfer Learning, Data Heterogeneity, Model Heterogeneity, Non-independent-identical Distribution

## 1  Introduction

The  abundance  and  diversity  of  data  available  today  have  made  big  data  a  critical resource  for  innovation  and  decision-making  in  various  sectors,  such  as  healthcare, finance,  and  retail  [1].  However,  processing  and  analyzing  large  and  complex  datasets promptly and effectively is challenging. One promising strategy to address these difficulties is Federated Learning (FL), an approach to distributed machine learning that allows for training machine learning models on decentralized data without data centralization. In FL, several participants collaborate by exchanging model updates rather than sharing raw data, allowing them to train models on sensitive or privacy-preserving data while leveraging multiple participants' collective knowledge [2].

However, this approach poses several challenges, including the difficulty of aggregating  the  knowledge  learned  from  each  participant  [3].  Firstly,  clients'  data  is  typically non-independent and non-identically distributed (non-IID), which means local models can drift away from the global goal, leading to issues like client drift [4], slower convergence [5], and reduced model performance [6]. Additionally, constantly uploading local models can clog up communication channels [7, 8]. Furthermore, this parameter-sharing

© The Author(s) 2025. Open Access This article is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License, which permits any non-commercial use, sharing, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if you modified the licensed material. You do not have permission under this licence to share adapted material derived from this article or parts of it. The images or other third party material in this article are included in the article's Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit   h  t  t  p  :  /  /  c  r  e  a  t  i  v  e  c  o  m  m  o  n  s  .  o  r  g  /  l  i  c  e  n  s e  s /  b y  n  c  n  d  / 4  . 0  /.

approach does not handle different model structures well, does not personalize for individual  clients,  and  is  not  great  at  dealing  with  system  differences  [9].  Finally,  In  realworld  scenarios,  the  issue  of  dealing  with  diverse  and  possibly  conflicting  data  from different participants poses a significant challenge [10].

Researchers have suggested various methods to mitigate these challenges, including Knowledge Distillation (KD) [11]. KD involves enhancing the performance of a smaller model by imparting knowledge from a larger and more precise model, referred to as the teacher model, to a smaller model, referred to as the student model [12]. This process is called "knowledge transfer." This approach has been extensively applied in many machine  learning  circumstances,  including  model  compression  [13],  model  ensemble [14], and transfer learning [15].

In  FL,  KD  can  be  employed  to  consolidate  the  information  gathered  from  a  large model trained on a central server into a smaller model trained locally on a client device [16],  making  communication  between  clients  and  the  server  smoother.  Researchers looked deeper into using KD for different FL methods [17-19]. Most of these approaches focus on training models on the client side, using KD to streamline the FL network and enhance the global model by transferring knowledge. Such an approach can improve the efficiency and privacy of FL by allowing clients to participate without transmitting their data.  Although these methods provide some level of data privacy, they still face challenges, such as limiting the information shared with the teacher model and dealing with differences in client datasets.

In this survey, we introduce the concept of FL and its key challenges, including the difficulty of aggregating knowledge from decentralized data. We then present an overview of KD and its applications in machine learning tasks. Next, we delve into KD in FL and discuss the various methods and approaches proposed in the literature. Additionally, we will present the challenges and limitations of using KD in FL, highlighting issues such as privacy risks,  communication bottlenecks, and system heterogeneity. Finally, we thoroughly examine the experimental results mentioned in the literature, emphasizing the significant discoveries and trends.

## 2  Motivation

Numerous surveys have explored either FL or KD independently, and a few recent ones have begun to touch on KD in the context of FL [4, 20, 21]. However, the field has been racing, and the intersection of KD and FL has grown into a rich subfield with diverse techniques, use cases, and emerging challenges. Despite this progress, there remains a lack of a dedicated, up-to-date, and in-depth survey that focuses specifically on the role and evolution of KD in FL systems.

What sets our work apart is its focused and structured treatment of KD in FL, not just from a technical perspective but also from a systems and deployment standpoint. We go beyond simple categorization of existing methods by organizing them around realworld FL challenges such as communication constraints, model and data heterogeneity, personalization, and privacy. We also include an empirical synthesis of findings from the literature, which, to our knowledge, has not been systematically done before in this area.

Additionally, we analyze trade-offs that are often underexplored in prior surveys, such as how privacy-preserving KD affects model utility or how KD can be adapted to support real-world deployment constraints. These insights are intended to be actionable for both researchers and practitioners who are developing or deploying FL systems that rely on distillation-based approaches.

In summary, this paper fills an important gap by offering the first comprehensive, endto-end  survey  that  focuses  specifically  on  KD  in  FL,  covering  algorithmic  strategies, system-level  design  choices,  empirical  outcomes,  privacy  concerns,  and  deployment considerations. We believe this synthesis will serve as a valuable resource for advancing the field.

Figure 1 displays the organization of the proposed survey. Overall, the survey aims to make a substantial addition to the existing body of knowledge and serve as a resource for scholars in the future. Table 1 represents the list of abbreviations and their corresponding descriptions utilized in the survey.

## 3  Overview of federated learning

In this section, we deep dive into FL, elucidating with an introduction to what it actually is. We will first discuss the basic principles behind FL, followed by the details of the algorithmic level of the Federated Averaging (FedAvg) method underlying the approach.

## 3.1  Federated learning

To train a shared global model, FL allows several clients to share and learn from one another's data without requiring each client to provide its data to the network [22]. This process is made feasible by pooling their data sets. The FL process is divided into several stages, each administered and monitored by the same server. At the start of each cycle, each client receives updates on the global model from the server. Then, the model is trained locally on each client, and only model updates are sent to the server.

Fig. 1 Survey organization

<!-- image -->

Table 1 Acronyms and their descriptions

| Abbreviations   | Description                                 |
|-----------------|---------------------------------------------|
| FL              | Federated Learning                          |
| FL-HI           | Federated Learning-Healthcare Informatics   |
| KD              | Knowledge Distillation                      |
| KDM             | Knowledge Distillation Method               |
| FC              | Fully Connected Layer                       |
| ASR             | Automatic speech recognition                |
| NLP             | Natural Language Processing                 |
| Non-IID         | Non-independent and Identically Distributed |
| MH              | Model Heterogeneity                         |
| DH              | Data Heterogeneity                          |
| CC              | Communication Cost                          |

Fig. 2 A schematic diagram of the basic steps in Federated Learning

<!-- image -->

Hence, the server aggregates the alterations performed on all clients and generates an update to the global model, ending the current cycle. FL solves data privacy problems [23] and data transmission [24] by doing away with gathering all data on a single device, which  makes  it  possible  for  machine  learning  models  to  be  trained  using  decentralized data. In addition, this method makes it possible for all parties involved to connect, exchange data, and collaborate on reaching a common objective.

## 3.2  Basics of federated learning

FL method involves using a central server to coordinate the process of training a model using data that each remote device has locally collected. It is important to note that to ensure data privacy, the devices only exchange model gradients with the server. Next, the server aggregates the models from all devices to create a new global model, which is sent back to each device.

In practical applications [25], there are N users { U 1 , . . . , U N } who possess individual databases { D 1 , . . . , D N } but are not able to access one another's data directly. A highlevel architecture of FL process is presented in Fig. 2, which requires gathering data from multiple devices and consists of three key steps: (1) distributing the initial model to each device by the server, (2) the user U i trains its model W i using its local data D i without sharing the data, and (3) the global model W g is created by aggregating the local models { W 1 , . . . , W N } , which is used to update the local models of each user. Due to advancements in FL, federated training models are increasingly efficient and accurate compared to centralized training models [26] and are essential in fields where privacy is a concern.

## 3.3  Steps of federated learning

The three basic steps are summarized as follows:

Step I: Initializing the Global Model

The process begins with the server initializing  a  global  model,  denoted  as W 0 g .  The server then distributes this model to the local participants involved in the task.

Step II: Local Model Update

Every user utilizes their local data to develop a model. Following the distribution of the current global model W t g by  the  server (where t and g denote the training rounds and global model, respectively), each user U i updates their model W t i by optimizing the parameters to minimize the local loss function F u ( W t u ) . These updated local models are then sent back to the server.

$$
\begin{aligned}
\min _ { w } f ( w ) = \sum _ { u = 1 } ^ { N } \frac { n _ { u } } { N } F _ { u } ( w )
\end{aligned}
$$

After  the  central  server  receives  the  local  models  from  the  participants,  it  combines them to create an updated global model ( W t +1 g ) . The new global model is disseminated to all  participants, reiterating the process. The server continues to iterate this process until it reaches convergence, which is determined by minimizing the global loss function F ( W t g ) across all the local models, this can be represented in the following manner:

Where w is the model parameters, N represents the number of total users, and F u ( w ) refers to the anticipated loss of prediction when a sample input from the U th device is utilized.

## 3.4  Federated averaging

The fundamental algorithm that FL is based on is called Federated Averaging (FedAvg) [27]. In order to provide synchronous rounds of collaborative learning, FedAvg utilizes an architecture composed of client and server components. The server (or aggregator) broadcasts the current global model parameters (i.e., participants) to some online clients at the start of each cycle. Each student locally updates the model parameters using its  unique  information (e.g.,  the  difference  between  the  received  and  privately  refined model parameters) before sending an update to the server. The server takes updates from the feed and then aggregates the gathered contributions using an established method (in the case of FedAvg, a weighted average based on the number of local examples maintained by clients). Following this, an updated version of the global model is generated by applying a "pseudo-gradient" to the aggregated changes [28]. It is now possible to start a new round of FL using the innovative version of the global model that can be spread.

However,  certain  restrictions  are  associated  with  the  FedAvg  algorithm,  which  is responsible  for  averaging  the  model  parameters  as  provided  by  clients.  One  of  the problems with it is that it mandates that all clients utilize the same neural architecture, which may not be appropriate for clients with varying hardware capabilities. In addition, sharing model parameters and updates may be time-consuming and expensive in terms of communication. It also can potentially result in the disclosure of sensitive information [29]. There is also the problem of clients being unwilling to provide their model architecture for fear of infringement on their intellectual property. Moreover, in the case of clients possessing different data [30], the divergence of their local models during training can lead to a decline in the performance of the global model. This decline can be attributed to the global model becoming less precise than its initial state due to the loss incurred from the diverging local models.

This  work  investigates  federated  versions  of  conventional  KD  methods  to  solve  the drawbacks of FL parameter-averaging and aggregation procedures discussed before. KDbased strategies were initially developed to permit model heterogeneity and minimize the process's communication cost. The achievement was made through the exchange of outputs generated by the model or intermediate representations that are not specific to the model instead of directly transferring the actual parameters or updates of the model. Another motivation for  developing  these  strategies  was  to  encourage  privacy  properties [31]. In order to improve model function in the presence of heterogeneous data and to  permit  model  heterogeneity,  a  series  of  techniques  [32-35]  was  then  presented  to improve the aggregation step of FedAvg with a server-side ensemble distillation phase, which was done in order to allow for model heterogeneity.

## 4  Overview of knowledge distillation

This  section  delves  deep  into  KD.  It  contains  a  brief  introduction  to  the  main  idea, the various methodologies, and then the training methods for the student and teacher model. Finally, we will go through the different ways KD can be applied, where personalization can become very pivotal to these practices.

## 4.1  Knowledge distillation

KD is  a  transfer  learning  method  introduced  by  [36],  using  knowledge  from  a  more robust teacher network to train a smaller student network. Initially, the idea was that deeper networks continuously acquire better representations. However, later research, such  as  FitNets  [37],  showed  that  shallower  networks  could  also  approximate  deeper ones  without  loss  in  accuracy.  Introducing  highway  and  residual  networks  [38,  39] allowed  for  training  deeper  architectures  with  higher  accuracy  on  various  datasets. However, it was later found that after a certain depth, improvements in accuracy were mainly due to the enhanced network capacity (number of parameters). A deeper residual network with just 16 layers may learn as well or more effectively representations than a much thinner network with 1000 levels if the two networks were trained with equivalent parameters [40]. As previously stated, the objective of KD is to acquire a small network of students from a large teacher network. The general framework for knowledge transfer is shown in Fig. 3 below.

In KD, the type of knowledge, transfer method, and teacher-student architecture play an important role, and different types of knowledge [12], as shown in Fig. 4, can be classified as follows:

- 1-  Response Based.
- 2-  Feature Based.
- 3-  Relation Based.

Fig. 3 A general framework teacher-student for knowledge transfer

<!-- image -->

Fig. 4 Different types of knowledge for transfer

<!-- image -->

## 4.1.1  Response-based method

The fundamental objective is to facilitate the student model in emulating the ultimate output of the teacher model with utmost accuracy and fidelity. This method, while simple, is usually very effective and can be seen in many jobs. Suppose logit z is the output vector of the last FC layer in a deep model. The following loss function can represent Response-based knowledge extraction:

$$
L _ { R e s D } \left ( \mathcal { Z } _ { t } , \mathcal { Z } _ { s } \right ) = \mathcal { L } _ { \mathcal { R } } ( \mathcal { Z } _ { t } , \mathcal { Z } _ { s } )
$$

In the relation, L R ( . ) Indicates the distance between the z t and z s of both teacher and student logits, respectively.

## 4.1.2  Feature-based method

$$
L _ { F e a D } \left ( f _ { t } \left ( x \right ) , f _ { s } ( x ) \right ) = \mathcal { L } _ { F } \left ( \Phi _ { t } \left ( f _ { t } \left ( x \right ) \right ) , \Phi _ { s } \left ( f _ { s } \left ( x \right )
$$

Deep neural networks learn several levels of feature representation well from data (from low-level features to semantic and abstract features). Therefore, the last layer of feature maps and the middle layers can be used as knowledge to guide the student model. Therefore, multi-level feature learning can be considered the development of response-based methods. In general, the cost function of feature-based methods can be formulated as follows:

In the relation, f t ( x ) and f s ( x ) are the characteristics of the middle or last layers of the student and teacher, and the functions Φ t ( f t ( x )) and Φ s ( f s ( x )) are  usually used when the maps and the properties f t ( x ) and f s ( x ) do not have the exact dimensions. The following Fig. 5 illustrates this transfer method well:

## 4.1.3  Relation-based method

Both  previous  methods  used  the  output  of  specific  layers  of  the  teacher  network  to transfer knowledge. While here, the goal is to transfer the connection between different layers (or even educational examples) to the student network. For example, in [15], the hot matrix describes the relationship between different layers. This matrix calculates the relationship between two feature maps using internal multiplication. In general, the loss function of knowledge-based relationship transfer can be written as follows:

$$
\begin{aligned}
L _ { R e l D } \left ( f _ { t } , f _ { s } \right ) & = \mathcal { L } _ { \mathcal { R } ^ { 1 } } \left ( \psi _ { t } \left ( \hat { f } _ { t } , \check { f } _ { t } \right ) , \psi _ { s } \left ( \hat { f } _ { s } , \check { f } _ { s } \right ) \right ) \\ & = \left ( \psi _ { t } \right ) ^ { \vee } \left ( \hat { f } _ { t } , \check { f } _ { t } \right ) , \psi _ { s } \left ( \hat { f } _ { s } , \check { f } _ { s } \right )
\end{aligned}
$$

In the relation, ̂ f t and ∨ f t are a pair of teacher network feature maps, and so ̂ f s and ∨ f s represent a pair in the student network. ψ t ( . ) and ψ s ( . ) are similarity functions of pairs. Finally,  the L R 1 function  represents  the  correlation  between  the  teacher  and  student network pairs.

## Feature-BasedKnowledgeDistillation

Fig. 5 Characteristic Knowledge Distillation Method

<!-- image -->

## Relation-BasedKnowledgeDistillation

Fig. 6 Communication knowledge distillation process between samples

<!-- image -->

Fig. 7 General framework for knowledge distillation through several teacher networks

<!-- image -->

In addition to learning the relation between different layers, many tasks aim to distill the knowledge of the relation between the samples in the teacher and student networks. This transfer process is shown in Fig. 6 below.

## 4.1.4  Multi-teacher knowledge distillation

Each teacher can provide helpful knowledge for the student network. Multi-teacher networks can be used separately or together to transfer knowledge. The simplest way to transfer knowledge from several teachers is to use their average response as an observer signal [11]. The general framework of KD architecture from several teachers is shown in the following Fig. 7:

Using multiple teacher networks to transfer response knowledge or characteristics has been reported to be effective in various researches. For example, [41] intermediate layer feature vectors are also included in knowledge transfer. In [42], two teacher networks are used, the first to transfer response knowledge and the second to transfer feature-based knowledge.  In  some  tasks  to  transfer  knowledge  from  multiple  teachers,  additional branches  of  teacher  networks  have  been  added  to  the  student  network  to  mimic  the behavior of teachers' intermediate characteristics [43]. In rebirth networks [44], a stepby-step method is used, and the student is employed in step t as a teacher in step t +1 .

In some cases, several teachers have been simulated by adding noise to a teacher network or skipping connections, and random blocks have been used for this purpose [45].

Combining the knowledge of several teachers at the feature level has been used in some methods for network training. For example, in [46], the output of several teacher networks is  extracted  and  combined.  For  this  purpose,  a  two-step  solution  is  presented. First,  a  compressed display of teachers is obtained; in the second stage, a compressed display of their parameters is obtained at the layer level. This information is used to build a  student  network (without tags and annotations). In [47], KD is done only using the teacher network with the least ambiguity in its classification for unlabeled data.

## 4.2  Training student and teacher models

The training of student and teacher models can be grouped into three main techniques based on whether the teacher model is modified simultaneously with the student model. These are known as offline, online, and self-distillation, as illustrated in Fig. 8.

## 4.3  Offline distillation

In  deep  learning,  offline  knowledge  transfer  is  a  standard  method  involving  a  skilled teacher  guiding  an  untrained  student  model.  A  common  technique  used  in  KD  is  to train the student model using the extracted knowledge distilled from the teacher model's learning  on  a  training  dataset  to  achieve  this  objective.  Recent  developments  in  deep learning have made available for use as teachers a large number of pre-trained neural network  models.  In  deep  learning,  offline  distillation  is  a  common  approach,  and  its implementation is not overly complicated.

## 4.4  Online distillation

Online distillation can be used instead of offline when a pre-trained model is unavailable. This approach uses parallel computing to update the teacher and student models simultaneously throughout a complete training cycle. Given that the instructor model is often a big-capacity deep neural network, online distillation has the potential to be a practical option for a wide variety of uses.

Fig. 8 Red represents 'pre-trained' networks acquired before distillation, and yellow represents 'to be trained' networks learned during distillation

<!-- image -->

## 4.5  Self-distillation

Figure 7 shows that the teacher and student models in self-distillation follow the same structure that permits a deep neural network's higher-layer information to instruct the network's lower-level information. In addition, it is a form of online distillation that may be applied in various contexts, such as when information is sent from the teacher model's earlier to its later epochs.

## 4.6  KD in different scenarios

Numerous machine learning and deep learning applications presented in Fig. 9, including image recognition, natural language processing, and speech recognition, have succeeded with KD. Here, we will discuss how information distillation methods are now being used and the possibilities for their future expansion.

## 4.7  KD in computer vision

The discipline of  computer vision has several opportunities for KD. Deep neural networks are rapidly used as the foundation for state-of-the-art computer vision models. However, compressing their configuration settings before deployment may significantly improve these models. Image classification [11], face recognition [48], image segmentation [49], object detection [50], and image retrieval [51, 52] are just a few regions where KD has been put to good use. In addition, because of its adaptability in using a wide variety of data, including cross-modal, multi-domain, multi-tasking, and minimal-resolution data, it is possible to train many distilled student models for a wide variety of recognition of visual cases.

## 4.8  KD in NLP

Natural  language  processing  (NLP)  is  another  application  where  KD  is  proper.  The teacher model is a big transformer-based language model like BERT [53], whereas the student model is a smaller transformer-based model. KD has been proven to increase the  performance of smaller models and enable them to be deployed in resource-constrained  situations  for  a  variety  of  natural  language  processing  tasks,  including  text classification [54], machine translation [55], and document retrieval [56]. KD is used to create NLP models that are both lightweight and efficient, using fewer resources when implemented. In addition, information gained through multilingual models may be conveyed and exchanged between students and teachers via training, which is beneficial to the challenges involving multilingual NLP. Furthermore, in low-resource scenarios such as sentiment analysis in underrepresented languages, combining transfer learning with KD has demonstrated promising improvements in performance [57].

Fig. 9 A Comprehensive overview of knowledge distillation in diverse machine and deep learning applications

<!-- image -->

## 4.9  KD in speech recognition

Smaller models' performance on voice recognition tasks was also boosted with the help of KD. Automatic speech recognition (ASR) systems are one of the most common applications of KD in voice recognition [58]. In training, contemporary ASR models, convolutional layers models with attention, and current-transforming algorithms are all essential components to consider. Achieving smaller, quicker models for effective performance is crucial for real-time, on-device speech recognition.

KD in speech recognition may also be used in the following scenarios: speaker verification [59], audio classification [60], and spoken language identification [61].

## 5  Benefits and limitations of KD in FL

FL systems aim to tackle three key challenges related to personalization. Firstly, the systems must account for the differences in device storage, computation, and communication capabilities. Secondly, the non-IID distribution of data results in heterogeneous data presents a challenge for the system. Finally, the model must be customized to each client's unique environment, leading to model heterogeneity [62].

To overcome these challenges, personalization of the global model is necessary. Most personalization  approaches  involve  two  stages.  First,  a  global  model  is  constructed through  suitable  means.  Second,  leveraging  their  private  data,  the  global  model  is adapted to clients' needs. Addressing the issues posed by statistical heterogeneity and non-IID distribution of data requires this personalized approach. Table 2 below presents various techniques for tailoring global models to particular clients.

Further, this review article differentiates two significant lines of research on using KD in FL: firstly, enabling model heterogeneity and secondly, limiting the influence of data heterogeneity on overall model performance. Facilitating model heterogeneity is the first of  these  two  main  lines  of  study.  After  that,  the  review  is  arranged  according  to  how these goals are achieved. Another approach for permitting model heterogeneity on the server  side  while  sharing  locally  generated  statistics,  model  outputs,  or  intermediate characteristics rather than model parameters is to use ensemble distillation in conjunction with FedAvg. This method may be found in Ensemble distillation [43]. To handle non-IID data [30], server-side techniques may be utilized to streamline the aggregation process. Client-side methods can be implemented to use global knowledge to solve client drift locally. Both of these approaches can be used in conjunction with one another. The bulk of potential answers include a combination of the two techniques. This revised article  intends  to  provide  readers  with  an  up-to-date  summary  of  the  most  recent advancements made in this area, which are critically important to the FL community.

Table 2 Various personalization techniques

| PUBLICATIONS   | TECHNIQUES                       |
|----------------|----------------------------------|
| [63-65]        | Transfer Learning                |
| [66, 67]       | Multi-Task Learning              |
| [68-72]        | Meta-Learning                    |
| [73, 74]       | Mixture of Global & Local Models |

## 5.1  Model heterogeneity

Multiple participants, each with their own data and computational capabilities, collaborate in FL to train a common global model. Nevertheless, different participants' data distributions and learning capacities may vary, resulting in heterogeneous and suboptimal global models. KD solves this challenge by transferring knowledge from the larger and more competent models to the smaller and less capable ones. This technique enhances the overall performance of the FL system by allowing smaller models to benefit from the knowledge of larger models.

Using server-side ensemble distillation as an additional step on top of the aggregation stage is one way the FedAvg protocol may be improved to support model heterogeneity [32, 75]. In order to achieve this goal, the server may keep a collection of conventional models, each of which may represent all learners using the same architecture. Once the server  has  collected  updates  from  its  clients,  it  performs  a  per-prototype  aggregation before generating soft targets for each model it receives from the clients. The soft targets are produced by utilizing unlabeled data or creating synthetic examples, which is done after the server has finished collecting updates. Substantially, clients with different model architectures can share and learn from one another by averaging these soft targets and using them to fine-tune each aggregated model prototype. Instead of parameter-averaging algorithms like FedAvg, other feasible ways to permit model heterogeneity consist of utilizing distributed modifications of online distillation [76], as will be given in the next section. Again, this approach allows for model heterogeneity.

## 5.1.1  Broadcasting model outputs (soft labels)

Compared to parameter-based systems, federated modifications of KD may minimize communication  needs  while  allowing  for  model  heterogeneity.  In  [77],  researchers offer  the  FedDistill  algorithm,  Fig.  10,  which  presents  a  personalized  model  based  on the assumption that the fundamental process of FedAvg already exists. In this approach, every  node  has  two  models:  a  copy  of  the  recorded  global  model  and  a  customized model designed individually by each node. The teacher network employs a personalized model to guide the student network through KD. After collaborative training, the collective model is returned to the central server for aggregation. In [19], an approach similar to that is provided. It is essential to point out that FedDistill communication efficiency is exceptionally high compared to other parameter-based methods.

A wider variety of approaches is mentioned in the literature. However, a rough outline presented in Fig. 11 of algorithmic processes might be as follows:

- i. Broadcasting: Clients receive the latest global logits/soft targets during communication.
- ii. Local distillation : Clients use global logits and soft labels to refine their local models in part of their private dataset.
- iii. Local training: The client then fine-tunes the refined model using their data.
- iv. Local prediction: Clients use the dataset to generate their local logits/soft targets.
- v. Aggregation: The server then gathers the logits, and a new round begins.

Fig. 11 Broadcasting Model Soft-Labels/Logits considering models heterogeneity

<!-- image -->

Recent approaches include an additional server distillation phase to the previous steps to distill a server-side model, which may be utilized to build the global logits/soft targets to broadcast [78-80]. At the same time, other systems employ the server entity as an aggregator for locally computed model outputs [17, 81, 82]. In addition, the training process may be enhanced by learning a model to be executed on the server [78]. Finally, algorithm development may also be affected by whether or not a dataset is labeled.

Before beginning the process correctly, clients in FedMD [17] undergo a preliminary pre-training phase using a labeled dataset. To improve training, particularly in non-IID

environments, [81] proposes a change to the aggregation stage called Entropy Reduction Aggregation (ERA), which involves applying SoftMax to the aggregated logits with a temperature lower than 1. This process reduces the entropy of global soft targets. Before sending  or  receiving  data,  the  client  and  server  in  Compressed  Federated  Distillation (CFD) [78] use a highly efficient compression method based on quantization and delta coding to reduce the size of incoming or outgoing data related to soft targets. By expeditiously  performing  training  on  the  merged  private  dataset  and  the  soft-labeled  public dataset, [82] combines the local distillation and local training phase into a single operation. They also use the high-dimensional method [83] and aggregate the soft targets to make them more durable.

[79] is similar to the Cronus method [82] in that it allows clients to train in collaboration using public data, private data, and public data annotated with global soft goals. In order to train its server model, MHAT [79] considers the existence of a labeled dataset and then trains on the mixture of this public dataset and the soft-labeled version of it. FedGEMS [80] uses a protocol similar to FedMD and improves upon it by adopting a server paradigm closer to CFD. Making use of a robust model server is vital to the FedGEMS concept. In order to promote knowledge transfer, the FedGEMS variant uses the labels included in the public transfer set to impose a selection and weighting technique [80].

Table 3 below compares the analysis of the previously discussed methods describing the knowledge, the type of auxiliary dataset, and the advantage of the used approach in terms of whether the server was used. Logits refer to the output of the last layer before the activation function, and the results of the SoftMax function are Soft-Labels.

## 5.1.2  Taking advantage of intermediate features

Several proposed methods emphasize the benefits of employing intermediate features instead  of  or  in  addition  to  soft  labels.  To  further  response-based  KD,  FedAD  [87] employs intermediate features in addition to model output, as shown in Fig. 12. As long as there is agreement on the form of the attention map, model diversity may be maintained via model-agnostic intermediate characteristics such as attention maps [88, 89]. As FedAD is a one-shot FL system, clients may join asynchronously without distilling their local model at the start of each cycle. Federated Group Knowledge Transfer [90] employs intermediate characteristics within an asynchronous split learning model [91]. The feature extractor on the edge device generates intermediate feature maps, while the classifier on the device generates soft targets 1 .

Table 3 Comparative analysis of federated knowledge distillation approaches that use soft-labels and logits

|                 | Knowledge     | Dataset                                      | Advantages                                                          | Notes (Server/Client)   |
|-----------------|---------------|----------------------------------------------|---------------------------------------------------------------------|-------------------------|
| FedDistill [77] | Logits a      | CIFAR10                                      | High Communication ef- ficiency, personalized model                 | Both                    |
| CFD [78]        | Soft-Labels b | CIFAR-10 / STL-10 [84]                       | Efficient data compression                                          | Both                    |
| MHAT [79]       | Soft-Labels   | MNIST                                        | Takes into account a labeled dataset                                | Both                    |
| FedGEMS [80]    | Soft-Labels   | CIFAR10/100                                  | Selection and weighting technique for knowledge transfer            | Server                  |
| FedMD [17]      | Soft-Labels   | MNIST/FEMNIST, CIFAR 10/ 100                 | Improved training in non-IID environments                           | Client                  |
| DS-FL [81]      | Soft-Labels   | MNIST/Fashion-MNIST                          | Reduces entropy of global soft targets                              | Both                    |
| Cronus [82]     | Soft-Labels   | SVHN [85] / PURCHASE [86] / MNIST / CIFAR 10 | Combines distillation and training phase, high dimen- sional method | Both                    |

Fig. 12 Overview of intermediate features framework

<!-- image -->

Similarly, the server uses a classifier and a more complex network. Clients share the intermediate features generated during local training, the anticipated soft targets, and associated ground truth labels for each local example. The retrieved features generated locally are sent into the server's deeper network, generating soft targets for the global model. The client and server adopt an objective function comprising a linear combination of the conventional cross-entropy loss and the KD-based loss. The first considers soft targets and ground truth labels, while the second calculates the difference between local and global logits. Using FedDKC [92] also provides server-side knowledge refinement methodologies built on top of an architecture similar to their own.

Table  4  below  summarizes  various  approaches  that  use  intermediate  features  to improve response-based KD in FL. The intermediate features refer to the output of specific layers between the input and the output layers in the neural network rather than soft labels or logits.

## 5.2  Data heterogeneity

Data heterogeneity is a frequent issue in FL due to the variation in statistical properties of data across local devices or nodes. These properties include sample sizes, feature spaces,  label  spaces,  and  data  distributions.  This  heterogeneity  can  adversely  affect model generalization, update quality, and convergence speed. KD can be applied in different  ways  to  handle  this  problem,  such  as  on  the  server  side,  by  refining  the  global model using ensemble distillation on a proxy dataset [32, 35, 75] or a data-free generator [33, 34] or at the client side, by using on-device regularizers or synthetically-generated data to distill global knowledge and control client drift. Various studies have proposed these techniques, including [95-98].

1 Intermediate features are representations extracted from hidden layers of neural networks before the final classification layer; they contain higher-dimensional structural knowledge about the data that isn't captured in either logits or soft labels.

Table 4 Comparative analysis of federated knowledge distillation methods using intermediate features

|             | Knowledge   | Dataset                                    | Advantages                                                    | Notes (Server/Client)   |
|-------------|-------------|--------------------------------------------|---------------------------------------------------------------|-------------------------|
| FedAD [87]  | Features    | CIFAR10/100, NIH CXR14 [93], CheXpert [94] | Model Diversity                                               | Client                  |
| FedGKT [90] | Features    | CIFAR10/100, CINIC10                       | Efficient Communication and Asynchronous Split Learning Model | Both                    |
| FedDKC [92] | Features    | MNIST, CIFAR10, CINIC10                    | Knowledge Refinement                                          | Server                  |

Table 5 Comparison of federated knowledge distillation techniques using model parameters and server-side refinement

|             | Knowledge   | Dataset                    | Advantages                                                                                                                         | Notes (Server/Client)   |
|-------------|-------------|----------------------------|------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| FedDF [32]  | ω           | CIFAR10/100, AG news, SST2 | Enables data heterogeneity and enhances FedAvg's aggregation                                                                       | Server                  |
| FedFTG [34] | ω           | CIFAR10/100                | Fine-tune global model with synthetic data                                                                                         | Server                  |
| FedBE [35]  | ω           | CIFAR10                    | Improves overall model performance using the Bayesian approach                                                                     | Server                  |
| FedAUX [75] | ω           | CIFAR10                    | Incorporates unsupervised pre-training for better initialization and ( ϵ, δ ) -differential privacy to weight ensemble predictions | Server                  |

## 5.2.1  Refining global models on the server-side

The authors of [32] proposed FedDF to enable data heterogeneity and enhance FedAvg's aggregation on the server side. In FedDF, a proxy dataset is used to fine-tune the global model by simulating the client models' ensemble output. FedAUX [75] further improves FedDF by incorporating unsupervised pre-training on auxiliary data to select an appropriate initialization for the client-side feature extractor. Additionally, FedAUX leverages the ( ϵ, δ ) - differentially private [99] certainty score of each participant model to weight the ensemble predictions on the proxy data. FedBE [35] suggests combining client predictions with a Bayesian model ensemble to enhance the robustness of the aggregate. Unlike  averaging  model  predictions,  FedBE  uses  a  Bayesian  approach  to  improve  the model's overall  performance.  FedFTG [34] fine-tunes the global model with synthetic data  using  data-free  KD.  This  approach  differs  from  server-side  ensemble  distillation, which requires a proxy dataset. Another proposal [33] suggests using a data-free generator-based approach to refine the global model. It is noteworthy that server-side global model rectifications can be used with client-side approaches to control model drift [34]. Table 5 below compares these methods to the dataset used and the advantage of using each one, where ω refers to the model parameters.

## 5.2.2  Refining local knowledge into global insights with REGULARIZATION

Recent studies, referenced as [3] and [95], have drawn inspiration from fine-tuning optimization methods and continual learning research. They have found that incorporating local  KD-based regularization reduces the impact of non-independent and identically distributed  (non-IID)  data  in  FL  settings.  In  the  local-global  distillation  process,  the clients' objective function comprises a cross-entropy loss and a KD-based loss synthesized through a composite function. The KD-based loss function quantifies the discrepancy between the output of the teacher model and the student model on sensitive data, thereby enabling the assessment of the student model's ability to capture the knowledge distilled  from  the  teacher  model.  The  loss  function  is  a  valuable  metric  in  evaluating the effectiveness of KD. It measures the student model's proficiency in replicating the teacher model's output, thereby providing a means to assess the quality of the distillation process. This difference is measured using mathematical concepts, such as the KullbackLeibler divergence.

$$
\begin{aligned}
L _ { c l i n t } = \sum _ { i = 1 } ^ { N } \alpha l _ { k } \left ( C _ { i } , G _ { i } \right ) + \left ( 1 - \alpha \right ) l _ { C } \left ( t _ { i } , y _ { i } \right )
\end{aligned}
$$

for N data points, where C i refers to the client model's output and G i refers to the global server model's output, l k and l c refers to KD and classification loss, respectively, t i truth labels and y i indicates the predicted value with learning rate α .

The  Cross-Entropy  loss  is  considered  a  classification  loss.  For  example,  the  binary cross-entropy loss that is calculated as the average cross-entropy across all data samples can be represented as follows:

$$
\begin{aligned}
L = - \frac { 1 } { N } \left [ \sum _ { j = 1 } ^ { N } [ t _ { j } \log \left ( p _ { j } \right ) + ( 1 - t _ { j } ) \log ( 1 - p _ { j } ) ] \right ]
\end{aligned}
$$

where N data points, t i is the truth labels and P i is the SoftMax probability for the i th data point.

In  addition,  KD  loss  can  be  considered  a  regression  problem.  Different  formulas measure the divergence, such as mean squared error, Huber loss, or Kullback-Leibler divergence.

$$
\begin{aligned}
D _ { K L } \left ( P | | Q \right ) = \sum _ { x \in X } P \left ( x \right ) \log \left ( \frac { P \left ( x \right ) } { Q \left ( x \right ) } \right )
\end{aligned}
$$

where P and Q denote the probability distributions for a set of points X .

Figure 13 depicts the fundamental framework for local-global distillation through a regularization term. The inspiration for this framework comes from two sources. First, in [3], authors draw from the concept of self-distillation mechanisms from [100], which demonstrate improved fine-tuning of pre-trained models like BERT [101] in a non-federated  setting.  Self-distillation  involves  utilizing  knowledge  from  previous  snapshots [102] produced during previous training stages of the in-training model to assist in the current model training step. In [95], the authors also observe a phenomenon akin to catastrophic forgetting [103] in continual learning research. When heterogeneous data is present, global models trained using FedAvg tend to generate inconsistent predictions on test  data  during  subsequent  rounds,  meaning that the global model's performance on certain classes decreases, which it had predicted accurately in the previous round. However, it has been shown that local distillation of global knowledge can prevent forgetting among the subsequent rounds, which can mitigate the detrimental effects of data heterogeneity (as demonstrated in [95]).

Fig. 13 Distilling Global Knowledge using regularization terms during local training where D G , D c are global and client databases, respectively. W g and W t refers to the global and client models with t number of rounds

<!-- image -->

The approach known as FedGKD, as proposed in [3], utilizes a set of previous global models as teachers to facilitate KD-based regularization. FedGKD-VOTE is a variation of this approach that leverages the discrepancy average of all previous models as the regularization term. The communication cost for FedGKD is equivalent to that of FedAvg in the simplest form, where M = 1. However, for M &gt; 1, the communication cost is doubled, and FedGKD-VOTE scales with M. FedNTD [95] use the framework depicted in Fig. 12. However, it disregards the logits generated by the actual classes when calculating the SoftMax score for the KD-based loss.

Building on previous research, the author of [104] notes that utilizing an imprecise global model as a teacher for specific classification classes can lead to inaccurate local training.  FedCAD  [96]  addresses  this  issue  by  introducing  a  class-specific  adaptive weight to regulate the impact of distillation. If the global model is accurate for a particular class, more distilled knowledge is used to train the local models for that class. FedCAD derives class-wise  adaptive  weights  by  measuring  the  global  model's  performance on an additional dataset, which is sent periodically by the server together with the model's parameters. FedSSD [97] is an extension of FedCAD that improves its performance by considering the credibility of the global model at the instance level during the calculation of the distillation term in local training.

FedMLB [105] enhances local-global distillation by utilizing intermediate representations.  It  creates  hybrid  local  and  global  subnetworks  pathways,  with  non-trainable global blocks following local network blocks. In addition to the standard cross-entropy approach, FedMLB incorporates the average cross-entropy derived from hybrid paths and  the  average  Kullback-Leibler  divergence  between  the  outputs  generated  by  the hybrid paths and the primary path as a regularization parameter. However, integrating hybrid pathways into the backpropagation process introduces a moderate computation overhead in FedMLB.

Previous studies, such as [3, 98], have employed FedDistill + as a substitute benchmark. This method builds upon the research conducted in [19, 106], as it incorporates per-label local logits on the training dataset and exchanges model parameters. In the framework illustrated in Fig. 13, FedDistill + calculates the KD loss using the globally averaged logits for each label received rather than relying on the global model's output on private data.

Table 6 below presents the different datasets for each method that uses regularization terms for KD with their advantages, where ω also  refers  to  the  model  parameters,  as mentioned before.

## 5.2.3  Data-free generator models for local-global knowledge distillation

Unlike the methods discussed before, the approach used in FedGen [98] trains a lightweight server-side generator to send round-by-round to clients for local-global distillation.  The  generator  produces  enriched  training  instances  using  global  information as inductive biases for local learning. FedGen needs local model parameters (classifier weights and label count) to build the generator. It only trains generative models using user model prediction rules. This generator creates feature representations that match user  expectations  for  a  target  label.  Users  train  their  models  across  the  latent  space, which captures peer knowledge, using the generator's enhanced samples. FedGen's lightweight generator uses a latent space substantially smaller than the input space.

In order to address the challenges posed by heterogeneous devices, FedZKT [33] has developed a framework that employs Zero-shot Knowledge Transfer to enable on-device models to be tailored to specific device types. Furthermore, this methodology enables autonomous devices to identify the most appropriate on-device models based on local resources.

In  order  to  promote  sharing  knowledge  across  these  varied  models,  the  FedZKT framework has developed a zero-shot distillation technique that does not require access to  private  on-device  data,  which  significantly  improved  over  previous  research  that relied on public datasets or pre-trained data generators. Moreover, to enable resourceconstrained  devices  to  participate,  the  computationally  intensive  distillation  task  is offloaded to the server, which adversarially learns a generator using models stored on the device. Finally, the resultant distilled central knowledge is transmitted back to the devices as model parameters that can be seamlessly integrated.

Table  7  below  lists  the  analysis  of  all  the  previously  discussed  methods,  describing  their  knowledge,  limitations  and  potential  solutions.  For  the  primary  purpose,  we have narrowed the proposed KD methods (KDM) into three main groups, i.e., reducing communication cost (CC), handling model, and data heterogeneity (MH and DH), respectively.

Table 6 Comparative analysis of federated knowledge distillation methods using regularization

term

|              | Knowledge   | Dataset                    | Advantages                                                | Notes (Server/Client)   |
|--------------|-------------|----------------------------|-----------------------------------------------------------|-------------------------|
| FedGKD [3]   | ω           | CIFAR10/100 AG News, SST-5 | Communication Efficient and Prevents forgetting           | Server                  |
| FedCAD [96]  | ω           | CIFAR10/100, FEMNIST       | Improve the accuracy of local models even when the global | Server                  |
| FedSSD [97]  | ω           | CIFAR10/100                | model is imprecise                                        | Server                  |
| FedMLB [105] | ω           | CIFAR100, Tiny-ImageNet    | More accurate and robust models                           | Client                  |

Table 7 Limitations analysis of the discussed methods

| KDM             | Purpose   | Limitations                                                                                                                                                                                                                                                       | Potential Solutions                                                                                                                                                                                                                  |
|-----------------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FedDistill [77] | CC,MH     | An extra model per client increases com- putational cost, training time and energy usage. Having to maintain an additional model can make deployment unfeasible or lower battery life by 40% for resource- constrained devices like mobile phones and IoT sensors | Lightweight model architectures for edge devices; progressive knowledge transfer strategies that eliminate model duplica- tion; hardware-aware model optimization that adjusts to client resources                                   |
| CFD [78]        | CC,MH     | The compression technique is efficient, but it can cause information loss in complex, high-dimensional feature spaces, especially impacting performance on fine-grained classification tasks (noted degradation of 5-8% on fine-grained datasets in compari-      | Adaptive compression rates depend on data complexity and job requirements; hybrid techniques selectively compress various model parts depending on sensi- tivity assessments                                                         |
| MHAT [79]       | CC,MH     | The requirement for labeled datasets creates a dependency on high-quality annotations, which can be expensive and time-consuming to obtain                                                                                                                        | Semi-supervised learning to reduce label- ing; active learning to discover the most valuable data points for labeling; syn- thetic data production with automated labeling; transfer learning from similar domains with labeled data |
| Fed- GEMS [80]  | CC,MH     | Reliance on publicly labeled datasets intro- duces bias when the public data distribu- tion differs from private client data. Studies show performance gaps of 12-18% when domain shift exists between public and private datasets                                | Domain adaptation to align feature spaces; diverse public data collecting tactics; dataset bias detection and mitiga- tion technologies                                                                                              |
| FedMD [17]      | CC,MH     | Pre-training restrictions hamper implemen- tation, especially on-device, where initial model weights cannot be transmitted or changed. Compared to approaches without pre-training, setup time is 150-200% longer                                                 | Transfer learning from general-purpose foundation models; distributed pre- training procedures; gradual learning that reduces initial knowledge; zero-shot or few-shot learning                                                      |
| DS-FL [81]      | CC,MH     | In specialized domains like healthcare and finance, unlabeled data may contain sensi- tive information or be regulated, limiting its use                                                                                                                          | Self-supervised learning methods that generate synthetic training instances, privacy-preserving data generation, and federated self-supervision using client data                                                                    |
| Cronus [82]     | CC,MH     | The server-side model introduces privacy vulnerabilities through potential inference attacks. Experiments show that with just 10% of leaked information, adversaries can reconstruct up to 40% of sensitive client data features in specific scenarios            | Integration with formal privacy guar- antees through differential privacy; safe multi-party computation proto- cols to improve server model security; decentralized designs to reduce single vulnerabilities                         |
| FedAD [87]      | CC,MH     | Synchronizing attention maps across various client architectures might cause compatibility concerns. Attention map mis- alignment reduces information transmis- sion effectiveness by 25%                                                                         | Architecture-agnostic attention mecha- nisms, adaptive attention matching algorithms, hierarchical attention distil- lation for architectural variations, and brain architecture search for compatible attention structures          |
| FedGKT [90]     | CC,MH     | The complex architecture increases deploy- ment challenges, particularly for resource- constrained environments. Implementation studies show 2-3×longer development cycles and 30-40% higher maintenance costs                                                    | Modular design patterns reduce imple- mentation; automated architectural search finds simpler but equally effective structures; hardware-software co-design optimizes deployment platforms                                           |
| FedDKC [92]     | CC,MH     | Limited information on datasets creates un- predictable performance variations across different data distributions. Experiments demonstrate performance fluctuations of 8-22% when applied to datasets with unexpected characteristics                            | Robust KD techniques that adapt to data distribution shifts; federated dataset profiling without compromising privacy; adaptive hyperparameter tuning based on dataset characteristics                                               |

Table 7 (continued)

| KDM          | Purpose   | Limitations                                                                                                                                                                                                                                              | Potential Solutions                                                                                                                                                                                                                                                        |
|--------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FedDF [32]   | MH, DH    | Proxy datasets often miss client data's statistical features, especially in domain shift situations.When utilizing CIFAR-10 as a proxy for specialized medical imaging datasets, the model may learn irrelevant features, degrading client task accuracy | Adaptive proxy generation methods modify synthetic data to better approxi- mate client data distributions without compromising privacy. Federated dataset condensation and differential privacy- preserving dataset synthesis can reduce the proxy-client distribution gap |
| FedFTG [34]  | DH        | When client data is not used, models may fail to capture important local patterns or specialized knowledge, resulting in ac- curacy losses of 10-15% on client-specific tasks compared to methods that leverage local data                               | Synthetic data generation techniques that approximate client data patterns, personalization layers that adapt generic models to local contexts, and hybrid approaches that balance privacy with performance                                                                |
| FedBE [35]   | DH        | Shared predictions expose clients to model inversion attacks, compromising privacy. According to research, adversaries can recreate 35% of training data features with enough prediction outputs                                                         | Differential privacy guarantees on shared predictions; secure aggregation protocols for prediction sharing; confidence masking techniques that reveal limited information; cryptographic approaches for secure prediction sharing                                          |
| FedAUX [75]  | MH, DH    | Additional unsupervised pre-training data increases storage and processing require- ments, creating a barrier for deployment on edge devices with limited capacity                                                                                       | Data pruning and condensation; online streaming without persistent storage; progressively updating pre-training with new data; federated unsupervised learn- ing to eliminate centralized pre-training data                                                                |
| FedGKD [3]   | DH        | Communication cost doubling forM>1 creates significant scalability challenges in bandwidth-constrained environments                                                                                                                                      | Network-based adaptive communica- tion scheduling, gradient compression, importance sampling to prioritize vital updates, and asynchronous communica- tion protocols to control bandwidth                                                                                  |
| FedCAD [96]  | DH        | Periodic sending of additional datasets for adaptive weight measurement increases privacy risks and communication overhead                                                                                                                               | Local weight adaptation without dataset sharing; privacy-preserving dataset statis- tics instead of raw data; secure multi-par- ty computation for weight measurement                                                                                                      |
| FedSSD [97]  | DH        | In heterogeneous data contexts, instance- level credibility measurement is com- putationally complex and biased. Such limitations increase processing time by 15-20% and cause fairness difficulties across client data distributions                    | Batch-level credibility approximations; fairness-aware credibility metrics; light- weight credibility estimation techniques; federated credibility validation to ensure consistent standards across clients                                                                |
| FedMLB [105] | DH        | Introduces computation overhead that disproportionately affects resource-con- strained client devices                                                                                                                                                    | Efficient implementation techniques like mixed-precision arithmetic and quantiza- tion to reduce computational demands; asynchronous computation scheduling to better distribute workload over time                                                                        |

## 5.3  Empirical synthesis of KD Performance in FL scenarios

To give a clearer picture of where KD proves most useful in FL, we summarize some of the performance results reported in recent studies. These findings highlight how KD helps tackle two major challenges in FL: model heterogeneity and data heterogeneity.

One of the key advantages of KD is that it allows clients with different model architectures to participate without needing to match a fixed global design. This flexibility has shown clear performance benefits. For example, FedGEMS [80] supports heterogeneous clients and improves accuracy by around 12-20% compared to FedAvg when working with non-IID data on CIFAR-100. Another method, FedAKD [107], lets clients choose their model architectures and still manage to outperform FedAvg by up to 20% in human activity recognition tasks-all while significantly reducing communication costs (more than 90% less data transmitted). These results show that KD can help FL systems remain both accurate and efficient, even when clients have very different capabilities.

Handling non-IID data remains one of the most persistent problems in FL. KD-based approaches offer  a  practical  way  to  smooth  out  the  differences  between  local  models and help stabilize training. FedCAD [96], which adjusts how much trust is placed in the global model for each class, reports up to 15% better accuracy in non-IID setups using datasets like  CIFAR-10 and FEMNIST. FedSSD [97] builds on this by measuring how credible  the  global  model's  predictions  are  at  the  instance  level  to  fine-tune  the  local distillation  process.  On  the  server  side,  methods  like  FedDF  [32]  and  FedBE  [35]  use ensemble learning over proxy datasets to improve global model performance, showing 5-10% improvements across tasks such as image classification and text categorization. These examples highlight how KD not only helps align client and server learning but also leads to more robust outcomes in diverse environments.

## 5.4  Real-world deployment challenges

While  KD  in  FL  demonstrates  considerable  theoretical  potential,  its  practical  implementation presents numerous problems that extend beyond the previously mentioned methodological  constraints.  Implementing  KD-FL  systems  in  practical  applications poses technological, computational, and operational challenges that practitioners must resolve.

## 5.4.1  Communication overhead

KD techniques  inherently  introduce  computational  overhead  to  the  FL  process.  This overhead manifests in several ways:

1.  Client-Side Computation: Distillation operations on edge devices necessitate additional compute  cycles  for  the  generation  or  processing  of  soft  labels,  the  calculation  of distillation losses, and the execution of multi-model training. This overhead can be significant for resource-limited devices such as IoT sensors or mobile phones. In real applications,  researchers  have  noted  a  30-70%  increase  in  training  duration  when applying KD on client devices relative to conventional FL methods [108, 109].
2.  Memory Requirements: Simultaneously operating both instructor and student models on client devices substantially increases memory demands. For instance, FedDistill's methodology, necessitating the maintenance of two models per client, can elevate peak memory use by 1.5-2 × , rendering implementation unfeasible for devices with limited memory [76].
3.  Battery Consumption: The increased computational load directly translates to higher energy consumption. In mobile device deployments, KD-enhanced FL protocols have been shown to increase battery usage by 25-40% compared to standard FL [110], thereby limiting user engagement in real-world applications.
4.  Server-Side  Scaling:  Server-side  distillation  approaches  like  FedDF  [32]  and  FedBE [35] face scaling challenges when dealing with hundreds or thousands of clients. The computational complexity of ensemble distillation grows linearly or super linearly with the number of participating clients, creating bottlenecks in large-scale deployments.