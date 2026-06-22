## Air Quality Prediction using CommunicationEfficient Federated Learning with Compressed Deep Learning Models

Iwin Thanakumar Joseph Department of Computer Science and Engineering, School of Advanced Computing, Alliance University Bengaluru, India iwin.joseph@alliance.edu.in

Babu K Department of Computational intelligence, SRM Institute of Science and Technology Kattankulathur, india babukumarit@gmail.com G Prabaharan Department of CSE, Vel Tech Rangarajan DrVel Tech Rangarajan Dr.Sagunthala R&amp;D Institute of Science and Technology) Avadi, Chennai prabaharang@gmail.com D Soundaravalli Department of AI and Data Science Kings Engineering College Kanchipuram, india soundaravalli@gmail.com

Abstract Accurate air quality forecasting is very significant for  smart  cities,  healthcare,  and  eco-friendly  urbanization.  In conventional air quality forecasting systems, data gathering is based  on  a  centralized  system,  and  it  leads  to  potential  data privacy  concerns,  high  communication  overhead,  and  poor scaling capabilities for a federated network of sensors. Federated Learning (FL) is a novel technique to train a model without  exchanging  data.  However,  frequent  transmission  of high-dimensional model parameters in FL introduces substantial  communication  costs,  particularly  in  bandwidthconstrained  Internet  of  Things  (IoT)  and  edge  computing settings. This paper presents a Federated Learning with Model Compression  (FL-CM)  framework  for  air  quality  prediction that  addresses  both  privacy  preservation  and  communication efficiency.  In  the  proposed  approach,  distributed  IoT-based sensing  nodes  locally  train  deep  learning  models  using  air pollutant and meteorological data, while only compressed model updates are communicated to a central server for aggregation. Model  compression  techniques  are  employed  to  significantly reduce communication payloads without compromising predictive accuracy. Experimental evaluation using a distributed air quality dataset demonstrates that the proposed FL-CM framework outperforms traditional machine learning, centralized deep learning, and standard FL approaches. Furthermore, communication overhead is reduced by 71.70% compared to the FL-only approach. These results confirm that integrating model compression with federated learning enables accurate,  scalable,  and  privacy-aware  air  quality  prediction suitable for real-world smart city deployments.

Keywords -Air Quality Prediction; PM2.5; Federated Learning;  Model  Compression;  Deep  Learning;  Internet  of Things; Smart Cities; Privacy-Preserving Learning

## I. INTRODUCTION

Air pollution poses a pressing global challenge, profoundly affecting human health, ecosystems, and economic progress. Fine particulate matter like PM2.5 proves especially dangerous, as it infiltrates deep into the lungs and bloodstream, heightening risks of respiratory illnesses, heart disease,  strokes,  and  lung  cancer.  Recent  data  reveals  that over 90% of people worldwide breathe air exceeding WHO

Velliangiri Sarveshwaran Department of Computational Intelligence, SRM Institute of Science and Technology Kattankulathur, india velliangiris@gmail.com

safety thresholds, with urban areas hit hardest by this crisis [1][2].

Traditional models in the prediction of air quality rely mostly on  centralized  predictions  and  data  processing.  Regional prediction of air pollution  index  has  been  done  using statistical  techniques  and  fuzzy  time  series  model,  and  has shown a decent performance in structured settings [3]. The latter has been recently surpassed by deep learning models, such as convolutional neural networks (CNNs), long shortterm  memory  (LSTM)  networks,  deep  belief  networks (DBNs), and hybrid architectures, which possess an ability to predict air quality values by imitating more complex spatialtemporal patterns in the available stream of air quality data [4], [5]. Image-based prediction schemes have also been used in  improving  PM  2.5  prediction  with  the  use  of  visual atmospheric  indicators  which  have  added  to  the  predictive capability  of  the  models  and  their  predictability.  Although they  are  effective,  the  methods  are  typically  based  on centralized access to large amounts of data, which would be a cause of concern with regard to data privacy, communication overhead, and scalability.

FL has now become an exciting paradigm of decentralized learning,  which  can  help  overcome  these  shortcomings, allowing the joint training of models without access to raw data.  Under FL-based air quality monitoring systems, edge devices  or  sensing  nodes  locally  train  models  and  only transmit model updates to a central server. The paradigm has been effectively utilized in prediction of PM 2.5 in smart city sensing  applications  to  achieve  better  privacy  preservation and  minimized  data  exposure  [6].  Moreover,  it  has  been suggested  that  aerial-ground  FL  using  unmanned  aerial vehicles  (UAVs)  should  be  used  to  offer  coverage  of  air quality monitoring and at the same time retain decentralized learning functions [7], [8]. Extensive literature has emphasized  the  increasing  applicability  of  FL  with  multiaccess edge computing to scalable air quality monitoring as well as privacy-conscious.

Santosh Kumar Perumal Tata Consultancy Services London UK santosh.personalid@gmail.com Nonetheless,  one  of  the  major  bottlenecks  of  FL-based systems  is  that  the  cost  of  communication  is  high,  when exchanging the model parameters between distributed clients and  the  aggregation  server  frequently.  This  is  especially  a significant problem in air quality monitoring applications of a  bandwidth  limited  network,  and  resource  limited  edge devices. To address this issue, quantization, deep compression are some of the model compression techniques that  have  been  proposed  to  achieve  a  large  reduction  in communication payloads without loss in the learning accuracy and convergence  rate. Compression  strategies combined  with  FL  have  been  proved  to  be  very  useful  in edge-based environmental sensing systems to facilitate effective and privacy preserving collaborative learning.

Driven by these trends, next generation air quality prediction with compressed models through Federated Learning is also a  robust  and  scalable  model  capable  of  supporting  next generation environmental monitoring. These frameworks are capable of providing precise air quality forecasts by combining decentralized learning, model updates with communication  efficiency,  and  state-of-the-art  predictive architectures using participant privacy and network overhead minimization. This work is based on the current developments in the field of FL-based air quality prediction and compressed learning models to overcome the practical opportunities of implementing intelligent air quality monitoring systems in the actual smart city conditions.

## Research Contributions:

1. A communication-efficient federated learning framework with model compression is proposed for privacy-preserving air quality prediction.
2. The approach achieves higher prediction accuracy with significantly reduced communication overhead.
3. The  framework  is  scalable  and  suitable  for  IoTenabled smart city air quality monitoring.

This  paper  is  arranged  as  follows.  Section  II  discusses existing  studies  related  to  air  quality  prediction,  federated learning,  and  model  compression.  Section  III  discuss  the proposed air quality prediction approach based on federated learning.  Section  IV  outlines  the  datasets,  experimental design, and evaluation criteria. Section V  reports the experimental  findings  and  provides  performance  analysis. Finally,  Section  VI  summarizes  the  work  and  highlights possible directions for future studies.

## II. RELATED WORK

Mengara  et  al.    study  proposes  a  novel  attention-based convolutional BiLSTM autoencoder framework for predicting particulate matter concentrations (PM₂.₅ and PM₁₀), using Busan, South Korea as a case study. The model is trained in a distributed data-parallel framework to capture spatiotemporal  correlations  of  air  pollution  across  multiple locations. It integrates meteorological data and traffic information,  where  vehicle  counts  are  extracted  using  the YOLO algorithm and encoded alongside weather variables through a stacked deep autoencoder [9].

Dong et al. proposes a hybrid quantum deep learning method applicable to air quality forecasting. The model integrates the quantum-classical convolutional neural network to learn the spatial  correlations  over  time  and  the  quantum  activation function to address the vanishing gradient and ReLU 'dead neuron' issue and improve the spatial feature learning. Then, the  proposed  model  uses  a  LSTM  network  to  learn  the temporal patterns of the air quality data. Experiments were performed  on  various  datasets  on  the  effectiveness  of  the model on the accuracy of the results. The proposed methodology has shown significant improvements over other methods  on  accuracy  [10].The  Long  short-term  memory (LSTM) network has become the standard architecture for the AQ forecasting task due to its capability to learn effectively the long-term dependencies over sequential data [11]assessed the effectiveness of various AI methods on the Air Quality Index  (AQI)  forecasting  and  found  that  the  results  were significantly accurate with the R-squared of 0.701, RMSE of 0.087, and MAE of 0.056. The researchers found that LSTMs' gating structure was successful at extracting the dependencies over AQ data.

Continuing on the conventional LSTM models, scholars have investigated  more  specialized  variants  with  a  forward  and backward temporal context to be represented Bharathi et al. [12]  implemented  a  BiLSTM  in  a  fog  computing  layer  to predict  AQI  and  reported  superior  forecast  accuracy  of  its model compared to those based on meteorological predictors in terms of MAE and RMSE. On the same note, CNN-BiGRU was  found  to  be  the  most  effective  architecture  to  predict PM2.5 concentrations by Mazinani et al. [13], who incorporated the spatial features retrieval of CNNs with the bi-directional temporal learning of Gated Recurrent Units.

Bhardwaj  et  al.  [14]  conducted  a  comparative  analysis  of LSTM  variants,  including  standard  LSTM,  LSTM  with attention mechanism, and BiLSTM, across different sequence lengths (30 and 365 days). Their findings indicated that LSTM with attention mechanism consistently outperformed other variants, particularly with shorter sequence lengths of 30 days, suggesting that longer sequences might introduce unnecessary complexity or overfitting.  Chen  et  al.  [15]  proposed  an  integrated  dual LSTM  model  combining  Seq2Seq  technology  for  singlefactor prediction with an attention-enhanced LSTM for multifactor  prediction,  integrated  using  XGBoosting  trees  to achieve improved precision over various baseline models.

CNN  have also been widely used to derive spatial information of air  quality  data,  especially  where  there  is  a multi-station  monitoring  network  or  grid-based  pollution map.  The  combination  of  CNN  and  LSTM  has  been especially successful in simultaneous feature of spatial and temporal dependence. Wardana et al. [16] introduced a hybrid 1D CNN-LSTM model that is edge device-optimized, and the results showed that the model was more efficient than other deep  learning  models  when  measured  with  the  RMSE  and MAE metrics. The hybrid model takes advantage of the CNN capability of extracting local patterns and LSTM capability to model the temporal sequence. The Deep-AIR by Zhang et al.  [17]  is  a  hybrid  CNN-LSTM  architecture  with  1x1 convolution layers to estimate air pollution of cities at a finer scale and forecast over stations. This framework obtained the improvement of the accuracy of Hong Kong by 1.5, 2.7, and

24-hour in comparison with Beijing by 1.4, 1.4, and 3.3 at various prediction horizons (1-hour estimation, 1-hour forecast, and 24-hour forecast).Abimannan et al. [18] applied a hybrid CNN-LSTM model in a federated learning system to predict spatiotemporal PM2.5 concentrations 24hr ahead with MAE of 0.466, RMSE of 0.522, and R2 of 0.9877, which is better than SVR, GRU, and BiLSTM.

## III. METHODOLOGY

The proposed method uses a federated learning framework with model compression to predict air quality while protecting  data  privacy  and  reducing  communication  cost. Air quality data collected from distributed IoT sensing nodes are  processed  locally,  and  only  compressed  model  updates are  sent  to  a  central  server.  As  shown  in  Figure  1,  each sensing node trains the model using its own data and shares only  compressed  updates  for  aggregation.  This  approach keeps raw data private and lowers communication overhead. The final global model is then used to predict the air quality index (AQI) for new data, supporting scalable and privacyaware air quality monitoring in smart cities.

Fig.1.  Federated Learning based Air Quality Prediction Architecture with Model Compression.

<!-- image -->

<!-- image -->

## A. Local Data Representation

Let  each  client 𝑘 ∈ {1,2, … , 𝐾} possess  a  local  air  quality dataset. The local data distribution at each client is defined as shown in Eq. (1), where the dataset consists of feature -label pairs representing air pollution observations.

$$
\begin{aligned}
\mathcal { D } _ { k } = \{ ( x _ { i } ^ { k } , y _ { i } ^ { k } ) \} _ { i = 1 } ^ { N _ { k } } \quad ( 1 )
\end{aligned}
$$

where 𝑥𝑖 𝑘 ∈ ℝ 𝑑 represents  input  features  such  as 𝑃𝑀2.5 , 𝑃𝑀10 ,  CO, 𝑁𝑂2 , 𝑆𝑂2 ,  temperature,  and  humidity,  and 𝑦𝑖 𝑘 denotes the corresponding AQI.

## B. Local Model Optimization

Each  federated  client  independently  optimizes  its  local  air quality  prediction  model  by  minimizing  an  empirical  loss function based on locally available data. The local training  objective  is  formally  defined  in  Eq.  (2) , where  the  loss  is  computed  as  the  average  error  over  all samples in the client's dataset. As shown in Eq. (2) , the model parameters 𝑤 are learned by mapping the input features 𝑥𝑖 𝑘 to predicted air quality values using the prediction function 𝑓(⋅) . The loss function ℓ(⋅) in Eq. (2) quantifies the discrepancy between the predicted output and the true label  Air Quality Index 𝑦𝑖 𝑘 , commonly using mean squared error for regression task.

$$
\begin{aligned}
\mathcal { L } _ { k } ( w ) = \frac { 1 } { N _ { k } } \sum _ { i = 1 } ^ { N _ { k } } \ell ( f ( x _ { i } ^ { k } ; w ) , y _ { i } ^ { k } ) \quad ( 2 ) \\ \text {where} \text {presets model parameters } f ( \cdot ) \text {is the prediction}
\end{aligned}
$$

where 𝑤 represents model parameters, 𝑓(⋅) is the prediction function, and ℓ(⋅) denotes the loss function mean squared error.

## C. Gradient-Based Local Update

The parameter update  rule  at  client 𝑘 is  defined  in  Eq.  (3), where  the  parameters  are  adjusted  in  the  direction  of  the negative gradient of the local loss function. As expressed in Eq. (3) , the learning rate 𝜂 controls the step size  of  the  parameter  update  and  influences  convergence speed and stability.

$$
\begin{aligned}
w _ { k } ^ { ( t + 1 ) } & = w _ { k } ^ { ( t ) } - \eta \nabla \mathcal { L } _ { k } \left ( w _ { k } ^ { ( t ) } \right ) \\ \dot { \cdot } & \cdot _ { 1 } \quad 1 \quad \dot { \cdot } \cdot _ { k } \cdot \quad \dot { \cdot } \cdot \cdot \quad \cdot \cdot \
\end{aligned}
$$

where 𝜂 is the learning rate and 𝑡 indicates the communication round.

## D. Model Compression Mechanism

To reduce communication cost, the updated parameters are compressed before transmission: The compression process applied to the locally updated parameters is mathematically defined in Eq. (4).

$$
\widetilde { w } _ { k } ^ { ( t + 1 ) } = \mathcal { C } ( w _ { k } ^ { ( t + 1 ) } )
$$

where 𝒞(⋅) denotes a compression operator such as quantization, sparsification, or low-rank approximation. E.Federated Aggregation at Server

The central server combines the compressed updates using weighted  averaging.  To  reduce  the  communication  cost  in FL, the model updates are compressed before being sent to the central server.

$$
w ^ { ( t + 1 ) } = \sum _ { k = 1 } ^ { K } \frac { N _ { k } } { \sum _ { j = 1 } ^ { K } N _ { j } } \, \widetilde { w } _ { k } ^ { ( t + 1 ) } \quad ( 5 )
$$

The client with larger data set contributes to more to the model predictions. Handing data in FL is complex process the model parameters need to be optimized for efficient communications.

## F. Global Air Quality Prediction

The final global model predicts air quality for a given input 𝑥 as:  After  completing  all  federated  training  rounds,  the aggregated  global  model  is  used  for  air  quality  prediction. The final prediction function is formally expressed in Eq. (6) , where  the  trained  global  parameters  are  applied  to  unseen input data.

$$
\hat { y } = f ( x ; w ^ { ( T ) } ) \quad ( 6 )
$$

where 𝑇 denotes the final federated training round and 𝑦 ̂ is the predicted AQI value or air quality class.

## IV. RESUTLS AND DISCUSSION

## A. Experimental Setup and Dataset Description

To  evaluate  the  effectiveness  of  the  proposed  Federated Learning with Model Compression (FL-CM) framework for air quality prediction, experiments were conducted using an air  quality  dataset  [19].  The  dataset  consists  of  10,000 samples distributed across 10 federated clients, each representing  an  IoT-based  air  quality  monitoring  station. Each sample includes pollutant concentrations ( 𝑃𝑀2.5 , 𝑃𝑀10 , CO, 𝑁𝑂2 ,  SO)  and  meteorological  attributes  (temperature and  humidity),  with  the  corresponding  AQI  as  the  target variable. Figure 2 depict the AQI class. Figure 2 depict the good  AQI  quality  images.  Table  1  present  the  simulation setup for the experiments.

Table 1 Simulation Setup for Federated Learning -Based Air Quality Prediction

Fig.2. Good AQI class images

| Parameter                          | Value                                   |
|------------------------------------|-----------------------------------------|
| Number of Clients ((K))            | 10                                      |
| Target Variable                    | Air Quality Index (AQI)                 |
| Data Distribution                  | Non-IID across clients                  |
| Training - Validation - Test Split | 70% - 15% - 15%                         |
| Local Training Algorithm           | SGD                                     |
| Learning Rate                      | 0.01                                    |
| Local Epochs per Round             | 5                                       |
| Batch Size                         | 32                                      |
| Number of FL Rounds ((T))          | 100                                     |
| Aggregation Method                 | Weighted Federated Averaging            |
| Compression Technique              | Quantization + Sparsification           |
| Baseline Models                    | Traditional ML, Centralized-DL, FL-only |
| Proposed Model                     | FL with Model Compression (FL-CM)       |
| Evaluation Metrics                 | RMSE, MAE, (R^2)                        |
| Communication Metric               | Average Model Update Size (MB)          |

<!-- image -->

## B. Quantitative Performance Analysis

The prediction performance of all evaluated models on the test  dataset  is  summarized  in  Table  3.  Traditional  ML approach exhibits the weakest performance, with the highest RMSE  (18.42)  and  MAE  (14.76)  values  and  the  lowest coefficient of determination ( 𝑅 2 = 0.78 ), indicating limited capability  in  capturing  complex  air  quality  patterns.  The Centralized-DL model significantly improves forecast accuracy, reducing RMSE to 13.65 and MAE to 10.21, while attaining an 𝑅 2 value of 0.86, highlighting the effectiveness of deep learning for air quality prediction when centralized data access is available.

Fig.3.  Performance comparison of Traditional ML, Centralized-DL, FLonly, and Proposed FL-CM models in terms of RMSE, MAE, and R².

<!-- image -->

The FL-only model demonstrates comparable performance to the centralized deep learning approach; however, its slightly higher RMSE (14.12) and MAE (10.89), as reported in Table 1 , suggest minor accuracy degradation due to decentralized training and heterogeneous local data distributions. In contrast,  the  proposed  FL-CM model  achieves  the  best comprehensive  performance,  with  the  lowermost  RMSE 11.08  and  MAE  8.35,  and  the  highest 𝑅 2 value  of  0.91,  as presented in Figure 3.

These  results  confirm  that  integrating  model  compression within  the  federated  learning  framework  not  only  reduces communication overhead but also enhances model generalization and convergence stability. The superior performance of the proposed FL-CM approach, as evidenced in  Table  2,  demonstrates  its  effectiveness  for  accurate, privacy-preserving,  and  scalable  air  quality  prediction  in distributed sensing environments.

TABLE II.  PERFORMANCES OF COMPARISONS OF DIFFERENT MODEL FOR AIR QUALITY PREDICTIONS ON THE TEST DATASET

| Model          |   RMSE ↓ |   MAE ↓ |   R 2 ↑ |
|----------------|----------|---------|---------|
| TraditionalML  |    18.42 |   14.76 |    0.78 |
| Centralized-DL |    13.65 |   10.21 |    0.86 |
| FL-only        |    14.12 |   10.89 |    0.84 |
| Proposed FL-CM |    11.08 |    8.35 |    0.91 |

## C. Communication Efficiency Analysis

The efficiency of the FL method, in terms of communication efficiency, is measured by comparing the average size of the model update in each communication round, which appeared in Table 2. Table 2 indicates that with the FL-only model, an average update of 12 MB will be needed each round, which is significant in terms of communication overhead especially where the bandwidth is limited in IoT and edge computing. Conversely, FL-CM model greatly decreases the update size by  3.4MB  with  a  communication  cutoff  of  71.70%  as compared  to  FL-only.  It  is  evident  in  Table  2  that  model compression is a successful concept when incorporated into the  federated  learning  paradigm.  The  proposed  FL-CM method  reduces  network  bandwidth  requests  significantly with minimal impacts on predictive performance because of sending compact model updates. The efficiency of communication in this manner renders the suggested framework  very  suitable  to  large-scale,  privacy-preserving air quality monitoring frameworks installed on heterogeneous and resource-constrained sensing infrastructures.

TABLE III.  COMMUNICATION EFFICIENCY

| Model          |   Avg. Update Size (MB) | Communication Reduction   |
|----------------|-------------------------|---------------------------|
| FL-only        |                      12 | -                         |
| Proposed FL-CM |                     3.4 | 71.70%                    |

## V. CONCLUSION

This research work proposed an efficient federated learning model based on model compression and communication  efficiency  in  air  quality  prediction  at  the distributed sensing setting. The given FL-CM solution can be considered effective in terms of overcoming the main issues related  to  privacy  preservation,  communication  overhead, and  scalability  with  the  help  of  decentralized  learning  and minimized model updates in monitoring systems for the smart city applications. The experimental findings indicate that the proposed  framework  has  a  high  predictive  performance relative  to  a  classical  centralized  deep  learning,  and  FL models, according to the values of lower RMSE and MAE

and a greater value of R. Besides the accuracy of the predicted value, when combined with model compression, the transmission cost is significantly decreased as well, which is why  the proposed framework  will be quite fitting in bandwidth-restricted and resource-constrained edge computing settings. The findings indicate that communication-efficient federated learning is able to provide strong and consistent predictive quality of air quality without compromising a centralized access to sensitive environmental  information.  In  the  future,  there  can  be  a discussion of adaptive compression techniques, heterogeneous model structure, and real-time implementation in  the  large-scale  smart  city  infrastructure  to  increase  the relevance of the presented approach.

## REFERENCES

- [1] K. T. Putra, H.-C. Chen, P. Prayitno, M. R. Ogiela, C.-L. Chou, C.-E. Weng, and Z.Y. Shae, 'Federated compressed learning edge  computing  framework  with  ensuring  data  privacy  for PM2.5 prediction in smart city sensing applications,' Sensors, vol. 21, no. 13, Art. no. 4586, 2021.
- [2] S. Utomo, A. John, A. Pratap, Z. S. Jiang, P. Karthikeyan, and P.  A.  Hsiung,  'AIX  implementation  in  image -based  PM2.5 estimation: Toward an AI model for better understanding,' in Proc. 15th Int. Conf. Knowledge and Smart Technology (KST), Feb. 2023, pp. 1 -6, doi: 10.1109/KST57286.2023.10086917.
- [3] S. Abimannan, E.-S. M. El-Alfy, S. Hussain, Y.-S. Chang, S. Shukla,  D.  Satheesh,  and  J.  G.  Breslin,  'Towards  federated learning  and  multi-access  edge  computing  for  air  quality monitoring: Literature review and assessment,' Sustainability , vol. 15, no. 18, Art. no. 13951, 2023.
- [4] C.C. Chang, V. Sarveshwaran, K. P., et al., 'TDPM -CNN: A comprehensive daytime and nighttime PM2.5  estimation method  using  multikernel  convolutional  neural  networks,' Earth  Science  Informatics,  vol.  18,  Art.  no.  129,  2025,  doi: 10.1007/s12145-024-01624-9.
- [5] Y. Feng, 'Participant privacy protection and air quality prediction based on FCM, PFI, LSTM, and DBN,' Results in Engineering, vol. 26, Art. no. 105496, 2025.
- [6] F.  M.  A.  Khan,  H.  AbouZeid,  and  S.  A.  Hassan,  'Deep compression for efficient and accelerated over-the-air federated learning,' IEEE Internet of Things Journal , vol. 11, no. 15, pp. 25802 -25817, 2024.
- [7] Y. Liu, J. Nie, X. Li, S. H. Ahmed, W. Y. B. Lim, and C. Miao, 'Federated  learning  in  the  sky:  Aerial -ground  air  quality sensing  framework  with  UAV  swarms,'  IEEE  Internet  of Things Journal, vol. 8, no. 12, pp. 9827 -9837, 2020.
- [8] J. W. Koo, S. W. Wong, G. Selvachandran, H. V. Long, and L. H.  Son,  'Prediction  of  air  pollution  index  in  Kuala  Lumpur using  fuzzy  time  series  and  statistical  models,'  Air  Quality, Atmosphere &amp; Health, vol. 13, no. 1, pp. 77 -88, 2020.
- [9] A.  M.  Mengara,  A.  Gedeon,  E.  Park,  J.  Jang,  and  Y.  Yoo, 'Attention -based distributed deep learning model for air quality forecasting,' Sustainability, vol. 14, no. 6, Art. no. 3269, 2022.
- [10] Y. Dong, F. Li, T. Zhu, and R. Yan, 'Air quality prediction based on quantum activation function optimized hybrid quantum -classical neural network,' Frontiers in Physics, vol. 12, Art. no. 1412664, 2024.
- [11] L. Andrade  et al., "A  comprehensive evaluation of ai techniques for air quality index prediction: RNNs  and transformers,"  Ingenius:  Revista  de  Ciencia  y  Tecnología, 2025. DOI: 10.17163/ings.n33.2025.06
- [12] B. Bharathi et al., "Fog  computing enabled air quality monitoring  and  prediction  leveraging  deep  learning  in  IoT," Journal of Intelligent and Fuzzy Systems, 2022. DOI: 10.3233/jifs-212713
- [13] S. M. Mazinani et al., "Air Quality Prediction via Embedded ML/DL  and  Quantized  Models,"  IEEE  Access,  2025.  DOI: 10.1109/access.2025.3603920
- [14] R. Bhardwaj et al., "A Deep Learning Approach to Enhance Air Quality  Prediction:  Comparative  Analysis  of  LSTM,  LSTM with  Attention  Mechanism  and  BiLSTM,"  in  2017  IEEE

- Region 10 Symposium (TENSYMP), 2024. DOI: 10.1109/tensymp61132.2024.10752321
- [15] Y.  Chen  et  al.,  "Air  Quality  Prediction  Based  on  Integrated Dual LSTM Model," IEEE Access, 2021. DOI: 10.1109/ACCESS.2021.3093430
- [16] A. P. Wardana et al., "Optimising deep learning at the edge for accurate hourly air quality prediction," Sensors, vol. 21, no. 4, 2021. DOI: 10.3390/S21041064
- [17] Y. Zhang et al., "Deep-AIR: A hybrid CNN-LSTM framework for fine-grained air pollution estimation and forecast in metropolitan cities," IEEE Access, 2022. DOI: 10.1109/ACCESS.2022.3174853
- [18] S. Abimannan  et al., "Spatiotemporal particulate matter pollution prediction using cloud-edge intelligence," Communications in Computer and Information Science, 2023. DOI: 10.1007/978-981-99-8145-8\_8
- [19] A. Rouniyar, S. Utomo, J. A., and P.A. Hsiung, 'Air pollution image dataset from India and Nepal,' Kaggle, 2023. [Online]. Available: https://www.kaggle.com/ds/3152196