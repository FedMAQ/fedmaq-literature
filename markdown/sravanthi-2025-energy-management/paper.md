## Federated Learning-Based Energy Management Systems for Privacy-Preserving Demand Forecasting in Smart Cities

K. Sravanthi Department of Electrical &amp; Electronics Engineering Vignan's Institute of information Technology, Visakhapatnam, Andhra Pradesh-530049, Vanshika Valikar UG Student Automation and Robotics KLE Technological University, Hubli, Karnataka

Syed Mohammad Ghouse School of Management, Cmr university, Bangalore, Karnataka A Mutharasan Department of Electronics and Communication Engineering Vel Tech Rangarajan Dr Sagunthala R&amp;D Institute of Science and Technology Chennai, India

Abstract -When it  comes  to  smart  cities,  efficient  energy management systems (EMS) that respect privacy are essential for balancing electricity demand and user data confidentiality. As  an  example of  an  EMS  in this  context,  we  present  here  a Federated Learning (FL)-based model using data from diverse urban  zones  to  forecast  demand  in  complete  privacy.  Using actual  consumption  data  from  the  Residential,  Commercial and Industrial sectors achieved system performance was compared with that of centralized deep learning networks. The FL-GRU  model  proposed  here  achieved  a  forecast  accuracy close  to  that  of  a  centralized  LSTM  counterpart.  This  new model  has  a  maximum  MAE  of  2.04  kW  and  RMSE  of  2.37 kW, while still  reducing  privacy  leaks  by  90%.  Although  the communication  overhead  for  FL  models  was  greater,  the models  proved  highly  scalable  and  converged  steadily  across the board at 48 to 57 training epochs. Furthermore the system remained  robust  under  data  heterogeneity,  retaining  86% accuracy even when non-IID. A comprehensive visual examination of the FL approach, including bar graphs, radar charts  and  area  plots,  shows  its  ability  to  balance  prediction accuracy with convergence efficiency and user data protection. These results vindicate FL's potential as an effective conclusion for scalable, secure and intelligent energy management systems throughout smart grid infrastructures - and open the path to practical application in real-world urban environments.

Keywords-Federated Learning (FL), Smart Grid, Energy  Management  System  (EMS),  Demand  Forecasting, Privacy Preservation

## I. INTRODUCTION

The arrival  of  renewable  energy  sources,  electric  vehicles, and  smart  appliances  into  modern  power  grids  has  greatly heightened the need for more intelligent and nimble Energy Management  Systems  (EMS).  The  development  of  urban areas  into  complex  cyber-physical  systems  has  brought  a daunting scalability challenge and increasing difficulties for traditional  centralized  prediction  and  control  systems  in terms of latency and user data security [1-3] .In this context, Federated Learning (FL) has emerged  as a promising alternative  in  which  model  training  is  done  collaboratively and on the edge itself. Thus it avoids sending raw data from client to client.[4-6].

Accurately forecasting power consumption is a key to the operation  of  smart  grids.  It  allows  the  utility  company  to optimize power generation sources, downsize peak times and implement demand-response strategies for its customers [7- S. Sri Nandhini Kowsalya, Department of Information Technology, Vel Tech Multi Tech Dr. Rangarajan Dr. Sakunthala Engineering College, Chennai 600062, Tamilnadu, India Ashok Vajravelu Faculty of Electrical &amp; Electronics Engineering Universiti tun Hussein onn Malaysia

9].  However,  when  large  amounts  of  data  from  individual customers  are  centralized  in  one  location,  the  privacy  risk becomes  very  high, particularly as regulations such as European Union General Data Protection Regulation (GDPR) strictly define how data can be used [10]. It tackles the  problem  that  arises  from  integrating  FL  and  energy management systems in  the  smart  city.  Federated  Learning (FL)  keeps  the  data  local  at  the  edge  so  as  to  achieve intelligence that is private but globally accurate in forecasting[11-14].

This  has  technical  challenges,  however,  for  FL  when applied  to  EMS  in  smart  city  environments.  These  include non-independent identically distributed (non-IID) data distributions across urban zones (e.g., residential vs industrial). There is also the communication overhead due to iterative  model  updates,  and  a  trade-off  between  model performance and privacy [15-16]. Recent works have tried to apply  FL  to  power  systems,but  few  have  looked  at  its performance systematically in terms of zone-specific heterogeneity;  privacy  leakage,  and  efficiency  in  training across an end-to-end EMS pipeline[17-18].

In  this  paper,  we  propose  a  Federated  Learning-based Energy Management System (FL-EMS) for privacypreserving demand  forecasting within smart cities.  Our approach leverages a distributed architecture  (FL-GRU) based on gated recurrent units and a data set that represents interactions between clients and servers. Each client interacts with  both  residential,  commercial,  and  industrial  zones  for input on  how the  system behaves in  those  settings [19-20]. We  compare  the  results  of  this  approach  to  those  from  a centralized  deep  learning  model  (CDL-LSTM)  in  terms  of forecasting accuracy, convergence behavior, communication efficiency and privacy leakage. We also perform stress tests under different levels of data heterogeneity, giving a comprehensive analysis of the method's robustness, scalability,  and  ability  to  preserve  privacy[21].  The  main contributions of this paper are as follows.

## The main contributions of this work are as follows:

1. We  present  a  full-scale  decentralized  framework for  FL-EMS  using  a  set  of  local  GRU-based models that are harmonized via Federated Averaging.

2. We do  a  performance  evaluation  of  our  FL-EMS alongside centralized models using key metrics like MAE, RMSE, convergence epochs and communication overhead.
3. We  quantitatively  evaluate  privacy  leakage  rates (PLRs) and model adaptability under both IID and non-IID data.
4. We also  visualize  the  results  through  stacked  bar charts, radar graphs, and area plots so as to support decision making for operators of smart city grids.

## II. METHODOLOGY

Now we will talk about the course frame being applied to build and test the Federated Learning-based Energy Management  Systems  (FL-EMS)  that  were  developed  for privacy-preserving  demand  forecasting  within  smart  cities. This method consists of six basic components:

## 2.1 Federated Learning Framework

The  system  uses  Federated  Averaging  (FedAvg)  to  enable distributed clients (e.g., smart meters, edge devices) to train models locally without sharing raw data. Aggregated model updates are securely sent to a central server for global model refinement,  ensuring  that  privacy  is  built  into  it  from  the start.

## 2.2 Forecasting Performance Evaluation

Performance assessment of predictive accuracy is conducted by  the  system,  whereby  the  Mean  Absolute  Error  (MAE) and Root Mean Square Error (RMSE) are calculated across different urban areasacross different urban zones --residential, commercial, industrial. These metrics are used to compare the FL model to a centralized basis.

## 2.3 Communication Overhead Analysis

Communication  overheads  are  analysed  by  counting  the volume  of  data  exchanged  during  model  updates  with different numbers of clients. This assessment will make sure that  the  FL  framework  remains  scalable  and  efficient,no matter how many devices are being used.

## 2.4 Privacy Leakage Assessment

It  should be noted that the implementation level of privacy depends  on  the  percentage  of  user  data  which  could  be reconstructed  from  shared  model  gradients.  This  leakage rate  gives  a  clear  picture  as  to  how  resistant  the  overall system is against adversarial attack.

## 2.5 Model Convergence and Training Efficiency

Alternately, the method  follows  the training efficiency through  convergence  epochs  and  total  time  to  train  in addition to tracking final model loss. These measures allow appraisals  of  whether  FL  models  can  attain  both  stability and central approaches effectively.

## 2.6 Impact of Data Heterogeneity

The system is evaluated at different levels of data heterogeneity, from iid ideal to highly non-IID. The framework's performance in terms of accuracy, stability and reactivity is monitored here, which validates how adaptable it is for practical applications.

## III. SYSTEM MODELING AND FORMULATION

This section presents a set of mathematical representations used  in  the  construction  of  Federated  Learn-Based  Energy Management  System  (FL-EMS),  which  can  be  applied  to privacy-respecting  demand  prediction  for  intelligent  cities.

Our  aim  is  to  optimize  the  forecasting  accuracy  of  the model, while protecting privacy, lowering training costs and reducing communication overheads as much as possible in a series of multi-objective optimization problems. All indicators  must  be  defined  in  accordance  with  which  ones are  related  or  required  to  calculate  the  product  their  result becomes.

## 3.1 Objective Function

The  primary  goal  is  to  minimize  forecast  error  as  well  as leakage of privacy and training load. The unified optimization problem is defined as follows:

$$
\min _ { \theta } [ \alpha _ { 1 } \cdot M A ( \theta ) + \alpha _ { 2 } \cdot R M S ( \theta ) + \alpha _ { 3 } \cdot P L R ( \theta ) + \alpha _ { 4 } \cdot T ( \theta ) - \alpha _ { 5 } \cdot A ( \theta ) ]
$$

## Where:

- : Global model parameters (weights)
- : Weighting coefficients for each objective term C.......5
- :  Mean  Absolute  Error  (forecast  error metric) MAE(e)
- : Root  Mean  Square  Error  (forecast deviation metric)
- : Privacy Leakage Rate (in %)
- : Total training time or convergence epochs
- : Forecasting accuracy of the federated model (%)

## 3.2 Forecasting Error Metrics

Two standard performance metrics are used to evaluate the forecasting performance:

Mean Absolute Error (MAE):

$$
M A E = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } | y _ { i } - \hat { y } _ { i } |
$$

- : Actual demand at time step
- : Predicted demand at time step
- : Total number of data points

Root Mean Square Error (RMSE):

<!-- image -->

- All variables as previously defined

## 3.3 FL Model Update: Federated Averaging (FedAvg)

The global model in FL is updated by averaging together the local models produced at each client's end:

<!-- image -->

## Where:

- : Global model weights at round
- : Weights from client at round
- : Number of local samples on client
- : Total number of samples E=1 n=
- : Total number of clients

## 3.4 Privacy Metric: Privacy Leakage Rate (PLR)

To quantify data exposure risks from gradients, we define:

$$
P L R = \frac { N _ { r } } { N } \times 1 0 0
$$

## Where:

- :  Number  of  successfully  reconstructed  inputs via gradient inversion
- : Total input samples shared in training
- PLR is expressed as a percentage

## 3.5 Model Convergence Criterion

It is possible to regard the model as having converged when the change of loss over multiple epochs becomes sufficiently small:

$$
| \mathcal { L } ^ { ( t ) } - \mathcal { L } ^ { ( t - 1 ) } | < \epsilon \ \text { for } E \text { cons}
$$

## Where:

- : Loss at epoch
- : Convergence threshold (e.g., 0.001)
- : Minimum number of stable epochs (e.g., 5)

## 3.6 Forecasting Accuracy of FL Model

We quantify accuracy by comparison of the mean absolute error of our model to the mean value of demand:

$$
A ( \theta ) = \left ( 1 - \frac { M A E } { \bar { \nu } } \right ) \times 1 0 0
$$

## Where:

- : Mean actual demand in the test set

## 3.7 Normalization for Multi-Metric Visualization

In  order  to  fairly  compare  those  different  metrics,  we  use min-max normalization in radar and area charts:

<!-- image -->

## Where:

- : Actual value
- :  Minimum and maximum values across the dataset min.X

## 3.8 Optimization Constraints

The  optimization  is  subject  to  the  following  operational constraints:

## Communication Budget:

$$
C _ { k } \leq C _ { \max } \ \forall k \in \{ 1 , \dots , 1 \}
$$

- : Communication overhead from client
- :  Maximum  allowable  communication  per client Gmax

## Client Participation Guarantee:

<!-- image -->

- : Indicator function (1 if client participates) Lactive (k)
- : Minimum number of active clients per round

## Convergence Time Limit:

## T(O) ≤ T

- :  Upper bound on training time or number of epochs max

The  unified  mathematical  framework  allows  a  systematic design, simulation and optimization of intelligent mini grid energy  forecasting  system  that  respects  privacy,  has  high accuracy  built  into  the  model  is  computationally  efficient and  can  be  deployed  across  decentralized  infrastructures serving smart grids.

## IV. RESULTS AND DISCUSSIONS

This  chapter  first  provides  an  in-depth  analysis  of  the performance  and  effectiveness  of  this  proposed  Federated Learning (FL)-based Energy Management System (EMS) in smart  cities.  The  Federated  Learning  model  was  deployed on 10 clients in each zoning area, each client representing a single  smart  meter.  The  FederationWithModelAveraging0 model is used as a comparison model, as well. Originating datasets come from three zones: Residential (RZ), Commercial (CZ), and Industrial (IZ). The baseline models for  comparison  include  Centralized  Deep  Learning  (CDL), Linear  Regression  (LR),  and  Long  Short-Term  Memory (LSTM) models trained in a centralized setting. All models were evaluated on key metrics: Mean  Absolute  Error (MAE), Root Mean Square Error (RMSE), privacy leakage rate (PLR), convergence time, and communication overhead.

## 4.1  Performance  Comparison  of  FL  vs  Centralized Approaches

The main objective is to examine the accuracy of forecasting  while  preserving  user  data  privacy.  Table  1 shows results of demand forecasting in all zones.

Table 1: Forecasting Accuracy Comparison Across Models and Zones

| Mode l   | Zon e   | MA E (kW )   | RMS E (kW)   | PL R (%)   | Trainin g Time (s)   | Accurac y (%)   |
|----------|---------|--------------|--------------|------------|----------------------|-----------------|

| Mode l      | Zon e   |   MA E (kW ) |   RMS E (kW) | PL R (%)   |   Trainin g Time (s) |   Accurac y (%) |
|-------------|---------|--------------|--------------|------------|----------------------|-----------------|
| FL- GRU     | RZ      |         0.92 |         1.18 | 2.1        |                  105 |            95.2 |
| FL- GRU     | CZ      |         1.34 |         1.66 | 2.6        |                  121 |            93.5 |
| FL- GRU     | IZ      |         2.04 |         2.37 | 3.2        |                  147 |            91.1 |
| CDL- LST M  | RZ      |         0.87 |         1.12 | 38. 4      |                   89 |            95.8 |
| CDL- LST M  | CZ      |         1.29 |         1.58 | 37. 9      |                   94 |            94.2 |
| CDL- LST M  | IZ      |         1.98 |         2.31 | 39. 6      |                  112 |            91.5 |
| Linea r Reg | All     |         3.71 |         4.29 | 0.0        |                   21 |            72.3 |

The FL  model actually outperforms the centralized LSTM  model  marginally  in  terms  of  accuracy.  The  FL model,  however,  drops  privacy  leakage  rates.  This  is  a demonstration of how FL can preserve privacy.

Figure 1: Forecasting Accuracy Comparison Across Models and Urban Zones

<!-- image -->

## 4.2 Communication Overhead and Scalability Analysis

Communication costs loom large in FL deployment. Table 2 gives per-round communication overhead (in KB) and total communication time.

Table 2: Communication Overhead Comparison Between Centralized and Federated Models

| Mod el   |   Clients |   Communicat ion per Round (KB) |   Total Communicat ion Time (s) |   Total Roun ds |
|----------|-----------|---------------------------------|---------------------------------|-----------------|
| FL- GRU  |        10 |                           1,560 |                            18.2 |              50 |
| FL- GRU  |        20 |                           3,050 |                            34.1 |              50 |
| FL- GRU  |        50 |                           7,460 |                            78.6 |              50 |

| Mod el      | Clients          | Communicat ion per Round (KB)   |   Total Communicat ion Time (s) |   Total Roun ds |
|-------------|------------------|---------------------------------|---------------------------------|-----------------|
| CDL - LST M | 1 (centraliz ed) | N/A                             |                             6.3 |               1 |

While on the one hand FL is more expensive to communicate,  its  distributed  nature  affords  both  parallel model  training  and scalability-it is all a question of balancing delay costs with the benefits.

Figure 2: Communication Overhead Distribution in Federated Learning Setups

<!-- image -->

## 4.3 Privacy Preservation and Data Leakage Analysis

To quantify privacy preservation, we carried out an adversarial  reconstruction  attack  simulation.  The  privacy leakage rate (PLR) was calculated as the average fraction of replicable input data to gradients.

Table 3: Privacy Leakage Rate Comparison

| Model     | Zone   |   PLR (%) |   Adversarial Reconstruction Accuracy (%) |
|-----------|--------|-----------|-------------------------------------------|
| FL- GRU   | RZ     |       2.1 |                                       3.4 |
| FL- GRU   | CZ     |       2.6 |                                       3.8 |
| FL- GRU   | IZ     |       3.2 |                                       4.1 |
| CDL- LSTM | RZ     |      38.4 |                                      41.9 |
| CDL- LSTM | CZ     |      37.9 |                                      42.3 |
| CDL- LSTM | IZ     |      39.6 |                                      43.1 |

Reconstructive potential  was  greatly  diminished  by  FLbased  EMS,  protecting  consumer  usage  patterns.  In  smart city  applications  where  sensitive  commercial  or  household data is involved, this is paramount.

Figure 3: Privacy Leakage Rate Comparison by Model and Zone

<!-- image -->

1) 4.4 Model Convergence and Computational Efficiency

Convergence  was  measured  in  terms  of  the  number  of training epochs required before loss stabilized (loss change 0.001 for 5 epochs). Table 4 gives convergence figures.

Table 4: Model Convergence Comparison

| Model     | Zone   |   Epochs to Converge |   Total Time (s) |   Final Loss |
|-----------|--------|----------------------|------------------|--------------|
| FL- GRU   | RZ     |                   48 |              105 |       0.0184 |
| FL- GRU   | CZ     |                   52 |              121 |       0.0211 |
| FL- GRU   | IZ     |                   57 |              147 |       0.0293 |
| CDL- LSTM | RZ     |                   43 |               89 |       0.0157 |
| CDL- LSTM | CZ     |                   47 |               94 |       0.0189 |
| CDL- LSTM | IZ     |                   53 |              112 |       0.0264 |

FL  models  converged  at  a  reasonable  pace  in  terms  of distributed synchronization. But future work might consider asynchronous FL variants in order to reduce delay further.

Figure 4: Model Convergence Metrics Across Zones 4.5 Robustness Against Data Heterogeneity

<!-- image -->

Data  is  never  uniform  throughout  smart  cities,  as  energy usage patterns vary enormously. We tested for the robustness of three degrees non-IID data with heterogeneity, low moderate and high and provided an evaluation.

Table 5: Impact of Data Heterogeneity on Forecasting Performance

| Heterogeneit y   |   MA E (kW) |   RMS E (kW) |   FL Accurac y (%) |   Convergenc e Epochs |
|------------------|-------------|--------------|--------------------|-----------------------|
| Low (IID)        |        0.92 |         1.18 |               95.2 |                    48 |
| Moderate         |        1.37 |         1.71 |               91.3 |                    55 |
| High (non- IID)  |        2.09 |         2.45 |               86.4 |                    63 |

FL  preserved  functional  accuracy  even  under  substantial heterogeneity,  demonstrating  adaptability.  But  under  high non-IID conditions its performance fell also slightly

Impact of Data Heterogeneity on Forecasting Performance (Area Graph)

Figure 5: Impact of Data Heterogeneity on Forecasting Performance

<!-- image -->

## Summary of Findings

The FL-based EMS architecture has the following features:

- Forecasting accuracy  comparable  to  centralized models while cutting privacy leakage by over 90%.
- Reasonable  torsioning  rates  given  the  increased cost of communication.
- Scalability  to  50+  clients  versus  10  in  the  initial model, with acceptable delay.

Robustness when given non-IID consumption patterns, so it is  applicable  in  most  sectors  of  Smart  City  development even now These results illustrate how FL can act as a power for privacy preserving EMS systems that are both accurate and can be widely distributed in a smart city setting.

## V. CONCLUSIONS

In  this  article  we  present  a  form  of  privacy-preserving demand prediction for intelligent cities called the Federated Learning-based  Energy  Management  System  (FL-EMS). The FL-GRU model showed strong generalization ability in Residential, Commercial, and Industrial areas, with prediction  accuracy  that  was  as  good  as  many  centralized deep learning models but leakage rates of privacy significantly  reduced.  FL  introduced  some  overhead  in communication  but  it  achieved  better  scalability  as  the tradeoff-I.e.,  through  it  model  supports  distributed  training of  many clients. Only a grogram of note can be written in two sentences. The convergence analysis indicated that FL models took a few more epochs in order to settle; however they did have good final loss figures, which suggests stable training behavior. The FL program was still effective under different  data  heterogeneity  settings;  even  when  the  data was not Independent and identically distributed, prediction accuracy remained above 86%. This shows that our model can handle real world urban energy datasets. The results are reported  in  charts  to  allow  a  visual  comparison  between three  different  metrics:  forecasting  accuracy,  number  of iterations required for convergence and information disclosure. She drew six charts altogether; one chart group of  three,  one  radar  chart  and  four  area  charts.  The  results show that FL offers a good compromise between accuracy and efficiency of convergence And its focus on user privacy. This  lends  support  to  FL  as  a  secure  method  of  energy management  in  smart  grid  applications,  where  it  can  take advantage of large-scale decentralized training while guarding sensitive user data.

## REFERENCES

- [1] Gurugubelli, V., Ghosh, A. and Panda, A.K., 2024. Improved Hopf Oscillator-based  VOC Method for Fast Synchronization of Parallel Inverters  in  Standalone  Microgrid. IEEE Journal of  Emerging  and Selected Topics in Power Electronics .
- [2] McMahan, B., Moore, E., Ramage, D., Hampson, S., &amp; y Arcas, B. A. (2017, April). Communication-efficient learning of deep networks  from  decentralized  data.  In  Artificial  intelligence  and statistics (pp. 1273-1282). PMLR.
- [3] Dora,  S.,  Gurugubelli,  V.,  Ghosh,  A.  and  Panda,  A.K.,  2022, November. Parallel operation of inverters by using Model Predictive Control  in  Islanded  Microgrid.  In 2022  IEEE  10th  Power  India International Conference (PIICON) (pp. 1-6). IEEE.
- [4] Yang, Q., Liu, Y., Chen, T., &amp; Tong, Y. (2019). Federated machine learning: Concept and applications. ACM Transactions on Intelligent Systems and Technology (TIST), 10(2), 1-19.
- [5] Gurugubelli, V., Ghosh, A., Panda, A.K., Ray, P.K. and Sarkar, I., 2023, March. Synchronization of Single-Phase Inverters using Deadzone  and  Hopf  Oscillator  based  Controllers  in  Standalone Microgrid.  In 2023  IEEE  IAS  Global  Conference  on  Renewable Energy and Hydrogen Technologies (GlobConHT) (pp. 1-6). IEEE.
- [6] Li,  H.,  Ota,  K.,  &amp;  Dong,  M.  (2018).  Learning  IoT  in  edge:  Deep learning  for  the  Internet  of  Things  with  edge  computing.  IEEE network, 32(1), 96-101.
- [7] Mahapatra,  P.K.,  Gudla,  S.K.,  Gurugubelli,  V.,  Ghosh,  A.  and Panda,  A.K.,  2022,  November.  Fuzzy  Adaptive  Droop  Controlled Parallel  Inverters  for  Microgrid  Applications.  In 2022  IEEE  10th Power India International Conference (PIICON) (pp. 1-6). IEEE.
- [8] R. S. S. Nuvvula et al ., "Federated Learning-Based Energy Forecasting  and  Trading  Platform  for  Decentralized  Renewable Energy  Markets," 2024  12th  International  Conference  on  Smart Grid (icSmartGrid) , Setubal, Portugal, 2024, pp. 277-283.
- [9] SLOUMA, Safa, Wael BOULARES, Souheil EL ALIMI, Abdelmajid  JEMNI,  and  Somnath  Maity.  "Optimal  Design  and Control of stand-alone photovoltaic system and analyzing its environmental and techno-economic aspects in Tunisia: A case study of Borj Cedria." International Journal of Renewable Energy Research (IJRER) 14, no. 4 (2024): 890-901.
- [10] Palanisamy, Rajakumar, Sibbala Bhargava Reddy, M. Senthil Kumar,  A.  Sakthidasan,  and  P.  Baburao.  "Optimal  Integration  of Multiple  Renewable  Energy  Distributed  Generations  using  Hybrid Optimization Technique." International Journal of Renewable Energy Research (IJRER) 14, no. 3 (2024): 491-502.
- [11] Varaprasad, Madisa VG, Ramakrishna SS Nuvvula, Polamarasetty P. Kumar, Neyara Radwan, C. Dhanamjayulu, Mohammed Rafi Shaik, and Baseem Khan. "Author Correction: Design and implementation
12. of  single  DC-link  based  three-phase  multilevel  inverter  with  CBPWM techniques." Scientific Reports 14 (2024): 21901.
- [12] Rao,  Nartu  Tejeswara,  Kalyana  Kiran  Kumar,  Polamarasetty  P. Kumar, Ramakrishna SS Nuvvula, A. Mutharasan, C. Dhanamjayulu, Mohammed Rafi Shaik, and Baseem Khan. "Multiobjective optimal TCSC placement using multiobjective grey wolf  optimizer  for  power  losses  reduction."  Scientific  Reports  14, no. 1 (2024): 21857
- [13] Sun,  Jiulong,  and  Xiao  Guo.  "Cooperative  scheduling  of  sourceload-storage for microgrids  with  electric springs." International Journal  of  Renewable  Energy  Research  (IJRER)  14,  no.  3  (2024): 575-586.
- [14] El  Alami,  Yassine,  Ali  Lamkaddem,  Houssam  Amiry,  Rachid Bendaoud, Fatima Chanaa, Said Bounouar, Said Dlimi et al. "Design and realization of a descretized PV system with an improved MPPT control  for  a  better  exploitation  of  the  PV  energy."  International Journal  of  Renewable  Energy  Research  (IJRER)  14,  no.  2  (2024): 224-237.
- [15] Swaminathan, B., S. Selvi, R. Jothilakshmi, D. Kirubakaran, and A. Rajaram. "Performance optimization of an interleaved boost converter with water cycle optimized PO algorithm-based MPPT for the  applications of solar-powered E-vehicles." International Journal of Renewable Energy Research (IJRER) 14, no. 2 (2024): 248-260.
- [16] D. Miyao and M. Nakamura, "Time and Day-Based Peak Electricity Demand Forecasting: A Comparative Analysis of Machine Learning Models for Peak Cut and Decarbonization," 2024 13th International Conference on Renewable Energy Research and Applications (ICRERA), Nagasaki, Japan, 2024, pp. 1429-1433, doi: 10.1109/ICRERA62673.2024.10815184.
- [17] P.  -E.  Lai,  S.  -C.  Chang  and  Y.  -S.  Chang,  "Implementation  of Heating  System  for  Lithium-Ion  Batteries  in  Low  Temperature Environments,"  2024  13th  International  Conference  on  Renewable Energy  Research  and  Applications  (ICRERA),  Nagasaki,  Japan, 2024, pp. 1363-1368, doi: 10.1109/ICRERA62673.2024.10815171.
- [18] Kotte,  Sowjanya,  Satish  Kumar  Injeti,  Vinod  Kumar  Thunuguntla, Polamarasetty P. Kumar, Ramakrishna SS Nuvvula, C. Dhanamjayulu,  Mostafizur  Rahaman,  and  Baseem  Khan.  "Energy curve  based  enhanced smell  agent optimizer  for optimal multilevel threshold  selection  of  thermographic  breast  image  segmentation." Scientific Reports 14, no. 1 (2024): 21833.
- [19] Varaprasad, Madisa VG, Ramakrishna SS Nuvvula, Polamarasetty P. Kumar, Neyara Radwan, C. Dhanamjayulu, Mohammed Rafi Shaik, and  Baseem  Khan.  "Design  and  implementation  of  single  DC-link based  three-phase  multilevel  inverter  with  CB-PWM  techniques." Scientific Reports 14, no. 1 (2024): 18078.
- [20] Y. Hou, J. Zhao, S. Zhu, H. Gu, C. Pan and Y. Zhang, "Analysis of DC  Building  Flexible  Capacity  Optimization  Configuration  Based on PEDF  Microgrid," 2024 13th International Conference on Renewable Energy Research and Applications (ICRERA), Nagasaki, Japan, 2024, pp. 661-666, doi: 10.1109/ICRERA62673.2024.10815217.
- [21] R. Takahashi, K. Abe, H. Takami and F. Ishibashi, "A Proposal of Optimal  Current  Control  for  Three-Phase  Grid-Connected  Inverter with  LCL-Filter  via  IRM-ILQ  Method,"  2024  13th  International Conference on Renewable Energy Research and Applications (ICRERA), Nagasaki, Japan, 2024, pp. 369-374, doi: 10.1109/ICRERA62673.2024.10815263.