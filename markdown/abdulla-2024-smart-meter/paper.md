<!-- image -->

Contents lists available at ScienceDirect

## Sustainable Energy, Grids and Networks

journal homepage: www.elsevier.com/locate/segan

## Smart meter-based energy consumption forecasting for smart cities using adaptive federated learning

Nawaf Abdulla a, ∗ , Mehmet Demirci b,a , Suat Ozdemir c

- a Department of Information Systems, Graduate School of Informatics, Gazi University, Ankara, Türkiye
- b Department of Computer Engineering, Faculty of Engineering, Gazi University, Ankara, Türkiye
- c Department of Computer Engineering, Faculty of Engineering, Hacettepe University, Ankara, Türkiye

## A R T I C L E I N F O

Keywords: Energy consumption forecasting Federated learning Adaptive learning LSTM IoT Smart grid

## 1. Introduction

Increasing demand and scarce energy resources require smart city applications to predict how much power an entire city or a single building will need. To accomplish this, we must first predict the short-term energy consumption (load) of a single household in a building and then the building's total energy consumption [1]. Consequently, it is possible to aggregate and forecast the demand for smart cities [2]. Short-term residential energy consumption forecasting stems from the decentralization of renewable energy generation, such as photovoltaic systems, which allow households to generate their energy and maximize its use to become increasingly self-sufficient [3].

Many studies in the literature have been conducted to forecast the energy consumption of households using artificial intelligence (AI) techniques and approaches [4]. For accurate model training, however, most AI models require vast quantities of historical data, which must be collected and stored in a centralized architecture [5]. Now, the Advanced Metering Infrastructure (AMI) facilitates this process. Installation of smart meters in customer homes is critical to the AMI, which allows for the reliable data collection on load consumption at intervals of no less than 15 min [6].

∗

## A B S T R A C T

Forecasting short-term residential energy consumption is critical in modern decentralized power systems. Deep learning-based prediction methods that can handle the high variability of residential electrical loads have made models more accurate. On the other hand, these methods need a lot of sensitive information about how much people use something gathered centrally to train a forecasting model. This is not good for privacy and scalability. Moreover, models may become less accurate over time due to changing conditions. In this work, we propose a framework for energy consumption forecasting that exploits adaptive learning, federated learning, and edge computing concepts. A central server aggregates numerous long short-term memory (LSTM) models that users at various locations train with their energy consumption data to create a generalized model that uses adaptive learning to detect data drifts and enhance forecasting at the edge layer. Our findings show that adaptive federated learning performs better than centralized learning while preserving privacy, reducing communication overhead, lowering the forecast error rate by 8%, and decreasing the training time by approximately 80%.

Regulatory agencies and others have raised privacy issues due to technological advancements that have introduced new choices for endusers and energy providers [7]. The central theme of these concerns revolves around using user data to uncover client behavior. In certain regions, privacy concerns may lead consumers to decline the usage of smart meters [8]. Numerous privacy-preserving technologies employ data aggregation and encryption techniques to mitigate the privacy above concerns. These solutions aim to balance safeguarding user privacy and enabling energy suppliers to access data. Integrating privacy-preserving approaches with residential short-term load forecasting systems creates a paradox. The latter employs granular household data, which is incompatible with privacy-preserving aggregated or masked data formats. The present state of AI-driven solutions results in an escalation in computational overhead for model training [6]. The significant processing intensity poses a limitation when expanding the system to accommodate millions of smart meters, a steadily growing need. It is possible to train AI models on smaller data portions to alleviate computational limitations. However, this approach diminishes their ability to generalize well across the entire data set.

Edge computing is a paradigm that decentralizes computational processes, bringing them closer to end users. Integrating cloud computing at the network edge enhances various aspects such as speed, efficiency, dependability, privacy, security, and scalability. It is ideal for energy consumption forecasting due to its privacy and scalability requirements [9]. Edge computing, similar to other architectural models, exhibits certain limitations. AI and machine learning models achieve optimal convergence when all training data from different users are processed centrally. However, edge computing models train locally to ensure client confidentiality, reducing their ability to generalize [1].

Corresponding author. E-mail addresses: nawaf.abdulla@gazi.edu.tr (N. Abdulla), mdemirci@gazi.edu.tr (M. Demirci), ozdemir@cs.hacettepe.edu.tr (S. Ozdemir).

<!-- image -->

<!-- image -->

On the other hand, in recent years, federated learning (FL) has garnered significant interest among researchers due to its capacity to integrate diverse data sources and facilitate the collaborative and privacy-preserving construction of machine learning models [10]. Thus, FL enables multiple parties to build a model collaboratively without disclosing data [11]. Additionally, it refers to the machine learning strategy that allows decentralized model training across massively separate data sources without privacy concerns [12]. These private data sets of many dispersed clients and institutions come from various sources, including smartphones and IoT devices [13]. Furthermore, the ability to generalize both machine learning and deep learning methods is frequently contingent on training on diverse data sets, such as data from multiple medical institutions, which is a restrictive requirement given the sensitive nature of medical data [14]. Additionally, the implementation of this approach has demonstrated success in various application fields [15], including human-computer interaction [16], language modeling [17], transportation [18], and Industry 4.0 [19], where privacy and scalability are essential.

Data undergoes temporal changes in many complex data analysis scenarios and necessitates near real-time analysis. The patterns and relationships in dynamic data often change over time, rendering models used for analysis obsolete. In machine learning and data mining, this occurrence is referred to as ''concept drift''. Specific domains, such as time series forecasting and predictions on streaming data, involve ordering predictions by time. In these domains, it is essential to explicitly test for and address the problem of concept drift, which is more likely to occur. One common challenge in data stream mining is the lack of strict stationarity, where the underlying distribution of incoming data unpredictably changes over time. There is a growing need to identify concept promptly drifts in data streams [20].

In this paper, we employ adaptive federated learning techniques within a distributed architecture for short-term energy load forecasting. In this framework, multiple people use smart meter data stored at the edge to train a global model. The model enhances customers' predictive capabilities by revealing unfamiliar local patterns that other users observe when it is returned to them. The forecasting model employed is LSTM, which leverages federated learning to uphold user privacy. Adaptive learning was incorporated into the overall architecture to enhance the accuracy and performance of the model. This was achieved by retraining the model whenever its accuracy fell below a predetermined threshold.

We evaluate our method carefully and show that it can make forecasts that are as good as, or even better than, an AI-based strategy that requires data centralization. Regarding how long it takes to train a model, the proposed framework is better than the centralized architecture, making it a more scalable and privacy-friendly alternative. Also, our strategy will reduce communication costs even mStrategy energy consumption measurements are taken more often than every hour (i.e., with a smaller interval).

The novel contribution of this study consists of the following facets:

- A cutting-edge deep learning model is developed, employing federated learning methodologies, to enhance the precision of energy consumption (load) prediction within smart grid systems. This reduces communication overhead and privacy concerns in energy consumption and demand applications.
- Adaptive learning is employed to enhance forecast accuracy by updating the model in response to changes in the data.
- To evaluate the efficacy of the suggested methodology compared to the centralized architecture, a real-world data set is employed, and extensive experiments are conducted.
- Ultimately, we proceed with the design, testing, and evaluation of two discrete input-output typologies for the forecasting models. The designs discussed in this context consist of two main approaches: the univariate method and the multivariate approach. The univariate approach focuses primarily on energy use as the input variable. On the other hand, the multivariate approach combines extra data, such as calendar and weather information, with energy consumption.

The remainder of the paper is organized as follows: Section 2 examines recent related research. Section 3 describes the methodology proposed and employed in this study. The results are presented in Section 4, along with a discussion of the key findings. Section 5 summarizes the main points of this work and outlines future work recommendations.

## 2. Literature review

Short-term energy consumption forecasting has been a popular topic for many years, with numerous techniques proposed to improve prediction accuracy. Moreover, selecting the most appropriate methodologies and models is another crucial factor. The surveys [21,22] analyzed the most pertinent papers on energy consumption forecasting, focusing on data-driven models. The majority of the reported works focused on non-residential scenarios, with artificial neural networks (ANNs) [23], deep learning (DL) [24], auto-regressive integrated moving average (ARIMA), and support vector machine (SVM) [25], or other regression methods [26] being the most frequently used models.

Another study provides a comprehensive overview of the methodologies, technologies, and applications of federated learning (FL) in the context of smart cities. The authors conduct a comprehensive analysis of FL in the context of smart cities, encompassing societal, industrial, and technological factors contributing to its advancement. The paper examines several intelligent city applications incorporating FL, including smart transportation systems, smart healthcare, smart grid, smart governance, smart disaster management, smart industries, and UAVs for smart city monitoring. Additionally, the authors discuss the concerns related to data leakage and privacy in the context of FL for smart city applications. It also focuses on the current and upcoming projects related to FL in the context of smart cities. Notable projects include those funded by the European Union (EU), the Defense Advanced Research Projects Agency (DARPA), and various industry and research initiatives. The authors conclude by examining the research challenges and prospects for the technological advancement of FL in the context of intelligent cities [27].

Deep learning is also widely employed in load forecasting. [28] proposed a method based on the conditional restricted Boltzmann machine, one of the most pertinent works in this context (CRBM). The authors provide evidence to support the superiority of their method over the currently employed techniques of Artificial Neural Networks (ANN) and Support Vector Machines (SVM). Similarly, [29] proposed a polling-based deep recurrent neural network (RNN) prevent (or severely restrict) overfitting. The method outperformed conventional techniques, including ARIMA, support vector regression, and RNN. The findings from both investigations were promising; nonetheless, the computational demands of deep learning techniques often exceed the capabilities of edge devices with constrained computing resources.

[30] is among the first works in this field to implement and evaluate LSTM. The research conducted by the authors showcases the higher performance of Long Short-Term Memory (LSTM) networks compared to standard back-propagation Artificial Neural Networks (ANNs). This superiority can be attributed to LSTM's enhanced capacity to identify and capture long-term temporal correlations. As well, [31] is a contemporary LSTM-based method. The suggested method is highly adaptable, as it can effectively forecast energy consumption even when the LSTM model is employed for load prediction in residential houses lacking historical data within the training set. Nevertheless, the attainment of such adaptability can solely be guaranteed by utilizing a substantial quantity of training data, hence becoming the procedure computationally demanding.

Although the previously mentioned machine-learning-based solutions differ in many ways, they all share one architectural characteristic: To train a global model in a centralized manner, it is necessary to establish a centralized body responsible for gathering energy consumption data from clients. This characteristic is known as centralized architecture, and we will use it as a benchmark for our proposed model and framework. According to previous research, two new academic papers have used federated learning methods for predicting load in edge computing settings [32,33].

Concerning load forecasting, a recent study [34] developed a novel algorithm named VMD-FK-SecureBoost to address the challenge of sharing data while preserving participants' privacy. The algorithm integrates three techniques to enhance prediction accuracy: variational mode decomposition (VMD), federated k-means clustering (FK), and SecureBoost. The utilization of VMD enables the decomposition of the initial dataset into smaller subsequences, hence facilitating the extraction of implicit characteristics and enhancing the accuracy of predictions. Then, FK is employed to recombine the subsequences into clusters with shared traits. Finally, SecureBoost is utilized to protect privacy during federated learning. The test results show that VMDFK-SecureBoost performs better than other algorithms like XGBoost and SecureBoost. It has the lowest mean absolute percentage errors (MAPEs) for one-step-ahead forecasting in Texas and Newcastle CBD. Therefore, this algorithm has the potential to enable data sharing while preserving the privacy of participants. However, their study differs from ours regarding the utility of the data sets and machine learning algorithms. That is, alternative machine learning methods were employed apart from LSTM, and a distinct data set, which was not accessible during the composition of our work, was utilized. Thus, maintaining a comparison with them poses a significant challenge.

A study [35] uses an expanded version of the federated averaging algorithm to train probabilistic neural networks and linear regression models in a way that saves communication and protects privacy in a problem that is similar to load forecasting. The experiments illustrate the superior performance of probabilistic prediction algorithms compared to deterministic prediction models when evaluated using appropriate scoring rules. Furthermore, their findings demonstrate the potential of federated learning to improve conventional driver-specific learning methods. The implementation of probabilistic predictions allows for the adjustment of safety margins based on the attainability of the destination. This leads to an improved driving range and reduced travel time. The results emphasize the potential of using federated learning and probabilistic prediction models to enhance driving efficiency and safety. Since their study [35] relies on probabilistic forecasting for electric vehicles (different applications), it cannot be directly assessed with the proposed model for our study.

In addition to load forecasting, another study [36] creates the first benchmark for detecting time series anomalies in federated learning frameworks. The benchmark includes five-time series anomaly detection algorithms, four federated learning frameworks, and three-time anomaly detection data sets. Through the experiments, they look at how federated learning works, what happens when data is not spread evenly, and how well different federated learning frameworks and time series work together. These analyses help researchers choose the right time series anomaly detection algorithms and federated learning frameworks when doing time series anomaly detection within federated learning frameworks. A concise overview of the previously described research is provided in Table 1 to offer readers a quick glimpse. The table shows that while there are several studies applying deep learning

(DL) to the problem of energy consumption forecasting in IoT, they have not enhanced their approaches by using federated learning (FL) and adaptive learning (AL) together.

In the current work, we utilize LSTM and distribute the training load across multiple edge nodes for energy consumption as applied in similar previous studies [32,33]. Even though both [32] and our study use univariate LSTM, i.e., the only input feature to the model is previous timesteps of energy consumption, we employ adaptive learning and federated learning to enhance model forecasting performance. Furthermore, our work differs from [33] by using both univariate and multivariate input architecture and then implementing adaptive learning to improve the model's overall performance, accuracy, and efficiency. The results demonstrate that our proposed model outperforms both models regarding Root Mean Squared Error (RMSE). Table 2 provides a comprehensive analysis of the most relevant literature to the current study.

## 3. Methodology

This section describes the proposed framework, provides details of the experimental design and evaluation metrics, and discusses data set collection and preprocessing.

## 3.1. Data set description and preprocessing

This study uses a data set titled ''SmartMeter Energy Consumption Data in London Households'', obtained from the London Datastore website [37] and imported into our data file. Between November 2011 and February 2014, 5567 samples were collected from London households as part of the UK Power Networks-led Low Carbon London project. Readings were taken every 30 min. The trial participants were selected to be representative of the Greater London population. Energy consumption in kWh (per half hour), date and time, and a unique household identifier are included in the data set. Unzipped, the CSV file contains about 167 million rows and is approximately 10 GB.

## 3.1.1. Extraction, Transformation, and Load (ETL)

Python and related libraries were used to manage raw data. For instance, we used Pandas for data extraction, exploration, cleaning, and loading. Also, the holidays library was used to extract the UK holidays for specific years, and the NumPy library was used for some arithmetic operations conducted on the data set. The process of the ETL is explained as follows: In the beginning, we extracted all of the features from the CSV file, which included the seven attributes above. Then, we narrowed our focus to the most pertinent characteristics for this study, including energy consumption, a unique household identifier, a date, and a time. Then, we divided the households that had samples and readings for the entire year of 2013. We made this decision because we intended to compare our findings to those of a previous related study [33]. As a result, we down-sampled the sampling rate from every half-hour to every hour for comparison. The modified data set was finally saved as a CSV file. The primary characteristics of the chosen data set are displayed in Table 3.

## 3.1.2. Supplementary features

This section explains how the data set was augmented with supplementary attributes, such as the timestamp of each sample's acquisition and the corresponding hourly energy consumption measurement. The study utilized three calendar attributes, namely the day of the year, weekday, and hour of the day, to investigate and understand the daily, weekly, and seasonal patterns. The attribute denoting the day of the year was expressed as an integer ranging from 0 to 365. Also, the weekday variable was represented as an integer within the range of 0 to 6. The hour variable was obtained by extracting information from the timestamp and then represented as a number ranging from 0 to 23. So, the hour and weekday variables were changed into sine and cosine Summary of the related literature.

Table 1

Table 2

| Ref.       |   Year | Preview                                                                                                                                                                          | Scope IoT   | Energy   | DL   | FL   | AL   |
|------------|--------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|----------|------|------|------|
| [27]       |   2023 | Explore FL-integrated smart city applications.                                                                                                                                   | ✓           |          | ✓    | ✓    |      |
| [28]       |   2016 | Show that the conditional-constrained Boltzmann machine surpasses ANN and SVM load forecasting.                                                                                  | ✓           | ✓        | ✓    |      |      |
| [29]       |   2017 | Suggest a polling-based deep RNN reduce overfitting in load forecasting.                                                                                                         | ✓           | ✓        | ✓    |      |      |
| [30]       |   2021 | Show that LSTM finds long-term temporal correlations better than standard back-propagation ANNs.                                                                                 | ✓           | ✓        | ✓    |      |      |
| [31]       |   2020 | Adaptable approach accurately predicts energy usage, even with no past data in training set for households using LSTM model.                                                     | ✓           | ✓        | ✓    |      | ✓    |
| [32]       |   2020 | Use univariate (lags) LSTM for energy consumption forecasting.                                                                                                                   | ✓           | ✓        | ✓    | ✓    |      |
| [33]       |   2021 | Implement multivariate LSTM models that are locally trained by users using their historical energy usage samples using Federated Learning and Edge Computing.                    | ✓           | ✓        | ✓    | ✓    |      |
| [34]       |   2023 | Develop the VMD-FK-Secure Boost algorithm that addresses data sharing while protecting participant privacy.                                                                      | ✓           | ✓        | ✓    | ✓    |      |
| [35]       |   2021 | Train probabilistic neural networks and linear regression models using an improved version of the federated averaging approach to maximize communication efficiency and privacy. | ✓           | ✓        | ✓    | ✓    |      |
| [36]       |   2022 | Create the first federated learning framework time series anomaly benchmark.                                                                                                     | ✓           |          | ✓    | ✓    |      |
| This study |   2023 | Add adaptive learning to LSTM for energy consumption utilizing univariate and multivariate models for FL and centralized systems.                                                | ✓           | ✓        | ✓    | ✓    | ✓    |

Comparison of the recent similar studies and our study.

Table 3

| Study      | Problem                                             | Proposed architecture      | LSTM type                                        | Input-Output architecture            | Data set(s)                                              | Metrics                               | Strengths                                                                                               | Limitations                                                                                          |
|------------|-----------------------------------------------------|----------------------------|--------------------------------------------------|--------------------------------------|----------------------------------------------------------|---------------------------------------|---------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| [32]       | Household electrical load forecasting               | Edge Computing (EC) and FL | Stacked (2- hidden-layered)                      | Single-input- single-output (SISO)   | 200 houses from Texas, USA.                              | RMSE, MAPE                            | The first application of federated learning to household load forecasting                               | Training inputs narrowed down to just energy consumption                                             |
| [33]       | Household short-term energy consumption forecasting | EC and FL                  | Stacked (2- hidden-layered)                      | Multiple-input- single-output (MISO) | Smart meter energy consumption data in London households | RMSE, Training time, Transmitted data | Strategy is based on customer clustering, with extra features related to weather and calendar as input. | No adaptive learning or other LSTM model types                                                       |
| This study | Household short-term energy consumption forecasting | EC and FL                  | Stacked (3- hidden-layered) + Bidirectional LSTM | SISO + MISO                          | Smart meter energy consumption data in London households | MAE, RMSE, Training time              | It implements adaptive learning and other LSTM model types                                              | It lacks deep investigation into other federated learning aspects, including communication overhead. |

The main information of the employed data set.

| Characteristic     | Description                                     |
|--------------------|-------------------------------------------------|
| Number of samples  | 241,714                                         |
| Missing values     | None                                            |
| Start date         | 01-01-2013                                      |
| End date           | 31-12-2013                                      |
| Number of features | 3                                               |
| Features           | Household ID, energy consumption, and timestamp |
| Data types         | int, float, date-time                           |
| Data set size      | 31.4 MB                                         |

functions. This made it possible to see the cyclical patterns as the hours and weeks went by.

In addition, lag variables representing past energy consumption, temperature, and humidity, all relevant to calendar functionality and meteorological conditions, were utilized. The lag variable was constructed by using the energy consumption value from the same hour of the previous day (lag-24). The temperature and humidity forecasts of the prediction hour are input features for the LSTM model.

All in all, the final data set we composed consists of the input features listed below, and the output target is the next hour's energy consumption. That being said, the ultimate data set was used in two ways: (i) univariate input forecasting, where the input data only relies on one feature, which is the previous energy consumption in the past hours; (ii) multivariate input forecasting, where the LSTM model is fed with more information than the previous energy consumption, including calendar data, weather data, etc. The input features and target list are:

- Timestamp
- Weekday (sin and cosine)
- Hour (sin and cosine)
- DayOfYear
- Humidity
- Temperature
- Lag-24 (energy consumption 24 h ago)
- Energy consumption (next hour - target variable)

## 3.2. Structure of proposed LSTM models

Machine learning (ML) enables a system to acquire information from historical or current data and utilize that knowledge to make informed predictions or judgments in the future. Deep learning (DL) is a sub-field within the broader domain of machine learning (ML) that enables training computer models characterized by multiple processing layers. These models are designed to acquire and internalize data representations. The various layers of features in deep learning are automatically identified and combined to get the desired outputs. In contrast to ML, DL offers a significant advantage in the form of automated feature extraction, which eliminates the need for manual effort in building feature representations. DL has received considerable acclaim in many fields, such as computer vision, natural language processing, and bioinformatics. In recent years, academia and industry have expanded the utilization of DL techniques to encompass a broader range of applications, including those within the Internet of Things (IoT) domain. IoT networks generate a substantial volume of data, necessitating alternative approaches to data collection, storage, and processing that can effectively handle this magnitude. DL models typically become more successful using such big data [38].

Based on the above reasoning, we propose implementing adaptive deep learning strategies in decentralized architectures to improve performance while preserving client privacy. To effectively address the dynamic nature of data streams and the scalability potential of IoT devices, it is imperative to enhance the precision of forecasting accuracy. A considerable corpus of literature on machine learning and deep learning algorithms, elucidating the progress made compared to preceding methodologies. Nevertheless, there has been a scarcity of research undertaken to assess and juxtapose the efficacy of models on this particular computational architecture.

We intend to address this issue by comparing the performance of centralized and decentralized learning. The forecasting model's input parameter is each household's hourly energy consumption as a separate client. The data set under consideration in this investigation is a time-series data set. Moreover, it is worth noting that forecasting can be categorized into two distinct types: univariate forecasting and multivariate forecasting. The former is employed to forecast a solitary independent variable. The latter refers to a method employed to forecast the value of a dependent variable by utilizing a collection of independent variables, wherein the model serves as a representation of their interrelation. The research commences by employing univariate forecasting approaches and multivariate forecasting methods to evaluate the model's predictive capacity.

This study introduced two state-of-the-art recurrent neural networks (RNN) models: Stacked LSTM and Bidirectional LSTM (BiLSTM). Despite belonging to the same category, the inner structures of the two models are quite distinct. However, the quantity of hidden layers differs among models, as explained in the subsequent discussion. Table 4 displays the hyper-parameter settings for the experimental models. The number of neurons is established at 50, as shown in Table 4. The reason for this observation is that reducing the value mentioned above leads Parameters selected for the LSTM models for comparison.

Table 4

| Hyper-parameters               | Applied value/function   |
|--------------------------------|--------------------------|
| Number of neurons              | 50                       |
| Number of units in dense layer | 1                        |
| Dropout                        | 20%                      |
| Activation function            | Relu                     |
| Loss function                  | Mean Square Error        |
| Optimization function          | ADAM                     |
| Epochs number                  | 50                       |
| Batch size                     | 100                      |

to a drop in precision while increasing it results in a more intricate framework that necessitates additional computational resources with minimal improvements in accuracy. In addition, because the output is always a single variable, the dense layer's unit count is set to 1 (energy consumption). Furthermore, the dropout rate is initially established at 20% to mitigate the issue of overfitting since it is often considered to strike a balance between minimizing overfitting and preserving high accuracy. Although we increased it to 50%, there was no further improvement so that we will maintain the previous value of 20%.

The Sigmoid, hyperbolic tangent, and Relu functions were also implemented as activation functions. However, we obtained the best results by utilizing the Relu function. Due to our model predictions, we employ the well-known optimizer Adam in such situations. The Adam optimizer generates results generally superior to those produced by conventional optimization methods, requires less computation time, and needs fewer tuning parameters. Adam is, therefore, recommended as our model's default optimizer. Similarly, the batch size selected for this study is 100 to reduce computation time, especially for edge devices.

The two diagrams below illustrate the structure of the two LSTM models utilized in this study. A Stacked LSTM model is initially constructed by stacking multiple hidden LSTM layers on top of one another. An LSTM layer requires a three-dimensional input and generates a twodimensional output at the end of a sequence by default. LSTM can generate a value for each time step in the input data with the return sequences = True option on the layer, which means the 3D output of a hidden LSTM layer becomes input to the subsequent layer. Generally, a single layer is sufficient for simple systems, whereas two or more layers are preferred for complex systems. Additionally, more nodes (neurons) may improve accuracy, whereas fewer nodes may cause an underfitting issue. Fig. 1 depicts the structure of the proposed stacked LSTM model for energy forecasting. We began with a single layer (vanilla LSTM), then added successive layers until we achieved the desired outcome. As depicted in Fig. 1, the stacked LSTM model employed in this study has four layers.

For some sequence prediction problems, it can be helpful to let the LSTM model learn the input sequence both forward and backward and then combine the two ways of understanding it. This kind of model is called a bidirectional LSTM model. Bidirectional LSTM can be used to improve how well a model does at predicting what will come next. When all of the time steps of the input sequence are known, bidirectional LSTM trains two LSTMs instead of one. The first is trained on the original input sequence, and the second is trained on a copy of the original input sequence with the time steps switched around. This can give the network more context, allowing it to learn the problem more quickly and thoroughly. We hypothesized that complexity is not always the solution because the simplest structure of a bidirectional LSTM achieves greater accuracy than a stacked LSTM. Despite this, we added layers but only saw minor improvements. As a result, we selected the simplest structure that provides the desired performance because it requires less time for model construction and, consequently, less time and memory for computations. Fig. 2 provides a detailed illustration.

Adaptive learning is used in both types of the LSTM model. This lets the models retrain and change their weights whenever a data drift and model performance worsens. We believe including this element represents a valuable contribution to the existing literature. The following subsection will explain the adopted adaptive mechanism.

Fig. 2. Proposed bidirectional LSTM structure.

<!-- image -->

## 3.3. Proposed adaptive model

In our earlier work [39], we designed the general adaptive learning framework depicted in Fig. 3. This adaptive approach is applied in the current study to predict energy consumption for the next hour (short-term forecasting).

The adaptability employed in this study functions in the following manner: Predictions are temporarily stored in a buffer. Depending upon the stakeholders ' preferences, the project's duration may vary, ranging from a single day to a week or even a month. Following a designated waiting period, such as one week, a comparative analysis is conducted between all projected forecasts and the corresponding actual values. Subsequently, an error score is calculated to assess the efficacy of the model's performance. Root mean square error (RMSE) is justified because it penalizes substantial errors. Finally, if the error rate surpasses the predetermined threshold, it signifies the occurrence of concept drift, necessitating an update to the model with the most recent historical data.

Fig. 3. The proposed adaptive model framework.

<!-- image -->

of local epochs and batches, which addresses the mentioned limitation. This approach aims to minimize the interaction between the server and devices by reducing the frequency of communication. Additionally, the weights are averaged to consider the number of training samples used locally.

For example, the RMSE of the forecast from the previous day is compared to the average RMSE over two weeks. If the RMSE observed yesterday is 10% greater than the average biweekly RMSE, the model undergoes retraining using the most up-to-date historical data. Otherwise, there is no adaptive training. This study's primary innovation lies in utilizing the suggested adaptive learning approach during the implementation of federated learning to forecast household energy consumption. This ensures that our model can effectively identify incidences of data drift while minimizing the associated expenses related to data transfer and retraining.

## 3.4. Federated architecture

Federated learning (FL) is a decentralized approach to training machine learning models, wherein remote devices utilize their data for local training. A centralized server aggregates the models into a global model, distributed to remote devices for additional training. The iterative nature of this process facilitates the inclusion of a substantial quantity of devices, enabling their participation without necessitating the transfer of data to a centralized location. FL demonstrates efficacy in handling non-independent and identically distributed (non-iid) data and presents advantages in terms of data exchange when compared to centralized training.

FL training workflow involves round-slotted time and two entities: end devices and a centralized server, with steps outlined in Fig. 4. First, initializing a machine-learning model on a centralized server involves randomly selecting weights. Additionally, the server transmits the model type and hyper-parameters to the end devices. Second, the training phase consists in selecting a subset of end devices, transmitting the global model to these devices, conducting local training on edge devices using their respective data, and aggregating all the received local models. Subsequently, the server generates a revised global model. The selection of end devices is reliant on the specific application at hand. Additionally, a commonly employed technique in this context is aggregating data on a centralized server, referred to as Federated Averaging (FedAVG) [40].

FedAVG is an extension of FedSGD, wherein the process entails the utilization of Stochastic Gradient Descent (SGD) by end devices, followed by transmitting model weights to the server [10]. This approach hinders the convergence of global model training by performing a single SGD step per round. The FedAVG approach introduces the ideas To optimize the performance of FedAVG, it is crucial to fine-tune three parameters [33]:

- Nround: The total number of rounds to be executed in the experiment. Item Nepoch: The total number of local epochs each chosen remote device will execute in a single round.
- Bsize: The number of local samples utilized for training in each round or epoch, commonly known as the batch size.

This study uses a federated short-term energy consumption forecasting architecture illustrated in Fig. 4, similar to that in [33]. By collaboratively training an LSTM network, this architecture enables accurate energy consumption forecasting for residential properties. As the FL approach prescribes, model aggregation and redistribution steps are used to train a global LSTM model. This is because many edge devices train local LSTM networks using locally generated energy consumption measurements. This architecture takes into account two entities:

- Energy company: It provides electricity to customers and, through smart grids, could increase the supply capacity. Residential customers: They use the electricity provided and receive bills from the energy company for it in their homes.

Three functional nodes are included and interact; their definitions are derived from previously defined federated learning functional nodes:

1. Smart Meters: All user premises have them installed, and the energy company is their owner. A smart meter logs energy consumption measurements (samples) with fine granularity (up to one per minute) [41].
2. Edge Computing Node: It might be a GPU board or PC that the energy provider owns and installs at the customer's location. It is linked to the smart meter via a suitable wireless or wired network connection and is responsible for storing energy consumption measurements. It is also connected to the aggregator for model exchange. It can interact with customers' end devices (such as smartphones) via special-purpose apps used for data visualization and as an alarm endpoint for the edge computing node.
3. Aggregator: The energy company owns this central server, which uses encrypted connections to communicate with the edge computing nodes. After each round, it aggregates local models trained by multiple edge computing nodes. This operation produces a global model, which is redistributed to edge computing nodes.

Fig. 4. Federated architecture for household energy consumption forecasting.

<!-- image -->

The significance of this study is rooted in introducing adaptive deep learning into the conventional federated learning framework. The purpose of this approach is to provide benefits in terms of data exchange when compared to centralized training while also ensuring privacy. The global model is updated only when data drifts, resulting in more accurate models with reduced data transmission and communication. Consequently, this approach leads to a decrease in forecasting errors and training time. Our approach distinguishes itself from existing methods in the academic literature on energy consumption by demonstrating a resilient and swift reaction to the drift concept. Previous studies [32, 33] have employed a basic federated architecture to predict energy consumption. However, these approaches have been unable to effectively account for the frequent occurrence of data drift throughout the forecasting process. Consequently, this resulted in a trade-off between the accuracy of deep learning models and the potential burden imposed on the network due to the vast amount of data required for repetitive training. In contrast, we have introduced the adaptive learning idea as a fundamental element inside the federated learning framework. We aim to ensure the freshness of current models (being up-to-date) while minimizing the need for extensive data transmission between local devices and the central node.

## 3.5. Experiment setup

We use Table 5 parameters to evaluate the proposed model within the federated architecture. To implement and test the LSTM model, we employed the TensorFlow Federated framework 1 and the Keras API, 2 and Google Colab Cloud 3 to perform the experiments.

As stated in Table 5, the training phase comprises the initial eleven months of 2013. The final month is then utilized as a test set. The rationale behind using only the most recent month is that the model must temporally observe the test set in time series during the training phase. In addition, the number of lags fluctuates during experimentation to determine which lags produce the most outstanding results. Instead, the prediction horizon is always the following hour (shortterm forecasting). In the simulation, the server and client settings are identical. In other words, the training and aggregation stages share the same parameters and functions. However, we altered the number of clients (participants) to test the influence of scalability on the system architecture, i.e., its impact on forecasting accuracy and complexity.

[1 https://www.tensorflow.org/federated/.](https://www.tensorflow.org/federated/)

[2 https://keras.io.](https://keras.io/)

[3 https://colab.research.google.com.](https://colab.research.google.com/)

Table 5 Model training, testing, and simulation settings.

| Parameters         | Applied value/function       |
|--------------------|------------------------------|
| Training period:   | 11 months                    |
| Testing period:    | 1 month (December)           |
| Time steps (lags): | 6, 12, 24, 48 h              |
| Forecast horizon   | Next hour                    |
| Forecast interval  | Hourly                       |
| Number of clients: | 2, 4, 6, 8, 10, 20, up to 50 |

## 3.6. Backtesting and evaluation metrics

The provided data consists of the energy consumption recorded over a specific period: the previous x hours. This data is in univariate mode, meaning it only includes information about energy consumption and does not consider any other variables. Rolling forecast validation, which is also referred to as walk-forward model validation, is utilized. The time steps of the test set are iterated sequentially. The model utilizes the extracted actual predicted value from the test set in the subsequent time step to facilitate forecasting. This scenario involves the simulation of hourly measurements of actual energy consumption, which are subsequently used to forecast the energy consumption for the upcoming hour or hours.

Consequently, due to the exclusive focus on energy consumption as the model's output, the feedback operation compares projected values and real-world observations. Initially, the input undergoes a process of filtration and preprocessing to adhere to the prescribed input structure of the trained model. The data is subsequently fed into the Long ShortTerm Memory (LSTM) model to forecast forthcoming energy loads. The predicted value is subsequently compared to the actual observed value to provide feedback. If the precision of the model is upheld, the process will proceed in its regular manner. Furthermore, the model takes into consideration seasonal variations.

Since the target variable acquires a real number value while dealing with regression models, it is doubtful that our model can precisely forecast the target variable. As a result, we employ a measure that considers the overall deviation to determine the distance between our predicted values and the actual values [42]. Several loss functions, such as the mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE), and many others, are used to measure the error between 𝑦 𝑖 and the forecast value ( ̂ 𝑦 𝑖 ) [39].

In this research, we assess the accuracy of energy consumption forecasting models using the MAE and RMSE metrics. While both attempt to quantify the accuracy of a regression model, they do so in different ways:

· The RMSE is more susceptible to outliers.

Table 6

The energy consumption statistics for all participating households in 2013.

| Statistic measure   |   Value (kWh) |
|---------------------|---------------|
| mean                |         0.239 |
| std                 |         0.381 |
| min                 |             0 |
| 25%                 |         0.062 |
| 50%                 |         0.131 |
| 75%                 |         0.256 |
| max                 |         6.528 |

- Because errors are squared initially, RMSE penalizes large errors more than MAE.
- MAE returns more interpretive results because it is simply the average of absolute error.

The following are the mathematical formulas for each of these metrics:

$$
\begin{aligned}
M _ { \ } A E = \frac { \sum _ { i = 1 } ^ { n } | y _ { i } - \hat { y } _ { i } | } { n } & & ( 1 ) & & \text {because}
\end{aligned}
$$

$$
\begin{aligned}
R M S E = \sqrt { \frac { \sum _ { i = 1 } ^ { n } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } } { n } } & & ( 2 ) & & 4 . 2 . \, S t a
\end{aligned}
$$

where 𝑦 𝑖 , ̂ 𝑦 𝑖 , and 𝑛 stand for observed value, forecast value, and the total number of observations. To evaluate the accuracy of the models, we employ a variation of the holdout method called the split method, in which the holdout method is run 𝑘 times, and the accuracy average is calculated.

## 4. Results and discussion

This section describes our experimental findings and discusses the positive and negative aspects. In addition, we contrast our proposed framework with state-of-the-art algorithms.

## 4.1. Exploratory Data Analysis (EDA)

In this study, univariate energy consumption is implemented. In other words, the only input to our model is the energy consumption of prior back steps (lags). Thus, we investigate and explore features of interest based on other temporal parameters. The primary characteristics of the chosen data set are displayed in Table 3. In addition, Table 6 illustrates the most popular energy consumption statistics to comprehend its behavior and implicitly understand subsequent outcomes from our proposed framework.

To gain a general understanding of the nature of the data set, we randomly selected one participant (household). We analyzed its energy consumption patterns, searching for trends, seasonality, and normal distribution. For example, Fig. 5 depicts the energy consumption ranges from 2012 to 2014. The line graph reveals that there are seasonal peaks that exceed 2.0 kWh on occasion. Although the maximum energy consumption for this data set is approximately 6 kWh, this household appears to be a moderate consumer.

Furthermore, Fig. 6 depicts the histogram of the same household, from which we can infer that the energy consumption is right-skewed. The vast majority of hourly energy consumption falls from 0 to 0.25 kWh. The remainder of the energy load ranges between 0.5 and 0.7 kWh, with the remaining values being exceptional or outliers.

To find out how far we should look back (lags), we resampled the data set (down-sample) to daily rather than hourly. Moreover, we calculated the auto-correlation and drew its chart to conclude how many lags could be beneficial during our experimentation. As seen from Fig. 7, up to day 24, the lags could be effective. Yet, looking back up to day 4 (i.e., 4 * 24 = 96 h) would be optimal.

Finally, Fig. 8 shows the selected household's observation, trends, seasonality, and residuals in the mentioned period. As seen, there are some increasing and decreasing trends throughout the two years, but it is evident that the consumption trends soar in the fall and winter when the weather is colder and decrease in the spring and summer. Also, from the seasonality chart, there is clear monthly seasonality. This leads us to investigate in the future if there is weekly and daily seasonality as well because it will help us understand the data better and, hence, design a better solution for it to maximize our objective.

Table 7 Comparing stacked LSTM and BiLSTM.

| Model        |    MAE |   RMSE |
|--------------|--------|--------|
| Stacked LSTM | 0.0859 | 0.1208 |
| BiLSTM       | 0.1576 | 0.2063 |

Table 8 Adaptive vs. non-adaptive learning.

| Model                     |    MAE |   RMSE |
|---------------------------|--------|--------|
| Non-adaptive stacked LSTM | 0.0892 | 0.1294 |
| Adaptive stacked LSTM     | 0.0859 | 0.1208 |

## 4.2. Stacked LSTM vs. BiLSTM

It is worth mentioning that the results reported in this subsection were obtained when adaptive learning was applied to both variations of the LSTM model in a federated architecture. The same model's hyperparameters and simulation settings were used for both, including the number of clients (participants), input features (lags), activation and optimization functions, etc.

Table 7 shows that the stacked LSTM model outperformed the BiLSTM model according to the MAE and RMSE metrics. The former reduced the MAE and RMSE by 45.5% and 41.4%, respectively. Accordingly, the stacked (4-layered) LSTM design was chosen for the remaining parts of our experimentation.

## 4.3. Impact of adaptive learning

In this part, we investigate the influence of adaptivity on the performance of models, specifically, and architecture, generally. As shown in Table 8, implementing adaptive learning with the deep federated learning architecture improved the system's performance by approximately 4% and 7% for MAE and RMSE, respectively. One could argue that the percentage of improvement is insignificant compared to the extra computational power, memory usage, and communication bandwidth needed for the retraining process when data drift is detected, and the model must react to it. There are two explanations for the lower improvement that adaptive learning produces in this specific experiment. First, the limited evaluation period (December 2013) prevented the occurrence of data drift as household energy consumption remains relatively consistent during the winter season. Therefore, it is unnecessary to update the model using our proposed algorithm. This resulted in a marginal improvement in the performance of the model. Furthermore, we expect that the effectiveness of adaptability will be demonstrated as the testing period is extended to encompass additional months and diverse seasons. Thus, adaptive learning may be less practical for timeseries data without concept or data drift. However, these drifts are common in time-series applications, so we argue that adaptive learning is worthwhile in this context.

## 4.4. Sensitivity to n-clients

We attempted to investigate the model's performance by increasing the number of participants in the federated learning environment and examining the architecture's scalability. As shown in Fig. 9, the MAE fluctuated as the number of clients increased. However, the Time (h)

<!-- image -->

Fig. 5. The energy consumption for a random household at an hourly rate.

<!-- image -->

Fig. 6. The histogram of energy consumption for a random household.

Fig. 7. The auto-correlation (lags) of daily energy consumption.

<!-- image -->

performance remained stable after 20 or more participants. The model appeared exposed to sufficient examples for its performance to be steady. Furthermore, the number of clients in this study was limited to 50 for easier comparison with [33]. However, we plan to repeat the experiment with more participants in future research, as was done in [31]. We adhered to n-clients = 50 in all other experiments.

## 4.5. Sensitivity to n-lags

Since the primary goal of this study is to predict univariate energy consumption, our main goal was to find out how the number of lags affects the model's overall performance and accuracy. Surprisingly, the model's accuracy did not change even after we increased the lags from DateTime six to forty-eight hours. No matter how far back we looked, the MAE and RMSE did not change. In contrast, training or building time is proportional to the number of lags. To optimize the model's overall performance, we opted to employ the most minor possible lag, which was six hours.

<!-- image -->

Fig. 8. The observations, trends, seasonality, and residuals of energy consumption in London households.

Fig. 9. Comparison of MAE for different numbers of clients (participants).

<!-- image -->

## 4.6. Centralized vs. Decentralized (Federated) architecture

The performance of the traditional centralized architecture and the federated architecture utilized in our framework are compared in this subsection. In the first method, all participants send their data to the energy company's server. Next, a global model is trained using all of the data sets that have been collected. Finally, the trained model is distributed to each participant so that they can make predictions about their local energy consumption [43]. In contrast, every participant in federated learning keeps their private data on-premises and receives a shared structure of the designed model from the energy company. This arrangement is known as distributed learning. After that, the training process takes place simultaneously at each location. Then, the parameters of the constructed models, also known as weights, are sent to the server to be aggregated and transformed into a brand-new global model. In a manner analogous to the centralized architecture, the global model is disseminated to all participants (clients) to produce forecasts of later local energy consumption.

Overall, as seen in Table 9, the performance of the federated architecture turned out to be superior to that of the centralized architecture. If we look at the RMSE, we see that the adaptive FL system reduced the error in forecasting by almost 8%. In addition, it is essential to Federated vs. centralized architecture.

Table 9

| Architecture         |    MAE |   RMSE |   Training time (min) |
|----------------------|--------|--------|-----------------------|
| Adaptive federated   | 0.0859 | 0.1208 |                  5.73 |
| Adaptive centralized | 0.1117 | 0.1310 |                 30.46 |

Table 10

| Univariate vs. Multivariate input-output architecture for adaptive federated learning.   | Univariate vs. Multivariate input-output architecture for adaptive federated learning.   | Univariate vs. Multivariate input-output architecture for adaptive federated learning.   |
|------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| Model                                                                                    | Input-output architecture                                                                | RMSE                                                                                     |
| Stacked LSTM                                                                             | SISO                                                                                     | 0.1208                                                                                   |
| Stacked LSTM                                                                             | MISO                                                                                     | 0.0810                                                                                   |

note that the difference in training time between the two architectures shows that adaptive FL required less time to converge, 5.73 min, as opposed to 30.46 min. The adaptive FL architecture is better than the current state-of-the-art architecture in more ways than we have already discussed. Some advantages are that user privacy is better protected and the amount of data sent over the network is lessened.

## 4.7. Adaptive federated learning: Univariate vs. Multivariate models

In mathematics, a univariate entity refers to an expression, equation, function, or polynomial that exclusively involves a single variable. Multivariate refers to objects that involve multiple variables. In time series analysis, the term ''variable'' refers to the entire time series. Specifically, a univariate time series represents the sequence of values over a period about a singular quantity. Similarly, a ''multivariate time series'' pertains to the dynamic evolution of multiple variables over a given period [33].

In this subsection, we address the impact of using either a univariate or multivariate input architecture. In our case, the former is denoted as single input, single output (SISO), whereas the latter is multiple input, single output (MISO). The last section found stacked LSTM works better than BiLSTM in SISO adaptive federated learning. Therefore, after engineering the input features and adding extra attributes like calendar and weather data, experiments were done to find the best input-output architecture to make the proposed model better at making predictions.

The experimental findings indicate that the MISO-stacked LSTM model exhibits superior performance compared to the SISO model, reducing the forecasting error from 0.1208 to 0.0810 as shown in Table 10. Nevertheless, it is worth noting that the MISO technique requires a longer training and convergence duration than the SISO approach.

## 4.8. Comparison with the literature

For comparison, we compared our suggested model to previous cutting-edge studies in edge computing and federated learning, in general, and energy consumption, in particular. As previously stated, few studies have addressed the application of federated learning to residential energy consumption. To our knowledge, none of these studies have considered adaptive learning as a complementary technique for this problem. So, we are trying to combine adaptive learning with federated learning to make the suggested model work better when measuring how much energy a home uses. So far, we have been able to compare and contrast our work with two previous models from the literature, as shown in Table 11, demonstrating the RMSE values achieved in those studies and ours.

In comparison, study [32] forecast energy consumption using the univariate method and followed a similar methodology to our own. Table 11 shows that their RMSE was roughly four times as large as ours. However, it is important to note that their data set differs from ours, as they used 200 houses from Texas, USA, while we used 100 houses from Performance comparison with state-of-the-art models.

Table 11

| Model                     |   RMSE |
|---------------------------|--------|
| Model proposed in [32]    |  0.509 |
| Model proposed in [33]    |  0.133 |
| Our proposed model (SISO) |  0.121 |
| Our proposed model (MISO) |  0.081 |

London, UK. On the other hand, study [33] utilized the same data set as we did, and our proposed model slightly outperformed theirs according to the RMSE metric.

In terms of how models were built, study [33] used MISO architecture, whereas we used both SISO and MISO. Moreover, we attempted two variations of LSTM, stacked LSTM and bi-LSTM, to investigate which design achieves better outcomes with the problem. Furthermore, our approach strengthens the proposed model with adaptive learning, which was not applied in either of the compared studies. Overall, Table 11 illustrates that our proposed model compared favorably to similar studies in the literature and reduced the error rate by 9% when SISO was implemented compared to another study [33] using the same data set. Additionally, the error reduction rate is enhanced to 39% when Multiple-Input-Single-Output (MISO) is implemented in our proposed framework.

## 5. Conclusion

Federated learning architectures have been used in the literature to tackle privacy problems and network communication overhead. This work focuses on adaptively applying federated learning to the problem of energy consumption forecasting in intelligent cities. We designed and built robust stacked LSTM and BiLSTM recurrent neural network models. The former outperformed the latter in our experiments. We set up a distributed architecture with a stacked LSTM model. We added adaptive learning methods to improve model performance and lower the amount of daily learning needed. Adaptivity improved architectural performance and reduced the root mean square error (RMSE) by 8% The impact of participant count and lags on the proposed framework's accuracy was minimal. Therefore, we opted for smaller values to minimize training time complexity. During our investigation, it was observed that the model exhibited enhanced stability and robustness with an increase in the number of participants.

We compared our approach with the current centralized architecture. The adaptive federated architecture outperformed the adaptive centralized architecture, reducing training time by nearly 80% and improving accuracy by 8% and 38% for single-input-single-output (SISO) and multiple-input-single-output (MISO), respectively. However, particular areas and aspects require additional clarification. In future work, we plan to investigate significant challenges such as privacy protection, network communication overhead, and non-IID (non-independent and identically distributed) data in federated learning. Also, to ensure a comprehensive analysis of the impact of adaptive learning across various seasons and data drifts, we aim to extend the training and testing set to a two-year timeframe. Subsequently, the suggested framework can be employed to analyze another recent data set to assess its robustness.

## CRediT authorship contribution statement

Nawaf Abdulla: Conceptualization, Data curation, Formal analysis, Methodology, Software, Validation, Investigation, Writing -original draft, Visualization, Project administration. Mehmet Demirci: Conceptualization, Resources, Formal analysis, Writing - review &amp; editing, Project administration, Supervision. Suat Ozdemir: Writing - review &amp; editing, Project administration, Supervision.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

The data used in this study is publicly accessible.

## References

- [1] [Y. Peng, Y. Wang, X. Lu, H. Li, D. Shi, Z. Wang, J. Li, Short-term load forecasting at different aggregation levels with predictability analysis, in: 2019 IEEE Innovative Smart Grid Technologies-Asia (ISGT Asia), IEEE, 2019, pp. 3385-3390.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb1)
- [2] [A.R. Khan, A. Mahmood, A. Safdar, Z.A. Khan, N.A. Khan, Load forecasting, dynamic pricing and DSM in smart grid: A review, Renew. Sustain. Energy Rev. 54 (2016) 1311-1322.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb2)
- [3] [S.N. Fallah, R.C. Deo, M. Shojafar, M. Conti, S. Shamshirband, Computational intelligence approaches for energy load forecasting in smart energy management grids: state of the art, future challenges, and research directions, Energies 11 (3) (2018) 596.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb3)
- [4] [B. Yildiz, J.I. Bilbao, J. Dore, A.B. Sproul, Recent advances in the analysis of residential electricity consumption and applications of smart meter data, Appl. Energy 208 (2017) 402-427.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb4)
- [5] [P. Zhang, X. Wu, X. Wang, S. Bi, Short-term load forecasting based on big data technologies, CSEE J. Power Energy Syst. 1 (3) (2015) 59-67.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb5)
- [6] [M.R. Asghar, G. Dán, D. Miorandi, I. Chlamtac, Smart meter data privacy: A survey, IEEE Commun. Surv. Tutor. 19 (4) (2017) 2820-2835.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb6)
- [7] [Z. Fan, G. Kalogridis, C. Efthymiou, M. Sooriyabandara, M. Serizawa, J. McGeehan, The new frontier of communications research: smart grid and smart metering, in: Proceedings of the 1st International Conference on Energy-Efficient Computing and Networking, 2010, pp. 115-118.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb7)
- [8] [F.G. Mármol, C. Sorge, O. Ugus, G.M. Pérez, Do not snoop my habits: preserving privacy in the smart grid, IEEE Commun. Mag. 50 (5) (2012) 166-172.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb8)
- [9] [W.Z. Khan, E. Ahmed, S. Hakak, I. Yaqoob, A. Ahmed, Edge computing: A survey, Future Gener. Comput. Syst. 97 (2019) 219-235.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb9)
- [10] [B. McMahan, E. Moore, D. Ramage, S. Hampson, B.A. y Arcas, Communicationefficient learning of deep networks from decentralized data, in: Artificial Intelligence and Statistics, PMLR, 2017, pp. 1273-1282.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb10)
- [11] Y. Kang, Y. Liu, T. Chen, Fedmvt: Semi-supervised vertical federated learning with multiview training, 2020, arXiv preprint arXiv:2008.10838.
- [12] [X. Liu, L. Zhu, S.-T. Xia, Y. Jiang, X. Yang, GDST: Global distillation self-training for semi-supervised federated learning, in: 2021 IEEE Global Communications Conference, GLOBECOM, IEEE, 2021, pp. 1-6.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb12)
- [13] E. Diao, J. Ding, V. Tarokh, SemiFL: Communication efficient semi-supervised federated learning with unlabeled clients, 2021, arXiv preprint arXiv:2106.01432.
- [14] [H. Kassem, D. Alapatt, P. Mascagni, A. Karargyris, N. Padoy, Federated cycling (FedCy): Semi-supervised federated learning of surgical phases, IEEE Trans. Med. Imaging (2022).](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb14)
- [15] [Q. Yang, Y. Liu, T. Chen, Y. Tong, Federated machine learning: Concept and applications, ACM Trans. Intell. Syst. Technol. 10 (2) (2019) 1-19.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb15)
- [16] T. Yang, G. Andrew, H. Eichner, H. Sun, W. Li, N. Kong, D. Ramage, F. Beaufays, Applied federated learning: Improving google keyboard query suggestions, 2018, arXiv preprint arXiv:1812.02903.
- [17] [X. Wu, Z. Liang, J. Wang, Fedmed: A federated learning framework for language modeling, Sensors 20 (14) (2020) 4048.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb17)
- [18] [S.R. Pokhrel, J. Choi, Federated learning with blockchain for autonomous vehicles: Analysis and design challenges, IEEE Trans. Commun. 68 (8) (2020) 4734-4746.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb18)
- [19] [R. Cioffi, M. Travaglioni, G. Piscitelli, A. Petrillo, F. De Felice, Artificial intelligence and machine learning applications in smart production: Progress, trends, and directions, Sustainability 12 (2) (2020) 492.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb19)
- [20] J. Brownlee, A gentle introduction to concept drift in machine learning, 2017, URL https://machinelearningmastery.com/gentle-introduction-conceptdrift-machine-learning/, Accessed December 20, 2022.
- [21] [K. Amasyali, N.M. El-Gohary, A review of data-driven building energy consumption prediction studies, Renew. Sustain. Energy Rev. 81 (2018) 1192-1205.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb21)
- [22] [F. Kaytez, M.C. Taplamacioglu, E. Cam, F. Hardalac, Forecasting electricity consumption: A comparison of regression analysis, neural networks and least squares support vector machines, Int. J. Electr. Power Energy Syst. 67 (2015) 431-438.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb22)
- [23] [H. Chitsaz, H. Shaker, H. Zareipour, D. Wood, N. Amjady, Short-term electricity load forecasting of buildings in microgrids, Energy Build. 99 (2015) 50-60.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb23)
- [24] [K. Aurangzeb, S. Aslam, S.I. Haider, S.M. Mohsin, S.u. Islam, H.A. Khattak, S. Shah, Energy forecasting using multiheaded convolutional neural networks in efficient renewable energy resources equipped with energy storage system, Trans. Emerg. Telecommun. Technol. 33 (2) (2022) e3837.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb24)
- [25] [H. Nie, G. Liu, X. Liu, Y. Wang, Hybrid of ARIMA and SVMs for short-term load forecasting, Energy Procedia 16 (2012) 1455-1460.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb25)
- [26] [Y. Hong, Y. Zhou, Q. Li, W. Xu, X. Zheng, A deep learning method for short-term residential load forecasting in smart grid, IEEE Access 8 (2020) 55785-55797.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb26)
- [27] [S. Pandya, G. Srivastava, R. Jhaveri, M.R. Babu, S. Bhattacharya, P.K.R. Maddikunta, S. Mastorakis, M.J. Piran, T.R. Gadekallu, Federated learning for smart cities: A comprehensive survey, Sustain. Energy Technol. Assess. 55 (2023) 102987.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb27)
- [28] [E. Mocanu, P.H. Nguyen, M. Gibescu, W.L. Kling, Deep learning for estimating building energy consumption, Sustain. Energy Grids Netw. 6 (2016) 91-99.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb28)
- [29] [H. Shi, M. Xu, R. Li, Deep learning for household load forecasting-A novel pooling deep RNN, IEEE Trans. Smart Grid 9 (5) (2017) 5271-5280.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb29)
- [30] [M.N. Fekri, H. Patel, K. Grolinger, V. Sharma, Deep learning for load forecasting with smart meter data: Online adaptive recurrent neural network, Appl. Energy 282 (2021) 116177.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb30)
- [31] [A.M. Alonso, F.J. Nogales, C. Ruiz, A single scalable LSTM model for short-term forecasting of massive electricity time series, Energies 13 (20) (2020) 5328.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb31)
- [32] [A. Taïk, S. Cherkaoui, Electrical load forecasting using edge computing and federated learning, in: ICC 2020-2020 IEEE International Conference on Communications, ICC, IEEE, 2020, pp. 1-6.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb32)
- [33] [M. Savi, F. Olivadese, Short-term energy consumption forecasting at the edge: A federated learning approach, IEEE Access 9 (2021) 95949-95969.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb33)
- [34] [Y. Yang, Z. Wang, S. Zhao, J. Wu, An integrated federated learning algorithm for short-term load forecasting, Electr. Power Syst. Res. 214 (2023) 108830.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb34)
- [35] [A.T. Thorgeirsson, S. Scheubner, S. Fünfgeld, F. Gauterin, Probabilistic prediction of energy demand and driving range for electric vehicles with federated learning, IEEE Open J. Veh. Technol. 2 (2021) 151-161.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb35)
- [36] [F. Liu, C. Zeng, L. Zhang, Y. Zhou, Q. Mu, Y. Zhang, L. Zhang, C. Zhu, FedTADBench: Federated time-series anomaly detection benchmark, in: 2022 IEEE 24th Int Conf on High Performance Computing &amp; Communications; 8th Int Conf on Data Science &amp; Systems; 20th Int Conf on Smart City; 8th Int Conf on Dependability in Sensor, Cloud &amp; Big Data Systems &amp; Application (HPCC/DSS/SmartCity/DependSys), IEEE, 2022, pp. 303-310.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb36)
- [37] SmartMeter energy consumption data in London households, 2022, URL https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-londonhouseholds, Accessed December 14, 2022.
- [38] [Y. Liu, Y. Zhou, K. Yang, X. Wang, Unsupervised deep learning for IoT time series, IEEE Internet Things J. (2023).](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb38)
- [39] N. Abdulla, M. Demirci, S. Ozdemir, Design and evaluation of adaptive deep learning models for weather forecasting, Eng. Appl. Artif. Intell. 116 (2022) 105440.
- [40] [K. Bonawitz, H. Eichner, W. Grieskamp, D. Huba, A. Ingerman, V. Ivanov, C. Kiddon, J. Konečn` y, S. Mazzocchi, B. McMahan, et al., Towards federated learning at scale: System design, Proc. Mach. Learn. Syst. 1 (2019) 374-388.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb40)
- [41] [Y. Wang, Q. Chen, T. Hong, C. Kang, Review of smart meter data analytics: Applications, methodologies, and challenges, IEEE Trans. Smart Grid 10 (3) (2018) 3125-3148.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb41)
- [42] [N. Pentreath, Machine Learning with Spark, Packt Publishing Ltd, 2015.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb42)
- [43] [N. Abdulla, M. Demirci, S. Özdemir, Adaptive learning on fog-cloud collaborative architecture for stream data processing, in: 2021 International Symposium on Networks, Computers and Communications, ISNCC, IEEE, 2021, pp. 1-6.](http://refhub.elsevier.com/S2352-4677(24)00071-7/sb43)