## A Survey of Model Compression and Its Feedback Mechanism in Federated Learning

Duy-Dong Le dongld@ueh.edu.vn Industrial University of Ho Chi Minh City (IUH) University of Economics Ho Chi Minh City (UEH) Ho Chi Minh City, Vietnam

The-Bao Pham ∗ ptbao@sgu.edu.vn Sai Gon University (SGU) Ho Chi Minh City, Vietnam

## ABSTRACT

In this paper, we review various model compression methods used in extensive neural networks, such as Quantization, Pruning, Knowledge Distillation, and Weight Sharing. We also focus on their implementation in federated learning environments. Especially, we delve into the feedback model compression mechanism in federated learning. This survey provides valuable insights into the potential advantages and challenges of this approach. Furthermore, the paper presents forward-looking perspectives, charting potential future developments in this dynamic field. It serves as a guide for researchers and practitioners aiming to refine model compression strategies in federated learning, contributing to the growth and practicality of this field.

## CCS CONCEPTS

- General and reference → Surveys and overviews .

## KEYWORDS

Model Compression, Feedback Model Compression, Federated Learning, Big Data, Decentralized Analysis

## ACMReference Format:

Duy-Dong Le, Anh-Khoa Tran, The-Bao Pham, and Tuong-Nguyen Huynh. 2024. A Survey of Model Compression and Its Feedback Mechanism in Federated Learning. In The Fifth Workshop on Intelligent Cross-Data Analysis and Retrieval (ICDAR '24), June 10-14, 2024, Phuket, Thailand. ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/3643488.3660293

## 1 INTRODUCTION

Federated learning (FL) [29], also referred to as "collaborative machine learning," is a transformative concept in machine learning that trains models using decentralized data sources, thus prioritizing

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

ICDAR '24, June 10-14, 2024, Phuket, Thailand

© 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-0549-6/24/06

https://doi.org/10.1145/3643488.3660293

## Anh-Khoa Tran ∗

tran@nict.go.jp National Institute of Information and Communications Technology (NICT) Tokyo, Japan

Tuong-Nguyen Huynh ∗ htnguyen@iuh.edu.vn Industrial University of Ho Chi Minh City (IUH) Ho Chi Minh City, Vietnam

user privacy. This technique involves local device training, which significantly enhances data privacy and security. FL is applicable in various sectors, including healthcare, finance, and smart agriculture [18]. A primary challenge in FL is the efficient transfer of model updates between clients and the central server, which is complicated by the distributed nature of the system. The complexity and size of models intensify this challenge, highlighting the need for improved communication strategies [22]. In this context, model compression methods are increasingly important. These methods aim to reduce the size of models with minimal impact on their performance, offering a vital solution in contexts like multimedia applications where large models can burden communication networks [36].

This review paper provides an in-depth exploration of model compression and its feedback mechanism, specifically within the realm of FL. It thoroughly investigates the latest strategies for compressing large neural networks, critically evaluating the advantages and disadvantages of these techniques. This analysis also aims to deliver a holistic understanding of feedback model compression in FL environments. Furthermore, the paper presents forward-thinking perspectives, aiming to direct researchers and practitioners towards novel approaches for tackling upcoming challenges and fully leveraging the capabilities of FL in a range of applications.

## 2 MODEL COMPRESSION

Model compression plays a pivotal role in both machine learning and deep learning. It focuses on minimizing the size of neural network models while striving to maintain or enhance their performance [10]. This process is fundamental for its multitude of advantages. It facilitates resource efficiency, especially for devices with limited resources, accelerates inference times for applications requiring real-time processing, and decreases storage demands. In the context of FL, it contributes to heightened privacy and security. Furthermore, model compression is beneficial for improving energy efficiency in edge computing and Internet of Things (IoT) environments and for reducing costs in extensive deployment scenarios. By optimizing the use of resources, reducing latency, and cutting down expenses, model compression renders machine learning models more feasible and accessible in various applications, ranging from mobile devices [43] to edge computing.

Several comprehensive surveys have delved into various aspects of the field. Berthelier et al. [4] provided an in-depth analysis of implementing deep neural networks on devices with limited resources. Their study concentrated on addressing issues related to model size, computation time, and memory efficiency. In their categorization, they distinguished between methods like compression (including pruning and quantization) and architecture optimization, highlighting the criticality of computational efficiency and model optimization. On the other hand, Li Zhuo et al. [23] explored the intricacies of deep neural networks in the domain of computer vision. Their discussion encompassed model compression strategies such as pruning, quantization, and knowledge distillation. They also charted potential future research trajectories, focusing on aspects like size reduction, enhancing understanding through pruning and decomposition, and integrating various techniques for more efficient compression. Particularly, they emphasized the role of neural architecture search in developing lightweight models.

In the preceding analysis, we delve into different model compression techniques, including:

## 2.1 Quantization

Quantization, a notable technique in model compression, aims to decrease the memory and computational requirements of deep neural networks (DNNs). It achieves this by representing parameters like weights and activations with fewer bits compared to standard full-precision floating-point numbers [5, 25]. As explained by Siddegowda et al. [37], quantization transitions from floating-point to more compact fixed-point representations, yielding substantial power savings and expedited inference, especially when paired with hardware optimized for fixed-point operations. The paper elaborates on Post-Training Quantization (PTQ), which involves converting already trained networks to fixed-point formats without modifying the original training process, and Quantization-Aware Training (QAT). QAT incorporates the effects of quantization noise during the training phase to achieve enhanced outcomes. The paper also discusses recent advancements in PTQ that attain near-floatingpoint accuracy with 8-bit quantization and provides insights into debugging methods and established procedures in the QAT pipeline.

Quantization within model compression follows a structured approach. It begins with the training of a deep neural network (DNN) using either 32-bit or 64-bit floating-point numbers. While these formats are known for their high precision, they are also resourceintensive [32]. In the subsequent quantization phase, the model's weights and/or activations are converted from these high-precision floating-point values to lower-precision fixed-point representations. These representations often use formats like 16-bit, 8-bit, or even binary (1-bit) [5, 53]. The transition to lower precision introduces quantization errors. Mitigating these errors involves careful selection of the quantization bit-width and the application of techniques like rounding or scaling [8]. After the quantization step, a fine-tuning phase usually follows. This phase might use a smaller dataset or a reduced learning rate to regain some accuracy lost during quantization and help the model adapt to its new quantized state [25].

Quantized models offer numerous benefits. Their primary advantage is a reduced memory footprint, as they require significantly less memory for storing parameters. This aspect makes them wellsuited for deployment on devices with limited resources [6, 31].

Additionally, their operations at lower precision enable faster inference, particularly on hardware designed for efficient fixed-point or integer operations. Quantized models also contribute to enhanced energy efficiency, a crucial factor for deployment on edge devices and in energy-sensitive environments. Finally, these models are compatible with specialized hardware accelerators like GPUs and TPUs, which are tailored for lower-precision arithmetic, thereby boosting their overall performance.

## 2.2 Pruning

Pruning, a vital model compression technique, focuses on selectively eliminating neurons, connections, or weights from a deep neural network (DNN). The aim is to trim the model's size while maintaining its performance [15, 50]. This procedure is usually applied after the DNN has been trained, a stage at which models are often over-parameterized [41]. Pruning algorithms employ specific criteria, such as the magnitude of weights or activation values, to identify and discard elements deemed less critical. This results in a leaner and more sparse network [47, 54]. To counter any loss in accuracy, pruned models are typically subjected to fine-tuning or retraining [42]. Iterative pruning, a process involving successive rounds of pruning and fine-tuning, progressively increases model efficiency by further eliminating unnecessary components.

Pruning offers several key benefits in the realm of model compression. It significantly reduces the size of models by removing superfluous parameters, thus improving memory efficiency. It also speeds up inference by lowering the number of computations required, a critical aspect for real-time applications. Furthermore, pruning enhances energy efficiency, making it a suitable approach for edge devices operating under constrained power conditions [55]. However, pruning comes with its challenges. The fine-tuning process following pruning can be time-consuming, and there might be limitations in fully restoring the accuracy lost during pruning. The effectiveness of pruning is also sensitive to the chosen criteria and thresholds for parameter elimination, which can affect the balance between model size and accuracy. Moreover, handling the resulting sparsity in pruned models often requires specialized hardware or software optimizations [50]. Despite these hurdles, pruning remains a popular and extensively researched technique in model compression, with continuous efforts directed toward refining pruning algorithms to achieve an ideal equilibrium between minimized model size and preserved performance.

## 2.3 Knowledge Distillation

Knowledge distillation, a technique in model compression, involves transferring knowledge from a larger, more complex 'teacher' model to a smaller, more streamlined 'student' model. This method is essential for reducing the model size and computational requirements while striving to preserve performance, as highlighted in [11, 32]. It is particularly beneficial in scenarios where deploying a large model is not feasible due to constraints in memory or computational power. The technique begins with a well-trained teacher model, which could be a sophisticated deep neural network, such as a convolutional neural network (CNN) or a transformer-based model. This teacher model acts as the source of knowledge. Simultaneously, a smaller and computationally efficient student model is developed with fewer parameters. Despite its inherent limitations in accuracy compared to the teacher model, the student model is trained to emulate and assimilate knowledge from the teacher, a process extensively discussed in [44].

In knowledge distillation, the training of the student model differs from conventional methods. Rather than training directly on the original dataset with one-hot encoded labels, the approach includes two main components. Firstly, the teacher model generates soft targets'-probability distributions over classes for each input-acting as the guiding signals for the student model. The student is then trained to replicate these probability distributions, aligning with the teacher's outputs [21]. Secondly, the loss function includes a regularization term, which encourages the student model to internalize the teacher's knowledge. This term penalizes any significant differences between the student's and teacher's predictions, often employing loss functions like Kullback-Leibler divergence or mean squared error to measure the disparity between their output distributions [40].

During training, the student model assimilates knowledge using soft targets and aims to minimize the loss function, which includes a regularization term. A key element in this process is the introduction of a 'temperature' hyperparameter [49], which regulates the softness of the probability distributions. Higher temperatures result in softer distributions, enabling the student model to grasp more expansive patterns in the teacher's knowledge. Post-training, the student model's efficacy is assessed on the intended task using standard labels. Knowledge distillation not only facilitates model compression and faster inference but also enhances generalization. It is particularly advantageous in scenarios with limited resources while still achieving competitive performance.

## 2.4 Weight Sharing

Weight sharing is an effective model compression technique that minimizes the number of parameters in a neural network. This method works by allocating the same weights across different parts or layers of the network, thereby optimizing the model for better memory and computational efficiency. This is particularly advantageous in scenarios with limited resources. The application of weight sharing varies based on the neural network's architecture and the specific task at hand.

Various neural network architectures incorporate weight-sharing techniques to boost efficiency and simplify model complexity. In convolutional neural networks (CNNs) [13, 17], strategies such as using tied weights allow multiple filters to utilize identical learnable weights. This approach reduces the number of parameters while still effectively capturing diverse input patterns. Depthwise separable convolutions [7, 26] streamline the convolution process by dividing it into depthwise and pointwise stages, thus diminishing parameters by separately convolving input channels with shared filters. In Recurrent Neural Networks (RNNs), the usage of shared recurrent weights [9, 16] across time steps and the implementation of tied embedding weights [33, 35] in sequence-to-sequence models are strategies to limit the parameter count. Neural Architecture Search (NAS) [45, 51] benefits from weight sharing as it allows for the sharing of weights during architecture evaluation. In knowledge distillation contexts [2, 52], smaller student models often employ shared parameters. Similarly, strategies that involve pruning and sparse models [2, 38] integrate weight sharing to create networks that are both efficient and sparser. For Graph Neural Networks (GNNs) [12, 28], weight sharing is utilized in convolutional layers to reduce parameter counts, such as by sharing weights among different nodes in a graph or across various layers.

Thecentral principle of weight-sharing techniques is to capitalize on redundancies within a model's architecture, thereby decreasing the number of unique, learnable parameters. This leads to models that are more memory and computation-efficient, fitting for deployment in environments with limited resources, or in applications where speed and memory efficiency are paramount. However, it is crucial to maintain a balance between the degree of compression and the performance of the model. Excessive weight sharing might result in reduced expressiveness and potentially lower accuracy.

In the sections above, we have provided a detailed examination of a range of model compression techniques prevalent in machine learning. These methods offer vital strategies to improve the efficiency, communication, and deployment of machine learning models across a diverse spectrum of applications. For an easy reference and comparison of these techniques, Table 1 is presented, briefly summarizing their key advantages and challenges.

Table 1 shows that each model compression technique has its pros and cons. Quantization significantly reduces model size and memory use, with faster inference and minimal accuracy loss, but fine-tuning these models can be challenging. Pruning, while reducing computation and preserving accuracy, requires retraining and complex optimization. Knowledge distillation maintains accuracy in a smaller model but needs a large teacher model and extra training time. Weight sharing, efficient for structured tasks via parameter sharing, is constrained by network architecture and less adept at handling complex tasks. The choice of technique depends on application-specific needs and constraints.

## 3 FEEDBACK MODEL COMPRESSION IN FL

## 3.1 Federated Learning

FL is a collaborative training approach where multiple entities, like individual devices or organizations (clients), participate actively without sharing data. In each training iteration, clients receive global model parameters, conduct local training on their data, and then share updated parameters for global model aggregation, as per the specified protocol in Formula. (1).

$$
\min F ( w ) , \text {where} \, F ( w ) \coloneqq \sum _ { k = 1 } ^ { m } \left ( p _ { k } F _ { k } ( w ) \right ) \quad ( 1 )
$$

$$
a n d \ F _ { k } ( w ) = \frac { 1 } { n _ { k } } \sum _ { j _ { k } = 1 } ^ { n _ { k } } ( w ; x _ { j _ { k } } , y _ { j _ { k } } )
$$

where:

𝑚 : 𝑛𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝑝𝑎𝑟𝑡𝑖𝑐𝑖𝑝𝑎𝑛𝑡𝑠

𝑝 𝑘 ≥ 0 and ˝ 𝑘 𝑝𝑘 =

1

𝐹 𝑘 : Local optimization function on participant k in formula 2

𝑛 𝑘 : number of data samples When considering the participants involved, we classify the connection as "cross-devices" when clients consist of terminal devices like mobiles, computers, or unmanned aerial vehicles (UAVs), abbreviated as UAVs. Conversely, when clients represent organizations such as hospitals, banks, or schools, we refer to this relationship as "cross-silos" [18]. The communication infrastructure between these clients can be organized into three architectural approaches: centralized, decentralized, and hierarchical [34]. Moreover, this architectural framework can also be classified according to the distributed data design of the system, which includes vertical federated learning (VFL), horizontal federated learning (HFL), and transfer federated learning (TFL). [22].

Table 1: Model Compression Techniques

| Technique                                                                | Idea                                                                    | Pros                                                                                                                                   | Cons                                                                                                                     |
|--------------------------------------------------------------------------|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| Quantization [5, 6, 8, 25, 31, 32, 37, 53]                               | Reduce the precision of weights/ acti- vations (e.g., 32-bit to 8-bit). | Significant size and memory re- duction. Faster inference. There is little to no accuracy loss, especially post-training quantization. | Challenging fine-tuning. Increased preci- sion reduction may impact accuracy.                                            |
| Pruning [15, 41, 42, 47, 50, 54, 55]                                     | Remove unimpor- tant weights/ neurons.                                  | Considerable model and computation size reduction. Minimal accu- racy impacts with careful pruning.                                    | Requires re- training for accuracy recov- ery. Pruning criteria for optimization can be complex.                         |
| Knowledge Distillation [11, 21, 40, 44, 49]                              | Train smaller "student" mod- els to mimic larger "teacher" models.      | Significant size reduction while maintain- ing accuracy. Knowledge transfer from larger to smaller model.                              | Requires large teacher model and additional training time. Less effec- tive when the teacher- student size gap is small. |
| Weight Sharing [2, 7, 9, 12, 13, 16, 17, 26, 28, 33, 35, 38, 45, 51, 52] | Share weights/ parameters be- tween network parts.                      | Drastic model size reduc- tion through weight sharing. Efficient for structured tasks.                                                 | Limited appli- cability, best for specific architectures. Reduced model capacity may not suit com- plex tasks.           |

## 3.2 Feedback compression in FL

In FL, the training process remains on individual devices, sending the full model compression from each client to the central server can be inefficient and privacy-invasive. To combat this, compressed gradients are employed. Methods such as quantization or pruning diminish the magnitude of the gradient update, retaining crucial information necessary for enhancing the model. [14, 19, 39]. This translates to less data transmission, saving bandwidth, and potentially cloaking sensitive details within the smaller compressed gradients. This trade-off between communication efficiency and accuracy requires careful balancing but holds promise for scaling FL to resource-constrained devices and increasing its applicability to privacy-sensitive settings [46, 48].

Here, we examine several notable works within this domain:

Alyazeed Albasyoni et al. [3] explored the inherent trade-off between the required number of bits for encoding compressed vectors and the resulting compression error. They conducted comprehensive analyses, including worst-case and average-case scenarios, and established precise lower bounds. In their worst-case analysis, they introduced an effective compression operator called Sparse Dithering, which closely approaches the theoretical lower bound. Additionally, in the average-case analysis, they devised a straightforward compression operator known as spherical compression, which naturally attains the lower bound. Consequently, their novel compression methods notably surpass existing techniques. They validated these advancements through numerical experiments to demonstrate their efficacy.

Inspired by the observation that gradients in stochastic gradient descent scenarios often exhibit a high correlation between adjacent rounds due to their shared objective of learning the same model, Kai Liang et al. [24] introduced a practical gradient compression scheme tailored for FL. Their approach utilizes historical gradients for compression and is grounded in Wyner-Ziv coding, all without reliance on probabilistic assumptions. Furthermore, they validated their gradient quantization method using real-world datasets, demonstrating its superior performance compared to previous schemes.

Luke Melas-Kyriazi and Franklyn Wang [30] tackled the challenge of high communication costs in FL by optimizing networks within a subset of their complete parameter space, a concept known as intrinsic dimension in the machine learning community. We leverage the relationship between intrinsic dimension and gradient compressibility to formulate a range of low-bandwidth optimization algorithms, termed intrinsic gradient compression algorithms. They introduced three algorithms within this family, each offering distinct levels of upload and download bandwidth suitable for diverse federated learning scenarios, accompanied by theoretical assurances regarding their efficacy. Furthermore, through large-scale FL experiments involving models with up to 100M parameters, they demonstrated the superior performance of our algorithms compared to existing state-of-the-art gradient compression techniques.

We infer from [3, 24, 30] algorithm 1 which shows how clients give feedback compressed gradients to the server in FL.

## 4 DISCUSSION AND FUTURE LOOK

In [1, 27], they discussed the significance of the divergence threshold, a key factor in deciding when a local model in a multi-model FL setup should transmit its error information to the global model. This concept is particularly relevant in multi-model FL scenarios [22], raising two important questions. The first pertains to identifying the optimal number of models to include in the global ensemble, a goal that might be achieved using genetic algorithms [20]. The second question involves examining whether the assessment of model divergence should be based solely on local model accuracy. There are various methods available to explore the divergence threshold, including approaches like regression, clustering, and classification.

| Algorithm 1 Feedback Compression in FL   | Algorithm 1 Feedback Compression in FL                                              |
|------------------------------------------|-------------------------------------------------------------------------------------|
| 1:                                       | Input: Local datasets, Local model parameters, Global model parameters              |
| 2:                                       | Output: Updated global model parameters                                             |
| 3:                                       |                                                                                     |
| 4:                                       | procedure ClientSideTraining                                                        |
| 5:                                       | Each client device trains a local model using its dataset                           |
| 6:                                       | After a certain number of local iterations, compute gradients of model param- eters |
| 7:                                       | end procedure                                                                       |
| 8:                                       |                                                                                     |
| 9:                                       | procedure CompressGradient                                                          |
| 10:                                      | Compress gradient information before transmission to central server                 |
| 11:                                      | Techniques: quantization, sparsification, or other compression algorithms           |
| 12:                                      | end procedure                                                                       |
| 13:                                      |                                                                                     |
| 14:                                      | procedure FeedbackToCentralServer                                                   |
| 15:                                      | Send compressed gradients to central server                                         |
| 16:                                      | end procedure                                                                       |
| 17:                                      |                                                                                     |
| 18:                                      | procedure ServerSideAggregation                                                     |
| 19:                                      | Decompress and Aggregate gradients at central server (e.g., gradient averaging)     |
| 20:                                      | end procedure                                                                       |
| 21:                                      |                                                                                     |
| 22:                                      | procedure GlobalModelUpdate                                                         |
| 23:                                      | Apply aggregated gradients to global model weights                                  |
| 24:                                      | Send updated global model back to clients for next round of local training          |
| 25:                                      | end procedure                                                                       |

Looking ahead, research in feedback model compression in FL will likely focus on dynamic compression techniques that can adapt to changing environments, with an emphasis on preserving privacy through strategies such as secure aggregation and federated differential privacy. Maintaining communication efficiency is essential, particularly in settings with limited bandwidth or unreliable networks [36, 48]. Key areas for future development include creating compression methods tailored to edge devices, extending FL to cross-domain applications, and devising adaptive, energy-efficient approaches. Moving forward, standardization and benchmarking will become increasingly important, as will considerations of security and robustness. Equally crucial will be the human-centric aspects, such as enhancing user-friendliness and ensuring accountability. The path forward for FL entails continuous innovation to meet the growing and diverse needs of technology while maintaining a strong commitment to privacy and efficiency.

## 5 CONCLUSION

This review has offered an extensive analysis of model compression and its feedback mechanism within FL, highlighting its significance in the broader realm of machine learning. We have delved into a range of compression techniques, paying particular attention to their application in large neural networks and FL environments. This exploration has illuminated the strengths and challenges of these techniques. Looking forward, the review identifies key areas for future research, emphasizing the development of dynamic compression methods that can adapt to changing environments. A strong focus is placed on privacy-enhancing strategies, ensuring efficient communication, optimizing for edge devices, exploring cross-domain applications, and integrating user-centric aspects. This trajectory is defined by a dedication to continuous innovation, custom-tailored to address the ever-evolving requirements of technological progress and varied application necessities, all the while upholding steadfast adherence to the fundamental principles of privacy and efficiency.

## ACKNOWLEDGMENTS

This research is funded by the University of Economics Ho Chi Minh City (UEH) Vietnam

## REFERENCES

- [1] Muntadher Qasim Abdulhasan, Mustafa Ismael Salman, Chee Kyun Ng, Nor Kamariah Noordin, Shaiful Jahari Hashim, and Fazirulhisham Hashim. 2015. An adaptive threshold feedback compression scheme based on channel quality indicator (CQI) in long term evolution (LTE) system. Wireless Personal Communications 82 (2015), 2323-2349.
- [2] Nima Aghli and Eraldo Ribeiro. 2021. Combining Weight Pruning and Knowledge Distillation For CNN Compression. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) (2021), 3185-3192.
- [3] Alyazeed Albasyoni, Mher Safaryan, Laurent Condat, and Peter Richtárik. 2020. Optimal gradient compression for distributed and federated learning. arXiv preprint arXiv:2010.03246 (2020).
- [4] Anthony Berthelier, Thierry Chateau, Stefan Duffner, Christophe Garcia, and Christophe Blanc. 2020. Deep Model Compression and Architecture Optimization for Embedded Systems: A Survey. Journal of Signal Processing Systems 93 (2020), 863 - 878.
- [5] Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami, Michael W Mahoney, and Kurt Keutzer. 2020. ZeroQ: A Novel Zero Shot Quantization Framework. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) . 13169-13178.
- [6] Song Cheng, Zixuan Li, Yongsen Wang, Wanbing Zou, Yumei Zhou, Delong Shang, and Shushan Qiao. 2021. Gradient Corrected Approximation for Binary Neural Networks. IEICE TRANSACTIONS on Information and Systems 104, 10 (2021), 1784-1788.
- [7] François Chollet. 2016. Xception: Deep Learning with Depthwise Separable Convolutions. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016), 1800-1807.
- [8] Wesley Cooke, Zihao Mo, and Weiming Xiang. 2023. Guaranteed Quantization Error Computation for Neural Network Model Compression. 2023 IEEE International Conference on Industrial Technology (ICIT) (2023), 1-4.
- [9] Greg Diamos, Shubho Sengupta, Bryan Catanzaro, Mike Chrzanowski, Adam Coates, Erich Elsen, Jesse Engel, Awni Hannun, and Sanjeev Satheesh. 2016. Persistent rnns: Stashing recurrent weights on-chip. In International Conference on Machine Learning . PMLR, 2024-2033.
- [10] Shiming Ge, Zhao Luo, Shengwei Zhao, Xin Jin, and Xiao-Yu Zhang. 2017. Compressing deep neural networks for efficient visual inference. In 2017 IEEE International Conference on Multimedia and Expo (ICME) . 667-672. https: //doi.org/10.1109/ICME.2017.8019465
- [11] Jianping Gou, Baosheng Yu, Stephen J Maybank, and Dacheng Tao. 2021. Knowledge distillation: A survey. International Journal of Computer Vision 129 (2021), 1789-1819.
- [12] Xiaotian Han, Tong Zhao, Yozen Liu, Xia Hu, and Neil Shah. 2022. Mlpinit: Embarrassingly simple gnn training acceleration with mlp initialization. arXiv preprint arXiv:2210.00102 (2022).
- [13] Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. 2017. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861 (2017).
- [14] Shengyuan Hu, Jack Goetz, Kshitiz Malik, Hongyuan Zhan, Zhe Liu, and Yue Liu. 2022. Fedsynth: Gradient compression via synthetic data in federated learning. arXiv preprint arXiv:2204.01273 (2022).
- [15] Berivan Isik, Albert No, and Tsachy Weissman. 2021. Rate-Distortion Theoretic Model Compression: Successive Refinement for Pruning.
- [16] Qinjun Jiang and Matthew D. Sinclair. 2021. Reducing Synchronization Overhead for Persistent RNNs.
- [17] Rui-Yang Ju, Ting-Yu Lin, Jia-Hao Jian, and Jen-Shiun Chiang. 2023. Efficient convolutional neural networks on Raspberry Pi for image classification. Journal of Real-Time Image Processing 20, 2 (2023), 21.
- [18] Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode,

Rachel Cummings, et al. 2021. Advances and open problems in federated learning. Foundations and Trends® in Machine Learning 14, 1-2 (2021), 1-210.

- [19] Sai Praneeth Karimireddy, Quentin Rebjock, Sebastian Stich, and Martin Jaggi. 2019. Error feedback fixes signsgd and other gradient compression schemes. (2019), 3252-3261.
- [20] Sourabh Katoch, Sumit Singh Chauhan, and Vijay Kumar. 2020. A review on genetic algorithm: past, present, and future. Multimedia Tools and Applications 80 (2020), 8091 - 8126.
- [21] Petros Katsileros, Nikiforos Mandilaras, Dimitrios Mallis, Vassilis Pitsikalis, Stavros Theodorakis, and Gil Chamiel. 2022. An Incremental Learning framework for Large-scale CTR Prediction. (2022), 490-493.
- [22] Duy-Dong Le, Anh-Khoa Tran, Minh-Son Dao, Kieu-Chinh Nguyen-Ly, HoangSon Le, Xuan-Dao Nguyen-Thi, Thanh-Qui Pham, Van-Luong Nguyen, and BachYen Nguyen-Thi. 2022. Insights into multi-model federated learning: An advanced approach for air quality index forecasting. Algorithms 15, 11 (2022), 434.
- [23] Zhuo Li, Hengyi Li, and Lin Meng. 2023. Model Compression for Deep Neural Networks: A Survey. Comput. 12 (2023), 60.
- [24] Kai Liang, Huiru Zhong, Haoning Chen, and Youlong Wu. 2021. Wyner-Ziv gradient compression for federated learning. arXiv preprint arXiv:2111.08277 (2021).
- [25] Yuang Liu, Wei Zhang, and Jun Wang. 2021. Zero-shot Adversarial Quantization. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2021), 1512-1521.
- [26] Gangzhao Lu, Weizhe Zhang, and Zheng Wang. 2021. Optimizing depthwise separable convolution operations on gpus. IEEE Transactions on Parallel and Distributed Systems 33, 1 (2021), 70-87.
- [27] Yuanhua Lv and ChengXiang Zhai. 2014. Revisiting the Divergence Minimization Feedback Model. Proceedings of the 23rd ACM International Conference on Conference on Information and Knowledge Management (2014).
- [28] Xiaojun Ma, Qin Chen, Yuanyi Ren, Guojie Song, and Liang Wang. 2022. Metaweight graph neural network: Push the limits beyond global homophily. (2022), 1270-1280.
- [29] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. 2017. Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics . PMLR, 1273-1282.
- [30] Luke Melas-Kyriazi and Franklyn Wang. 2021. Intrinisic Gradient Compression for Federated Learning. arXiv preprint arXiv:2112.02656 (2021).
- [31] Georgii Sergeevich Novikov, Daniel Bershatsky, Julia Gusak, Alex Shonenkov, Denis Valerievich Dimitrov, and Ivan Oseledets. 2023. Few-bit backward: Quantized gradients of activation functions for memory footprint reduction. (2023), 26363-26381.
- [32] Antonio Polino, Razvan Pascanu, and Dan Alistarh. 2018. Model compression via distillation and quantization. ArXiv abs/1802.05668 (2018).
- [33] Ofir Press and Lior Wolf. 2016. Using the Output Embedding to Improve Language Models. In Conference of the European Chapter of the Association for Computational Linguistics .
- [34] Nicola Rieke, Jonny Hancox, Wenqi Li, Fausto Milletarì, Holger R Roth, Shadi Albarqouni, Spyridon Bakas, Mathieu N Galtier, Bennett A Landman, Klaus MaierHein, et al. 2020. The future of digital health with federated learning. NPJ Digital Medicine, 3, 119. (2020).
- [35] Mohammed Saeed and Paolo Papotti. 2022. You Are My Type! Type Embeddings for Pre-trained Language Models. In Conference on Empirical Methods in Natural Language Processing .
- [36] Suhail Mohmad Shah and Vincent KN Lau. 2021. Model compression for communication efficient federated learning. IEEE Transactions on Neural Networks and Learning Systems (2021).
- [37] Sangeetha Siddegowda, Marios Fournarakis, Markus Nagel, Tijmen Blankevoort, Chirag Patel, and Abhijit Khobare. 2022. Neural network quantization with ai model efficiency toolkit (aimet). arXiv preprint arXiv:2201.08442 (2022).
- [38] Suraj Srinivas, Andrey Kuzmin, Markus Nagel, Mart van Baalen, Andrii Skliar, and Tijmen Blankevoort. 2022. Cyclical pruning for sparse neural networks. (2022), 2762-2771.
- [39] Sebastian U Stich and Sai Praneeth Karimireddy. 2020. The error-feedback framework: Better rates for sgd with delayed gradients and compressed updates. The Journal of Machine Learning Research 21, 1 (2020), 9613-9648.
- [40] Ye Tian, Liguo Zhang, Jianguo Sun, Guisheng Yin, and Yuxin Dong. 2022. Consistency regularization teacher-student semi-supervised learning method for target recognition in SAR images. The Visual Computer 38, 12 (2022), 4179-4192.
- [41] Sunil Vadera and Salem Ameen. 2022. Methods for Pruning Deep Neural Networks. IEEE Access 10 (2022), 63280-63300. https://doi.org/10.1109/ACCESS. 2022.3182659
- [42] Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, et al. 2022. Model soups: averaging weights of multiple finetuned models improves accuracy without increasing inference time. (2022), 23965-23998.
- [43] Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng. 2015. Quantized Convolutional Neural Networks for Mobile Devices. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2015), 4820-4828.
- [44] Xiang Wu, Ran He, Yibo Hu, and Zhenan Sun. 2020. Learning an evolutionary embedding via massive knowledge distillation. International Journal of Computer Vision 128 (2020), 2089-2106.
- [45] Lingxi Xie, Xin Chen, Kaifeng Bi, Longhui Wei, Yuhui Xu, Lanfei Wang, Zhengsu Chen, An Xiao, Jianlong Chang, Xiaopeng Zhang, et al. 2021. Weight-sharing neural architecture search: A battle to shrink the optimization gap. ACM Computing Surveys (CSUR) 54, 9 (2021), 1-37.
- [46] Ye Xue, Liqun Su, and Vincent KN Lau. 2022. FedOComp: Two-timescale online gradient compression for over-the-air federated learning. IEEE Internet of Things Journal 9, 19 (2022), 19330-19345.
- [47] Nakyeong Yang, Yunah Jang, Hwanhee Lee, Seohyeong Jeong, and Kyomin Jung. 2023. Task-specific Compression for Multi-task Language Models using Attribution-based Pruning. In Findings of the Association for Computational Linguistics: EACL 2023 . 582-592.
- [48] TJ Yang, Y Xiao, G Motta, F Beaufays, R Mathews, and M Chen. 2022. Online Model Compression for Federated Learning with Large Models. ArXiv abs/2205.03494 (2022).
- [49] Mengyang Yuan, Bo Lang, and Fengnan Quan. 2023. Student-friendly Knowledge Distillation. ArXiv abs/2305.10893 (2023).
- [50] Mingyang Zhang, Xinyi Yu, Jingtao Rong, and Linlin Ou. 2022. Graph pruning for model compression. Applied Intelligence 52, 10 (2022), 11244-11256.
- [51] Tunhou Zhang, Dehua Cheng, Yuchen He, Zhengxing Chen, Xiaoliang Dai, Liang Xiong, Feng Yan, Hai Li, Yiran Chen, and Wei Wen. 2023. NASRec: weight sharing neural architecture search for recommender systems. (2023), 1199-1207.
- [52] Qi Zhao, Shuchang Lyu, Lijiang Chen, Binghao Liu, Ting-Bing Xu, Guangliang Cheng, and Wenquan Feng. 2023. Learn by Oneself: Exploiting Weight-Sharing Potential in Knowledge Distillation Guided Ensemble Network. IEEE Transactions on Circuits and Systems for Video Technology (2023).
- [53] Kai Zhen, Hieu Duy Nguyen, Raviteja Chinta, Nathan Susanj, Athanasios Mouchtaris, Tariq Afzal, and Ariya Rastrow. 2022. Sub-8-Bit Quantization Aware Training for 8-Bit Neural Network Accelerator with On-Device Speech Recognition. In Interspeech .
- [54] Qinghe Zheng, Xinyu Tian, Mingqiang Yang, Yulin Wu, and Huake Su. 2019. PACBayesian framework based drop-path method for 2D discriminative convolutional network pruning. Multidimensional Systems and Signal Processing 31 (2019), 793 - 827.
- [55] Michael Zhu and Suyog Gupta. 2017. To prune, or not to prune: exploring the efficacy of pruning for model compression. ArXiv abs/1710.01878 (2017).