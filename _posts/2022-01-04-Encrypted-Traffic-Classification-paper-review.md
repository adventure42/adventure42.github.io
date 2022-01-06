---
layout: post                          # (require) default post layout
title: "Encrypted Traffic Classification paper review"                   # (require) a string title
date: 2022-01-04       # (require) a post date
categories: [paperreview]          # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [EncyptedTrafficClassification]                      # (custom) tags only for meta `property="article:tag"`

---

<br>

# Encrypted Traffic Classification paper review

<br>

주요 papers:

1. Deep learning for Network Traffic Classification (Bayat, 2021)
2. A Survey of HTTPS Traffic and Services Identification approaches(Shbair, 2020)
3. Service Level Monitoring of HTTPS Traffic(Shbair, 2017)
4. A Multi-Level Framework to Identify HTTPS Services (Shbair, 2016)
5. Encrypted Internet traffic classification using a supervised Spiking Neural Network (Rasteh, 2021)
6. Network Traffic Classifier With Convolutional and Recurrent Neural Networks for Internet of Things (Lopez-Martin, 2017)
7. FlowPic: Encrypted internet traffic classification is as easy as image recognition (Shapira, 2019)
8. Deep Packet: A Novel Approach For Encrypted Traffic Classification Using Deep Learning(Lotfollahi, 2018)

<br>

<br>

## Deep learning for Network Traffic Classification (Bayat, 2021) 

**source code available at: https://github.com/niloofarbayat/NetworkClassification**



min connections의 값을 변경해가면서 10-fold cross validation result를 확보함. network traffic classifier에 가장 현실적이고, 가장 어려운 scenario가 min connections =100. 이 조건은 Shbair의 연구에서도 동일한 same dataset을 사용하기위한 threshold로 사용되었음.  100 min connections 사용 시, 532 possible SNI classes로 분류하게됨. (when we restrict ourselves to just Google Chrome data.)

이 조건에서 Random Forest Classifier는 **92.2%**의 10-fold cross validation accuracy를 확보함. 

preliminary baseline RNN model은 (trained on packet sequences) 67.8% accuracy를 확보함. 

two-layer baseline CNN trained on packet sequences achieves 62.4 % accuracy. 

The combined CNN-RNNs for packet, payload, and inter-arrival time sequences achieve 77.1%, 78.1%, and 63.2%

그리고 이들을 통합한 ensemble CNN-RNN는 **82.3%** accuracy를 확보함.

결국 deep learning architecure을 활용한것 만으로는 machine learning (random forset)모델의 성능 대비 분류성능이 더 좋아지지 못함. 

그러나 최종적으로 ranodm forest classifier와 CNN+RNN network의 ensemble 모델로 단순 Random Forest classifier보다 더 좋은 best performance를 확보할 수 있음.

<br>

<br>

## A Survey of HTTPS Traffic and Services Identification approaches(Shbair, 2020)

TLS protocol을 사용하는 application중에서 HTTP가 가장 많이 사용된다.

HTTPS identification = detecting and identifying the HTTP traffic inside TLS

TLS identification은 주로 다음과 같이 2개의 방식으로 나뉜다:

- using the TLS record format
- employing machine learning approach over the encrypted payload

basic requirements : training dataset (solved examples와 같은), statistical features, algorithms, evaluation techniques

learning process는 다음과 같이 3개의 단계로 나누어짐 : training, classification, validation

- training

  statistical features and machine learning algorithms are trained to make prediction

- classification

  output of training phase = model. use this model to identify unseen data

- validation

  result of classificaiton are validated to measure the performance of the classification model (accuracy used as metric)

![](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/HTTPS_identification_methods.PNG)

우려사항:

Many recent approaches, intend to identify HTTPS services based on the plain-text information that appears in the TLS handshake phase or based on the statistical signature of HTTPS web services. However, the reliability of handshake information for identification still need improvement, as discussed with SNI and SSL certificate. 

The reliability of machine learning method has a challenge with the increased complexity of web applications that can be easily extended with new functionalities that may change the application behaviour and the statistical-signature. 

This complexity creates an overhead to the machine learning based identification methods to re-evaluate their statistical features and re-train classification models regularly to keep their methods effective with updated changes.

<br>

<br>

## Service Level Monitoring of HTTPS Traffic(Shbair, 2017)

traffic flow의 data를 잘 활용하면 매우 높은 accuracy로 분류가 가능해보인다. 

machine learning based methods

- **packet payload에서 extract statistical signature** (Haffner, 2005)  handshake phase에서부터 words를 extract한다. 

  first 64-Bytes of a reassembled TCP data stream is encoded in a binary vector and used as input for different machine learning algorithms (Naïve Bayes, AdaBoost and Maximum Entropy). The evaluation over dataset from ISP shows that AdaBoost identifies HTTPS traffic with 99.2% accuracy.)

- web application behavior를 signature로 사용해서 accessed website를 identify한다. use the fact that some information remains intact after encryption like **packet size, timing, and direction**, to identify the common application protocols by applying **k-nearest neighbors algorithm (KNN)** and Hidden Markov Model (HMM). The KNN algorithm detects HTTPS flows with 100% accuracy and the HMM algorithm performs 88% accuracy.

- 먼저, protocol-format을 기반으로 **TLS traffic을 감지**한다. **그 다음, HTTP traffic in TLS channels is recognized** by applying machine learning methods. use the size of the first five packets of a TCP connection to identify HTTPS applications with 81.8% accuracy rate. 

  The performance of their classifier has been improved (up to 85%) in [47] by adding a pre-processing phase, where they first detect the TLS traffic based on protocol-format and then identify the HTTP traffic within TLS.

- 위 방법에서 더 개선된 방안으로, ml algorithm으로 Naive Bayes를 활용하고, **8개의 statistical features를 활용**한다. Mean, Maximum, Minimum of packet length, and Mean, Maximum, Minimum of Inter-Arrival time, flow duration and number of packets.  

  Using a private dataset, results show the ability to recognize over 99% of TLS traffic and to detect the HTTPS traffic with 93.13% accuracy.

<br>

### issues

1. 공식적인 성능 평가 기준/ dataset 부족

위에서 언급한 연구 결과 모두 private HTTPS dataset을 바탕으로 얻은 성능 결과이다. (due to privacy, security issues) 공정하게 다른 모델들을 비교/평가할 수 있는 공식적인 public dataset이 없다. (as of 2017)

Levillain et al. [59] analyse some of the existing public HTTPS datasets published between 2010 and 2015. Their investigation shows that some datasets contain only SSL certificates information, while in the other ones the whole TLS answers were truncated

2. the third party의 존재

facebook의 경우 content delivery를 third party인  Akamai Content Delivery Network (CDN)를 통해 수행한다. 그래서 facebook의 content는 facebook.com이 아닌  "akamaihd.net"로 확인되기때문에 website fingerprinting에는 문제가 발생할 수 있다. 

<br>

### machine learning based method

The possible techniques can be based on website fingerprinting or machine learning approaches, on values extracted from: 

- DNS, 
- IP address, 
- SNI, or SSL certificate, on an HTTPS proxy server or on the acquisition of TLS encryption keys

3 novel features가 활용된다:

1. service proximity - the existence of a POP, IMAP or SMTP server within a domain is a strong indication that a mail server exists.
2. activity profiles
3. periodicity

<br>

<br>

## A Multi-Level Framework to Identify HTTPS Services (Shbair, 2016)

이 논문에서 저자의 contribution includes: 

- database: collecting HTTPS traces from user sessions and using the SNI extension for labelling each connection. 
- model: 새로운 statistical framework을 제안했다. their framework includes the standard packet and inter-arrival time statistics, as well as additional statistical features related to the encrypted payload. 

They achieve their best results using **Decision Tree** and **Random Forest classifiers.**

<br>

<br>

## Encrypted Internet traffic classification using a supervised Spiking Neural Network (Rasteh, 2021)

기존의 encrypted traffic classification/ recognition technique인 payload inspection technique가 더 이상 effective하지 않음. 

이 논문에서는 packet size와 time of arrival만 특성으로 활용하고 (ANN이 아닌) SNN(spiking neural networks)를 훈련시켜서 encrypte traffic 분류 모델을 만들었다. 

성능을 확인할 때에 Tor, VPN traffic의 분류에서도 accuracy를 확인했음. --> The average accuracy for Tor traffic rises from 67.8% to 98.6%, and for unencrypted traffic from 85% to 99.4%. For VPN traffic it rises from 98.4% to 99.8%. The number of errors is thus divided by a factor 8 for VPN traffic and by a factor 20 for Tor and unencrypted traffic.

SNN : ANN inspired by how biological neurons operate, used for two reasons - 

1) SNN is able to recognize time-related data packet features
2) can be implemented efficiently on neuromorphic hardware with a low energy footprint

여기에서는 a very simple feedforward SNN, with only one fully-connected hidden layer, and trained in a supervised manner using the newly introduced method known as Surrogate Gradient Learning. 

Surprisingly, such a simple SNN reached an accuracy of 95.9% on ISCX datasets.

더 높은 accuracy외에도 simplicity에 큰 개선을 만들었다. input size, number of neurons, trainable parameters are all reduced by one to four orders of magnitude.

such good accuracy의 원인을 분석 -  It turns out that, beyond spatial (i.e. packet size) features, the SNN also exploits temporal ones, mostly the nearly synchronous (i.e, within a 200ms range) arrival times of packets with certain sizes. Taken together, these results show that SNNs are an excellent fit for encrypted internet traffic classification: they can be more accurate than conventional artificial neural networks (ANN), and they could be implemented efficiently on low power embedded systems.

what is SNN?

biological neural network을 imitate하는 neural network이다. neuron은 membrane potential이 specific value에 도달하면, "spike"라고 불리는 electrical impulse를  fire한다. 이는 spikes로 구성된 signal을 만들고 다음 neuron에게도 propagate된다. SNN에서는 its binary nature때문에 classic training방식인 back-propagation gradient descent가 적용될 수 없다. SNN은 time-related pattern을 감지하는데에 유용한다. 

SNN에는 back propagation대신에 rate-based (meaning they only consider the number of spikes inside a temporal window) ignoring their times. 이런점은 servere limitation을 발생시킨다 - different spike trains can have the same number of spikes but present distinct temporal patterns. 

이를 해결하기위해 surrogate gradient algorithm과 같이 더 local한 방식을 활용하는 algorithm이 적용되었다. (Neftci et al. 2019 [18]의 논문에서 제안되었던 방법) It consists of approximating the true gradient of the loss function while keeping good optimization properties.

<br>

<br>

## Network Traffic Classifier With Convolutional and Recurrent Neural Networks for Internet of Things (Lopez-Martin, 2017)

one of the first to apply RNN and CNN for application level identification problem.

Their CNN-LSTM architecture uses source port, destination port, packet size, TCP window size, and inter-arrival times as features, and beats the standard Random Forest classifier. Importantly, they also find that a large number packets is not necessary, as between 5-15 packets is sufficient to achieve excellent results. 

<br>

<br>

## FlowPic: Encrypted internet traffic classification is as easy as image recognition (Shapira, 2019)

Basic flow data를 picture("FlowPic")로 transform해서  image classification CNN model을 훈련시켜서 flow의 category(browsing, chat, video, etc,...)를 분류하고 application in use를 identify했다.

UNB ISCX dataset를 사용해서 traffic classification을 높은 accuracy의 성능을 검증했다. We can identify a category with very high accuracy even for VPN and Tor traffic. We classified with high success VPN traffic when the training was done for a non-VPN traffic. Our categorization can identify with good success new applications that were not part of the training phase. We can also use the same CNN to classify applications with an accuracy of **99.7%**. 

<br>

<br>

## Deep Packet: A Novel Approach For Encrypted Traffic Classification Using Deep Learning(Lotfollahi, 2018)

**source code available at: https://github.com/PrivPkt/PrivPkt**

### 요약

deep learning을 통해 feature extraction/selection 과정이 따로 없이 autoencoder와 CNN을 사용해서 두 단계 (traffic categorization과 application identification)로 구성된 traffic classifier 모델을 구현 했음. 

성능: After an initial pre-processing phase on data, packets are fed into Deep Packet framework that embeds stacked autoencoder and convolution neural network in order to classify network traffic. Deep packet with CNN as its classification model achieved **recall of 0.98** in application identification task and **0.94** in traffic categorization task.

### why unique

이 논문의 연구 결과가 superior한 이유:

- (network traffic관련 expertise가 기반이 되어야하는) feature extraction 단계 없이 data preprocessing만 진행한 후, deep learning model 훈련을 진행했다. ("cumbersome step of finding and extracting distinguishing features has been omitted.")
- traffic characterization과 application identification을 모두 진행할 수 있다. "can identify traffic at both granular levels (application identification and traffic characterization) with state-of-the-art results compared to the other works conducted on similar dataset" [16, 47]
- can accurately classify one of the hardest class of applications, known to be P2P [20]. This kind of applications routinely uses advanced port obfuscation techniques, embedding their information in well-known protocols’ packets and using random ports to circumvent ISPs’ controlling processes.