---
layout: post                          # (require) default post layout
title: "Time Series Classifications Review"   # (require) a string title
date: 2022-04-22       # (require) a post date
categories: [RCAClassification]          # (custom) some categories, but make sure these categories already exists inside path of `category/`
tags: [RCAClassification]                      # (custom) tags only for meta `property="article:tag"`
---

<br>

# Time-series Classification

## Traditional TSC

1. NN-DTW (Nearest Neighbor coupled with Dynamic Time Warping)

   TSC classifier coupled with a distance function - Dynamic Time Warping(DTW) distance used with a NN classifier. 

2. ensemble

   ensemble of NN classifiers each with different distance measures outperforms individual NN classifier. 

   ensemble of decision trees (RandomForest) or ensemble of different types of discriminant classifiers (SVM, 여러가지 distance 방식을 사용하는 NN, 등)
   여기에선 times series data를 new feature space로 transform하는 단계가 활용된다. 예를들어, shapelets transform 또는 DTW features를 사용하는 방식, 등

3. COTE (Collective Of Transformation-based Ensembles)

   위 ensemble방식을 기반으로 더 발전된 ensemble임. 35 classifiers로 구성되고, 동일 transformation을 통해 difference classifier들을 ensemble하는 방식이 아니라, instead ensemble different classifiers over difference time representations

4. HIVE-COTE (COTE with hierarchical vote system)

   위 COTE에서 hierarchical vote system과 함께 확장된 방식임. probabilistic voting과 함께 hierarchical structure을 leverage해서 COTE보다 더 개선된 성능이 확보된다. two new classifiers, two additional representation transformation domains가 포함되게 됨.
HIVE-COTE는 computationally intensive해서 현실적인 real big data mining 문제에서는 활용되기 어려운것이 단점임. HIVE-COTE는 37 classifiers가 필요하고 algorithm의 hyperparameter의 cross-validating까지 수행되어야함. 
예를 들어 37개의 classifier중 하나를 Shapelet Transform을 수행한다면, time complexity가 O(n^2*l^4)수준까지 올라가게됨. (n=number of time series in the dataset, l=length of time series)
또한, HIVE-COTE의 기반이되는 nearest neighbor algorithm이 classification에 많은 시간을 소모한다. 그래서 real-time setting에서는 HIVE-COTE를 적용하기 어렵다.

<br>

<br>

## DNN architecture TSC

### Deep learning for time series classification

다음 그림과 같이 TSC deep learning framework이 구성된다:

![overview](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/DL_framework_for_TSC.PNG)

deep learning을 사용하면 특히 multivariate time series data를 다룰때에 문제가 되는 "curse of dimensionality"의 영향을 완화할 수 있다 - by leveraging different degree of smoothness in compositional function as well as the parallel computations of the GPUs
또 다른 장점은 NN-DTW와 같은 non-probabilistic classifier과는 다르게 probabilistic decision이 deep learning network으로 인해 만들어진다는 것이다. algorithm이 제공하는 예측값의 confidence를 가늠할 수 있다. 

Deep learning TSC의 family overview:

![overview_DeepLearning_TSC](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/timeseries_classification_model_family.PNG)



TSC family는 크게 generative와 discriminative model로 나뉜다.

Discriminative model은 directly learns the mapping between raw input of a time series (or its hand engineered features) and outputs a probability distribution over the class variables in a dataset.
Distriminative model은 다음과 같이 sub divide 된다:

1) deep learning models with hand engineered features
2) end-to-end deep learning models


end-to-end deep learning model의 경우에는 feature engineering 과정이 필요하지 않은 경우가 많기때문에 domain-specific한 data preprocessing과정이 필수가 아니다.

end-to-end model의 종류로는 MLP, CNN, Hybrid가 있다. MLP의 경우에는 time series data의 temporal information이 lost되기때문에 학습된 features가 충분하지 못한 문제가 있다. CNN의 경우에는 spatially invariant filters (또는 features)를 raw input time series 데이터로부터 추출하고 학습할 수 있다.
CNN 기반의 모델 중, 논문의 실험 결과 가장 성능이 높은 architecture로 ResNet, FCN, Encoder가 확인되었다. 

<br>

<br>

# References

1. Deep learning for time series classification: a review(2019) by Hassan Ismail Fawaz, et al (The final authenticated version is available online at: https://doi.org/10.1007/s10618-019-00619-1.)
2. Johann Faouzi. Time Series Classification: A review of Algorithms and Implementations. (2022) Machine Learning (Emerging Trends and Applications), In press. ffhal-03558165f 
