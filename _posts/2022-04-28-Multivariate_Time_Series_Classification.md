---
layout: post                          # (require) default post layout
title: "Multivariate Time-series Classification"   # (require) a string title
date: 2022-04-28       # (require) a post date
categories: [RCAClassification]          # (custom) some categories, but make sure these categories already exists inside path of `category/`
tags: [RCAClassification]                      # (custom) tags only for meta `property="article:tag"`

---

<br>

# Multivariate Time Series Classification 

## Multivariate Time Series (MTS) data

시계열 데이터중, 여러개의 feature dimension을 가진 데이터는 multivariate time series data로 분류된다. 

연구에 주로 활용되는 public dataset은 다음과 같다.

![MTS dataset](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/multivariate_time_series_dataset.PNG)

<br>

<Br>

## TSC model's performance 

MTS data는 특히 "curse of dimensionality"의 문제를 deep learning model을 통해 해소할 수 있다 - by leveraging different degree of smoothness in compositional function (also using multiple GPU's for distributed computation)

위 12가지의 multivariate time series datasets를 기반으로 end-to-end deep learning TSC의 성능을 확인한 결과이다. end-to-end deep learning model로는 MLP, FCN, ResNet, Encoder, MCNN, t-LeNet, MCDCNN, Time-CNN가 시험되었고, 이중에서는 ResNet, FCN, Encoder의 성능이 가능 우월했다.

Deep learning TSC model의 구조 및 hyperparameter:

![TSC](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/TSC_experiment_conditions.PNG)

<br>

deep learning model 성능 비교:

![result](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/result.PNG)



<br>

Dataset의 테마, 시계열 time length, 그리고 dataset의 size에 따라서 각각 다른 성능이 확인되지만, ResNet과 FCN에서 가장 높은 성능이 확인된다. 

dataset의 themes에 따른 차이:

![by_theme](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/TSC_performance_by_dataset_themes.PNG)



dataset의 time series length에 따른 차이:

![by_timeseries_length](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/TSC_performance_by_timeseries_length.PNG)



dataset의 train size에 따른 차이:

![by_trainsize](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/TSC_performance_by_trainsize.PNG)

<br>

<Br>

# GTN for MTS classification

GTN(gated transformer network)를 사용해서 multivariate time series classification problem을 해결했다. 

그림과 같이 two towers of transformer가 활용되어서 channel-wise 그리고 step-wise correlations를 model할 수 있다.

<img src="https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/GTN.PNG" alt="GTN" style="zoom:80%;" />

13개의 MTS 데이터셋을 기반으로 확인한 분류성능은 다음과 같다.

![GTN result](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/GTN_result.PNG)

현재 분석하고있는 데이터 셋과 비슷한 특성을 가진 ArabicDigits 데이터 셋으로는 GTN보다 ResNet에서 더 높은 분류 정확도가 확인되었다.

<br>

<br>

# References

1. [paper] Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline by Zhiguang Wang, et al (2016) [link](https://arxiv.org/pdf/1611.06455.pdf)

   [git repo] [UCR_Time_Series_Classification_Deep_Learning_Baseline](https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline)

2. [paper] Gated Transformer Networks for Multivariate Time Series Classification by Minghao Liu, etal (2021) [link](https://arxiv.org/pdf/2103.14438.pdf) 

   [git repo] [Gated-Transformer-on-MTS][https://github.com/ZZUFaceBookDL/GTN]

