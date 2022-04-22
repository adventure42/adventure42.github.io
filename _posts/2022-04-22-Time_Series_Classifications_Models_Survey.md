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

   ensemble of NN classifiers each with different distance measures outperforms individual classifier. 

   ensemble of decision trees (RandomForest) or ensemble of different types of discriminant classifiers (SVM, ?)

3. COTE (Collective Of Transformation-based Ensembles)

   ensemble of 35 classifiers, not only ensemble different classifiers over the same transformation, but instead ensemble different classifiers over different time series representations

   Bagnall A, Lines J, Hills J, Bostrom A (2016) Time-series classification with COTE: The collective of transformation-based ensembles. In: International Conference on Data Engineering, pp 1548-1549

4. HIVE-COTE (COTE with hierarchical vote system)

   COTE?? ? ??? ??. probabilistic voting? ?? ??? hierarchical structure? leverage?. HIVE-COTE? ?? ?? accuracy? ????? computationally intensive?? ?. real big data mining ?? ???? ???? ??. (??? ???? Shapelet Transform? computation time complexity? O(n 2 · l 4 )???? ?? ??. n being the number of time series in the dataset and l being the length of a time series.)

   Lines J, Taylor S, Bagnall A (2016) HIVE-COTE: The hierarchical vote collective of transformationbased ensembles for time series classification. In: IEEE International Conference on Data Mining, pp 1041â€“1046



## DNN architecture TSC

### Deep learning for time series classification

???? deep learning framework for TSC? ?? ??? ??:

![overview](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/DL_framework_for_TSC.PNG)

time series data? hierarchical representations? network? ????? design????. .

Deep learning TSC ?? overview:

![overview_DeepLearning_TSC](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/timeseries_classification_model_family.PNG)



TSC family?? discriminative model? raw input of a time series (?? ?? engineer? features)? mapping? ???? ????? class variables?? probability distribution? output??. ? ??? ?? discriminative model? ? ?? ??? sub-divide? ? ??. 

1) deep learning models with hand engineered features
2) end-to-end deep learning models

End-to-end model? classifier? fine-tuning??? ??? ???? feature learning process? ????. ??? domain? ??? ?? pre-processing??? ?? ???? ??? ??. end-to-end model? neural network? architecture? ?? MLP, CNN, Hybrid? ?? ? ??. 

??? CNN??? 3?? ?? - FCN, ResNet, Encoder? ?? ????.

<br>

<br>

### FCN(Fully Convolutional Neural Network)

FCN? convolutional network? ?????, local pooling layer? ?????? ??. ??? length of time series? convolution?? ??? ???. ??, ?? architecture?? final layer? ???? FC layer ??? GAP(global average pooling) layer? ?????, number of parameters? ?? ??????.  

3?? convolutional blocks? ????, ? block? 3?? operations? ?????? -

convolution followed by a batch normalization whose result is fed to a ReLU activation function.

third convolutional block? ??? time dimension ??? ???? average??. (corresponds to the GAP layer) ????? traditional softmax classifier is fully connected to the GAP layer's output.

?? convolutions? stride=1? padding? zero? ???? convolution? ??? ??? exact length of the time series? ??? ? ??. The first convolution contains 128 filters with a filter length equal to 8, followed by a second convolution of 256 filters with a filter length equal to 5 which in turn is fed to a third and final convolutional layer composed of 128 filters, each one with a length equal to 3.

FCN? pooling ?? regularization operation? ??. 

<br>

<br>

### Residual Network (ResNet)

TSC? ?? deep? architecture?? - composed of 11 layers of which the first 9 layers are convolutional followed by a GAP layer that averages the time series across the time dimension. 

The Residual Network's architecture for time series classification:

![ResNet](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/TSC_ResNet.PNG)

ResNet architecture? ?? ??:

- shortcut residual connection between consecutive convolutional layers 

  ResNet? (FCN? ??)?? convolutions? ???? linear shortcut? ???? residual block? output? input? ????, ? connection? ?? ????? gradient? flow?? DNN? vanishing gradient effect? ????? ????? ?? ??? ? ?????? ???.

- residual blocks

  network? 3?? residual block?? ?????? a GAP layer and a final softmax classifier(whose number of neurons is equal to the number of classes in the dataset)? ????. ? residual block? 3?? convolutions? ???????, convolution? output? residual block? input?? ???? next layer? feed??. ?? convolutions?? filter ??? 64? fix??, activation function??? ReLU?, ??? ? ??? batch normalization operation? ????. ? residual block? 1st, 2nd, 3rd convolutions? ???? filter? length? 8, 5, 3?? ????.

- number of parameters

  ??? FCN? ????, (? ??? layer? ????) network? layer?? invariant number of parameters (across different datasets)? ?????. ??? ?? source dataset? model? pre-train??, transfering? fine-tuning? ?? target dataset? ?? ? ??. 

<br>

<br>

### Encoder 

Encoder? hybrid deep CNN?? FCN? ???? ???? architecture??. ?, GAP layer? attention layer? ?????. ? ??? variants? ??:

- train the model from scratch (in end-to-end fashion on a target dataset)
- pre-train the architecture on a source dataset, and then fine-tune it on a target dataset

(??? variant?? transfer learning technique? ???? ??? variant? ? ?? accuracy? ??? ? ??.)

FCN? ????, first three layers are convolutional with some relatively small modifications. 

first convolution - composed of 128 filters of length 5

second convolution - composed of 256 filters of length 11

third convolution - composed of 512 filters of length 21

Each convolution followed by an instance normalization operation, then output is fed to the PReLU activation function. then followed by a dropout operation (rate=0.2) and a final max pooling of length 2.

third convolutional layer? attention mechaanism? feed???, network? time series? ??? ???? ??? ????? ??. 

More precisely, to implement this technique, the input MTS is multiplied with a second MTS of the same length and number of channels, except that the latter has gone through the softmax function. Each element in the second MTS will act as a weight for the first MTS, thus enabling the network to learn the importance of each element (time stamp).

?????, latter layer? softmax classifier? fully connected??? ???? ????. 

GAP layer? attention layer? ???? ? ??? FCN? ????? ??? ??:

- PReLU activation function where an additional parameter is added for each filter to enable learning the slope of the function
- dropout regularization technique 
- max pooling operation

<br>

<br>

## Multivariate Time Series (MTS) data

MTS dataset? unequal length time series? ??? ? ??. Deep learning model? architecture? length of the input time series? ???????, ?? ?? GPU? ???? parallel computation? ??? ? ?????, time series length? ??? ??? ??? ?? ????. ????? ??? ????, ? time series? ?? ? time series' length? ????? linearly interpolate the time series of each dimension for every given MTS.

<br>

<Br>

# TSC model's performance 

12?? multivariate time series datasets? ???? end-to-end deep learning TSC ?? ??? ?????. 

MLP, FCN, ResNet, Encoder, MCNN, t-LeNet, MCDCNN, Time-CNN? ??? ?? ?? ? ??, deep CNN??? **ResNet, FCN, Encoder**?? ? superior? ???? ?? ???. 

??? ? deep learning TSC model? ???:

![TSC](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/TSC_experiment_conditions.PNG)

<br>

Dataset? ??? ??? ??? ?? ? deep learning TSC model? ??? ??? ?????. 

dataset theme? ??:

![by_theme](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/TSC_performance_by_dataset_themes.PNG)



dataset? time series length? ??:

![by_timeseries_length](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/TSC_performance_by_timeseries_length.PNG)



dataset? train size? ??:

![by_trainsize](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/TSC_performance_by_trainsize.PNG)



<br>

<Br>

# References

1. [GitHub - hfawaz/dl-4-tsc: Deep Learning for Time Series Classification](https://github.com/hfawaz/dl-4-tsc)
1. Deep learning for time series classification: a review(2019) by Hassan Ismail Fawaz, et al (The final authenticated version is available online at: https://doi.org/10.1007/s10618-019-00619-1.)



