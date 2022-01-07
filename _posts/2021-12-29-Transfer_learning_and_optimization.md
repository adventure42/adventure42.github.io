---
layout: post                          # (require) default post layout
title: "Transfer learning and optimization"                   # (require) a string title
date: 2021-12-29       # (require) a post date
categories: [deeplearning]          # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [CNN]                      # (custom) tags only for meta `property="article:tag"`
---



# Transfer learning & Optimization

**transfer learning vs. retraining**

![transfer_learning_vs_retraining](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/transfer_learning_vs_retraining.PNG)



## Transfer training/ learning

이미 훈련된 CNN을 가져와서 내가 원하는 부분만 내가 해결하려는 문제에 맞게 변경하여 예측값을 찾는 것이다. 훈련된 CNN의 마지막 layer를(classification을 위한 softmax가 있는 층) 잘라서 내가 원하는 층으로 바꾸는 것이다. 가져온 CNN내의 trained weights와 biases는 그대로 유지되고, 내가 마지막 부분으로 추가한 층만 훈련이 되어서 내가 원하는 문제에 활용 될 수 있다. 단, 이 방법을 사용하기 적절한 경우는 가져온 CNN이 훈련되었었던 dataset이 내 문제의 dataset과 "close enough"할때이므로 주의해야한다. 

가져와서 그대로 사용하려는 층을 "freeze"하고, 내 문제에 맞게 변경해서 사용하려는 층은 "unfreeze"한다고 표현한다. 그래서 frozen initial layers (top conv layers) + unfrozen layer의 구성으로 training을 진행하여 원하는 모델을 구현할 수 있다. 

![전이학습](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/transfer_learning_diagram.png)

Neural network 하위층의 처음 few layers에서 network은 image의 edges (horizontal, vertical lines)와 shapes부터 특성들을 학습한다. 그렇게 상위층으로 올라가면서 specific layer에서 network은 classification의 목적에 알맞게 이미지의 category를 예측할 수 있게된다.

<br>

<br>

### Examples

1. keras에서 제공하는 practice ipynb(via Google Colab): 

[https://colab.research.google.com/drive/1mL7gQo3ynZqOmsHB4Y8bhQ4-L3KplNOb#scrollTo=XLJNVGwHUDy1](https://colab.research.google.com/drive/1mL7gQo3ynZqOmsHB4Y8bhQ4-L3KplNOb#scrollTo=XLJNVGwHUDy1)

<br>

2. ResNet-50 Enhancement

Imagenet classes의 1 million이상의 이미지를 미리 학습한 ResNet-50을 전이학습해서 hyperparameter optimization을 진행 함.

이 예시에선 다음 그림과 같이 pretrained model에 몇가지 layer를 추가했다 - Global Average Pooling layer + Fully Connected Layer + Dropout Layer + sigmoid function (for output).

추가적으로,

만약 output 성능이 부족하다면, regularization(규제) 또는 hyperparameter optimization technique를 진행할 수있다.

그리고 data augmentation(shear_range=0.2, horizonal_flip=True, zoom_range=0.2), dropout(rate = 0.3), early stopping(w/ patience=10)을 regularzation techniue로 설정할 수 있다. (early stopping을 위한 patience = number of epochs with no improvement after which training will be stopped.)

![](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/NeuralNetwork_through_Transferlearning.png)

2256 Images (1127 Graffiti, 1129 No-Graffiti Images)로 구성된 이미지 collection을 train : test : validation으로 0.8 : 0.1 : 0.1로 나누어서 image classifier를 통해 graffiti인지 non-graffiti인지 binary 분류를 예측하는 모델을 만들고 다음과 같이 최적화 실험을 진행했다.

Hyperparameter tuning 및 transfer learning optimizations:

- learning rate optimization - 

  batch size 32로 fix하고 optimization function으로 Adam을 적용하여 learning rate variation으로 0.1, 0.01, 0.001, 0.0001, 0.00001를 적용한 결과,  0.0001에서 가장 좋은 성능을 보임.

- batch size optimization - 

  learning rate은 0.0001로 fix하고 batch size 16, 32, 64, 100을 적용함. batch size=64인 경우에 가장 높은 recall 값(실제 graffiti인 case를 맞게 예측한 경우)이 확인되고 낮은 수준의 false negative rate이 확인되었다.  

- best freezing layer selection - 

  다음과 같이 dataset size-similarity matrix를 기반으로 transfer해오는 model의 얼만큼을 freeze하고 얼만큼을 훈련시킬지를 선택할 수 있다. transfer해오는 model이 pre-train된 dataset과 내가 훈련시키려는 현dataset 사이의 similarity와 현 dataset size에 따라 알맞는 case를 선택할 수 있는 matrix이다.

  ![matrix](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/size_similarity_matrix.png)

  dataset size가 작은 경우, first few layers를 freeze하고, remaining layer를 훈련시키는 (matrix의 3rd quadrant) case에 해당된다. 모델이 결국 현 dataset을 더 학습할 수 있는 방향이다. 

  transfer할 model에서 몇개의 layer만 제외하고 freeze할지는 variation으로 두고 실험을 했다. Freezing the model's 모든 layers except the last 15, 25, 32, 40, 100, 150 layers. (즉 마지막 몇개 layer를 현 dataset으로 훈련시킬지를 15~150까지 실험해 봄.)

  결과, training last 100 layers가 test accuracy 87%와 f1-score 87%로 가장 높은 성능을 보였다.

  

Overall, learning rate = 0.0001, batch size = 64, training last 100 layers의 조건으로 가장 높은 성능을 확보할 수 있었다.

<br>

<br>

# Retraining

모델에 input되는 data의 특성이 바뀌어서 (due to data drift or concept drift), 모델이 이전과 완전히 다른 결과를 output할때에는 모델을 fine-tuning하거나 아얘 처음부터 다시 훈련시켜야한다 (training from scratch). 이런 과정을 retraining이라고 한다. 



retraining vs. transfer learning의 차이 review: https://www.youtube.com/watch?v=XJHmD6SKMyw

(github: https://github.com/sohiniroych/Knowledge-transfer-for-ML)

<br>

<br>

# Reference

1. "Improving the performance of ResNet50 Graffiti Image Classifier with Hyperparameter Tuning in Keras" by Silaja Konda Apr 2020, from [https://towardsdatascience.com/improving-the-performance-of-resnet50-graffiti-image-classifier-with-hyperparameter-tuning-in-keras-dbb59f43c6f7](https://towardsdatascience.com/improving-the-performance-of-resnet50-graffiti-image-classifier-with-hyperparameter-tuning-in-keras-dbb59f43c6f7)

2. Tenorio, et al. 2019. Improving Transfer Learning Performance: An Application in the Classification of Remote Sensing Data [https://www.semanticscholar.org/paper/Improving-Transfer-Learning-Performance%3A-An-in-the-Ten%C3%B3rio-Villalobos/640ea588ac02dcf73def8a0b80890302d12e6b73](https://www.semanticscholar.org/paper/Improving-Transfer-Learning-Performance%3A-An-in-the-Ten%C3%B3rio-Villalobos/640ea588ac02dcf73def8a0b80890302d12e6b73)