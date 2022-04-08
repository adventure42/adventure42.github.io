---
layout: post                          # (require) default post layout
title: "Multiclass Classification: Precision & Recall"   # (require) a string title
date: 2022-03-14     # (require) a post date
categories: [RCAClassification]          # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [RCAClassification]                      # (custom) tags only for meta `property="article:tag"`

---

<br>

# Precision & Recall (Multiclass Classification)

binary classification과는 다르게 multiclass의 경우 positive/negative class로 나뉘어지지 않는다. multiclass를 다룰때에는 각 individual class별로 TP, TN, FP, FN을 찾아야 한다.

![multiclass_confusion_matrix](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/multiclass_confusion_matrix.png)

예를 들어서 Apple 클래스에 대한 variable들과 분류성능 지표를 찾아본다면, 다음과 같다.

TP = 7
TN = (2+3+2+1) = 8
FP = (8+9) = 17
FN = (1+3) = 4

Precision = TP/(TP+FP) = 7/(7+17) = 0.29
Recall =  TP/(TP+FN) = 7/(7+4) = 0.64
F1-score = harmonic mean of precision & recall = 0.40

<br>

## Scikit-Learn metrics

Scikit-Learn을 통해서 confusion matrix를 확인하면 주의해야할 사항이있다. 주로 교재에서는 matrix의 row가 predicted values, column이 actual values로 설명하지만, scikit-learn에서 확인하는 confusion matrix는 이 둘이 바뀌여 있다. scikit-learn에서는 row가 actual values고, column이 predicted values이다. 이를 기억하고 recall, precision,등의 metric을 계산해야 한다.

<br>

## Micro F1

"micro-averaged F1-score"이며, total TP, total FP, total FN을 기반으로 계산된다. 그래서 각각을 class를 따로 계산하지않고 global한 값을 추출한다. globally 추출된 값이기때문에 precision, recall, micro-F1, accuracy가 모두 동일한 값이다. 

Total TP = (7+2+1) = 10
Total FP = (8+9)+(1+3)+(3+2) = 26
Total FN = (1+3)+(8+2)+(9+3) = 26

Precision = 10/(10+26) = 0.28

Recall = 10/(10+26) = 0.28

F1-score = 0.28

## Macro F1

"macro-averaged F1-score"이며, 각가의 class의 metric을 계산하고 unweighted mean을 추출한다. 

제일 처음 Apple 클래스에 대한 metric을 추출한 방식과 같이, Orange, Mango 클래스들의 metric들도 계산해보면 다음과 같다.

Class Apple F1-score = 0.40
Class Orange F1-score = 0.22
Class Mango F1-score = 0.11

Macro F1 = (0.40+0.22+0.11)/3 = 0.24

<br>

## Weighted F1

Weighted F1-score은 macro F1과 다르게 weighted mean of measures를 계산한다. 각 class별 sample수 만큼의 weight를 가하게 된다. Apple은 11개, Orange는 12개, Mango는 13이므로,

Weighted F1 = ((0.40\*11)+(0.22\*12)+(0.11\*13))/(11+12+13) = 0.24 이다.

<br>

<br>

# Precision vs. Recall

classification 문제를 해결하다보면, classifier 모델의 성능이 low precision & high recall 또는 low recall & high precision로 확인될때가 종종 있다. 각각의 현상이 의미하는 classifier의 성능은 다음과 같다.

![venndiagram](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/recall_precision_venn.png)

solid line은 ground truth set을 의미하고, dotted line은 classifier set을 의미한다. 

(Binary classifier의 경우에는 다음 venn diagram 하나로 표현이 되지만, multiclass classifier의 경우에는 각각의 class 분류 성능을 하나의 venn diagram으로 판단할 수 있겠다.)

<br>

## low precision & high recall

![venn1](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/high_recall_low_precision_venn.png)

샘플의 클래스를 ground truth와 맞게 예측할 확률이 높지만 (True positive가 많지만), truth가 아닌 것들도 맞다고 예측할 확률 또한 높다. (False positive가 많다.) Classifier를 물고기를 잡기위해 만든 그물로 표현한다면, high recall & low precision의 성능을 내는 classifier는 물고기를 많이 잡는 그물이기는 하지만, 타겟한 물고기 말고도 다른 원하지 않은 것까지도 많이 잡는다고 생각하면 될것이다. 

<br>

## high precision & low recall

![venn2](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/low_recall_high_precision_venn.png)

샘플의 클래스를 맞게 분류할 확률이 상대적으로 낮지만 (true positive가 적지만), 클래스를 틀리게 분류할 확률 또한 낮다.(false positive가 적다). 이런 classifier는 매우 까다로운 highly specialized 그물로 생각하면 될것이다. 원하는 물고기를 많이 잡지는 못하지만, 원하지 않는 다른것들 또한 거의 잡지 않는다. 

<br>

## high precision & high recall

![venn3](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/high_recall_high_precision_venn.png)

일반적으로 가장 성능이 높은 classifier이다. 매우 까다롭지만 (high precision), 높은 확률로 샘플들의 클래스들을 맞게 예측한다(high recall).

<br>

머신 러닝으로 해결하려는 문제가 무엇이냐에 따라서 precision과 recall이 모두 높은 classifier가 필요하지 않을 수도 있다. 샘플의 클래스를 맞게 예측하는것이 모든 샘플을 맞게 예측하는것보다 더 중요하다면 high precision에 더 비중을 두어야 하고, 샘플들의 클래스를 모두 맞추는것이 틀린 예측이 발생하지 않게 하는것보다 더 중요하다면 high recall에 더 비중을 두어야 할 것이다.

"An example of this compromise exists in our domain, media monitoring (at least traditionally), many customers expect an almost perfect recall. They never want to miss an article about the subjects they are interested in. However precision is not as important (albeit highly desirable), if you get a bit of noise in the article feed, you are usually fine."

<br>

<br>

# References

1. Confusion Matrix for Your Multi-Class Machine Learning Model by Joydwip Mohajon https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
1. Precision and recall — a simplified view by Arjun Kashyap https://towardsdatascience.com/precision-and-recall-a-simplified-view-bc25978d81e6
1. Explaining precision and recall by Andreas Klintberg https://medium.com/@klintcho/explaining-precision-and-recall-c770eb9c69e9

