---
layout: post                          # (require) default post layout
title: "Evaluation metrics"                   # (require) a string title
date: 2022-01-06       # (require) a post date
categories: [machinelearning]          # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [evaluation]                      # (custom) tags only for meta `property="article:tag"`
---



# Accuracy, Precision, Recall

<br>

## Confusion matrix for binary classification

binary classification의 결과를 평가하기위해 사용하는 metric이다.

![confusion matrix](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/confusion_matrix.png)

**accuracy = true positive + true negative / (all types)**

--> "모든 cases들 중에 true로 (맞게) 예측한 경우"

number of correct predictions over the output size

<br>

**precision = true positive / (true positive + false positive)**

--> "positive로 예측한것들 중에 맞게 예측한 경우"

특정 domain에서는 false positive가 false negative를 얻는것보다 더 치명적인 에러이다. (e.g. spam detection - 매우 중요한 이메일을 spam으로 잘못 구분하여 못보는것이 spam 이메일 하나가 inbox에 잘못 들어오는것 보다 더 큰 에러임.) 

만약 precision이 accuracy보다 낮은 경우, false positive가 현재 가지고있는 에러의 큰 비중을 차지하고 있을 수 있다. (The greater false positive will render greater denominator, hence smaller precision value.)

<br>

**recall = true positive / (true positive + false negative)**

--> "실제 positive인것들 중에 맞게 예측한 경우"

Precision과는 "반대"를 의미한다고 볼 수 있다. False negative vs. true positive를 보여주는 metric이다. 병에 걸렸는데에도 안결린것으로 잘못 예측하는 false negative가 더 치명적인 error인 안전과 같은 문제에서 매우 critical한 평가 metric이다. 그래서 disease diagnosis와 같은 문제에서는 recall이 accuracy or precision보다 높은 결과를 선호한다. 

<br>

**precision-recall curve**

unbalanced dataset이 주어진 경우, precision과 recall 사이의 tradeoff를 보여주는 유용한 metric이다. (when one class is dominant while the other class is under represented in the dataset)

<br>

**f1-score = 2 * [ (recall*precision)/(recall + precision) ]**

만약 precision과 recall 모두 높은 값을 얻어야한다면, recall과 precision의 harmonic mean인 f1-score를 통해 판단할 수 있다. recall과 precision의 average보다 harmonic mean은 outliers 덜 민감하다.  F1-score는 여러 domain에서 적합한 평가 metric으로 활용되고있다. 

"The F1-score is a balanced metric that appropriately quantifies the correctness of models across many domains."

<br>

**AUC(area under the curve)**

AUC는 ROC curve아래의 면적을 의미하고, ROC는 TPR(Recall)과 FPR(specificity) 사이의 tradeoff를 표현해준다. 

Precision-recall curve와는 다르게 ROC(receiver operator characteristic) curve는 balanced dataset의 domain에 적합하다. 

AUC의 값은 다른 metric과 동일하게 ranges over 0 and 1. (0.5는 random prediction을 expected value로 여겨진다.) 

![AUC](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/AUC.PNG)

**TPR** (True Positive Rate)은 recall과 같고 sensitivity로도 불린다. 

**TPR = true positive / (true positive + false negative)**



**FPR** (False Positive Rate)은 specificity로도 불리며 다음과 같다:

**FPR = false positive / (false positive + true negative)**

--> "실제 negative인것들 중에 맞게 예측한 경우"

FPR은 classifier model의 "false alarm metric"이다. 이 metric은 얼마나 자주 classifier가 negative여야하는 case를 positive로 잘못 예측하는지를 알려준다. 

<br>

<br>

## Metrics for other classification

multiple class들을 예측해야하는 경우, accuracy는 다음과 같이 찾을 수 있다.

**accuracy = correct predictions / all predictions**

```python
# Accuracy for non-binary predictions
def my_general_accuracy_score(actual, predicted):
    correct = len([a for a, p in zip(actual, predicted) if a == p])
    wrong = len([a for a, p in zip(actual, predicted) if a != p])
    return correct / (correct + wrong)
```

<br>

precision과 recall은 false positive와 negative를 측정해기때문에 다음과 같이 general classifier에 해당할 수 있도록 precision값을 찾을 수 있다.

```python
def my_general_precision_score(actual, predicted, value):
    true_positives = len([a for a, p in zip(actual, predicted) if a == p and p == value])
    false_positives = len([a for a, p in zip(actual, predicted) if a != p and p == value])
    return true_positives / (true_positives + false_positives)
```

<br>

만약 discrete categories or classes가 아닌, continuous prediction이 요구되는 domain이라면, threshold parameter를 설정해서 continuous prediction에 대한 accuracy를 계산할 수 있다. (해당 domain knowledge를 기반으로 적절한 threshold가 설정되어야 함.)

```python
# Accuracy for continuous with threshold
def my_threshold_accuracy_score(actual, predicted, threshold):
    a = [0 if x >= threshold else 1 for x in actual]
    p = [0 if x >= threshold else 1 for x in predicted]
    return my_accuracy_score(a, p)
```

<br>

<br>

# Reference

1. A Pirate's Guide to Accuracy, Precision, Recall, and Other Scores by Philip Kiely. Oct 2019, from https://blog.floydhub.com/a-pirates-guide-to-accuracy-precision-recall-and-other-scores/
2. Geron, Aurelien. Hands on Machine Learning. O'Reilly, 2019 