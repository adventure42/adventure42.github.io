---
layout: post                          # (require) default post layout
title: "Transfer learning"                   # (require) a string title
date: 2021-12-29       # (require) a post date
categories: [deeplearning]          # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [CNN]                      # (custom) tags only for meta `property="article:tag"`

---



# Transfer learning

**transfer learning vs. retraining**

![transfer_learning_vs_retraining](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/transfer_learning_vs_retraining.PNG)



## **transfer training/ transfer learning**

이미 훈련된 CNN을 가져와서 내가 원하는 부분만 내가 해결하려는 문제에 맞게 변경하여 예측값을 찾는 것이다. 훈련된 CNN의 마지막 layer를(classification을 위한 softmax가 있는 층) 잘라서 내가 원하는 층으로 바꾸는 것이다. 가져온 CNN내의 trained weights와 biases는 그대로 유지되고, 내가 마지막 부분으로 추가한 층만 훈련이 되어서 내가 원하는 문제에 활용 될 수 있다. 단, 이 방법을 사용하기 적절한 경우는 가져온 CNN이 훈련되었었던 dataset이 내 문제의 dataset과 "close enough"할때이므로 주의해야한다. 

가져와서 그대로 사용하려는 층을 "freeze"하고, 내 문제에 맞게 변경해서 사용하려는 층은 "unfreeze"한다고 표현한다. 그래서 frozen initial layers (top conv layers) + unfrozen layer의 구성으로 training을 진행하여 원하는 모델을 구현할 수 있다. 

![전이학습](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/transfer_learning_diagram.png)

keras에서 제공하는 practice ipynb: https://colab.research.google.com/drive/1mL7gQo3ynZqOmsHB4Y8bhQ4-L3KplNOb#scrollTo=XLJNVGwHUDy1



## **retraining**

모델에 input되는 data의 특성이 바뀌어서 (due to data drift or concept drift), 모델이 이전과 완전히 다른 결과를 output할때에는 모델을 fine-tuning하거나 아얘 처음부터 다시 훈련시켜야한다 (training from scratch). 이런 과정을 retraining이라고 한다. 



retraining vs. transfer learning의 차이 review: https://www.youtube.com/watch?v=XJHmD6SKMyw

(github: https://github.com/sohiniroych/Knowledge-transfer-for-ML)