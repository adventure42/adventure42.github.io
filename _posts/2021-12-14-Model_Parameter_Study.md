---
layout: post                          # (require) default post layout
title: "Parameters"                   # (require) a string title
date: 2021-12-14       # (require) a post date
categories: [python]          # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [test]                      # (custom) tags only for meta `property="article:tag"`

---

## Parameter란?

Parameter는 data로 부터 estimate 또는 학습될 수 있고, **model의 내부적인 configuration역할**을 하는 variable들을 가리킨다. 

**Model = hypothesis역할**

**Parameter = hypothesis가 특정 data set에 대해 맞춤(tailor)되도록 하는 역할** (parameter는 part of the model that is learned from historical training data.)

optimization algorithm을 사용해서 model parameter를 estimate한다. 크게 두 종류의 optimization algorithm이 있다:

- statistics - in statistics, Gaussian과 같이 variable의 distribution을 assume할 수 있다. (e.g, Gaussian에서는 mu, sigma와 같은 두가지 parameter를 주어진 data를 바탕으로 계산하여 distribution형태를 확보할 수 있다.) machine learning에서도 동일하게 주어진 dataset의 data를 기반으로 parameter가 estimate되어서 prediction을 output할 수 있는 model의 부분이 된다. 

- programming - function에 parameter를 pass한다. 이 경우에는 parameter는 function argument로서 range of values를 가질 수 있다.새로운 data가 주어졌을 때에 model이 prediction(output)을 만들어낸다. machine learning에서는 model이 function역할을 하고 parameter가 주어져야 새로운 data를 통해 prediction output을 만들어 낼 수 있다.

Parameter의 예시로는: weights in ANN, coefficients in linear regression or logistic regression, support vectors in SVM, 등이 있다.

