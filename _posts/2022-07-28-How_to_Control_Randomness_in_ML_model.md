---
layout: post                          # (require) default post layout
title: "How to Control Reproducibility in ML Model"   # (require) a string title
date: 2022-07-28      # (require) a post date
categories: [RCAClassification]          # (custom) some categories, but make sure these categories already exists inside path of `category/`
tags: [RCAClassification]                      # (custom) tags only for meta `property="article:tag"`

---

<br>

# Where does the randomness occur & how to control them

Machine learning algorithms 또는 model이 매번 똑같은 결과를 만들지 못하는 경우가 존재한다. 

보통 applied machine learning에서는 dataset에 machine learning algorithm을 run해서 machine learning model을 얻어내는데, 이런 learning 과정을 통해 dataset의 inputs과 outputs를 mapping하는 function이 approximate된다. 

Learning 과정에 사용되는 data에 algorithm이 얼마나 sensitive한지를 보여주는 지표가 variance이다. Variance는 algorithm의 hyperparameter를 tuning하여 감소시킬 수 있다. 

## stochastic nature of algorithms

어떤 machine learning algorithm은 deterministic하다. 즉, 같은 data로 같은 algorithm을 훈련시키면 항상 똑같은 model을 얻는다. 그러나 같은 training data에 같은 algorithm을 사용하여도 다른 결과가 나온다면, 해당 algorithm의 non-deterministic 또는 stochastic nature 때문일 수 있다. 

Stochastic nature은 randomness와는 조금 다르다. stochastic machine learning algorithm은 제공된 historical data에 의존하며 model이 학습되며, 이 때 specific small decision들이 random하게 달라질 수 있다. 이렇게 각각의 작은 decision을 만들때에 randomness가 더해지면 종종 매우 어려운 문제를 해결하는데에 도움이 될 수 있다. 최상의 mapping function for dataset을 찾는 것은 일종의 search problem이다. Randomness를 통해 "good" solution보다 더 좋은 "great" solution을 찾을 수 있는 search space를 확보할 수 있다.

Neural network에서는 random initial weights를 통해서 model이 different starting point에서부터 search space를 탐색해나아갈 수 있도록 한다. 

Ensemble에서는 bagging의 경우 randomness를 활용한다. Training dataset에서 sampling procedure를 통해 각각의 다른 decision tree들을 확보하여 ensemble을 형성하도록 한다. ("In ensemble learning, this is called ensemble diversity and is an approach to simulating independent predictions from a single training dataset.")

이런 경우, randomness는, seed를 fix함으로서 constant하게 유지될 수 있다. 

<br>

## different evaluation procedures

train-test split 또는 k-fold cross validation을 구현할때에 random하게 rows of data samples를 assign하게 된다. 이런 randomness는 model performance가 specific한 samples에 의존할 수 있는 것은 방지해준다. 

이런 경우에도, random seed를 fix해서 constant한 randomness generator를 통해 randomness를 제어할 수 있다.

<br>

## different platforms

Random seeds를 fix해도 다음과 같은 differences로 인해, 다른 result를 얻게될 수 있다:

- Differences in the **system architecture**, e.g. CPU or GPU.
- Differences in the **operating system**, e.g. MacOS or Linux.
- Differences in the **underlying math libraries**, e.g. LAPACK or BLAS.
- Differences in the **Python version**, e.g. 3.6 or 3.7.
- Differences in the **library version**, e.g. scikit-learn 0.22 or 0.23.

machine learning algorithm은 numerical 연산의 한 종류이기때문에, a lot of math with floating point values가 involve되어있다. 그래서 다른 architecture 또는 operating system으로 인해 round errors와 같은 부분에서 차이가 발생할 수 있고 이들이 여러번의 연산을 통해 더 크게 다른 result를 줄 수 있다. 같은 dataset과 같은 parameter가 사용된 model의 결과임에도 불구하고 difference가 발생될 수 있다. 

다른 language (R vs. Python), 또는 Python library의 다른 version으로 인해 different result가 발생하기도 한다. 

이런 경우와 같은 difference를 발생시키는 요소들은 machine learning evaluation을 수행하기 전에 미리 fix되어야한다. Machine learning project에서 development에서부터 deployment/production까지 가는 과정동안 full reproducibility를 보장하기위해서는 platform과 같은 aspect는 fix되어야 한다. 

다른 방법으로는 docker or virtual machine과 같은 virtualization을 활용하는 것이다. 이런 virtual 방법을 통해 environment를 일정하게 fix할 수 있다. 

<br>

<br>

# References

1. https://pytorch.org/docs/stable/notes/randomness.html
1. https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
1. https://machinelearningmastery.com/different-results-each-time-in-machine-learning/
1. https://machinelearningmastery.com/randomness-in-machine-learning/
1. https://machinelearningmastery.com/how-to-reduce-model-variance/
1. https://machinelearningmastery.com/stochastic-in-machine-learning/
