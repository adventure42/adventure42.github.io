---
layout: post                          # (require) default post layout
title: "Probability Density Estimation"   # (require) a string title
date: 2022-07-21       # (require) a post date
categories: [RCAClassification]          # (custom) some categories, but make sure these categories already exists inside path of `category/`
tags: [RCAClassification]                      # (custom) tags only for meta `property="article:tag"`

---

<br>

# Probability Density Estimation

Random variable의 outcomes와 이런 outcomes가 나올 probability 사이의 상관관계를 probability density (또는 간략하게 "density")라고 한다. 

Random variable x가 가진 probability distribution을 p(x)로 표기한다.

Random variable x가 continuous하다면, probability를 pdf(probability density function)을 통해 계산할 수 있다.

probability distribution의 shape을 확인해서 the most likely values, the spread of values, 등 다양한 properties를 확보할 수 있다. 또한, a sample observation이 outlier 또는 anomaly인지 결정할 수 있다.

Dataset에 주어진 sample들로 제한되어서 random variable의 모든 possible outcome을 확보할 수는 없다, 그래서 sample of observations를 기반으로 probability distribution을 estimate한다. 그래서 probability density estimation이라고 한다. 

process of density estimation for random variable:

Review the density of observations in the random sample with histogram. Distribution의 형태를 보고 normal인지, 또는 model을 fit해서 distribution을 estimate해야하는지 확인한다.

<br>

histogram을 확인하는 방법

```python
# plotting a histogram of a random "sample" data
from matplotlib import pyplot
pyplot.hist(sample, bins=10)
pyplot.show()
```

hist() 함수에서 bins parameter를 어떻게 설정하냐에따라서 distribution density가 달라질 수 있다. 

numpy.histogram_bin_edges를 통해서 미리 설정된 binning strategies중 하나를 선택하여 사용할 수 있다.

[`numpy.histogram_bin_edges`](https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges): 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.

hist()함수의 bins parameter를 위 edges들 중 하나의 string으로 설정하면된다. 

## parametric density estimation

가장 흔하게 볼 수 있는 probability distribution은 normal distribution이다. Normal distribution은 두개의 parameters를 가지고있다 - mean and standard deviation. 이 두 parameter만으로 probability distribution function을 알 수 있다. sample mean과 sample standard deviation을 calculate하여 variable 의 parametric density estimation을 수행할 수 있다. 

만약 data가 bell curve가 아니고, left or right으로 shift된 skewed data일 수 있다. 이런 경우 parameters를 estimate하기 전에, data 를 transform해야한다. transformation 방식으로는 taking log or square root, 또는 Box-Cox transform과 같은 과정들이 있다. 

- Loop Until Fit of Distribution to Data is Good Enough:
  - Estimating distribution parameters
  - Reviewing the resulting PDF against the data
  - Transforming the data to better fit the distribution

<br>

<br>

# References

1. https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
1. https://machinelearningmastery.com/probability-density-estimation/
1. https://medium.com/@gianlucamalato/how-to-choose-the-bins-of-a-histogram-865c2042c0ce
1. https://machinelearningmastery.com/continuous-probability-distributions-for-machine-learning/
