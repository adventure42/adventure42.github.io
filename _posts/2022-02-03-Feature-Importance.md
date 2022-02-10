---
layout: post                          # (require) default post layout
title: "Feature Importance"                   # (require) a string title
date: 2022-02-03       # (require) a post date
categories: [RCAClassification]          # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [RCAClassification]                      # (custom) tags only for meta `property="article:tag"`

---

<br>

# Feature Importance

Feature Importance: Feature importance is calculated as the decrease in node impurity weighted by the probability of reaching that node. The node probability can be calculated by the number of samples that reach the node, divided by the total number of samples. *The higher the value the more important the feature.*

<br><br>

## Gradient Boosting Feature Importance

간단한 tree-based model을 만들어서 prediction 값을 만들고 standard feature importance estimation을 계산할 수 있다. 이 feature importance를 통해 standard correlation index보다 더 많은 정보를 알 수 있다. Feature importance는 model 훈련 phase에서 internal space partition이 실행되는 동안 특정 하나의 feature이 point되었을 때에 trees 전체에 대한 impurity index의 reduction을 summarize해준다. sklearn은 output의 sum이 1이되도록 normalization을 적용해서 결과를 보여준다.

```python
gb = GradientBoostingRegressor(n_estimators=100)
gb.fit(X_train, y_train.values.ravel())

plt.bar(range(X_train.shape[1]), gb.feature_importances_)
plt.xticks(range(X_train.shape[1]), ['AT','V','AP','RH'])
```

위 코드로 graph되는 feature_importances_는 특정 feature의 값이 더 높을 수록 prediction에 더 큰 영향을 주는 것을 보여준다.

<br>

<br>

## Permutation Importance

permutation importance는 model이 fit된 후에 계산할 수 있다.

This method works on a simple principle: *If I randomly shuffle a single feature in the data, leaving the target and all others in place, how would that affect the final prediction performances*?

데이터에서 특정 하나의 feature만 random하게 shuffle한다면, final prediction 성능에 어떤 영향을 끼칠지 살펴보는 것이다. 가장 중요한 feature일수록, shuffling으로 인해 가장 나쁜 prediction 성능이 확인될 것이다.

This is because we are corrupting the natural structure of data. If we, with our shuffle, break a strong relationship we’ll compromise what our model has learned during training, resulting in higher errors (**high error = high importance**).

<br>

<br>

## RandomForestRegressor

scikit-learn의 class중 하나로 존재하며, dataset의 feature reduction을 수행하기위해 feature들의 importance value와 같이 reduction의 기준으로 사용할 수 있는 parameter를 RandomForestRegressor를 통해 생성할 수 있다.

<br>

<br>

# Feature Selection

Correlation coefficient, variance threshold, mean absolute difference, dispersion ratio와 같은 technique를 사용해서 more relevant feature를 선택할 수 있다.

Correlation is a measure of the linear relationship of 2 or more variables. Through correlation, we can predict one variable from the other. The logic behind using correlation for feature selection is that the good variables are highly correlated with the target. Furthermore, variables should be correlated with the target but should be uncorrelated among themselves.

If two variables are correlated, we can predict one from the other. Therefore, if two features are correlated, the model only really needs one of them, as the second one does not add additional information

<br>

<br>

# References

1. "Feature Importance with Neural Network" by Marco Cerliani from https://towardsdatascience.com/feature-importance-with-neural-network-346eb6205743
2. "Feature Selection: Beyond feature importance?" by Dor Amir from https://medium.com/fiverr-engineering/feature-selection-beyond-feature-importance-9b97e5a842f
3. "The Mathematics of Decision Trees, Random Forest and Feature Importance in Scikit-learn and Spark" by Stacey Ronaghan from https://towardsdatascience.com/the-mathematics-of-decision-trees-random-forest-and-feature-importance-in-scikit-learn-and-spark-f2861df67e3

3. scikit-learn 1.10 Decision Trees from https://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart