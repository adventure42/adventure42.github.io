---
layout: post                          # (require) default post layout
title: "Parameters vs. Hyperparameter"                   # (require) a string title
date: 2021-12-07       # (require) a post date
categories: [deeplearning]          # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [deeplearning]                      # (custom) tags only for meta `property="article:tag"`

---
<br>


# Parameter란?

Parameter는 data로 부터 estimate 또는 학습될 수 있고, **model의 내부적인 configuration역할**을 하는 variable들을 가리킨다. 


<br>
**Model = hypothesis역할**



**Parameter = hypothesis가 특정 data set에 대해 맞춤(tailor)되도록 하는 역할** (parameter는 part of the model that is learned from historical training data.)



optimization algorithm을 사용해서 model parameter를 estimate한다. 크게 두 종류의 optimization algorithm이 있다:

- statistics - in statistics, Gaussian과 같이 variable의 distribution을 assume할 수 있다. (e.g, Gaussian에서는 mu, sigma와 같은 두가지 parameter를 주어진 data를 바탕으로 계산하여 distribution형태를 확보할 수 있다.) machine learning에서도 동일하게 주어진 dataset의 data를 기반으로 parameter가 estimate되어서 prediction을 output할 수 있는 model의 부분이 된다. 

- programming - function에 parameter를 pass한다. 이 경우에는 parameter는 function argument로서 range of values를 가질 수 있다.새로운 data가 주어졌을 때에 model이 prediction(output)을 만들어낸다. machine learning에서는 model이 function역할을 하고 parameter가 주어져야 새로운 data를 통해 prediction output을 만들어 낼 수 있다.



Parameter의 예시로는: weights in ANN, coefficients in linear regression or logistic regression, support vectors in SVM, 등이 있다.


<br>  
<br>  

# Hyperparameter란?

Model의 hyperparameter는 model의 **configuration역할**을 수행하는 **외적부분**으로 (external to the model) data를 기반으로 estimate될 수 있는 값이 아니다. 

주로 model parameter들을 estimate하는 과정에 사용되고, 사용자/개발자에 의해 종종 heuristics(가장 최적의 완벽한 방법은 아닐 수 있지만, 즉각적이거나 단기적인 목적을 이루기에는 충분한)방법을 기반으로 설정된다. 특정 predictive modeling problem을 다룰때에는 이미 hyperparameter들이 tune되어 있는 경우도 있다.

*"We cannot know the best value for a model hyperparameter on a given problem. We may use rules of thumb, copy values used on other problems, or search for the best value by trial and error."*

특정 문제에 맞게 machine learning algorithm이 tuning될 때, model이 최선의 prediction을 만들 수 있도록 parameter를 설정하기위해서 grid search나 random search를 통해 hyperparameter를 tuning한다. 

Parameter와 hyperparameter를 model parameter로 통합하여 명칭하는 경우도 있다. 그러나 사용자가 manually 설정한 model parameter이라면, 해당 parameter는 hyperparameter로 구분할 수 있다. hyperparameter는 parameter와는 다르게 analytical formula를 통해서 적절한 값을 계산하기가 어려운 variable이다.  



Hyperparameter의 예시로는: ANN(Artificial Neural Network)의 learning rate, SVM(Support Vector Machine)의 C와 sigma, KNN(k-Nearest Neighbor)의 k, 등이 있다.

<br>
<br>


## Hyperparameter optimization

<br>

### 최적화 libraries

- **Hyperopt**

[https://teddylee777.github.io/thoughts/hyper-opt](https://teddylee777.github.io/thoughts/hyper-opt)



- **Hyperas, kopt, Talos**

Hyperas : [https://github.com/maxpumperla/hyperas](https://github.com/maxpumperla/hyperas)

kopt : [https://github.com/Avsecz/kopt](https://github.com/Avsecz/kopt)

Talos : [https://github.com/autonomio/talos](https://github.com/autonomio/talos)



- **Keras Tuner**

[https://www.tensorflow.org/tutorials/keras/keras_tuner?hl=ko](https://www.tensorflow.org/tutorials/keras/keras_tuner?hl=ko)

 

- **Scikit-Optimize(skopt)**

[https://scikit-optimize.github.io/stable/](https://scikit-optimize.github.io/stable/)

 

- **Spearmint**

[https://github.com/HIPS/Spearmint](https://github.com/HIPS/Spearmint)



- **Hyperband**

official : [https://keras.io/api/keras_tuner/tuners/hyperband/](https://keras.io/api/keras_tuner/tuners/hyperband/)

개인 블로그 : [https://iyk2h.tistory.com/143](https://iyk2h.tistory.com/143)



- **Sklearn-Deap**

[https://github.com/rsteca/sklearn-deap](https://github.com/rsteca/sklearn-deap)

<br>
<br>


### Tuning guide

최적의 학습률은 다른 hyperparameter에 의존적이다. 특히 batch size에 영향을 많이 받는다. 따라서 다른 hyperparameter를 수정하면 학습률도 반드시 tuning해야한다. 어떤것을 어떻게 조정해야할까?

<br>

#### hidden layer 개수

복잡한 문제일수록 심층 신경망이 (얕은 신경망보다) parameter efficiency가 훨씬 좋다.
심층 신경만은 복잡한 함수를 modeling하는데에 얕은 신경망보다 훨씬 적은 neuron을 사용하기때문에 동일한 양의 훈련 데이터에서 더 높은 성능을 확보할 수 있다.

예를들어 복/붙 기능이 없는 drawing SW에서 숲을 그려야한다고 생각해보자. 나무,가지,잎 들을 전부 하나하나 그려야하기때문에 시간이 오래 소모된다. 만약 잎은 하나그려서 복/붙으로 가지에 붙히    고, 이 가지를 또 여러번 복붙해서 나무를 하나 그리고, 그 다음 이 나무를 여러번 복/붙해서 숲을 그리는것은 훨씬 빠르게 진행될 수 있다. 심층 신경망은 이렇게 계층 구조를 형성해서 더 효율적이게 진행된다. 아래쪽 은닉층은 저수준의 구조를 모델링하고 (방향,모양,선, 등), 중간 은닉층은 저수준 구조를 연결해서 중간 수준의 구조를 모델링한다.(사각형, 원,등) 그리고 가장 위쪽의 은닉층과 출력층은 중간 수준의 구조를 연결해서 고수준의 구조를 모델링한다(얼굴, 머리카락, 등)

또한 새로운 데이터에 일반화되는 능력도 향상시켜준다. 예를들어 얼굴을 인식하는 모델을 훈련한 후, 헤어스타일을 인식하는 신경망을 새로 훈련하려면 첫번째 네트워크의 하위 층을 재사용하여 훈련을 시작할 수 있다. 저수준 구조가 미리 학습되어있는 상태에서 고수준만 학습하면 되도록 "전이 학습"을 하는것이다. 

<br>

#### hidden layer의 neuron 개수

은닉층의 구성 방식은 일반적으로 각 층의 뉴런을 점점 줄여서 깔대기 모양을 구성한다.
(e.g., MNIST문제의 신경망의 경우에는 각 은닉층의 뉴런수를 300, 200, 100으로 차례대로 구성했음.)
지금은 대부분의 경우 모든 은닉층에 같은 크기를 사용해도 동일하거나 더 나은 성능을 낸다. 

 실전에서는 필요한것보다 더 많은 층과 뉴런수를 가진 모델을 선택하고, 그런 다음 과대적합되지 않도록 조기 종료나 규제 기법을 사용한다. 
 "stretch pants" 기법 활용
 맞는 사이즈를 찾기위해 시간을 낭비하지 않도록, 그냥 큰 stretch pants를 사고 나중에 알맞게 줄이는 방식이다.

<br>

#### learning rate

매우 낮은 학습률에서 시작해서 점진적으로 매우 큰 학습률까지 수백번 반복하여 모델을 훈련하는것이다. 
e.g., 10^-5부터 시작해서 10까지 exp(log(10^6)/500)를 500번 반복

<br>     

#### optimizer

SGD보다 더 좋은 optimizer

<br>    

#### batch size

GPU RAM에 맞는 가장 큰 batch 크기를 권장한다. 단, 주의할 점은 일반화 성능이 떨어질 수도 있다는 점이다. 큰 batch를 사용해서 훈련하면, 작은 batch로 훈련된 모델만큼 일반화 성능을 내지 못할 수 있다. 

한가지 전략은 학습률 예열을 사용해 큰 배치 크기를 시도해보고 만약 훈련이 불안정하거나 최종 성능이 만족스럽지 못하면 작은 batch size를 사용해보는것이다.

<br>     

#### activation function

일반적으로 ReLU가 모든 은닉층에 좋은 기본값이다.
출력층의 활성화 함수는 수행하는 작업에 따라 달라진다.

<br>  

#### iteration 횟수

대부분의 경우 훈련 반복횟수는 튜닝할 필요가 없다. 대신 조기 종료를 사용한다.

<br><br>  

# References

1. comparison between parameters vs. hyperparamenter: [https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/)
1. Geron, Aurelien. Hands on Machine Learning. O'Reilly, 2019 
