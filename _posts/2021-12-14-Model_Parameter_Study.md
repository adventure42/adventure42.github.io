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



#### ANN (artificial neural network)

##### Percepton 

perceptron - 가장 간단한 인공 신경만 구조 중 하나임. perceptron은 층이 하나인 TLU(threshold logic unit)로 구성된다. 각 TLU는 모든 input(입력)에 연결되어있다.

![](C:\SJL\VQML_VQA\VQML\figures\perceptron.png)그림1

입력에 bias라는 편향값이 더해져서 다음과 같은 공식으로 perceptron이 표현된다. 

![perceptron_eqn](C:\SJL\VQML_VQA\VQML\figures\perceptron_eqn.png)

**w** : vector of weights

**x** : vector of inputs

**b** : bias 

**φ** : non-linear activation function.

Input data 로 위그림과 같이 TLU를 훈련시켜서 최적의 parameters (w1,w2,w3,w4)를 찾는다. 여기에서 weights는 각 corresponding input이 output에 대해 가지고있는 "importance"(중요도)를 의미한다.

![activation](C:\SJL\VQML_VQA\VQML\figures\neuron_activation.png)

각 intput에 weight만큼 가중치를 설정해서 구한 이들의 sum이 특정 threshold보다 크면 function의 결과가 1, 보다 작으면 function의 결과가 0이 된다.  위 그림에서 보이는 바와 같이 bias라는 값을 통해 threshold 기준을 0으로 맞추어서 step function을 구현했다. 각 neuron을 activate할지 deactivate할지를 이 기준을 통해 설정할 수 있기때문에 이 단계는 활성화(activation) 함수라고 부른다. weights와 threshold를 변경해서 각 input이 ouput에 얼마나 영향을 줄지를 제어하고 결국 model의 output을 제어할 수 있다. (You can think of the bias as a measure of how easy it is to get the perceptron to output a 1.)



##### Multilayer perceptron

single-layered perceptron으로는 non-linearity나 data의 complexity가 포함된 문제를 해결할 수 없다. 그래서 researcher들이 multilayer perceptron을 만들어서 여러개의 hidden layer를 구성하고 data속에 숨어있는 non-linearity를 찾아낸다. 

아래 그림과 같이 input layer(입력층)와 output layer(출력층)사이에 hidden layer(은닉층)가 존재한다. 

![multilayer-perceptron](C:\SJL\VQML_VQA\VQML\figures\multilayer_perceptron.jpg)

![](C:\SJL\VQML_VQA\VQML\figures\neuralnetwork.png)

입력층과 가까운 층은 하위층(lower layers)이라고 부르고, 출력층과 가까운 층은 상위층(upper layers)이라고 부른다. 이렇게 은닉층을 여러개 쌓아 올린 인공 신경망을 심층 신경망 (Deep neural network)이라고 한다. 

multilayer perceptron은 intput-output pairs로 구성된 data set으로 훈련을 하면서 결국 input들과 output들사이의 dependencies (or correlation)을 model한다. 이 훈련과정에서 weights and biases와 같은 parameter들을 보정해서 model의 error를 최소화 한다. 

Hidden layer들이 input data의 feature를 담고있다. 예를 들어서 MNIST data set를 입력해서 손글씨 숫자를 구분하는 model을 표현하는 neural network이 있다고 가정하면, 숫자 이미지의 특징들을 sub components로 나누어서 network가 학습하는 것이다. 



다음과 같이 layer의 각 neuron들이 specific 특성을 인식하도록 학습하는 것이다. matrix, vector로 표현하는 network:

<img src="C:\SJL\VQML_VQA\VQML\figures\neuralNetwork_matrix.PNG" alt="matrix_vector" style="zoom:67%;" />

Here, sigmoid is the "cost function" that adjusts the outcome of the neural network for the desired scale. sigmoid는 small changes in weights and bias로 output에 small changes를 만들 수 있는 function이다.  

sigmoid function:
$$
{\sigma}(z) = \frac{1}{1+e^{-z}}
$$
![sigmoid function](C:\SJL\VQML_VQA\VQML\figures\sigmoid.PNG)

step function과는 다르게 0,1 으로 구분되는 binary가 아니라 0과 1 사이에서 어떤 값이든 가능하다. sigmoid function을 사용하는 neuron의 output은 다음과 같은 공식으로 결정된다.
$$
\frac{1}{1+exp(-\sum_jw_jx_j-b)}
$$
**deep neural network에서 network & parameter들의 역할 및 operation: https://www.youtube.com/watch?v=aircAruvnKk** 





##### 경사하강법 Gradient descent 

**gradient descent explained: https://www.youtube.com/watch?v=IHZwWFHWa-w**

gradient descent algorithm은 stochastic gradient descent(or incremental gradient descent)와 batch gradient descent로 두가지 방법이 존재한다. 

Batch gradient descent는 매번 update를 진행할때에 entire training dataset를 모두 훌터보기 때문에 특히 training dataset이 크다면 속도와 cost가 나쁜편이다.

stochastic gradient descent는 random으로 dataset을 보기때문에 보통 cost function의 minimum에 근접하는 θ를 더 빠르게 찾는편이다. 그래서 training data set이 큰 경우에는 stochastic gradient descent를 선호하는 편이다.

![](C:\SJL\VQML_VQA\VQML\figures\Gradient_Descent_graph.PNG)



(Note however that it may never “converge” to the minimum, and the parameters θ will keep oscillating around the minimum of J(θ); but in practice most of the values near the minimum will be reasonably good approximations to the true minimum. By slowly letting the learning rate α decrease to zero as the algorithm runs, it is also possible to ensure that the parameters will converge to the global minimum rather than merely oscillate around the minimum)



##### 역전파(back propagation)

backpropagation(역전파)를 사용해서 model에서 감지된 error를 바탕으로 weights와 bias를 보정한다. 

model들이 어떻게 자신의 error/mistakes를 바탕으로 학습하는 과정은 두가지 방법으로 진행된다. 

- forward propagation(정방향 계산):

  signal flow가 input layer에서부터 hidden layer를 통과하여 output layer까지 흘러간다. output layer의 decision은 ground truth labels를 기준으로 평가되고 output layer가 ground truth와 비교했을때에 얼만큼의 error(오차)가 발생했는지를 감지한다.

- backward propagation(역전파):

  model의 모든 parameter에 대한 network 오차의 gradient를 계산할 수 있다. 즉, 오차를 감소 시키기위해서 각 연결 가중치와 편향값이 어떻게 바뀌어야하는지를 찾는 것이다. gradient를 구하고나면 평범한 경사 하강법을 수행한다. 전체 과정은 network가 어떤 해결책으로 converge될때까지 반복한다.

forward + backward propagation은 간단하게 다음과 같이 진행된다:

-각 훈련 샘플에대한 역전파 알고리즘이 먼저 예측을 만들고 오차를 측정한다. (정방향 계산)

-역 방향으로 각 층을 거치면서 각 연결이 오차에 기여한 정도를 측정한다. (역방향 계산)

-이 오차가 감소하도록 가중치를 조정한다. (경사 하강법 단계)

상세 과정:

1. 하나의 mini batch씩 반복적으로(iteratively) 진행해서 전체 훈련세트를 처리하며, 하나의 iteration을 epoch라고 부르는데, 각 mini batch는 network의 입력층으로 전달되어 첫번째 은닉층으로 보내진다. mini batch에 있는 모든 sample에 대해 해당 층에 있는 모든 뉴런의 출력을 계산한다. 계산된 결과는 다음 층으로 전달된다.
2. 이런식으로 마지막 층인 출력증의 출력을 계산할때 까지 계속된다. 이렇게 정방향으로 진행하면서 중간 계산값을 모두 저장한다. (역방향 계산을 위해 저장)
3. algorithm이 network의 출력 오차를 측정한다. (손실 함수를 사용해서 기대하는 출력과 network의 실제 출력을 비교하고 오차값을 반환한다.)
4. 각 출력 연결이 이 오차에 기여하는 정도를 계산한다.(미적분의 가장 기본 규칙인 연쇄법칙을 사용해서 빠르고 효율적으로 진행한다.)
5. algorithm이 또 다시 연쇄 법칙을 사용해서 이전 층의 연결 가중치가 이 오차의 기여 정도에 얼마나 기여했는지를 확인한다. 이런식으로 입력층에 도달할때까지 역방향으로 계속 반복한다. 이렇게 역방향 진행단계에서 오차 gradient를 거꾸로 전파함으로서 효율적으로 network에 있는 모든 연결 가중치에 대한 오차 gradient를 측정한다. 
6. 마지막으로 algorithm은 경사 하강법을 수행해서 방금 계산한 오차 gradient를 사용해서 network에 있는 모든 연결 가중치를 수정한다.

**backpropagation explained with graphics: https://www.youtube.com/watch?v=Ilg3gGewQ5U**

note: hidden layer의 연결 가중치를 random하게 초기화 하는것이 매우 중요함. 그렇지 못하면 훈련을 실패할 것임. 예를 들어 모든 가중치와 편향을 0으로 초기화하면 -> 층의 모든 뉴런이 같아진다. 따라서 역전파도 뉴런을 동일하게 바꾸어서 모든 뉴런이 똑같아진 채로 남고, 뉴런이 마치 하나인것처럼 작동하게 됨. 그래서 가중치를 random하게 초기화해서 대칭성을 깨고 역전파가 뉴런을 다양하게 훈련시키는 것임.



#### LMS(Least Mean Square) algorithm

아주 간단한 supervised learning중 하나인, linear regression문제에서 model의 parameter들이 어떻게 설정되는지 알아볼 수 있다.

다음과 같이 hypotheses h를 통해 output y를 예측하려한다면 linear 함수로 표현할 수 있다. 

Input dataset에 두개의 features가 주어졌을 때, (x_0 =1)
$$
h_{\theta}(x) = {\theta}_0+{\theta}_1x_1+{\theta}_2x_2
$$
input x와 output y를 mapping하는 linear 함수이고 여기에서 θ가 weights 또는 parameter를 의미한다. (위와 같은 공식에서 θi가 space of linear functions를 parameterize해서 x와 y의 mapping을 수행하기때문에 θ를 "parameter"라고 불린다.)

주어진 training data를 바탕으로 어떻게 적절한 parameter들을 찾을 수 있을까?

supervised learning에서는 우리가 y를 이미 알고있기때문에 y에 가장 가까운 h를 만드는 θ를 찾아볼 수 있다. Hypotheses h와 y의 차이를 계산하는 함수를 cost function J 라고 부르고 다음과 같이 정의한다. (input dataset에 총 n개의 samples/instances가 있고, ith번째 sample에 대한 예측값과 ground truth값의 차이를 계산하는 방식)
$$
J({\theta}) =\frac{1}{2}{\sum_{i=1}^{n}}(h_{\theta}(x^{(i)})-y^{(i)})^2
$$
J는 linear regression에서 주로 사용되는 least-square cost function이고 이는 ordinary least squares regression model에서 cost function으로 사용된다. 우리는 J를 가장 최소화 시킬 수 있는 θ를 찾아야 한다.

gradient descent algorithm은 우리가 원하는 θ를 찾는 방법 중 하나이다. 처음 특정 값을 하나 guess를 해서 cost를 계산하고 이를 점차 줄여가는 방향으로 update를 반복적으로 진행하는 방법이다. 
$$
{\theta}_j := {\theta}_j - {\alpha}\frac{\partial}{{\partial}{\theta}_j}J({\theta})
$$
(여기에서 :=는 θ에 계산된 새로운 값을 assign하여 update한다는 것을 의미하고 α는 learning rate을 의미한다.) θ는 J가 가장 가파르게 감소하는 방향으로 update된다. 위 공식에서 partial derivative term을 x,y로 표현하면 다음과 같이 LMS update rule을 찾을 수 있다.
$$
{\theta}_j := {\theta}_j + {\alpha}(y^{(i)}-h_{\theta}(x^{(i)}))x_j^{(i)}
$$
note: the update의 크기는 the error term (y-h) 즉, y와 h의 차이와 비례한다.





#### Normal equations

반드시 경사 하강법을 통해 반복적으로 parameter θ를 update하는 방식을 사용해야하는 것은 아니다. Parameter optimization을 위해 iteration없이 analytical 방식인 normal equation을 사용할 수 있다. Normal equation을 통한 optimization은 matrix를 사용하기때문에 multiple linear regression의 parameter를 한번에 계산해준다.

Normal equation 방식에서는 우리에게 주어진 input, output data set을 matrix형태로 다음과 같이 표현하고, 

X (input feature 값들)

y (output 값들)

Xtheta - y

matrix와 linear algebra를 사용해서 cost function J를 minimize하는 θ를 찾는다. 특히 소수의 features를 가진 dataset을 기반으로 model을 훈련하는 과정에서는 normal equation을 사용해서 더 빠르게 최적의 parameter θ를 찾을 수 있다.

Matrix의 특성 중 다음과 같은 특성을 사용하면,
$$
z^Tz = \sum_iz_i^2
$$
cost function J와 J의 derivative를 다음과 같이 표현할 수 있다.
$$
J({\theta}) = \frac{1}{2}\sum_{i=1}^n(h_{\theta}(x^{(i)})-y^{(i)})^2 =  \frac{1}{2}(X{\theta}-\vec{y})^T(X{\theta}-\vec{y})\\
{\gradient}_{\theta}J({\theta}) = X^TX{\theta} - X^T\vec{y}
$$
J의 derivative를 0으로 set해서 minimum cost function에 해당하는 parameter θ를 찾을 수 있다. matrix derivative 사용해서 얻을 수 있고 다음과 같이 찾은 θ를 normal equation이라고한다. 
$$
{\theta} = (X^TX)^{-1}X^T\vec{y}
$$

Normal equations are equations obtained by setting equal to zero the partial derivatives of the sum of squared errors or cost function; normal equations allow one to estimate the parameters of multiple linear regression.

example code:

https://www.geeksforgeeks.org/ml-normal-equation-in-linear-regression/



#### Gradient descent vs. Normal equation

|      | Gradient  Descent                                           | Normal Equation                                              |
| ---- | ----------------------------------------------------------- | ------------------------------------------------------------ |
| 1    | In gradient descenet , we need to choose learning  rate.    | In normal equation , no need to choose learning rate.        |
| 2    | It is an iterative algorithm.                               | It is analytical approach.                                   |
| 3    | Gradient descent works well with large number of  features. | Normal equation works well with small number of  features.   |
| 4    | Feature scaling can be used.                                | No need for feature scaling.                                 |
| 5    | No need to handle non-invertibility case.                   | If (X) is non-invertible , regularization can    be used to handle this. |
| 6    | Algorithm complexity is O(k). n is the number of features.  | Algorithm complexity is O(). n is the number of features.    |





#### Probabilistic interpretation

The notation “p(y (i) |x (i) ; θ)” indicates that this is the distribution of y (i) given x (i) and parameterized by θ.

**likelihood function L(θ)**
$$
L({\theta}) = \prod_{i=1}^{n}{p(y^{(i)} | x^{(i)};{\theta})}
$$

L(θ)는 output과 input(y's and the x's)를 relate하는 probabilistic model이다. maximum likelihood principal을 기반으로 가장 best parameter θ값을 찾을 수 있는데, 이 principal에서는 data로 가장 높은 probability를 만드는 θ를 찾아야 한다고 한다. 즉, L(θ)의 최대값을 주는 θ를 찾는 것이다. 

L(θ)를 maximize하기위해서 항상 증가하는 L(θ)의 함수(e.g., log(L(θ)))를 maximized하는 방법을 인용하면, log likelihood의 최대치를 계산하여 L(θ)의 maximum을 찾을 수 있다.

Gaussian Discriminant Analysis (GDA) <-- ??



#### Logistic regression

note: types of regression 복습

1. linear regression
   - simple linear regression: predict output based on one feature
   - multiple linear regression: predict output based on multiple features
2. logistic regression
   - binary: based on features, predict an output that is one of two possible classes (e.g., 0 or 1) logistic regression은 회귀이지만 확률값을 계산해서 분류를 하는 모델이다.
   - multinomial: based on features, predict an outputs that is one of many possible classes (i.e., multiple categories, two or more discrete outcomes) (e.g., predict what will be the most used transportation type in 2030 - possible outputs can be train, bus, tram, bikes.)



**cost function**

logistic의  cost function은 linear regression의 cost function과는 조금의 차이가있다.
$$
J(W,b) = \frac{1}{m}\sum_{i=1}^m(H(x^{(i)})-y^{(1)})^2
$$
linear regression 문제를 해결할때와 동일하게 input feature와 parameter를 linearly combine하지만, natural logarithm을 사용한다. linearyl combined된 input + parameter를 sigmoid function에 plug-in해서 probability(확률값)을 찾는다. 그래서 다음과 같이 공식으로 hypothesis를 찾을 수 있다.
$$
H(X) = \frac{1}{1+e^{-W^TX}}
$$
이를 기반으로 optimization을 진행한다. best parameter는 gradient ascent 또는 gradient descent를 통해서 찾을 수 있다. 

binary logistics 문제의 경우
$$
P(y_i = 1 | x_i,{\theta}) = {\sigma}(z_i)=\frac{1}{1+e^{z_i}}\\
P(y_i = 0 | x_i,{\theta}) = 1-{\sigma}(z_i)\\
z_i = \hat{y}_i = log(odds_i)  = log(\frac{p_i}{1-p_i})
$$
**logit transformation**

logit transformation= processing of wrapping log around odds or odds ratio. probability를 계산하기위해 0에서 1사이의 값으로 제한해야하는데, log-odds가 linear과 probability form 사이의 gap을 매꾸어준다. 

linear regression function을 사용하여 estimate한  y ("y-hat")의 값을 log-odds하고 한다. 

note: what is log-odds? odds= probability of success divided by failure = P(success)/P(failure)

![log_odds](C:\SJL\VQML_VQA\VQML\figures\logistic_regression_log_odds.png)

다음과 같은 순서로 변환해 나아간다.

probability increase-> odds increase -> log-odds increase ("monotonic relationship")

![probabillity](C:\SJL\VQML_VQA\VQML\figures\logistic_regression_probability.png)

likelihood (L(θ))

training set의 sample(or instance)마다 randomly estimated parameters θ를 사용해서 log odds를 계산한다. 그리고 sigmoid function을 통해 probability를 예측한다. 모든 probabilities를 곱해서 likelihood를 찾을 수 있다. 

![likelihood](C:\SJL\VQML_VQA\VQML\figures\logistic_regression_likelihood.png)

likelihood를 maximize해서 optimal parameters로 converge할 수 있다. likelihood를 maximize해서 best parameter를 찾으면서 probability of Y를 maximize하게 된다. 이 방식은 MLE(Maximum Likelihood Estimation)으로 불린다. maximum에 도달하게되면 처음 설정된 initial parameter값이 최적의 값으로 수렴된다. gradient descent/ gradient ascent와 같은 optimization algorithm으로 인해 이 수렴하는 과정이 진행된다. 

maximum을 찾는 방법:

log-likelhood의 partial derivative를 (with respect to each θ)계산한다. 즉 각 parameter의 gradient를 찾아서 optimal 방향으로 수렴하기 위한 방향으로 magnitude와 direction을 찾아간다.

linear regression때와 동일하게 learning rate (eta)으로 gradient ascent algorithm이 iteration마다 얼마나 큰 step으로 이동할지를 설정한다. (don’t want the learning rate to be too low, which will take a long time to converge, and we don’t want the learning rate to be too high, which can overshoot and jump around)

![gradients](C:\SJL\VQML_VQA\VQML\figures\logistic_regression_gradient.png)

**cost function** 

gradient descent algorithm을 통해 다음과 같이 반복적으로 parameter를 update해서 cost function을 최소화할 수 있는 optimal parameter를 찾는다.

![cost function](C:\SJL\VQML_VQA\VQML\figures\logistic_regression_cost.png)

cost function의 partial derivative (with respect to parameter)를 활용하여 parameter들이 optimal될때까지 parameter를 update한다.

![optimal parameter](C:\SJL\VQML_VQA\VQML\figures\logistic_regression_gradient_descent.png)

Cross entropy의 경우, convex graph이기때문에 gobal minimum을 보다 쉽게 찾을 있다.



- 

  

### 큰 DNN의 훈련 효율/성능을 높이는 방법

1. 연결가중치에 좋은 초기화 전략 적용
2. 좋은 활성화 함수 사용
3. batch normalization 사용
4. 보조작업 또는 비지도 학습을 통해 사전훈련된 network의 일부 재사용 (skipped less relevant to parameters)
5. 고속 optimizer사용
6. 희소 모델 사용
7. 학습률 scheduling
8. 규제



#### 연결가중치 초기화 설정

##### gradient 손실

출력층에서 입력층으로 오차 gradients를 전파하면서 역전파 algorithm을 진행한다. 신경망의 모든 parameter에 대한 오차 함수의 gradient를 계산하면 SGD단계에서 이 gradient를 사용하여 각 parameter를 수정한다. 이때에 algorithm이 하위층으로 진행할 수 록 gradient가 점점 작아지는 경우가 발생할 수 있다. SGD를 통해 연결 가중치를 변경하지 않은채로 두게되어 결국 훈련이 좋은 솔루션으로 수렴되지 않는다.

##### gradient 폭주

gradient 손실과는 반대로 gradient가 너무 커져서 비이상적으로 큰 가중치로 갱신되고 역전파 algorithm이 진행되다가 발산(diverse/explode)해 버리는 경우가 발생할 수 있다. Gradient 폭주는 주로 순환신경망에서 많이 발생한다. gradient 폭주의 경우, 불안정한 gradient로 인해 층마다 학습 속도가 매우 달라져서 심층 신경망의 훈련이 어려워 진다.

##### 적절한 gradient 역전파

활성화 함수를 잘 선택해야 gradient 손실이나 폭주를 방지할 수 있다.

적절한 gradient 역전파 algorithm 진행을 위해 다음 2 가지가 지켜져야한다:

1. gradient를 역전파 할때에는 영방향으로 양방향 신호가 적절한 수준으로 전달되어야한다. (신호의 폭주/소멸을 방지해야함.) 적절한 수준을 유지하기 위해서는 각 층의 출력에 대한 분산이 입력의 분산과 동일해야한다.
2. 역방향에서 층이 통과하기 전과 후의 gradient 분산이 동일해야한다. 

fan_in: 입력의 연결 개수

fan_out: 출력의 연결 개수

fan_in과 fan_out이 같지 않다면, 위 두가지 사항이 지켜지기 어렵다.

fan_avg = (fan_in + fan_out)/2

이런 현상을 도입하기 위해, 각 층의 연결 가중치를 다음 공식대로 무작위로 초기화한다. 이를 Xavier initialization 또는 Glorot initialization이라고 부른다. 
$$
normal{\space}distribution{\space}where:{\space}{\space}{\space}mean = 0{\space}{\space}{\space}and{\space}{\space}{\space}
variance={\space}{\sigma}^2 = \frac{1}{fan_{avg}}\\ 
or{\space}{\space}{\space}uniform{\space}distribution{\space}over{\space}range(-r,+r){\space}where:{\space}{\space}{\space}
r = \sqrt{\frac{3}{fan_{avg}}}
$$
활성화 함수마다 적절한 초기화 전략이 있다.

| 초기화 전략 | 활성화 함수                                | 정규분포(sigma^2) |
| ----------- | ------------------------------------------ | ----------------- |
| Glorot      | 활성화 함수 없음/ tanh/  logistic/ softmax | 1/fan_avg         |
| He          | ReLU(ELU를 포함한 ReLU의 변종들)           | 2/fan_in          |
| LeCun       | SELU                                       | 1/fan_in          |



#### 활성화 함수 

적절한 활성화 함수를 선택해서 실행 속도 향상, 과대적합 억제, 등 network의 훈련과정과 성능을 개선할 수 있는 여러가지 효과를 만들어 낼 수 있다.

일반적으로 주로 선호하는 활성화 함수 순서대로 나열해보면:

**SELU > ELU > LeakyReLU(그리고 변종들) > ReLU > tanh > logistics**



**ReLU**

ReLU는 continuous한 함수이지만, z=0에서 미분가능하지 않다. (기울기가 갑자기 높아져서 경사 하강법이 어뚱한 곳으로 튈 수 있음) 그리고 z<0일 경우에도 함수는 0이지만, 실제로 잘 작동하고, 계산 속도가 빨라서 기본적인 활성화 함수로 많이 사용됨. 가장 큰 장점으로 출력에 최대값이 없다는 점이 경사 하강법에 있는 문제를 일부 완화해줌.



**LeakyReLU**

ReLu함수를 주로 사용하지만, "죽은 ReLU(dying ReLU)"로 알려진 문제가 있다. 훈련하는 동안 일부 뉴런이 0 이외의 값을 출력하지 않는다는 의미임. 특히 큰 학습률을 사용하면, 신경망의 뉴런 절반이 죽어있기도 함. (모든 샘플에 대해 입력의 가중치 합이 음수가되면, 뉴런이 죽게된다. 가중치 합이 음수이면 ReLU함수의 gradient가 0이 되기때문에 SGD가 더 작동하지 않음.)

이런 경우 문제해결을 위해 LeakyReLU와 같은 ReLU의 변종을 사용한다. 
$$
LeakyReLU_{\alpha}(z) = max({\alpha}z, z)
$$
Hyperparameter alpha가 이 함수가 '새는(leaky)' 정도를 결정한다. (새는 정도: z<0일때에 이 함수의 기울기이며, 일반적으로 0.01로 설정함. 이 작은 기울기때문에 LeakyReLU가 절대 죽지않는다. 즉 혼수상태로 떨어지지만, 죽지는않고 다시 깨어날 가능성을 유지하는 것이다.) 다음 graph와 같이 음수부분이 작은 기울기를 가지게되어 0이 되지는 않는다.

<img src="C:\SJL\VQML_VQA\VQML\figures\leakyReLU.png" alt="leakyReLU" style="zoom: 50%;" />

LeakyReLU의 종류로는 RReLU(Randomized leaky ReLU)와 PReLU(parametric leaky ReLU)가 있다.

**RReLU(Randomized leaky ReLU)** - 훈련하는 동안 주어진 범위에서 alpha를 무작위로 선택하고 테스트한다.

**PReLU(parametric leaky ReLU)** - alpha가 hyperparameter가 아니고 다른 model parameter들 처럼 역전파에 의해 변경된다. 소규모 데이터 input에서는 훈련세트에 과대적합 될 위험이 있다.



**ELU (Exponential linear unit)** -  

<img src="C:\SJL\VQML_VQA\VQML\figures\ELU.png" alt="ELU" style="zoom:50%;" />

![ELU_formula](C:\SJL\VQML_VQA\VQML\figures\ELU_formla.png)

ELU의 장점:

- gradient 손실 문제 방지: z<0일 때에 (음수값이 들어오기때문에) 활성화 함수의 평균 출력이 0에 더 가까워짐. gradient가 0이 되지 않기때문에 neuron이 죽지는 않고, hyperparameter alpha를 통해 ELU가 수렴할 값을 정의할 수 있다. (보통 alpha를 1로 설정하지만, 양수의 다른 값으로 설정할 수 있음.)
- alpha=1이면, z=0에서 급격하게 변동하지 않기때문에 z=0을 포함해서 모든 구간에서 매끄러워 경사 하강법의 속도를 높여준다.

단점:

- 지수함수를 사용하기때문에 속도가 느린편이다. (ReLU나 그 변종들보다 계산이 느림.) 훈련하는 동안에는 수렴 속도가 빨라서 느린 계산이 상쇄되지만, test시에는 ELU를 사용한 network가 ReLU를 사용한 network보다 느리다.



**SELU(scaled ELU):**

SELU를 뉴런의 활성화 함수로 사용하면, 훈련하는 동안 각 층의 풀력이 평균0과 표준편차 1을 유지한다. 그래서 gradient의 손실과 폭주가 방지된다. 다음과 같은 사항들이 만족되면 network가 자기 정규화 (self-normalize)된다.

- input feature들이 표준화(mean=0, standard deviation=1)되어야한다.
- 모든 은닉층이 가중치가 lecun_normal로 초기화되어야함. (kernel_initializer="lecun_normal"로 설정)
- network는 일렬로 쌓은 층으로 구성되어야한다. 순환 신경망이나 skip connection과 같이 순차적이지 않은 구조에서는 SELU로 self-normalize하는 것이 보장되지 못한다.

perceptron의 activation 함수로 다음과 같이 다양한 option을 활용해서 알고리즘이 원하는 방향으로 작동할 수 있도록 유도할 수 있다.



**Logistics:**

그림 1의 TLU에서 계단 함수는 수평선으로 되어있어서 gradient를 계산할 수 없기때문에 계단 함수를 logistic(sigmoid) 함수로 바꾼다. logistics 함수는 어디서든지 0이 아닌 continuous 값을 가지고있어서 gradient가 잘 정의될 수 있다.



**tanh (hyperbolic tangent function)**:

logistic 함수처럼, 이 활성화함수 모양이 S이고, continuous해서 derivative를 찾을 수 있다. 출력 범위가 -1에서 1사이이고 훈련 초기에 각 층의 출력을 원점 근처로 모으는 경향이 있어서 model이 빠르게 수렴되도록 도와줌.



**softmax:**

출력 0에서 1사이 실수. softmax함수 출력의 총합은 1이다. 출력의 총 합이 1이 된다는것은 softmax의 매우 중요한 성질이다. 이 성질을 기반으로 softmax 함수의 출력을 "확률"로 해석할 수 있기때문이다.

```python
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y
```



**언제 어떤 활성화 함수를 사용해야할까?**

활성화 함수를 잘못 선택하면 gradient손실/ 폭주를 발생시킬 수 있다. 주의해야 함.

network가 self-normalize되지 못하는 구조 --> SELU보다 ELU(SELU가 z=0에서 연속적이지 않기때문)

실행 속도가 중요하다면 --> LeakyReLU(hyperparameter를 더 추가하고 싶지 않다면 케라스에서 사용하는 기본 alpha를 사용)

신경망이 과대적합 되어있다면 --> RReLU

훈련 세트가 매우 크다면 --> PReLU

보통 가장 널리 사용되는 활성화 함수이며, 많은 라이브러리와 hardware 가속기에 특화 되어있는 --> ReLU

(source: https://himanshuxd.medium.com/activation-functions-sigmoid-relu-leaky-relu-and-softmax-basics-for-neural-networks-and-deep-8d9c70eed91e)



#### Batch normalization (배치 정규화)

gradient 손실과 폭주를 방지하기위해 개발한 기법임. ELU와 함께 He initialization을 사용하면 훈련 초기 단계에서 gradient 손실/폭주 문제를 억제할 수 있지만, 훈련하는동안 이런 문제가 아얘 발생하지 않는것이 보장되지는 못한다. 최근 Hongyi Zhang 등의 최근 논문에는 batch normalization 없이 가중치 초기화 기법만으로 심층 신경만을 훈련시켜서 매우 복잡한 이미지 분류 작업의 최고 성능을 확보했다(2019). 그러나 아직 이를 뒷받침할 추가 논문을 통해 타당성을 확인 해야함.

**how-to:**

각층에서 활성화 함수를 통과하기 전 or 후에 모델에 연산을 하나 추가한다:

- 단순하게 입력을 원점에 맞추고 정규화한 다음, 
- 각 층에서 두 개의 새로운 parameter로 결과값의 스케일을 조정하고 이동시킨다. (하나는 scale 조정에, 다른 하나는 이동에 사용한다.)

만약 신경망의 첫번째 층으로 batch normalization을 추가하면, 훈련 세트를 표준화(e.g., via StandardScaler)할 필요가 없다. Batch normalization층이 표준화 역할을 대신한다. 

입력을 원점에 맞추고 정규화하려면 - algorithm이 입력의 평균과 표준편차를 추정해야함. 이를 위해서 주어진 mini batch에서 입력의 평균과 표준편차를 평가한다. (batch의 normalization을 계산하기때문에 hence the name "batch normalization") 

훈련하는 동안에는 입력을 전규화 한다음, scale을 조정하고 이동시킨다. 반면, test시에는 샘플 하나에 대한 예측을 찾아야하기 때문에, 더 복잡하다. 입력에 대한 평균과 표준편차를 계산할 수 없기때문. IID 조건을 만족하지 못함. 

keras에서는 batch normalization층 마다 입력 평균과 표준편차의 이동평균 (moving average)를 이용해서 훈련하는 동안 ''최종'' 통계를 추정한다. ('최종' 통계 - test시, 각 샘플의 대변하기위함. 전체 데이터 셋이 신경망을 통과해서 batch normlization층의 각 입력에 대한 평균과 표준편차를 계산하는 것과 같은 '최종' 통계를 나타내도록 다음 4개의 paraemter vectors를 사용)

batch normalization층 마다 4 prameter vectors가 학습된다.

- 출력 scale vector - gamma
- 출력 이동 vector - beta
- 최종 입력 평균 vector - mu ("이동평균")
- 최종 입력 표준편차 vector - sigma ("이동평균")

여기에서 mu와 sigma는 훈련하는 동안 추정되지만, 훈련이 끝난 후에 사용됨.



**활성화함수 이전/이후:**

활성화 함수 이전에 batch normalization을 추가하려면 은닉층에서 활성화 함수를 지정하지말고 batch normalization 층 뒤에 별도의 층으로 추가해야함. batch normalization층 입력마다 이동 parameter를 포함하기 때문에 이전 층에서 편향을 뺄 수 있다. (층을 만들때, use_bias=False로 설정.)



**매개변수(p.425~6 review again later regarding dimensions):**

BatchNormalization class에서 조정할 수 있는 hyperparameter:

- momentum

  적절한 값은 대부분 1에 가까움. (e.g., 0.9, 0.99, 0.999, 데이터셋이 크고 mini batch가 작으면, 1더 가깝게 증가시킨다.)

- axis - 정규화할 축을 결정한다. 기본값은 -1

- 이동 평균 V_hat을 다음 공식을 사용해서 update한다.
  $$
  \hat{v} \leftarrow \hat{v} \cross momentum + v\cross(1-momentum)
  $$

**장점:**

- gradient손실, 폭주 문제 감소. tanh나 logistic 함수와 같이 수렴성을 가진 활성와 함수 사용 가능.
- 가중치 초기화에 network가 훨씬 덜 민감해짐
- 훨씬 큰 학습률을 사용하여 학습과정의 속도를 크게 높일 수 있음. (e.g., 이미지 분류 모델에 적용하면 정규화가 14배나 적은 훈련 단계에서도 같은 정확도를 달성한다.)
- 규제와 동일한 역할을 하기도 함.

**단점:**

- 모델의 복잡도를 키워서 실행시간 면에서 손해, 층마다 추가되는 계산이 신경망 예측을 느리게 한다. 그러나 batch normalization을 사용하면 수렴이 훨씬 빨라지기때문에 보통 상쇄됨. 오히려 더 적은 epoch로 동일한 성능을 확보할 수도 있음.

##### gradient clipping

gradient 폭주문제를 완화시키는 방식. 역전파가 진행될때 일정 임계값을 넘지못하게 gradient을 잘라내는것이다.

순환 신경망에서는 batch normalization을 적용하기 어려워서 gradient clipping 방식을 많이 사용한다. 

model compile시, optimizer를 생성할때에 clipvalue와 clipnorm 매개변수를 지정하면 됨. clipvalue=1.0 지정 시, optimizer은 gradient vector의 모든 원소를 -1.0과 1.0사이로 clipping한다. 즉, 훈련되는 parameter에 대한 손실함수의 모든 편미분값을 -1.0에서 1.0안에 들어오도록 잘라내는것. 이 기능을 통해 gradient의 방향을 바꿀 수도 있다. 이런 문제를 방지하려면 clipnorm을 설정하면된다. clipnorm=1.0을 지정하면, gradient vector의 원소값들을 normalize해서 방향이 바뀌는 문제는 발생하지 않는다.

e.g. if gradient vector = [0.9, 100.0], then clipvalue=1.0 매개변수로 optimize가 정의되면, gradient vector = [0.9, 1.0]이 되어서 원래 두번째 축 방향을 향해야하는 것이 첫번째와 두번째의 대각선으로 바뀌여버린다. 만약 대신 clipnorm=1.0을 설정하면, gradient vector=[0.00899964, 0.9999595]로 방향이 유지된채 gradient값이 clipping될 수 있다.



#### 고속 optimizer

여기에서 논의하는 최적화 기법은 1차 편미분(Jacobian)에만 의존한다. 최적화 이론에는 2차 편미분(Hessian)을 기반으로한 뛰어난 algorithm들이 많다. BUT! Hessian algorithm들은 심층 신경망에 적용하기 어렵다. 2차 편미분 알고리즘을 사용하게되면 하나의 출력마다 n개의 1차 편미분이 아니라 n^2개의 2차 편미분을 계산해야하기 때문.(where n=parameter 개수). 심층 신경망은 보통 수만개의 parameter를 가지므로 2차 편미분 최적화 알고리즘은 memory 용량을 넘어서는 경우가 많다.

1. momentum optimization(모맨텀 최적화)

   경사하강법(SGD)에서는 이전 gradient가 얼마였는지 고려하지않는다.(그래서 gradient가 아주 작으면 매우 느려지는 문제 발생). Momentum optimization에서는 gradient가 얼마였는지는 매우 중요하게 고려한다. 

   모멘텀 알고리즘:
   $$
   1. {\space}m \leftarrow {\beta}m-{\eta}\grad_{\theta}J({\theta})\\
   2. {\space}{\theta}\leftarrow{\theta+m}
   $$
   매 반복에서 현재 gradient를 학습률을 곱한 후, momentum vector m에 더하고 이 값을 빼는 방식으로 가중치를 갱신한다. 즉 gradient가 속도(velocity)가 아니라 가속도(acceleration)로 사용되는 것이다. (momentum의 차이 만큼 gradient가 변하기때문에, velocity의 차이만큼 acceleration이 변하는 것과 동등하다고 보면 된다?) 여기에서 Beta는 일종의 마찰저항을 표현하고 momentum이 너무 커지는것을 막아준다. Beta=(0,1) 일반적인 momentum값은 0.9이다.

   terminal velocity (종단속도)를 구할때에 위 공식에서 1번의 좌우변을 equal하게 set해서 m을 구해보면 --> 종단속도는 학습률을 곱한 gradient에 (1/(1-beta))를 곱한것과 같은을 확인할 수 있다. beta가 0.9라면,  (1/(1-beta))는 10이 되고, momentum 최적화가 SGD보다 10배는 더 빠르게 진행된다는것을 확인할 수 있다. 

   **code 구현 방법:**

   compile시, SGD optimizer를 정의할때에 매개변수로 momentum을 전달하면된다.

   ```Py
   optimizer = keras.optimizers.SGD(lr=0.001, momentun=0.9)
   ```

   

2. Nesterov accelerated gradient (NAG)

   기본 momentum 방식에서 변종된 기법이다. 기본 momentum기법보다 더 빠르다. 현재 위치가 기존 gradient가 아니라 momentum 방향으로 조금 더 앞선 theta = theta + beta*m 에서 비용함수의 gradient를 계산한다.
   $$
   1. {\space}m \leftarrow {\beta}m-{\eta}\grad_{\theta}J({\theta+{\beta}m})\\
   2. {\space}{\theta}\leftarrow{\theta+m}
   $$
   NAG는 진동을 감소시키고 수렴을 빠르게 만들어준다. 

   **code 구현 방법:**

   ```py
   optimizer = keras.optimizers.SGD(lr=0.001, momentun=0.9, nesterov=True)
   ```

   

3. AdaGrad

   기본 SGD는 가장 가파른 경사를 따라 빠르게 내려가기 시작한다. AdaGrad는 이와 다르게 좀 더 정확한 방향으로 이동한다. 가장 가파른 차원을 따라 gradient vector의 scale을 감소시켜서 전역 최적점 쪽으로 좀 더 정확한 방향을 잡는다.
   $$
   1.{\space}s\leftarrow s+ \grad_{\theta}J({\theta})\cross\grad_{\theta}J({\theta})\\
   2.{\space}{\theta}\leftarrow{\theta}-{\eta}\grad_{\theta}J({\theta})\div\sqrt{s+{\epsilon}}
   $$
   NOTE: 여기에서 1의 multiply와 2의 divide는 각각 원소별 곱셈과 원소별 나눗셈을 의미한다.

   첫번째 단계에서는 gradient의 제곱을 vector s에 누적한다. vector화된 식은 vector s의 각 원소 s_i는 parameter theta_i에 대한 비용함수의 편미분을 제곱하여 누적한다. (비용함수가 i번째 차원을 따라 가파르다면 s_i는 반복이 진행됨에 따라 점점 더 커질것임.)

   두번째 단계에서는 기존 SGD와 비슷함. 한가지 파이는 gradient vector를 sqrt(s+e)로 나누어서 scale을 조정하는 것이다. (제곱을 했기때문에 sqrt(s)로 나누어서 scale 원복해야함. 여기에서 e는 0으로 나누게되는 것을 방지하기위해 작은 값이 더해진것이다.)

   AdaGrad는 학습률을 감소시키지만 경사가 완만한 차원보다 가파른 차원에 대해 더 빠르게 감소된다. (이를 adaptive learning rate이라고 한다.) 전역 최적점 방향으로 더 곧장 가도록 갱신되는데에 도움이 된다. 그래서 AdaGrad에서는 학습률을 덜 tuning해도 된다. 

   AdaGrad는 신경망을 훈련할때에 너무 일찍 멈춰버리는 경향이 있다. 학습률이 너무 감소되어서 전역 최적점에 도착하기전에 알고리즘이 멈춰버린다. But linear regression과 같이 간단한 작업에는 효과적일 수 있다. keras에 포함된 optimizer이지만, 실제 심층 신경망을 훈련할때에는 사용하지 않는다. 

    

4. RMSProp

   AdaGrad가 너무 빨리 느려져서 최적점에 수렴하지 못하는 위험이 있다. RMSProp은 훈련 시작부터 모든 gradient가 아닌, 가장 최근 반복에서 비롯된 graidnet만 누적한다. 그래서 알고리즘의 첫번째 단계에세 지수 감소를 사용한다.
   $$
   1.{\space}s\leftarrow {\beta}s+ (1-{\beta})\grad_{\theta}J({\theta})\cross\grad_{\theta}J({\theta})\\
   2.{\space}{\theta}\leftarrow{\theta}-{\eta}\grad_{\theta}J({\theta})\div\sqrt{s+{\epsilon}}
   $$
   code 구현:

   ```Python
   optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)
   ```

   아주 간단한 문제를 제외하고는 RMSProp가 AdaGrad보다 성능이 더 좋다.

5. Adam

   Adam = (적응적 모멘텀 최적화) Adaptive momtum optimizer (=momentum최적화 +RMSProp)
   $$
   1. {\space}m \leftarrow {\beta}_1{m}-(1-{\beta}_1)\grad_{\theta}J({\theta})\\
   
   2.{\space}s\leftarrow {\beta}_2s+ (1-{\beta}_2)\grad_{\theta}J({\theta})\cross\grad_{\theta}J({\theta})\\
   
   3.{\space}\hat{m}\leftarrow\frac{m}{1-{\beta}_1^t} \\
   
   4.{\space}\hat{s}\leftarrow\frac{s}{1-{\beta}_2^t} \\
   
   5.{\space}{\theta}\leftarrow{\theta}+{\eta}\hat{m}\div\sqrt{\hat{s}+{\epsilon}}
   $$
   t는 (1부터 시작하는) 반복횟수를 의미한다.

   beta_1는 momentum 감쇠 hyperparameter이고

   beta_2는 scale감회 hyperparamter이다.

   3,4번은 m과 s가 0으로 초기화되기때문에, 훈련 초기에 0으로 쏠리게되어있어서 이 둘의 값을 증폭시키는 역할을 한다.

   1,2,5번은 RMSProp과 momentum 최적화 방식과 비슷함. 단, 1에서 지수 감소의 평균을 구한다. 

   code 구현:

   ```Python
   optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
   ```

   Adam에서도 RMSProp과 AdaGrad에서 처럼 적응적 학습률/최적화 알고리즘이기때문에 학습률 hyperparameter (eta)를 튜닝할 필요가 적다.

   

6. Nadam

   Nadam (= Adaptive momtum optimizer + Nesterov 기법)

   Adam보다 조금 더 빠르게 수렴한다.

   

7. AdaMax

   Adam은 시간에 따라 감쇠된 gradient의 L2 norm으로 parameter update scale을 낮춘다. Adamax는 L2 norm에서 L_inf norm으로 바꾸는 것이다. (L_inf는 vector max norm을 계산하는것과 같음.) theta를 갱신할때에 s에 비례하여 gradient update의 scale을 낮춘다. 시간에 따라 감쇠된 gradient의 최대값이다. 실전에서는 AdaMax가 Adam보다 더 안정적이다. 데이터셋에따라 다르기때문에 Adam이 잘 동작하지 않는다면, AdaMax를 시도해볼 수 있다.

#### 회소 모델 훈련 (sparse model trianing)

모든 최적화 알고리즘은 대부분의 parameter가 0이 아닌 dense 모델을 만든다. 만약 엄청 빠르게 실행할 모델이 필요하거나 메모리를 적게 차지하는 모델이 필요하면 dense(밀집) model이 아닌, sparse(희소) model을 만들어서 훈련을 진행할 수 있다.

more info (chp.11 p.443)



#### 학습률 scheduling

한가지 전략은 - 큰 학습률로 시작하고 학습 속도가 느려질때 학습률을 낮추면 최적의 고정 학습률보다 좋은 솔루션을 더 빨리 발견할 수 있다. 훈련하는 동안 학습률을 어떻게 감소시킬지 - 감소시키는 전략에는 여러 방법이 있다. 이런 다양한 전략을 학습률 scheduling이라고 한다.

주로 사용되는 학습률 scheduling:

- 거듭제곱 기반 스케쥴링 (power scheduling)
- 지수기반 스케쥴링 (exponential scheduling)
- 구간별 고정 스케쥴링 (piecewise constant scheduling)
- 성능 기반 스케쥴링 (performance scheduling)
- 1 사이클 스케쥴링 (1cycle scheduling)



#### 규제(regularization)

dataset에서 feature들이 지나치게 많거나 training대비 testing 성능이 부족한경우, model의 generalization 부족하여 overfitting 이슈가 발생할 수 있다. 즉, 주어진 input에만 상세하게 맞춰진 model이 생성되어서 새로운 data가 주어졌을때에 정확도가 떨어지는 prediction output을 만들어내는 것이다. 이를 방지하기위해 cost function의 최소값을 위한 parameter를 계산할때에 규제를 적용한다. 

예측하려는 샘플의 분류가능한 class가 2개 이상일때에 다중 분류 모델을 활용한다.

- C매개변수 & max_iter

  - 반복적인 알고리즘을 사용하기 때문에 max_iter 매개변수의 값을 어느정도 큰 값으로 설정한다. (참고: 기본값 100에서 1,000으로 늘려야 경고가 발생하지 않는다.)

  - LogisticRegressioin은 기본적으로 릿지 회귀와 같이 계수의 제곱을 규제한다.(L2)

  - 릿지에서 alpha매개변수로 규제의 양을 조절했지만 LogisticRegression에서는 C매개변수를 사용한다.
  - C매개변수는 커지면 완화된다. 기본값이 1이지만, 완화하기위해 20으로 지정한다. (C 매개변수는 ridge의 alpha와는 반대 경향)

bias & variance & why use regularization: 

https://www.geeksforgeeks.org/lasso-vs-ridge-vs-elastic-net-ml/

https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b



- **Ridge** 

  L2 regularization. 기존 cost function에 다음과 같이 penalty를 더한다.
  $$
  \sum_{i=1}^{n}(y_i-\sum_{j=0}^{p}w_j\cross{x_{ij}})^2+\alpha\sum_{j=0}^{p}w_j^2
  $$
  ridge는 parameter (i.e. weights)에 규제를 더한다. penalty term lambda를 통해서 regression의 coefficient를 감소시킨다. 이는 model complexity와 multicollinearity를 감소시켜준다.

  when λ → 0 , the cost function becomes similar to the linear regression cost function (eq. 1.2). So *lower the constraint (low λ) on the features, the model will resemble linear regression model.* 

  Ridge는 coefficients를 zero에 가깝게는 감소시키지만, 완전히 zero로 만들어서 제외하지는 못한다.

  (official documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge)

- **Lasso**

  L1 regularization. Least absolute shrinkage와 selection operator를 통해서 다음과 같이 penalty를 더한다.
  $$
  \sum_{i=1}^{n}(y_i-\sum_{j=0}^{p}w_j\cross{x_{ij}})^2+\alpha\sum_{j=0}^{p}\abs{w_j}
  $$
  ridge와 비슷하지만 penalty로 가져가는 값이 squared가 아닌 magnitude라는 점이 다르다. Lasso와 같은 방식으로 규제를 하게되면 zero coefficient를 갖게될 수도 있다. 즉, 특정 feature이 output evaluation에서 완전히 제외되도록 설정할 수 있는것이다. Lasso는 overfitting을 방지하는 목적외에도 feature selection에도 활용될 수 있는 technique이다. (official documentation: https://scikit-learn.org/stable/modules/linear_model.html#lasso)

  feature를 selectively 사용할 수 있게 해주는것을 compressive sensing이라고도 부름.

  

  

  *need to review the following three topics*

- dropout

  dropout 비율 p를 설정해서 매 훈련 step에서 각 neuron이 임시적으로 dropout될 활률을 의미한다. (즉, 이번 step에서는 와전히 무시되지만, 다음 스텝에서는 활성화될 수 있다.) 보통 10~50%사이 값을 지정한다. 순환 신경망에서는 20~30%, 합곱신경망에서는 40~50%사이 값을 지정.

- MonteCarlo dropout(MC dropout)

  monte class에서 설정하는 dropout 

  dropout층 상속, call method override하고, training 매개변수를 True로 설정

- max-norm regularization

  불안정한 gradient를 완화하는데에 활용한다. 매개변수 bias constraints를 조정하여 편향을 조정한다.







#### 실용적 guideline

모든 case에 맞는 명확한 기준은 없지만, hyperparameter tuning을 크게 하지 않고 대부분의 경우에 잘 맞는 조건은 다음과 같다:

기본 DNN설정:

| hyperparameter | default                                                  |
| -------------- | -------------------------------------------------------- |
| 커널 초기화    | He 초기화                                                |
| 활성화 함수    | ELU                                                      |
| 정규화         | 얕은  신경망일 경우 없음 \ 깊은 신경망일 경우 배치정규화 |
| 규제           | 조기 종료(필요시 L2 규제 추가)                           |
| 옵티마이져     | momentum 최적화 (또는  RMSProp or Nadam)                 |
| 학습률 스케쥴  | 1 cycle                                                  |

만약 network가 완전 연결층을 쌓은 단순한 모델이라면, 다음과 같이 자기 정규화를 사용할 수 있다.

자기 정규화를 위한 설정:

| hyperparameter | default                                  |
| -------------- | ---------------------------------------- |
| 커널 초기화    | LeCun 초기화                             |
| 활성화 함수    | SELU                                     |
| 정규화         | 없음(자기  정규화)                       |
| 규제           | 필요시 alpha dropout                     |
| 옵티마이져     | momentum 최적화 (또는  RMSProp or Nadam) |
| 학습률 스케쥴  | 1 cycle                                  |

pytorch - compile 및 fit 방식 확인 필요  (build, etc)


## References

1. comparison between parameters vs. hyperparamenter: https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/

2. Neural Networks and Deep Learning (e-book) http://neuralnetworksanddeeplearning.com/index.html
3. constraints in regression models: https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/
4. Optimizing parameters in CNN: https://www.analyticsvidhya.com/blog/2021/06/create-convolutional-neural-network-model-and-optimize-using-keras-tuner-deep-learning/**
