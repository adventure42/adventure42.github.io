---
layout: post                          # (require) default post layout
title: "Dimension Reduction and Classifiers"                   # (require) a string title
date: 2022-01-28       # (require) a post date
categories: [RCAClassification]          # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [RCAClassification]                      # (custom) tags only for meta `property="article:tag"`
---

<br>

# Classifiers

Data에 대해 자세하게 조사하기전에 항상 test dataset를 따로 떼어놓아야함. (데이터에 대한 편견 방지)

SGDClassifier (included in scikit-learn)모델은 한번에 하나씩 훈련 샘플을 독립적으로 처리하기때문에 온라인 학습에 적절함.

## 성능 측정

### cross validation 

교차 검증을 사용한 정확도 측정할 수 있으나, 특히 불균형 dataset을 다뤄야할때에 정확도는 선호되지 않는 metric이다. (e.g., MNIST dataset에서 10%정도만 '5'인데, 5가 아님을 분류하는 정확도는 당연히 90%수준 또는 이상으로 높을 수 밖에 없다.)

분류기를 평가하는 더 좋은 방법 - 오차행렬 confusion matrix

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

```

cross_val_predict()는 cross_val_score와 동일하게 k-fold cross validation을 수행하지만, 평가점수대신에 각 test fold에서 얻은 prediction을 반환한다.

cross_val_predict()로 얻은 예측값을 가지고 confusion matrix를 확인할 수 있다. 

<br>

### SGDClassifier

#### decision_function()

SGDClassifier는 decision function을 사용해서 각 sample의 점수를 계산하고, 이 점수가 임계값보다 크면 sample을 양성 클래스에, 작으면 음성 클래스에 할당한다.

decision threshold (결정 임계값)

#### 정밀도/재현율 trade off

임계값을 내리면 재현율 높아지고, 정밀도 낮아짐. 임계값을 올리면, 그 반대.

#### ROC curve for binary classification

ROC = false positive rate vs. true positive rate graph 

<br>

### RandomForestClassifier

#### predict_proba()

RandomForestClassier에는 decision_function()이 없고 그 대신 predict_proba() 함수를 통해서 sample이 행, 클래스가 열이고, sample이 주어진 클래스에 속할 확률을 담은 배열을 반환한다. (e.g., 주어진 sample이 RCA#1에 해당할 확률 70%) 

참고: scikit-learn classifier은 일반적으로 decision_function과 predict_proba 둘 다 또는 둘중 하나를 가지고있다.

<br>

## 다중 분류

multiclass or multinomial 분류기 - SGDClassifier, Random Forest Classifier, Naive Bayes Classifier, etc... 또는 binary 분류기를 여러 개 사용하는 다중 분류기법도 있음. (사용 할 수 있는 binary 분류기 - SVM(support vector machine), logistics regression, etc...)

scikit-learn의 분류기는 OvO(One-versus-One) 또는 OvR(One-versus-Rest) 를 사용해서 분류를 수행한다. 

MNIST dataset (of 0~9 숫자를 분류하려는 class로 가진 dataset)를 기반으로 분류기를 만드는 경우,

OvR : 특정 숫자 하나만 구분하는 숫자별 binary classifier를 10개 만들어서 분류하려는 클래스가 10개인 숫자 image classifier system을 만든다. 

OvO : 0과 1을 구분하는, 0과 2를 구분하는, 0과 3을 구분하는등과 같은 방식으로 각 숫자의 조합마다 분류하는 binary classiers를 만드는 전략이다. (총 Nx(N-1)/2 개의 binary classifiers (e.g., MINIST dataset의 경우 (10x9)/2=45개의 binary classifier를 만들어야함.) OvO전략의 장점은 각 분류기의 훈련에 전체 훈련 세트중 구별할 두 클래스에 해당하는 samples만 필요하다는 것이다. 



대부분의 이진분류 알고리즘에서는 OvR를 선호하지만, scikit-learn의 suport vector machine과 같은 일부 algorithm은 훈련 세트의 크기에 민감해서 큰 훈련세트에서 몇개의 분류기를 훈련시키는 것보다, 작은 훈련 세트에서 많은 분류기를 훈련시키는 쪽이 빠르다. 그래서 OvO를 선호한다. 

support vector machine classifier의 경우 sklearn.svm.SVC를 import해서 사용할 수 있고 자동으로 OvO전략을 사용해서 각 숫자의 조합인 N개의 분류기를 훈련시키고 각각의 결정 점수를 얻어서 점수가 가장 높은 클래스를 선택한다. (실제 decision_function()을 호출하면, sample에 대한 10개의 점수를 반환한다. 이중 가장 높은 점수의 클래스로 분류하는 것이다.) 

scikit-learn에서 OvO나 OvR중 원하는 전략을 강제로 사용하도록 설정하려면, OneVsOneClassifier이나 OneVsRestClassifier를 사용하면 된다. 

```python
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(X_train, y_train)
ovr_clf.predict([some_digit]) # 분류기가 예측 클래스를 반환
```

참고: SGD classifier의 경우에는 직접 다중 클래스로 분류할 수 있기때문에 별도로 scikit-learn의 OvR 또는 OvO를 적용할 필요가 없음.

<br>

### 에러분석

오차행렬 - cross_val_predict() 함수를 사용해서 예측을 만들고 confusion_matrix() 함수를 호출해서 오차행렬을 확인할 수 있다. 

```python
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx # 오차행렬 출력

plt.matshow(conf_mx, cmap=plt.cm.gray) #오차 행렬을 이미지로 확인할 수 있음.
plt.show() #오차 행렬 이미지 출력
```

이렇게 오차행렬을 출력하면 에러에만 초점을 맞추어서 오차행렬 이미지를 확인하기 어렵다. (대부분의 이미지가 바르게 분류되어 주 대각선만 상대적으로 너무 밝기때문에) 다음과 같이 오차 행렬의 각 값을 대응되는 클래스의 이미지 개수로 나누어서 에러 비율을 비교할 수 있다. 다른 항목들은 그대로 유지하고, 주 대각선만 0으로 채워서 다시 오차행렬 이미지를 확인할 수 있다.

행 - 실제 클래스, 열 - 예측 클래스

어떤 숫자가 주로 잘못 예측되고있는지, 어떤 숫자가 어떤 숫자로 잘못 분류되고있는지 확인이 가능하다.

```python
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx/ row_sums 
np.fill_diagonal(norm_conf_mx,0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()
```

개개의 에러를 분석해보면, 분류기가 무슨일을 하고, 왜 잘못 되었는지에 대한 아이디어를 얻을 수 있지만, 상세한 이유를 알아내기는 어렵다. MNIST 손글씨 숫자의 분류기를 SGDClassifier로 분류한 경우, 선형 모델인 SGDClassifier를 사용한것이 에러의 주 원인이다. 선형 분류기는 클래스마다 pixel에 가중치를 할당하고 새로운 이미지에 대해 단순히 pixel 강도의 가중치 합을 클래스의 점수로 계산한다. 그래서 예를들어, 3과 5의 경우 몇개의 pixel만 다르기때문에 모델이 쉽게 혼동할 수 있다. 

3과 5의 주요 차이는 위쪽선과 아래쪽 호를 이어주는 작은 직선의 위치이다. 이 연결 부위가 조금 왼쪽으로 치우치면 분류기가 5로 분류할 수 있다. 즉, 이 분류기는 이미지의 위치나 회전 방향에 매우 민감하다. 3과 5를 혼동하는 에러를 줄이기위해서 한가지 방법은 이미지를 중앙에 위치시키고 회전되어 있지않도록 전처리 하는 것이다.  

<br>

<br>

# Dimension Reduction

MNIST 이미지를 생각해보면, 숫자가 그려진 중앙부분외의 가장자리 쪽의 픽셀은 blank 흰색으로, 제거된다고 해도 많은 정보를 잃지 않는다. 또한 인접한 두 픽셀은 종종 많이 연관되어있어서 두 픽셀을 하나로 합친다고해도 잃는 정보가 크게 영향을 주지 않는다. 

이렇게 차원을 축소시키는 방법은 품질이 감소될 수 있는 위험을 감수하는 대신에 훈련 속도를 빠르게 증가시킬 수 있다. 시스템의 성능이 나빠지거나, 작업 pipeline이 조금 더 복잡하게되고 유지 관리가 어려워 지는 단점이 있기때문에, 차원 축소를 고려하기 전에 훈련이 너무 느린지 원본 데이터로 시스템을 훈련시켜보아야 한다. 

어떤 경우에는 훈련 데이터의 차원을 축소시켜서 잡음이나 불필요한 세부 사항을 걸러내서 훈련 속도를 향상시키는 것 외에도 성능을 높이는데에 도움이 될 수 있다. 

종종 높은 차원수를 2D 또는 3D수준으로 낮추어서 graphical visualization을 통해 데이터의 형태를 시각화할 수 있다. 특히 군집과 같은 시각적인 패턴을 통해서 데이터에 대한 중요한 통찰을 얻을 수 있다. 

<br>

## 투영 (projection)

대부분의 훈련 sample이 모든 차원에 걸쳐 균일하게 퍼져있지 않다. Few or several specific feature들이 서로 강하게 연관되어 있다. 그래서 결과적으로 고차원 공간 안의 저차원 subspace(부분 공간)에 훈련 sample들으 놓여 있다. 이 형태를 graph해보면, 3D 공간에있는 2D 부분공간에 data sample들을 그려볼 수있다. 즉, 3D 데이터 sample들을 부분 공간에 수직으로 (sample과 평면 사이의 가장 짧은 직선을 따라) 투영하면, 2D 데이터셋을 얻을 수 있다. 기존 3D 공간에서 특성 x1, x2, x3를 축으로 위치했던 데이터 sample들은 (평면에 투영된 좌표인) 새로운 특성 z1, z2를 축으로 표현될 수 있다. 

<br>

## 매니폴드

만약 3D 공간에서 데이터 sample들이 스위스 롤 형태로 동그랗게 말린 roll형태를 가진 상태이라면, 그냥 평면에 투영시키면 roll 형태의 층이 서로 뭉개져버리는 문제가 발생한다. 이런 경우에는 매니폴드를 모델링하는 방식이 필요하다. 

많은 차원 축소 알고리즘이 훈련 샘플이 놓여있는 매니폴드를 모델링하는 방식으로 작동한다. 이를 매니폴드 학습 (manifold learning)이라고 한다. 

매니폴드는 바로 처리해야하는 작업(e.g., 분류, 회귀, 등)이 저차원의 매니폴드 공간에 표현되면 더 간단해질 것이라는 가정과 함께 병행된다. 이런 가정을 주어진 데이터셋에 따라 valid할 수 있고 아닐수도있다. 전적으로 데이터셋에 달려있다. 

<br>

## PCA

PCA(Principal Component Analysis)는 먼저 데이터에 가장 가까운 초평면(hyperplane)을 정의한 다음, 데이터를 이 평면에 투영시킨다. 올바른 hyperplane을 선택하는 것이 매우 중요한다. 

데이터의 분산이 최대로 보존되는 축을 선택하는것이 정보가 가장 적게 손실되므로 합리적인 선택이된다. (원본 데이터셋과 투영된 것 사이의 평균제곱 거리가 최소화되는 선택임.)

고차원 데이터셋이라면 PCA는 첫번째 축에 직교하고 남은 분산을 최대한 보존하는 두번째 축을 찾고, 그 다음 이전 두 축에 직교하는 세번째 축을 찾고, 데이터셋에 있는 차원의 수 만큼 네번째, 다섯번째, ... n번째 축을 찾는다. 

i번째 축을 이 데이터의 i번째 PC(주성분 principal component)라고 부른다. 

SVD(singular value decomposition)이라는 표준 행렬 분해 기술을 통해서 훈련 데이터 셋의 주성분을 찾을 수 있다. numpy의 svd() 함수를 사용해서 훈련 세트의 모든 주성분을 구한 후, 처음 두개의 주성분을 정의하는 두 개의 단위 vector를 다음과 같이 추출할 수 있다. 

```python
import numpy as np

X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:,0]
c2 = Vt.T[:,1]
```

참고: PCA는 데이터셋의 평균이 0이라고 가정한다. scikit-learn의 PCA python class는 이 작업을 자동으로 처리해준다. 위 코드와 같이 PCA를 직접 구현하거나, 다른 라이브러리를 사용한다면 먼저 데이터를 원점에 맞추어야한다. 

<br>

### d 차원으로 투영하기

주성분을 모두 추출해냈다면 처음 d개의 주성분으로 정의한 hyperplane에 투영해서 데이터셋의 차원을 d차원으로 축소시킬 수 있다. 이 hyperplane은 분산을 가능한 최대로 보존하는 투영이다. 

첫 두 개의 주성분으로 정의된 평면에 훈련 세트를 투영하는 코드는 다음과 같다.

```python
W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)
```

<br>

### scikit-learn 사용하기

PCA모델을 사용해서 데이터셋의 차원을 2로 줄이는 코드는 다음과 같다.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)
```

<br>

### 분산의 비율

explained_variance_ratio_ 변수에 저장된 주성분의 "설명된 분산의 비율"은 각 주성분의 축을 따라 있는 데이터셋의 분산 비율을 나타낸다. (e.g., 3D 데이터셋의 처음 두 주성분에 대한 설명된 분산의 비율은 - 예를 들어, 데이터셋 분산의 85%가 첫 번째 PC를 따라 놓여있고, 15%가 두번째 PC를 따라 놓여있음을 알려준다.)

<br>

### 적절한 차원 수 선택하기

축소할 차원 수를 임의로 정하기보다는 충분한 분산 (e.g., 95%)이 될 때까지 더해야 할 차원 수를 선택하는 것이 간단하다. 

```python
pca = PCA()
pca.fit(X_train)
cum_sum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cum_sum >= 0.95) + 1

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)
```

<br>

### random PCA

svd_solver 매개변수를 "randomized"로 지정하면 scikit-learn은 확률적 알고리즘인 random PCA를 사용해서 처음 d개의 주성분에 대한 근삿값을 빠르게 찾는다.

```python
rnd_pca = PCA(n_components=154, svd_solver="randomized")
X_reduced = rnd_pca.fit_transform(X_train)
```

<br>

### incremental PCA

전체 훈련 데이터셋이 너무 크기때문에 SVD 알고리즘 실행을 위해 메모리게 올리는것이 문제가 되는 경우, IPCA (점진적 incremental PCA)알고리즘을 활용할 수 있다. 훈련 데이터셋을 mini batch로 나눈 뒤, IPCA 알고리즘에 한번에 하나씩 주입한다. 이 방식은 데이터 셋이 너무 큰 경우외에도 온라인으로 새로운 데이터가 준비되는대로 실시간으로 PCA를 적용하려할때에 활용될 수 있다. 

예시로 MNIST 데이터셋을 100개의 mini batch로 나누고 scikit-learn의 IncrementalPCA 클래스에 주입하여 MINIST 데이터셋의 차원을 154개로 줄이는 코드가 다음과 같다.

```python
from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch) #전체 훈련세트가 아니기때문에 fit()대신 partial_fit()를 mini batch마다 호출해야함.
    
X_reduced = inc_pca.transform(X_train)
```

다른 방식으로는 numpy 라이브러리의 memmap 클래스를 함께 활용하는 방법이 있다. 하드 디스크의 binary 파일에 저장된 매우 큰 array를 메모리에 들어 있는 것처럼 다루는데, memmap 클래스가 필요한 데이터를 그때그때 메모리에 적재한다. IncrementalPCA는 특정 순간에 array의 일부만 사용하기때문에 메모리 부족 문제의 발생을 방지할 수 있다. 예시 코드는 다음과 같다.

```python
X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m,n))

n_batches = 100
batch_size = m // n_batches #전체 데이터셋의 m개의 samples를 n_batches개의 mini batch로 나눔.
inc_pca = IncrementalPCA(n_components=154, batch_size = batch_size)
inc_pca.fit(X_mm) #메모리에 올라온 데이터를 사용하는것이기때문에 fit()을 호출함.
```

<br>

### Kernel PCA

Kernel trick(커널 트릭)은 support vector machine의 비선형 분류와 회귀를 수행하기 위해 sample을 매우 높은 고차원 공간 (특성 공간 feature space)으로 mapping하는 수학적 기법이다. 이 기법을 통해 원본 공간에서는 복잡한 비선형 decision boundary (결정 경계)를 고차원 특성 공간에서의 선형 decision boundary로 표현 할 수 있다. 

이 기법을 PCA에 적용해서 차원 축소를 위한 복잡한 비선형 투영을 실행할 수 있다. 이 방법은 kernel PCA(kPCA)라고 부른다. 투영된 후에 sample의 군집을 유지하거나 꼬인 매니폴드에 가까운 데이터셋을 펼칠때에도 유용하다.

kPCA의 kernel로 선형 커널, RBF 커널, sigmoid 커널, 등을 적용할 수 있다.

RBF 커널을 지정하여 kPCA를 적용한 코드 예시는 다음과 같다.

```python
from sklearn.decomposition import kernelPCA

rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)
```

kPCA는 비지도 학습이기때문에 좋은 커널과 hyperparameter를 선택하기위한 명확한 성능 측정 기준은 없다. but, 차원 축소는 종종 지도 학습의 전처리 단계로 활용된다. 이때 grid search를 사용해서 주어진 문제에서 성능이 가장 좋은 커널과 hyperparameter를 선택할 수 있다.

다음 코드는 두 단계의 pipeline을 구축한다 - 먼저 첫번째 단계로 kPCA를 사용해서 차원을 2차원으로 축소하고, 분류를 위해 logistic regression을 적용한다. 그 다음 단계로 GridSearchCV를 사용해서 kPCA의 가장 좋은 커널과 gamma parameter를 찾아서 가장 높은 분류 정확도를 얻는다.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression())
])

param_grid = [{
    "kpca__gamma":np.linspace(0.03, 0.05, 10),
    "kpca__kernel":["rbf", "sigmoid"]
}]

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X,y)

print(grid_search.best_params_)
```

가장 좋은 커널과 hyperparameter는 **best_params_**변수에 저장된다.

<br>

## LLE (Locally Linear Embedding)

LLE(지역 선형 임베딩)은 또 다른 비선형 차원 축소 (nonlinear dimensionality reduction, NLDR) 기술이다. LLE는 투영에 의존하지 않는 매니폴드 학습방식이다. 먼저 각 훈련 sample이 가장 가까운 이웃에 얼마나 선형적으로 연관되어있는지 측정한다. 그 다음, 국부적인 관계가 가장 잘 보존되는 저 차원 표현을 찾는다. 이 방법은 잡음이 너무 많지 않다면, 꼬인 매니폴드을 펼치는데에 유용한 방법이다. 

```python
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)
```

LLE 작동 방식:

1. 알고리즘이 각 훈련 샘플에 대해 가장 가까운 k개의 샘플을 찾는다. (위 코드에서는 k=10)

2. 각 샘플과 나머지 샘플 사이의 제곱 거리가 최소가 되게 만들어주는 가중치 값을 구한다. (이웃에 대한 선형 함수로 각 샘플을 재구성한다. 나머지 샘플이 k개의 이웃에 해당되지 않는 경우에는 가중치값=0)

3. 가중치 행렬은 훈련 샘플사이에 있는 지역 선형 관계를 담고있다.

4. 가능한 이 관계가 보존되도록 훈련 샘플을 d차원 공간으로 mapping한다. (d < n)

5. d차원 공간에서 샘플을 고정하고 최적의 가중치를 찾는 대신, 가중치를 고정하고 저차원의 공간에서 샘플 이미지의 최적 위치를 찾는다.

계산복잡도는 가중치 최적화에 O(mnk^2), 저차원 표현에 O(dm^2)이므로 m^2 term때문에 대량의 데이터셋에 적용하기 어렵다.

<br>

## 다른 차원 축소 기법들

MDS(Multi-Dimensional sScaling), 

Isomap, 

t-SNE(t-Distributed Stochastic Neighbor Embedding), 

LDA(Linear Discriminant Analysis)

<Br>

<br>

# References

1. Geron, Aurelien. Hands on Machine Learning. O'Reilly, 2019 
