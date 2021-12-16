---
layout: post                          # (require) default post layout
title: "CNN_enhanced"                   # (require) a string title
date: 2021-12-17       # (require) a post date
categories: [machinelearning]          # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [test]                      # (custom) tags only for meta `property="article:tag"`



---



# CNN(convolutional neural network)

local receptive field(국부 수용장) - 뉴런들이 시야의 일부 범위 안에 있는 시각 자극에만 반응 한다. 시각 피질의 국부 수용장은 수용장이라 불리는 시야의 작은 영역에 있는 특정 패턴에 반응한다. 그리고 시각 신호가 연속적인 뇌모듈을 통과하면서 뉴런이 더 큰 수용장에 있는 더 복잡한 패턴에 반응한다.

이미지 인식 문제에서 일반적인 완전연결층의 심층 신경망을 사용하지 않는 이유 - 완전 연결층의 심층 신경망은 작은 이미지에서는 잘 작동하지만, 큰 이미지에서는 아주아주아주 많은 parameter들이 만들어지기 때문에 문제가 된다. (e.g., 100x100 pixel image의 경우 input layer의 neuron이 10,000개이며, 첫번째 은닉층에서 neuron을 1000개만 가져가도 연결이 총 1천만개가 되어버린다.) 그래서 CNN은 layer를 부분적으로 연결하고 가중치를 공유해서 parameter들이 너무 많아지는 것을 방지 한다.

또한, 완전연결층의 심층 신경망(fully connected neural network)의 문제는 spatial orientation을 잃어버린다는 것이다. 

e.g., 강아지의 눈, 코, 입이 제대로된 위치에 존재하는것을 인지해야지만 해당 이미지가 강아지얼굴이라는 결론을 내릴 수 있을것이다.



## Convolutional layer(합성곱층)

첫번째 합성곱층의 뉴런은 input layer(입력 이미지)의 모든 pixel에 연결되는것이 아니라 합성곱층 뉴런의 수용장 안에 있는 pixel에만 연결된다. --> 두번째 합성곱증에있는 각 뉴런은 첫번째층의 작은 시각 영역 안에 위치한 뉴런에 연결된다.

이런 network구조는 다음과 같은 방식을 가능하게 한다 - 첫번째 은닉층에서 작은 저수준 특성에 집중하고, 그 다음 은닉층에서는 더 큰 고수준 특성으로 조합해 나가도록 도와준다.

CNN에서는 각 층이 2D로 표현되므로 뉴런을 그에 상응하는 입력 (2D image)과 연결하기 더 쉽다. 

수용장 사이 간격을 두어서 더 큰 입력층을 훨씬 작은 층에 연결하는 것도 가능 함. 한 수용장과 다음 수용장 사이의 간격을 **stride**라고 부른다. 수평 stride는 s_w, 수직 stride는 s_h라고 표기한다면, 이전층(상위층)의 i행, j열에 있는 뉴런이 이전 층의 i x s_h에서 i x s_h + (f_h -1)까지의 행과 j x s_w에서 j x s_w + (f_w -1)까지의 열에 위치한 뉴런들과 연결된다. 

다음 그림과 같이 stride값이 더 크면, 더 큰 pixel 크기의 이미지가 더 작은 feature map으로 추려질 수 있다.

<img src="C:\SJL\VQML_VQA\VQML\figures\stride.gif" alt="stride" style="zoom:50%;" />

<img src="C:\SJL\VQML_VQA\VQML\figures\stride2.gif" alt="stride1" style="zoom:50%;" />

뉴런의 가중치는 수용장 크기의 작은 이미지로 표현될 수 있다. **filter (or convolutional kernel)**를 통해 이전층의 이미지에서 원하는 부분만 "filter"해서 학습할 수 있다 예를 들어서 수직 필터(가운데 열은 1로 채워져있고, 그 외에는 모두 0인 7x7 행렬)를 통해서 가운데 수직선 부분은 제외하고는 나머지는 모두 0이 곱해지기때문에 후속 층으로 전송되지 못하고 무시하게 된다.

다음 그림과 같이 filter-right sobel을 통해서 오른쪽 수직선만 filter하는 경우 extract된 결과물이 어떤지 실제 이미지로 보여준다. 이렇게 전체 뉴런에 적용된 하나의 filter는 하나의 feature map을 만든다. feature map을 보면 filter를 가장 크게 활성화 시키는 이미지 영역이 강조된것을 확인할 수 있다. ![filter](C:\SJL\VQML_VQA\VQML\figures\filter_dog.png)

![local_receptive](C:\SJL\VQML_VQA\VQML\figures\local_receptive_field.gif)

feature map 쌓기

실제 convolutional layers는 여러가지 filter를 가지고 filter마다 하나의 특성 map으로 출력함으로 3D로 표현하는것이 더 현실적이다. 각 특성 map (feature map)은 pixel하나의 뉴런에 해당하고 하나의 특성 map안에서는 모든 뉴런이 같은 parameter(동일한 wieghts & bias)를 공유하지만, 다른 특성 map에있는 뉴런은 다른 parameter를 사용한다. 한 뉴런의 수용장(receptive field)는 이전층에있는 모든 특성map에 걸쳐 확장된다. 즉, 하나의 합성곱층에 여러 필터를 동시에 적용하여 입력에 있는 여러 특성을 감지할 수 있는 것이다. 

입력 이미지는 color channel마다 하나씩 여러 서브 층으로 구성되기도 한다 - 주로 R,G,B(빨강, 초록, 파랑) 만약 이미지가 흑백이라면 color channel은 1 이다. 위성 이미지의 경우에는 다른 빛 파장로 기록하기때문에 3개 이상의 channel을 사용하기도 함.

공식 표현:
$$
z_{i,j,k}=b_k+\sum_{u=0}^{f_h-1}\sum_{v=0}^{f_w-1}\sum_{k'=0}^{f_{n'}-1}x_{i',j',k'}\cross{w_{u,v,k',k}}\\
where{\space}{\space}{\space} i'=i\cross{s_h+u}{\space}{\space}{\space} \\ and{\space}{\space}{\space}j'=j\cross{s_w+v}
$$
이미지에 filter를 적용해서 output을 만들기 위한 저수준 딥러닝 함수 - tf.nn.conv2d()에 다음과 같은 매개변수가 전달된다:

- images - 입력의 mini batch (4D tensor)

- filters - 적용될 일련의 filters (4D tensor). filter를 사용하는 이유는 입력 이미지속의 features들과 그들의 spatial orientation을 보존하기 위함이다. (e.g., 이미지에서 강아지의 얼굴, 다리, 꼬리를 인지해서 강아지라는결론을 내리는 것.) 아래 그림과 같이 이미지를 가로질러가며 filter(주황색)가 감지한 feature를 계산해서 output을 특성 map(분혹색)에 기록해둔다.

  ![filter](C:\SJL\VQML_VQA\VQML\figures\filter_applied_numbers.gif)

- strides - 1 or 4개의 원소를 갖는 1D 배열로 지정할 수 있음. 4개에서 중간 2개는 수직, 수평 stride (s_h, s_w)이고 첫번째와 마지막 원소는 나중에 batch stride(일부 sample 건너뛰기위함)와 channel stride(이전 층의 특성 map이나 channel 건너뛰기위함)로 사용될 수 있다.

- padding - "VALID" or "SAME" 둘중 하나 선택. 

  - "VALID"는 zero padding을 사용하지 않는것. 그래서 stride에 따라 입력 이미지의 아래/오른쪽 행과열이 무시될 수도 있음. 수용장이 입력의 안쪽에만 놓인다는 의미로 valid임. 
  - "SAME"은 필요한 경우 zero padding을 사용하는 것. (입력 크기가 13이고, stride가 5이면, 출력 크기는 3임. (13/5 = 2.6 -->올림해서 3)) 필요시 입력 데이터 주변에 가능한 양쪽 동일하게 0이 추가된다. padding을 추가함으로서 output dimension을 input dimension과 동일하게 가져갈 수 잇다.

훈련 가능한 변수로 filter를 정의해서 신경망에 가장 잘 맞는 filter를 학습할 수 있음. 변수를 직접 만들기보다는 다음과 같이 keras.layers.Conv2D 층을 사용한다.

```Python
conv = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                          padding="same", activation="relu")
```



합성곱 layer이  매우 큰 memory용량이 필요하다.

예시)

filter= 5x5, stride=1, padding="SAME", 입력 이미지=150x100(RGB image이기때문에 channel=3)

입력 이미지에 filter 적용되면 다음과 같은 합성곱층 (convolutional layer)이 만들어진다 - 

- 해당 합성곱층의 parameter개수: (5x5x3+1)x200 = 15,200개 (''+1"은 bias때문에 추가된것임. 200개의 feature map들은 각각 filter로 정의된 parameter를 가진 15,000개의 neuron을 갖고있기때문에, 각 feature map은 76 parameters를 갖고있다. 해당 층에는 총 200개의 feature map이 있기때문에 76x200 = 15,200개의 parameter가 존재한다.)

- 해당 합성곱층의 특성map 개수: 200개 (각 150x100 = 15,000개의 뉴런 포함하고있고, 각 뉴런은 5x5x3=75개의 입력에대한 가중치 합을 계산 해야 한다.)
- 이 합성곱증에서 수행되어야하는 실수 계산은 5x5x3x15000x200 = 2억2천5백만개

훈련을 할때에는 역방향 계산을 위해 정방향에서 계산햇던 모든 값을 가지고있어야 함. 그래서 사용하는 메모리양이 엄청나진다. 만약 메모리부족으로 훈련에 실패한다면, mini batch 크기를 줄여볼 수 있다. 또는 stride 조정해서 차원을 줄이거나 몇개의 층을 제거할 수도 있다.



**Pooling**

pooling층의 목적은 계산량, 메모리 사용양을 줄이는 목적으로 만들어졌는데, 결과적은 과대적합의 위험을 줄여준다. prameter수를 줄이기 위해 입력 이미지의 subsample(축소본)을 만드는 것이다. 

pooling은 일정 수준의 invariance(불변성)을 만들어 준다. 작은 변화에 둔감해지는것이다. 최대 풀링으로 인해 이동, 회전, 확대, 축소에대한 불병성을 얻을 수 있다. 분류작업처럼 작은 부분에 영향을 받지않는 경우에 최대 풀링이 유용하게 사용될 수 있다. 이름 그대로 가장 큰 특징만 유지하고 나머지는 버려버리는 조금 극단적인 방법이다.

최대 풀링은 파괴적이기때문에 시맨틱 분할(pixel이 속한 객체에 따라 pixel을 구분하는 작업)과 같은 경우에는 최대 풀링이 적절하지 않다. (입력의 작은 변화가 출력에서 그에 상응되는 작은 변화로 이어져야 하기 때문에) 평균 풀링을 사용하면 최대 풀링보다 정보손실이 적어진다. 



## CNN구조

![CNN architecture [7].](https://www.researchgate.net/publication/333168248/figure/fig1/AS:759564609798145@1558105723414/CNN-architecture-7.ppm)

```python
#MNIST dataset기반 문제 해결위한 CNN
model = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation="relu", padding="same", 
                       input_shape=[28,28,1]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flattten(),
    keras.layers.Dense(128, activation="relu")
    keras.layers.Dropout(0.5)
    keras.layers.Dense(64, activation="relu")
    keras.layers.Dropout(0.5)
    keras.layers.Dense(10, activation="softmax")
])
```

CNN출력층에 다다를수록 filter개수가 늘어난다. 

저수준 특성(작은 동심원, 수평선, 등)의 개수는 적지만 이를 연결해서 고수준 특성을 만들 수 있는 방법이 많고 이런 방법이 더 합리적이다. 일반적으로 pooling층 다음으로 filter 개수를 두배로 늘린다. (64--> 128 -->256개) pooling층이 공간 방향 차원을 절반으로 줄이므로 이어지는 층에서 parameter 개수, 메모리 사용량, 계산비용을 늘리지 않고 특성 map 개수를 두배로 늘릴수 있다.

convolution층 이후, 완전연결 network는 두 개의 은닉층과 하나의 출력층으로 구성되어있다. dense network는 sample의 특성으로 1D 배열을 기대하므로 입력을 일렬로 평쳐야 한다. 그리고 밀집층 사이에 과대적합을 줄이기 위해 50%의 dropout 비율을 가진 dropout층을 추가한다.



### Data augmentation

훈련이미지를 랜덤하게 이동하거나, 수평으로 뒤집고 조명을 바꾸는 등 data를 증식해서 더 다양한 case가 포함되도록 데이터 세트 크기를 늘리는 것이다.

예시) 조명 조건에 민감하지 않은 모델을 만들기 위해 비슷하게 여러가지 명암을 가진 이미지를 생성하는 data augmentation을 진행할 수 있다.



### LRN(local response normalization)

합성곱층의 ReLU 활성화 함수 단계이후 LRN라는 경쟁적인 정규화 단계를 사용하는것이다. 가장 강하게 활성화된 뉴런이 다른 특성 맵에 있는 같은 위치의 뉴런을 억제한다. 이는 특성 map을 각기 특별하게 다른것과 구분되게 하고, 더 넓은 시각에서 특징을 탐색할수 있도록 유도해서 결국 모델의 일반화 성능이 개선된다.



Image classification 성능 향샹 순서대로 model : **LeNet-5** --> **AlexNet** --> **GoogLeNet**



### GoogLeNet

깊이 연결 층(depth concatenation layer)

장점: 더 깊은 CNN network를 구현했다. GoogLeNet은 inception module를 sub network로 가지고 있어서 GoogLeNet이 이전의 구조보다 훨씬 효과적으로 parameter를 사용한다. (GoogLeNet은 AlexNet대비 10배 적은 parameter를 가짐)

![inception_module](C:\SJL\VQML_VQA\VQML\figures\inception_module.png)

An **Inception Module** is an image model block that aims to approximate an optimal local sparse structure in a CNN. Put simply, it allows for us to use multiple types of filter size, instead of being restricted to a single filter size, in a single image block, which we then concatenate and pass onto the next layer. Think of the inception module as 여러 크기의 복잡한 패턴이 담긴 특성 맵을 출력하는 합성곱 층 on steroids.

inception module은 1 x 1 kernel의 합성곱층을 가진다. 이 층은 한번에 하나의 pixel만 처리하는것인데 어떤 목적으로 사용되는지?: 

- 공간상의 pattern(e.g., spatial orientation)을 잡을 수는 없지만, 깊이 차원에 따라 놓인 pattern을 찾을 수있음.
- 입력보다 더 적은 특성맵을 출력하기때문에 차원을 줄이는 의미의 bottleneck layer역할 수행 (parameter 수 감소, 훈련 속도 향상, 일반화 성능 향상)
- 합성곱 쌍을 통해 더 강력한 합성곱층을 구현한다. (합성곱쌍은 단순한 선형분류기보다 더 깊이있게 이미지의 특성을 찾아낸다. 두개의 층을 가진 신경망으로 이미지를 훑는 것임.) 



단점: inception module에서 합성곱 층의 합성곱 kernel 수는 hyperparameter이다. 그래서 inception module이 하나 추가될때마다 hyperparameter가 6개가 추가된다.





### ResNet







### YOLO







## CNN의 Hyperparameters

- kernel수

