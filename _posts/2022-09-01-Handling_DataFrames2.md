---
layout: post                          # (require) default post layout
title: "Handling DataFrames II"   # (require) a string title
date: 2022-09-01       # (require) a post date
categories: [DataProcessing]          # (custom) some categories, but make sure these categories already exists inside path of `category/`
tags: [DataProcessing]                      # (custom) tags only for meta `property="article:tag"`

---

<br>

#  Handling DataFrames II

## 열 & 행

dataframe df의 행과 열 크기를 알려준다. tuple형태로, (행,열) 이 순서대로 출력.

```
dataframe.shape

```

## 데이터 타입

자료형을 구성하는 데이터 타입 확인

```
dataframe.dtypes
```


## column명 확인

dataframe의 columns의 이름 확인

```
dataframe,columns
```

## dataframe의 기본정보 확인

총 인덱스 수, column 수
column별 이름, non-null 데이터 수, 데이터 타입

```
dataframe.info()
```

## dataframe의 기본 통계 데이터 확인

dataframe의 평균, 표준편차, 최소값, 최대값, quartile, 데이터 타입
(문자열인 경우, count, unique, top, freq 알려줌)

```
dataframe.describe()

#dataframe의 description에 numberic값을 가진 column들만 포함
import numpy as np
dataframe.describe(include=[np.number]

#dataframe의 description에 object(문자열)을 가진 column들만 포함
import numpy as np
dataframe.describe(include=[np.object]
```

## loc vs. iloc

loc: index를 기준으로 행 데이터 추출 (index는 숫자, 문자 지정가능)
iloc: 행 번호를 기준으로 행 데이터 추출 (행 번호는 dataframe의 row 순서대로 매겨지는 번호, 임의로 지정할 수 없고 반드시 숫자이다)

iloc과 loc 둘다 찾아서 반환하는 결과가 1개의 행이면 series형태를 반환하고, 만약 결과가 여러개의 행이라면 dataframe형태를 반환한다.

df.loc[[행],[열]], df.iloc[[행],[열]]으로 특정 행&열을 지정해서 데이터를 가져올 수 있다. (열은 지정할때에는 숫자/ 열이름 모두 가능)


## dataframe 복사본 만들기

데이터프레임내 데이터를 수정/삭제하는 작업을 하는 동안 원본은 유지하도록 복사본을 생성해서 작업할 수 있다. 

```
#dataframe의 첫 10줄만 복사본을 생성해서 dataframeC에 저장함.
dataframeC = dataframe.head(10).copy()
```

## groupby()

columnA, columnB 별로(이 순서대로) 그룹화 하여 columnC와 columnD의 평균값 구할 수 있다. groupby()함수에 들어가는 column 순서대로 group된다. 먼저 column A의 종류별로 group되고 그 다음, 각 group내에서 columnB의 종류별로 group된다. 

```
multi_group_mean = dataframe.groupby(['columnA','ColumnB'])[['columnC','columnD']].mean()
```

## 빈도수

nunique()로 빈도수를 구할 수 있다. (아래 예시와 같이 그룹화 한 후 빈도수 구하는 경우가 많음)

```
df.groupby('columnA')['columnB'].nunique()
```

## Broadcasting - Series & DataFrame

vector 연산이다.

broadcasting : 시리즈나 데이터프레임에 있는 모든 데이터에 대해 한 번에 연산을 하는 것을 말한다.

```
#같은 길이의 백터로 더하기, 곱하기 연산은 같은 길이의 백터가 출력된다.
#백터 + 백터, 시리즈는 백터의 한 종류이다.
v1 = columnA + columnB 
```

## dataframe의 행/열 삭제

행과열을 삭제하려면 drop 메서드를 사용해야 한다.

drop메서드에서 : 첫번째 인자는 열이름, 두번째 인자는 axis=1은 **칼럼의 레이블을** 의미하고 axis=0은 **인덱스를** 의미한다.

```
#columnA를 삭제한 dataframe2 생성
dataframe2 = dataframe.drop(['columnA'], axis=1)

#인덱스번호 0해당하는 첫번째 행 삭제한 dataframe3 생성
dataframe3 = dataframe.drop([0], axis=0) 
```

## column datatype 변환

astype() 함수에 원하는 data type을 인자값으로 전달해서 열의 dtype을 바꾼다. 소숫점까지 중요한 돈(especially dollars vs. won)계산을 할때에 숫자 타입 변환에 주의를 기울여야함.

tip: 문자열(object)보다 category 데이터타입이 메모리를 적게 차지한다.

```
dataframe['columnC'] = dataframe['columnC'].astype(str)
dataframe['columnC'] = dataframe['columnC'].astype(int)
dataframe['columnC'] = dataframe['columnC'].astype(float)
```

<br>

<br>

# References

1. [15 ways to create a Pandas DataFrame by Joyjit Chowdhury from TowardsDataScience](https://towardsdatascience.com/15-ways-to-create-a-pandas-dataframe-754ecc082c17)
