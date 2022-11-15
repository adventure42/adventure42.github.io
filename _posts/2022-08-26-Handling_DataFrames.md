---
layout: post                          # (require) default post layout
title: "Handling DataFrames I"   # (require) a string title
date: 2022-08-26       # (require) a post date
categories: [DataProcessing]          # (custom) some categories, but make sure these categories already exists inside path of `category/`
tags: [DataProcessing]                      # (custom) tags only for meta `property="article:tag"`

---

<br>

# Handling DataFrames

pandas의 dataframe 생성 및 다루는 방법 정리

거의 pandas의 함수를 사용하기 때문에, 가장 먼저 pandas library를 가져온다. 
```
import pandas as pd
```

<br>

# dataframe 생성

## 1.파일에서 읽어오기
1. xlsx, csv, pickle 파일한번에 열기
read_csv() / read_excel() / read_pickle()
파일 경로는 full 경로로 넣어도된다. 또는 현재 위치라면 ./을 사용, 상위 폴더에서 경로를 지정하려면 ../을 사용한다.
```
# sep를 따로 지정하지 않으면 default는 comma이다.
dataframe = pd.read_csv('읽어오려는 파일 폴더 경로/file.csv', sep='\t')

dataframe = pd.read_pickle('읽어오려는 파일 폴더 경로/file.pickle')
```
2. pickle library사용
pickle.load()
```
import pickle
with open("읽어오려는 파일 폴더 경로/file.pkl", "rb") as f:
    dataframe = pickle.load(f)
```

3. json file
json file안에 한줄에 한개의 record가 있는경우, lines = True로 지정해서 한줄씩 json object를 읽어오게 한다.
```
dataframe = pd.read_json('data.json',lines=True)
```

## 2.constructor pd.DataFrame()
```
# 컬럼만 생성한 빈 dataframe 생성
dataframe = pd.DataFrame(columns=['columnA','columnB','columnC','columnD'])

# 한 행씩 데이터 입력
df.loc[0] = [2021,"shuttle","space",234.1] 
df.loc[1] = [2020,"drone","earth",432.9]
```

## 3.numpy array in DataFrame()
```
# 2 dimensional numpy array를 생성
data = np.array([[2021,"shuttle","sapce",234.1],
		[2020,"drone","earth",432.9],
                [2021,"submarine","ocean",89.5]])
                
dataframe = pd.DataFrame(data, columns = ['columnA','columnB','columnC','columnD'])
```

## 4.dictionary in DataFrame()
dictionary의 key들이 column이 되고, values이 각 column의 값들이 된다.

```
#인덱스를 따로 지정할 수 있다.
scientists = pd.DataFrame( 
    data={'Occupation': ['Chemist', 'Statistician'], 
          'Born': ['1920-07-25', '1876-06-13'], 
          'Died': ['1958-04-16', '1937-10-16'],
          'Age': [37, 61]},
    index=['Rosaline Franklin', 'William Gosset'],
    columns=['Occupation', 'Born', 'Age', 'Died'])  #columns 열 순서를 이렇게 따로 지정할 수 있다.


# 추가 내용: 순서보장 딕셔너리를 사용할 수도 있다.
from collections import OrderedDict 

# 괄호 표기에 주의!! 
# 보통 딕션어리에서는 (key: value)형태로 입력하지만, 
# ordered 딕션어리에서는 (key, value)와 같이 tuple 형태로 값들을 입력한다.
scientists = pd.DataFrame(OrderedDict([
    ('Name',['Robin Williams','Rosaline Franklin','William Gosset']),
    ('Occupation', ['Comedian/Actor','Chemist', 'Statistician']),
    ('Born',['1951-07-21','1920-07-25','1876-06-13']),
    ('Died',['2014-08-11','1958-04-16','1937-10-16']),
    ('Age',[46,37,61])
])
) 
```

## 5.list of dictionaries in DataFrame()
list내의 각 dictionary가 하나의 record(=row)가 된다.
각 dictionary의 key는 column명이 되고, value는 column의 값이 된다.
```
data = [{'columnA': 2014, 'columnB': "toyota", 'columnC':"corolla"}, 
        {'columnA': 2018, 'columnB': "honda", 'columnC':"civic"}, 
        {'columnA': 2020, 'columnB': "hyndai", 'columnC':"nissan"}, 
        {'columnA': 2017, 'columnB': "nissan" ,'columnC':"sentra"}]
       
dataframe = pd.DataFrame(data)
```

## 6.from_dict() method 사용
```
data = {'columnA': [2014,2018,2020,2017], 
        'columnB': ["toyota","honda","hyndai","nissan"],
        'columnC':["corolla","civic","accent","sentra"],
        'columnD':["space","earth","ocean","glacier"]}
 
df = pd.DataFrame.from_dict(data)
```
from_dict()를 사용하면 유용한 기능이 하나 있다. 쉽게 dataframe을 transpose할 수 있는것이다.

즉, column명들이 dataframe의 index가 되고, 각 행의 번호가 column명이 된다.

```
# columns에는 transpose한 후 column명을 지정해준다.
dataframe2 = pd.DataFrame.from_dict(
	data, orient='index',columns=['record1', 'record2', 'record3', 'record4'])
```


## 7.HTML page속 table
read_html() method를 통해 HTML page속의 table 태그를 찾아서 dataframe으로 받아온다. HTML contents를 받아와주는 requests library를 사용해서 이 방법을 활용할 수 있다.
```
import requests

url = 'https://www.abc.com/page-with-tables'
r = requests.get(url)
dataframe = pd.read_html(r.text)

```

## 8.vertical concatenation
one on top of the other (아래 예시를 보면, df1과 df2를 각각 만들고 이둘을 아래,위로 붙여서 df3를 생성한다)

```
data1 = [{'columnA': 2014, 'columnB': "drone", 'columnC':"space"}, 
        {'columnA': 2018, 'columnB': "shuttle", 'columnC':"earth"}, 
        {'columnA': 2020, 'columnB': "submarine", 'columnC':"ocean"}, 
        {'columnA': 2017, 'columnB': "yellow" ,'columnC':"glcier"}
       ]
       
df1 = pd.DataFrame(data1)

data2 = [{'columnA': 2019, 'columnB': "red", 'columnC':"magma"}]

df2 = pd.DataFrame(data2)

# 3 ways to concatenate vertically:

# axis = 'index' is same as axis = 0, and is the default 
df3 = pd.concat([df1,df2], axis = 'index') 

#OR
df3 = pd.concat([df1,df2], axis = 0)

# OR
df3 = pd.concat([df1,df2])
```

**참고:** 이렇게 concatenate하는 경우, 붙여서 새로 생성된 dataframe의 index번호가 뒤죽박죽일 수 있다. 그럴때는 reset_index()를 사용하거나, 아얘 concat 메소드의 index 파라미터에 True를 지정하면 새롭게 index가 0부터 순차적으로 매겨진다.
```
df3 = pd.concat([df1,df2]).reset_index()
#OR
df3 = pd.concat([df1,df2], ignore_index = True)
```

그리고 concatenation방식은 horizontal 방향으로도 가능하다.
Horizontal concatenation은 concat() 메소드를 사용하거나, merge()메소드를 사용하는 방삭이 있다. 

**concat()**
위의 예시와 동일하게 df1, df2를 생성한 후, 아래와 같이 axis 설정만 바꾸어준다.
```
df3 = pd.concat([df1,df2], axis = 'columns')
#OR
df3 = pd.concat([df1,df2], axis = 1)
```

**merge()**
merge()의 default 설정은 innerjoin이다. df1에 4 rows가 있지만, df2에 3 rows가 있기때문에 innerjoin으로 merge된 df3는 3 rows가 있다.
```
data1 = [{'columnA': 2014, 'columnB': "drone", 'columnC':"space"}, 
        {'columnA': 2018, 'columnB': "shuttle", 'columnC':"earth"}, 
        {'columnA': 2020, 'columnB': "submarine", 'columnC':"ocean"}, 
        {'columnA': 2017, 'columnB': "yellow" ,'columnC':"glcier"}
       ]
       
df1 = pd.DataFrame(data1)

data2 = [{'columnC': 'space', 'columnD': purple}, 
        {'columnC': 'earth', 'columnD': green}, 
        {'columnC': 'ocean', 'columnD': blue}
       ]
       
df2 = pd.DataFrame(data2)

# inner join on 'make'
# default가 inner join이기때문에 아래 두줄이 같다.
df3 = pd.merge(df1,df2,how = 'inner',on = ['columnC'])
df3 = pd.merge(df1,df2,on = ['columnC'])
```

만약 left join을 구현하려면,
```
# for a left join , use how = 'left'
df3 = pd.merge(df1,df2,how = 'left',on = ['columnC'])
```

## 9.transpose된 dataframe
```
# To transpose a dataframe - use .T method
df4 = df3.T

# To rename columns to anything else after the transpose
df4.columns = (['R1','R2','R3','R4'])
```

## 10.one-hot columns로 변환
One-Hot은 한 column의 값을 Binary Representation방식으로 표현하는 것이다. 해당하는 값이라면 column에 1이 표기되고, 나머지에는 0이 표기된다. 

예를 들어, 아래 dataframe을 보면
columnD에는 각 columnC의 값에 해당하는 색깔값이 주어져있다. (space는 purple, earth는 green, ocean은 blue가 각각의 record에 주어져있다.)

get_dummies()메소드로 column을 columnD(색깔)으로 지정하면, 
columnA, columnB, columnC 에 추가로 columnD_purple, columnD_green, columnD_blue가 생성된 dataframe이 만들어진다.
새로 추가된 column을 보면, 각각의 record에서 확인한 색깔 match를 기반으로, 해당이되는 column에만 1이 들어가고, 나머지 해당하지 않는 column에는 0이 들어간다.

```
data1 = [{'columnA': 2014, 'columnB': "drone", 'columnC':"space", 'columnD':"purple"}, 
        {'columnA': 2018, 'columnB': "shuttle", 'columnC':"earth", 'columnD':"green"}, 
        {'columnA': 2020, 'columnB': "submarine", 'columnC':"ocean", 'columnD':"blue"}]
       
df1 = pd.DataFrame(data1) 

dataframe = pd.get_dummies(df1,columns = ['columnD'])
```

<br>

<br>

# References

1. [15 ways to create a Pandas DataFrame by Joyjit Chowdhury from TowardsDataScience](https://towardsdatascience.com/15-ways-to-create-a-pandas-dataframe-754ecc082c17)
