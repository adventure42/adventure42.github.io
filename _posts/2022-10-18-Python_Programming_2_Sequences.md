---
layout: post                          # (require) default post layout
title: "Python Sequences"   # (require) a string title
date: 2022-10-18       # (require) a post date
categories: [PythonProgramming]          # (custom) some categories, but make sure these categories already exists inside path of `category/`
tags: [PythonProgramming]                      # (custom) tags only for meta `property="article:tag"`

---

<br>

# Sequences

sequence = generic term for an ordered set. 순서가 있는 포멧

Lists are the most versatile sequence type. The elements of a list can be any object, and lists are mutable (they can be changed) Elements can be reassigned or removed, and new elements can be inserted.

<br>

Python에서는 데이터 타입을 다음 2 가지 종류로 나눈다. There are two types of data format in python:

## container vs. flat

container: 서로 다른 자료형을 담을 수 있음. (e.g., list(element 수정 가능), tuple(element 수정 불가), collections.deque(collections package의 deque), 등)

flat: 한개의 자료형만 담을 수 있음. (e.g., str, bytes, bytearray, array.array, memoryview 등)

저장하고자 하는 데이터 타입에 맞는 형태를 사용하여 메모리를 더 효율적이게 사용하고 처리 속도를 향상할 수 있다. 

<br>

또는 다음과 같이 2 가지로 나눌 수 있다:

## 가변 vs. 불변

가변: 한번 선언한 후, 변경이 가능함. (e.g., list, bytearray, array.array, memoryview, deque, 등)

불변: 한번 선언하면, 변경할 수 없음. (e.g., tuple, str, bytes, 등)

```Python
# Mutable (가변형)  vs. Immutable(불변)
l = (15, 20, 25)
m = [15, 20, 25]
print(l, id(l))
print(m, id(m))

l = l * 2
m = m * 2
print(l, id(l))
print(m, id(m))

l *= 2
m *= 2
print(l, id(l))
print(m, id(m))
```

m(리스트)는 가변형이기때문에 동일 address(id)를 가지고 안에 원소들만 변경이됨.

l(튜플)은 불변형이기때문에 새로운 address(id)가 주어짐.

--> 활용에따라 이점을 사용하라. 

데이터 분석이나 머신러닝과정에서 여러가지 방안을 시도해볼때에는 가변형(리스트)으로 진행하는것이 메모리 사용에 더 효율적임. 매번 새롭게 할당하는 것을 방지.

<br>

## List 및 tuple 고급

### 지능형 list (comprehending lists)

특히, 데이터 전처리과정에 filter와 같은 함수를 다음과 같이 적용하여 효율적으로 진행할 수 있음.

```python
chars = '+_)(*&^%$#@!'
code_list1 = []
for s in chars:
    #유니코드 리스트
    code_list1.append(ord(s))
print(code_list1)

#지능형 리스트 (comprehending lists)
code_list2 = [ord(s) for s in chars]
print(code_list2)

# comprehending lists + map, filter
code_list3 = [ord(s) for s in chars if ord(s) > 40]
print(code_list3)
code_list4 = list(filter(lambda x : x > 40, map(ord, chars)))
print(code_list4)
```

<br>

### Generator

Python generator is a function that produces a sequence of results. It works by maintaining its local state, so that the function can resume again exactly where it left off when called subsequent times. 

You can think of generator as a powerful iterator.

연속되는 값을 반환하는데에 메모리를 좀 더 효율적으로 활용할 수 있도록 해준다. 

장점: 한번에 한개의 항목을 생성함. (즉, 연속되는 값들의 메모리를 유지할 필요 없음). 최소한의 메모리만 있으면 연속적인 수행 가능.

```python
# Generator 생성

# Generator: 한번에 한개의 항목을 생성함. (즉, 연속되는 값들의 메모리를 유지할 필요 없음). 
# 최소한의 메모리만 있으면 연속적인 수행 가능.
tuple_g = (ord(s) for s in chars)
# 연속된 값을 생성할 준비가 되어있어! 말 만해 바로 첫번째 값을 줄테니. 부릉부릉.
print(tuple_g)
print(type(tuple_g))

# 계속 연속되는 값들을 next()를 통해 확인할 수 있음.
print(next(tuple_g))
print(next(tuple_g))

# array 활용
import array
array_g = array.array('I',(ord(s) for s in chars))
print(array_g)
print(type(array_g))
# tolist()를 통해 리스트로 변경 가능함.
print(array_g.tolist())

# List vs. Generator 비교:
# 리스트로는 한번에 확보됨
print(['%s' % c + str(n) for c in ['A', 'B', 'C', 'D'] for n in range(1,21)])
# generator를 통해 한꺼번에가 아닌, 하나씩 반환됨. 
print(('%s' % c + str(n) for c in ['A', 'B', 'C', 'D'] for n in range(1,21)))
for s in ('%s' % c + str(n) for c in ['A', 'B', 'C', 'D'] for n in range(1,21)):
    print(s)
```

<br>

#### 주의할 점

list 사용시, 깊은 vs. 얕은 복사 주의해야함.

```python
# 리스트 사용시, 주의할점. 
# 다음 두가지 list는 동일할까??
marks1 = [['~'] * 3 for _ in range(4)] #반복은하되, 사용하지않는 variable은 _로 표기 가능.
print(marks1)
marks2 = [['~']*3]*4
print(marks2)

# 수정 시, 다른점이 확인 됨.
marks1[0][1] = 'X'
print(marks1)
# 하나의 주소값이 4번 복사된 경우이기때문에, 4번 모두 index=1의 원소가 X로 바뀜!
marks2[0][1] = 'X'
print(marks2)

# 증명 - id값으로 증명해볼 수 있음.
print([id(i) for i in marks1])
print([id(i) for i in marks2])
```

<br>

### Asterik

*asterik을 활용하여 유연한 unpacking/packing을 구현할 수 있음.

```Python
# *를 사용해서 unpacking이 가능 함.
# 그냥 divmod에 숫자 두 개를 주어서 실행.
print(divmod(100,9))
# unpacking된 100과 9가 divmod에 주어져서 실행 됨.
print(divmod(*(100,9)))
# 100과 9가 주어진 divmod의 결과 값(11과 1)을 unpacking해서 출력
print(*(divmod(100,9)))

# x, y, rest = range(10)
# 0~9까지의 숫자가 unpacking되기에는 부족함.
# unpacking시 변수 개수가 분명하지 못할때에 *로 변동가능한 상태를 설정할 수 있음.

# 0은 x, 1은 y, 그리고 나머지는 list로 packing되어서 rest에 주어짐.
x, y, *rest = range(10)
print(x, y, rest)

# 0은 x, 1은 y, 그리고 존재하지않는 나머지는 빈 list로 rest에 주어짐.
x, y, *rest = range(2)
print(x, y, rest)

# 0은 x, 1은 y, 그리고 나머지는 묵어서 list로 rest에 주어짐.
x, y, *rest = 0, 1, 2, 3, 4, 5
print(x, y, rest)

# 0은 x, 1은 y,... 차례대로 주어지고, 남은 1개는 list로 rest에 주어짐.
x, y, w, z, v, *rest = 0, 1, 2, 3, 4, 5
print(x, y, w, z, v, rest)
```

<br>

### 정렬

```Python
# sort vs. sorted
# reverse, key=len, key=str.lower, key=func...

f_list = ['orange', 'apple', 'mango', 'papaya', 'lemon', 'strawberry', 'coconut']
print(f_list)

# sorted: 정렬 수, 새로운 객체 반환. (원본이 수정되지 않는다.)
print('sorted-', sorted(f_list))
print('sorted-', sorted(f_list, reverse=True))
print('sorted-', sorted(f_list, key=len))
# key(기준)은 내가 직접 설정할 수 있음. 
# 원소의 마지막 글자를 기준으로 정렬
print('sorted-', sorted(f_list, key=lambda x: x[-1]))
print('sorted-', sorted(f_list, key=lambda x: x[-1], reverse=True))

# sort: 정렬 후, 객체 직접 변경 (원본이 수정되고 반환값이 None임.)
print('sort-', f_list.sort(), f_list)
print('sort-', f_list.sort(reverse=True), f_list)
print('sort-', f_list.sort(key=lambda x: x[-1], reverse=True), f_list)
```

<br>

### Lists vs. Arrays

List기반: 융통성 - 다양한 자료형, 범용적 사용, 속도도 빠른 편.

Array기반: 숫자 기반 머신러닝와 같은 연산이 필요한 경우. 보통 array는 리스트와 거의 호환됨.

<br>

<br>

## Hash

hash table: key에 value를 저장하는 구조. 적은 resource로 많은 데이터를 효율적으로 관리가능

- dict -> key 중복 허용 x

- set -> 중복 허용 x

면접에 자주 나오는 질문 중 하나- hash table의 key가 중복되는것을 어떻게 처리?

파이썬에서는 dict가 해쉬 테이블의 예이다.

왜 hash table을 사용하는지? --> 키 값의 연산 결과에 따라 직접 접근이 가능한 구조

접근하는 과정: 값을 해싱 함수에 넣고  -> 해쉬 주소값을 찾고 -> 주소값을 기반으로 key에 대한 value의 위치를 찾아서 value 참조

```Python
# Dict Setdefault 예제
source = (
    ('k1', 'val1'),
    ('k1', 'val2'),
    ('k2', 'val3'),
    ('k2', 'val4'),
    ('k2', 'val5')
)

new_dict1 = {}
new_dict2 = {}

# No use Setdefault
for k, v in source:
    if k in new_dict1:
        new_dict1[k].append(v)
    else:
        new_dict1[k] = [v]
print(new_dict1)

# Use Setdefault
for k, v in source:
    new_dict2.setdefault(k, []).append(v)
print(new_dict2)

# 주의 - 다음과 같이 같은 key의 value들을 모을 수 있을까? NOPE
new_dict3 = {k: v for k, v in source}
print(new_dict3)
# 출력해보면, key가 중복되는 경우, 덮어쓰기 때문에 가장 나중에 더하는 value 1개만 존재하게 됨.
```

<br>

### 불변

매우 중요한 정보를 보관해야하는 경우 활용 immutable dictionary 또는 immutable set을 다음과 같이 활용할 수 있음.

```Python
# Immutable dictionary (읽기 전용의 dictionary)
from types import MappingProxyType

d = {'key1': 'value1'}
# Read Only
d_frozen = MappingProxyType(d)
print(d, id(d))
print(d_frozen, id(d_frozen))

# 수정 가능
d['key2'] = 'value2'
print(d)

# 수정 불가
d_frozen['key2'] = 'value2'

# Immutable set (읽기 전용의 set)
# set도 read only로 바꿀 수 있음.
s1 = {'Apple', 'Orange', 'Apple', 'Orange', 'Kiwi'}
s2 = set(['Apple', 'Orange', 'Apple', 'Orange', 'Kiwi'])
s3 = {3}
s4 = {} # 이렇게하면 빈 set이 아닌, dictionary가 되어버림.
s4 = set()
s5 = frozenset({'Apple', 'Orange', 'Apple', 'Orange', 'Kiwi'})

# 수정 가능
s1.add('Melon')

# 수정 불가. frozenset에는 더할 수 없음.
s5.add('Melon')
```

<br>

<br>

# References

1. 우리를 위한 프로그래밍: 파이썬 중급
