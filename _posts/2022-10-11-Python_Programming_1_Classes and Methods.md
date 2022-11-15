---
layout: post                          # (require) default post layout
title: "Python Classes and Methods"   # (require) a string title
date: 2022-10-11       # (require) a post date
categories: [PythonProgramming]          # (custom) some categories, but make sure these categories already exists inside path of `category/`
tags: [PythonProgramming]                      # (custom) tags only for meta `property="article:tag"`

---

<br>

# Python Class & Methods

## Python의 핵심

- sequence

- iterator

- functions

- class

<br>

## Class

### 함수 중심 (절차형) vs. 클래스 중심

규모가 큰 프로젝트(프로그램) : 함수 중심 -> 데이터 방대 -> 복잡

프로그램이 개선될때 데이터가 더 커지고 복잡해지며, 한번 만들고나서 개선이 어렵다. 여럿이 협업을 하다보면 코드의 중복이 발생하기 쉽다.

클래스 중심 : 데이터 중심 -> 객체로 관리

함수의 parameter로 감소하게되고, 클래스의 구성 요소들이 객체로 관리되어서 코드의 재사용이 가능하고 중복을 방지할 수 있어서 관리가 쉬워진다.

==> OOP (Object Oriented Programming)객체지향 : 코드의 재사용, 코드 중복 방지, 유지보수, 대형 프로젝트에 적합함. 

단, 만들고자 하는 프로그램의 목적과 규모에 따라서 적절한것을 선택해야 함. 만약 소수의 기능한 실행하면되는 특정 utility(e.g., 단순 crawling)가 목적이라면 함수 중심도 OK. 

<br>

### 클래스 구조

구조 설계 후 재사용성 증가, 코드 반복 최소화, 메소드 활용

기본적으로 python에서 제공하는 메소드: str, repr, 등

<Br>

### 클래스 변수 선언

instance의 namespace에 없으면, 상위에서 검색 (dir() 메소드로 조회 가능)

즉, 동일한 이름으로 변수 생성 가능 (instance 검색 후 -> 상위(클래스 변수, 부모 클래서 변수))

<br>

## Methods

### 클래스 기반 메소드

- Class method

   첫번째 인자로 클래스를 의미하는 cls를 받는다.

   클래스 변수를 조정할때에는 class method를 통해 하는것을 권장함.

- Instance method

   self가 들어가있으면 instance method임.

   self : 객체의 고유한 속성 값을 사용한다는 것을 의미함.

- Static method

  does not take any specific parameter (Class vs. Static debatable)

  instance도 class도 애매한 method일것 같다면 static을 사용. 더 유연함.

각 종류별 method의 예시:

```python
class Car():
    # 클래스 변수 선언 (모든 인스턴스가 공유)
    price_per_raise = 1.0

    def __init__(self, company, details):
        self._company = company
        self._details = details

    def __str__(self):
        return 'str: {} - {}'.format(self._company, self._details)

    def __repr__(self):
        return 'repr: {} - {}'.format(self._company, self._details)

    # Instance Method
    # self가 들어가있으면 instance method임.
    # self : 객체의 고유한 속성 값을 사용한다는 것을 의미함.
    def detail_info(self):
        print('Current ID : {}'.format(id(self)))
        print('Car detail info: {}, {}'.format(self._company, self._details.get('price')))

    # Instance Method
    def get_price(self):
        return 'Before Car Price -> company:{}, price:{}'.format(self._company, self._details.get('price'))

    # Instance Method
    def get_price_calc(self):
        return 'After Car Price -> company:{}, price:{}'.format(self._company, self._details.get('price')*Car.price_per_raise)

    # Class Method
    # 첫번째 인자로 클래스를 의미하는 cls를 받는다.
    # 클래스 변수를 조정할때에는 class method를 통해 하는것을 권장함.
    @classmethod
    def raise_price(cls, per):
        if per <= 1:
            print('Please Enter 1 or more')
            return
        cls.price_per_raise = per
        print('Succeeded! price has increased.')

    # Static Method
    # does not take any specific parameter (Class vs. Static debatable)
    # instance도 class도 애매한 method일것 같다면 static을 사용. 더 유연함.
    @staticmethod
    def is_bmw(inst):
        if inst._company == 'Bmw':
            return 'OK. this car is {}'.format(inst._company)
        else:
            return 'sorry this car is {}'.format(inst._company)
```

<Br>

### Special method

special method = "magic method"

magic method : class안에 정의 할 수 있는 특별한 built-in method. 사용자가 원하는 대로 다시 설정하여 원하는 operation을 수행할 수 있다.

special method를 활용하는 예시:

```python
class Vector(object):
    def __init__(self, *args):
        '''Create a vector, example : v = Vector(5,10)'''
        if len(args) == 0:
            self._x, self._y = 0, 0
        else:
            self._x, self._y = args

    def __repr__(self):
        '''Returns the vector infomations'''
        return 'Vector(%r, %r)' % (self._x, self._y)

    def __add__(self, other):
        '''Returns the vector addition of self and other'''
        return Vector(self._x + other._x, self._y + other._y)
    
    def __mul__(self, y):
        return Vector(self._x * y, self._y * y)

    def __bool__(self):
        return bool(max(self._x, self._y))
```

<br>

### Data modeling 

data modeling : 데이터를 체계적으로 관리하기 위함.

객체 : 파이썬의 데이터를 추상화. 모든 객체 -> id, type -> value로 확인할 수 있음.

#### NamedTuple이란?

tuple : 변경되지 않아야하는 값들을 저장하는 데에 유용함.

named tuple = a type of container dictionaries. it exists under the "collections" module. Similar to a dictionary, named tuples contain keys that are hashed to a particular value. But on the other hand, it also supports access via index. index로도 key로도 store된 value에 접근할 수 있음. 

named tuple의 선언 방법 : class 형식으로  tuple을 추상화한다.

예시:

```python
# namedtuple 사용
from collections import namedtuple

# nametuple 선언
Point = namedtuple('Point','x y')
# class 형식으로  tuple을 추상화한다.(인스턴스 생성)
pt3 = Point(1.0, 5.0)
pt4 = Point(2.5, 1.5)

# use index to access value
print(pt3[0])
print(pt4[1])
# use key to access value
print(pt3.x)
print(pt4.y)

# namestuple의 활용 예시
l_leng2 = sqrt((pt3.x - pt4.x)**2 + (pt3.y - pt4.y)**2)
print(l_leng2)

# namedtuple 선언 방법에는 여러가지가 있음.
Point1 = namedtuple('Point', ['x', 'y'])
Point2 = namedtuple('Point', 'x, y')
Point3 = namedtuple('Point', 'x y')
Point4 = namedtuple('Point', 'x y x class', rename=True) # default는 False
# Dict to Unpacking
temp_dict = {'x': 75, 'y': 55}

# 선언한 namedtuple들의 객체 생성
p1 = Point1(x=10, y=35)
p2 = Point2(20, 40)
p3 = Point3(45, y=20)
p4 = Point4(10, 20, 30, 40)
p5 = Point1(**temp_dict)

# 사용
print(p1[0] + p2[1])
print(p1.x + p2.y)
x, y, w, z  = p4

# Built-in attributes and methods: _make(), _fields, asdict()
# _make(): 새로운 객체 생성
temp = [52, 38]
p4 = Point1._make(temp)
print(p4.x, p4.y)
print(p4)

# _fields : 필드 네임 확인
print(p1._fields, p2._fields, p3._fields, p4._fields)

# _asdict() : OrderedDict 반환
print(p1._asdict())

# 활용
temp = [12, 22, 45, 11]
p4 = Point4._make(temp)
print(p4)
print(p1._fields, p2._fields, p3._fields, p4._fields)
```

<br>

### 리스트 구조 vs. 딕션어리 구조

리스트는 index로 접근, 실수 가능성 높고, 중간에 element가 빠져야 할수도있는데, 삭제하기 어려워서 관리하기 불편함. 데이터량이 많아질수록, index번호를 알아야하는것이 문제가 됨.

단, 리스트도 목적에 적합하다면 함수로 warping하여 활용할 수 있음.

딕션어리는 코드 반복 지속, 중첩 문제 (동일 key가 반복될 수 없음), key로 조회했을때에 예외처리, 등 단점들이 존재함. 그러나 dictionary, nested dictionary, 등 dictionary기반의 형태가 자주 사용 됨. 특정 item을 삭제해야하는 경우, pop 또는 del 함수를 사용함. (e.g., pop(key, 'default')) 

<br>

<br>

# References

1. 우리를 위한 프로그래밍: 파이썬 중급
1. Python Like You Mean It : https://www.pythonlikeyoumeanit.com/Module4_OOP/Special_Methods.html
