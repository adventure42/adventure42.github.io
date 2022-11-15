---
layout: post                          # (require) default post layout
title: "Functional Programming I"   # (require) a string title
date: 2022-10-25       # (require) a post date
categories: [PythonProgramming]          # (custom) some categories, but make sure these categories already exists inside path of `category/`
tags: [PythonProgramming]                      # (custom) tags only for meta `property="article:tag"`

---

<br>

# Functional Programming

## 일급 함수

(First Class) 일급 함수 / 일급 객체

Python외에 타 언어에서도 일급함수 컨셉이 있음.

함수형 프로그래밍 장점: 코드를 간결하게 작성, 개발시간 단축 가능

순수 함수 (pure function)을 지향하여 동시에 여러 thread에서 문제없이 동작하는 프로그램을 작성할 수 있음 <--새로운 기능 추가/ 기존 기능 수정 용이함.

<br>

**Python의 함수 특징(일급 함수의 특징):**

1. 런타임 초기화 (실행 시점에 초기화 된다)

2. 변수 할당 가능 (함수를 변수에 할당하기)

3. 함수 인수 전달 가능 (함수를 다른 함수의 인수로 전달하기)

4. 함수 결과 반환 가능 (return) (함수를 자체를 결과로 반환하기)

<br>

```python
def factorial(n):
    '''Factorial Function -> n: int'''
    if n == 1: # n < 2
        return 1
    return n * factorial(n-1)

class A:
    pass

print(factorial(5))
print(factorial.__doc__)
# factorial은 함수, A는 class임이 확인 됨.
print(type(factorial), type(A))
# factorial은 함수이지만, dir()결과를 보면 객체와 같이 취급된다는것을 볼 수 있음.
print(dir(factorial))
# class function의 dir - class type의 dir = 순수 함수에만 해당하는 속성 확인.
print(set(sorted(dir(factorial))) - set(sorted(dir(A))))

#증명-1
# 변수 할당 가능 (함수를 변수에 할당하기)
var_func = factorial
print(var_func)
print(var_func(5))
print(list(map(var_func, range(1,11))))

# 증명-2
# 함수 인수 전달 및 함수로 결과 반환
# a.k.a 고위 함수 (higher-order function)
print([var_func(i) for i in range(1,6) if i % 2])
# map, filter를 활용해서 확인하기.
print(list(map(var_func, filter(lambda x: x % 2, range(1,6)))))
# sidenote: i % 2 는 홀수만 (짝수일때 remainder=0, 즉 False이기때문)
# filter 함수의 인자로 var_func 함수가 전달 됨.
# 실행시, var_func가 filter함수의 인자로 전달되어 lambda로 filter된 x가 들어가서 실행 됨.
# 위 comprehensive 코드를 이해하기 어렵다면, list, map, filter 순으로 하나씩 지워가면서 뭐가 달라지는지 확인하며 이해해볼 수 있음.

# reduce를 활용해서 확인하기.
from functools import reduce
from operator import add
print(sum(range(1,11))) # <- 이게 더 빠름.
print(reduce(add, [1,2,3,4,5,6,7,8,9,10]))
print(reduce(add, range(1,11)))
```

note: 익명 함수 (lambda): 가급적 주석을 꼭 작성하라(다른사람 가독성위해) 가급적 익명보다는 일반(이름이 있는) 함수를 작성하라.

note: partial 사용법

```python
# partial 사용법: 인수 고정 -> 콜백 함수 사용
from operator import mul
from functools import partial

print(mul(10,10))
# 인수 고정
# 함수 mul를 partial 함수의 인자로 전달했고, mul을 변수 five에 할당한다. 
five = partial(mul, 5) # 5*? 상태
print(five(10)) # 5*10 이 호출됨.
print(five(100)) # 5*100 이 호출됨.

six = partial(five, 6)
print(six()) # six만 호출하여도 5*6=30이 나옴. 
# five에서 mul의 a가 fix되고, six에서 mul의 b가 fix되어서 a*b결과가 반환됨.
```

<br>

<br>

# References

1. 우리를 위한 프로그래밍: 파이썬 중급 인프런 오리지널
