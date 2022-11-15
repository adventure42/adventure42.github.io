---
layout: post                          # (require) default post layout
title: "Functional Programming II"   # (require) a string title
date: 2022-11-07       # (require) a post date
categories: [PythonProgramming]          # (custom) some categories, but make sure these categories already exists inside path of `category/`
tags: [PythonProgramming]                      # (custom) tags only for meta `property="article:tag"`

---

<br>

# Functional Programming

## Closure

closure remembers values in enclosing scopes even if they are NOT present in memory

ex) 함수가 실행이 끝나도, 그 함수 내 local 변수의 값이 소멸되지 않고 기억됨. 서버 programming에서 동시성 제어가 중요함. (concurrency)

메모리 공간에 여러 자원이 접근 -> 교착 상태 (deadlock) 또는 race 상태 발생 방지를 위해 concurrency(동시성) 제어가 필요함. e.g. TomCat과 같은 검증된 open source로 동시성 구현 가능

메모리를 공유하지 않고 메세지 전달로 처리하기위한 언어도 만들어짐-> Erlang

python에서 closure : 공유하되, 변경되지 않는 (immutable, readonly) ->함수형 programming

closure는 불변자료구조 및 원자성, 일관성 (atomic, STM)을 통해 멀티스레드에 강점. (멀티스레드 프로그래밍 (coroutine))

여러 스레드로 일을 나누어서 진행할때에, 각 스레드가 어디까지 일을 했는지 scope밖에서도 변경 없이 기억되어야함. 한마디로, closure은 상태를 기억한다. (불변 상태)

<br>

<br>

결과를 누적해 나아가는 방식의 구현:

```Python
# 결과 누적 (함수 사용)
print(sum(range(1,51)))
print(sum(range(51,101)))

# 결과 누적 (클래스 사용)
class Averager:
    def __init__(self):
        self._series = []

    def __call__(self, v):
        self._series.append(v)
        print('inner >> {} / {}'. format(self._series, len(self._series)))
        return sum(self._series) / len(self._series)

# 인스턴스 생성
averager_cls = Averager()
print(dir(averager_cls))

# 누적
print(averager_cls(10))
print(averager_cls(30))
print(averager_cls(50))
print(averager_cls(193))
```

외부에서 호출된 함수의 변수 값, 상태(reference) 복사(snapshot)후 저장 -> 후에 접근(access)가능하도록 함.

정리: 처음 호출은 outer function(외부)의 호출로 시작됨. inner function(내부)에 내가 define해둔 task들을 수행하고, outer function에 선언된 자유 변수에 inner function의 수행을 통해 얻은 상태를 기억하도록 함.

```Python
# Closure 사용
def closure_ex1():
    # free variable(자유변수)
    # 자유변수는 내가 사용하려는 함수 바깥에서 선언된다.
    # 클로져 영억
    series=[]
    def averager(v):
        series.append(v)
        print('inner >>> {} / {}'. format(series, len(series)))
        return sum(series) / len(series)
    return averager # 함수 자체를 반환

avg_closure1 = closure_ex1()
print("what's returned when called for first time: ", avg_closure1)
# 함수 자체가 반환되었음이 확인 됨.
# <function closure_ex1.<locals>.averager at 0x00000287FE21DA60>

print(avg_closure1(10))
print(avg_closure1(30))
print(avg_closure1(50))
```



```Python
# 잘못된 closure 사용
def closure_ex2():
    # Free variable
    cnt = 0
    total = 0
    def averager(v):
        cnt += 1
        total += v
        return total / cnt
    return averager

avg_closure2 = closure_ex2()
#print(avg_closure2(10)) # UnboundLocalError: local variable 'cnt' referenced before assignment

# 잘못된 closure 사용 -> 개선 됨.
def closure_ex3():
    # Free variable
    cnt = 0
    total = 0
    def averager(v):
        nonlocal cnt, total #이 method 밖의 변수가 인식되도록 nonlocal설정 필요
        cnt += 1
        total += v
        return total / cnt
    return averager

avg_closure3 = closure_ex3()
print(avg_closure3(15))
print(avg_closure3(25))
print(avg_closure3(35))
```

<br>

<br>

## Decorator

decorator를 작성하기위해 이해해야 하는 것들:

1. closure

2. 함수를 일급 인사로 활용하는 법 (first class argument)

3. 가변 인자 (위치 가변인자: *args, 키워드 가변 인자: **kwargs)

4. 인자 풀기 (argument unpacking)

5. Python source code를 불러오는 자세한 과정

decorator의 장점:

1. 중복 제거, 코드 간결, 공통 함수 작성

2. 로깅, 프레임 워크, 유효성 체크(e.g., 비번입력시 조건에 충족하는지 확인) -> 공통 함수

3. 조합해서 사용 용이

decorator의 단점:

1. 가독성 떨어짐(?)

2. 특정 기능에 한정된 함수는 -> 단일 함수로 작성하는 것이 유리

3. 디버깅 불편



이미 사용해본 decorator의 예시) @classmethod, @staticmethod

```Python
# decorator 실습
import time

def perf_clock(func):
    def perf_clocked(*args):
        # 함수 시작 시간
        st = time.perf_counter()
        # 함수 실행
        result = func(*args)
        # 함수 종료 시간
        et = time.perf_counter()
        # 함수 실행 시간
        elapsed = et - st
        # 실행 함수명
        name  = func.__name__
        # 함수 매개변수
        arg_str = ','.join(repr(arg) for arg in args)
        # 결과 출력
        print('[%0.5fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
        return result
    return perf_clocked 

def time_func(seconds):
    time.sleep(seconds)
    
def sum_func(*numbers):
    return sum(numbers)

# decorator 미사용
non_deco1 = perf_clock(time_func)
non_deco2 = perf_clock(sum_func)
# 자유변수 확인
print(non_deco1, non_deco1.__code__.co_freevars)
print(non_deco2, non_deco2.__code__.co_freevars)
# decorator 없이 closure만 활용하여 결과 확인
print('-'*40, 'called non-decorator -> time_func')
print()
non_deco1(1.5)
print('-'*40, 'called non-decorator -> sum_func')
print()
non_deco2(100, 200, 300, 400, 500)

# decorator 사용
@perf_clock
def time_func(seconds):
    time.sleep(seconds)
@perf_clock
def sum_func(*numbers):
    return sum(numbers)
# 각 함수에 decorator로 달아놓았기때문에, 함수 자체를 호출하면 됨.
print('-'*40, 'called with-decorator -> time_func')
print()
time_func(1.5)
print('-'*40, 'called with-decorator -> sum_func')
print()
sum_func(100, 200, 300, 400, 500)
```

<br>

<br>

# References

1. 우리를 위한 프로그래밍: 파이썬 중급 인프런 오리지널
