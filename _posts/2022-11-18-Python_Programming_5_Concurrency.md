---
layout: post                          # (require) default post layout
title: "Concurrency"   # (require) a string title
date: 2022-11-18       # (require) a post date
categories: [PythonProgramming]          # (custom) some categories, but make sure these categories already exists inside path of `category/`
tags: [PythonProgramming]                      # (custom) tags only for meta `property="article:tag"`

---

<br>

# Concurrency

## Iterator, Generator

Python generators : simple way of creating iterators. A generator, in simple words, is a function that returns an object(iterator) which we can iterate over (one value at a time)

iterator : 반복 가능한 객체 (iterable objects) can be used in for loops.

examples of iterator: collections, text files, list, Dict, Set, Tuple, unpacking, *args...

<br>

Iterator

반복가능한 이유 : iter(x) 함수 호출 가능.  

내부적으로 다음과 같이 exception을 고려하여 수행 됨.

```python
t = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
print(dir(t)) # dir()로 attributes 모두 출력해서 iter()이 있는지 확인

w = iter(t)

while True:
    try:
        print(next(w))
    except StopIteration:
        break
```



generator 패턴

1. 지능형 리스트, 딕션어리, 집합 -> 데이터 양 증가 후, 메모리 사용량 증강 -> generator 사용 권장

2. 단위 실행 가능한 coroutine  구현과 연동

3. 작은 메모리 조각 사용 

Generator는 built-in keyword (yield)를 사용해서 다음에 return할 요소의 위치를 기억함. index를 굳이 사용하지 않아도 됨. 나중에 yield는 coroutine에 사용 됨.

```python
class WordSplitterGenerator: 
    def __init__(self, text):
        self._text = text.split(' ')

    # iter함수를 구현해주면, 내부적으로 이 yield라는 keyword를 통해 
    # 다음에 return될 원소의 위치값을 기억한다.
    def __iter__(self):
        for word in self._text: 
            yield word
        return

    def __repr__(self):
        return 'WordSplitterGenerator(%s)' % (self._text)
```

Generator의 주요 함수를 사용해서 데이터를 센스있게 잘 다룰 수 있음.

ex) count, takewhile, filterfalse, accmulate, chain, product, groupby, etc,,,

```python
import itertools

gen1 = itertools.count(1,2.5)
print(next(gen1))
print(next(gen1))
print(next(gen1))
print(next(gen1))
# ... 무한

# 조건
gen2 = itertools.takewhile(lambda n : n < 1000, itertools.count(1, 2.5))
for v in gen2:
    print(v)

# filterfalse : filter 조건의 반대에 해당하는 것들만 남겨!
gen3 = itertools.filterfalse(lambda n : n < 3, [1,2,3,4,5])
for v in gen3:
    print(v)

# 누적 합계
gen4 = itertools.accumulate([x for x in range(1,101)])
for v in gen4:
    print(v)

# 연결1
gen5 = itertools.chain('ABCDE', range(1,11,2))
print(list(gen5))

# 연결2
gen6 = itertools.chain(enumerate('ABCDE'))
print(list(gen6))

# 개별
gen7 = itertools.product('ABCDE')
print(list(gen7))

# 연산(경우의 수)
gen8 = itertools.product('ABCDE', repeat=3)
print(list(gen8))

# 그룹화(반복되는 원소의 그룹화)
gen9 = itertools.groupby('AAABBCCCDDEEE')
# print(list(gen9))
for chr, group in gen9:
    print(chr, ":", list(group))
```

<br>

<br>

## Concurrency vs. parallelism

**병행성(concurrency)** - 하나의 computer가 여러 일을 동시에 수행하는 것 (하나의 cpu, 하나의 thread, 등등). 내가 멈춘 위치를 잘 알고 그대로 pickup할 수 있어야 (closure, generator의 yield, 등 활용 가능)

예시) coroutine의 활용 - thread는 하나 이지만, 마치 여러 작업을 동시에 하는 듯

장점: 단일 프로그램안에서 여러 일을 해결

Concurrency가 더 적합한 경우:

실행하려는 task가 IO-bound operations (e.g., querying a web service or reading large files) 이라면, concurrency option이 더 적합하다. If we run two CPU bound operations as two threads then they will run sequentially and we will not yield any benefits in Python. IO-bound operations에는 external resources (e.g., hardware or network)와 communicate해야하는 과정이 요구되고, I/O bound operation이 I/O waiting 상태로 external resource로 부터 result를 반환 받기까지 기다려야하기 때문임. 또한 context switching이나 lock acquisition 때문에 여러 thread로 실행하게되면 오히려 더 긴 소요시간이 발생할 수 있다. 

<br>

**병렬성(parallelism)** - 여러 computer가 여러 작업을 동시에 수행. worker가 여러 작업을 동시에 수행

예시) Data scientist - 병렬로 동시에 여러 site에서 crawling작업 수행. 

동시에 다 작업해서 취합은 한곳에서

장점: 속도

Parallelism이 더 적합한 경우:

"At a high level, if your Python application is performing CPU bound operations such as number crunching or text manipulation then go for **parallelism**. Concurrency will not yield many benefits in those scenarios."

<br>

<br>

## Coroutine

**thread**: OS 에서 직접 관리함, CPU core에서 실시간, 시분할(시간을 서로 나눈) 비동기 작업을 "멀티쓰레드"라고 함. single thread 또는 multi thread로 사용 가능. 

thread ->복잡 ->공유되는 자원 ->교착 상태 (deadlock or race condition) 발생 가능성, context switching 비용 큼, 자원 소비 가능성 증가

switching 비용이 큰 경우가 종종 있기 때문에, multi thread보다 오히려 single thread가 더 효율이 높은 수 있음.

<br>

**coroutine**: 단일 (single) thread를 의미함. 메인과 서브가 서로 상호 작용하면서 stack을 기반으로 동작하는 비동기 작업. 단일 thread에서도 순차적으로 상호작용을 하면서 여러 작업이 진행될 수 있음. coroutine은 Python외에 Golang과 같은 다른 언어에서도 구현 가능함.

즉, main function 안에서 여러 sub routine을 실행 + 중지하는 과정을 구현해서 하나의 thread안에서 여러 작업이 동기화되어 진행 될 수 있도록 한다. 여기서 yield와 send를 통해 main과 sub가 서로 데이터를 주고 받을 수 있다.

<br>

**yield**: yield라는 keyword를 통해서 메인 <-> 서브 루틴이 서로 상호작용함. coroutine을 제어 할때에 yield keyword를 사용 함. yield와 send를 통해서 coroutine을 제어하고, 상태를 저장하고, 양방향으로 데이터 전송을 함.

subroutine: "흐름 제어" - main routine에서 호출하면 -> sub routine에서 수행

coroutine: "동시성 프로그래밍" - 루틴을 실행 중, 중지하고(상태를 기억하고), 다시 재 실행할 수 있음. 

coroutine의 장점: thread에 비해 overhead 감소 (단일 thread이기 때문에, 운영 체제에게 자원(thread)을 덜 요구하게 됨.)

NOTE: Python 3.5 이상에서는 def -> async, yield -> await 바꾸어서 사용할 수 있음.

```python
# Coroutine Ex1
# def를 통해 generator, coroutine(yield를 사용하는 generator에서 파생된)을 생성할 수 있음.
def coroutine1():
    print('>>> coroutine started.')
    i = yield
    print('>>> coroutine received : {}'.format(i))


# Generator 선언
cr1 = coroutine1() # Main routine이며, "일"하나에 해되며 coroutine1() 속에 define된 task들이 subroutine에 해당 됨.

# generator 객체라고 출력됨.
print(cr1, type(cr1)) # output = "<generator object coroutine1 at 0x0000020D7E9BB740> <class 'generator'>""


# 여기에서 subroutine은 main에서 반환값을 주는 것 밖에는 없었음. 수동적임.
# 첫번째 next(cr1)에서는 coroutine()이라는 generator내의 첫번째 yield 지점까지 subroutine을 수행 하고 정지. 여기 상태를 기억.
next(cr1)
# 두번째 next(cr1)에서는 기본 전달 값=None. 그리고 Stopiteration 예외가 발생함.
# next(cr1)

# 값 전송
# send(): main routine과 subroutine이 서로 data를 주고받을 수 있게 함. next()의 기능도 포함하고 있음.
# cr1.send(100) # coroutine이 100을 받는다. output = "">>> coroutine received : 100"

# 잘못된 사용
# 다음과 같이 generator 선언 후, 바로 send()의 parameter로 값을 전달하는 경우, 예외 발생
cr2 = coroutine1()
# cr2.send(100) # output = "TypeError: can't send non-None value to a just-started generator"

# 맞는 사용법
def coroutine():
    print('>>> coroutine started.')
    i = yield
    print('>>> coroutine received : {}'.format(i))
cr3 = coroutine()
next(cr3)
# cr3.send(50)
```

getgeneratorstate를 통해서 coroutine의 상태를 확인할 수 있다. 

generator states:

\# GEN_CREATED : 처음 대기 상태

\# GEN_RUNNING : 실행 상태

\# GEN_SUSPENDED : Yield 대기 상태 (send로 데이터를 보내거나 받을 수 있는 상태)

\# GEN_CLOSED : 실행 완료 상태

```python
# sub routine과 main routine사이 값이 오고 가는 과정 구현:

def coroutine2(x): # main routine에서 sub routine으로 x를 전달
    # x를 main routine으로부터 받아서 출력
    print('>>> coroutine started : {}'.format(x)) 
    # y는 main routine에서 send로 받을 예정. (x를 send를 통해 sub routine에서 main routine으로 전달 (양방향 데이터 주고받기: 동시성 가능))
    y = yield x 
    # y를 main routine으로부터 받아서 출력
    print('>>> coroutine received : {}'.format(y))
    # x+y를 subroutine에서 main routine으로 전달
    # main routine이 sub routine에게 넘긴것은 z
    # sub routine이 나에게 준것은 x+y
    z = yield x + y 

    # z를 main routine으로부터 받아서 출력
    print('>>> coroutine received : {}'.format(z))

    # 오른쪽 : sub routine이 나에게 주는거
    # 왼쪽: 입력을 받는거 (main routine에게 전달되는 값)


# cr3 = coroutine2(10)

# print(next(cr3)) # yield를 통해 x가 주어진 상태, y 값을 받기위해 대기 중
# cr3.send(100) # y는 100을 받았고 (x는 10이 주어진 상태) yield를 통해 x+y가 주어진 상태, z 값을 받기위해 대기 중
# cr3.send(100) # StopIteration이 걸림.

# 상태값 확인
from inspect import getgeneratorstate

cr3 = coroutine2(10)
print("상태 값 확인")
print(getgeneratorstate(cr3))
print(next(cr3))
print(getgeneratorstate(cr3))
# cr3.send(100)
# print(getgeneratorstate(cr3)) # send를 보내야하니까 여기서는 GEN_SUSPENDED

# x+y를 확인하기위해서는 print()로 확인가능
print("확인")
print(cr3.send(100)) # output = send의 결과로 ">>> coroutine received : 100" 그리고 subroutine이 나에게 주는 값 "110"은 print의 결과로 확인
```

<br>

Iterable한 object내 element를 순차적으로 꺼내기 위해서 다음과 같이 coroutine을 활용해볼 수 있다.

```python
# Coroutine Ex3
def generator1():
    for x in 'AB': # iterable한 string에서 순차적으로 끝날때까지
        yield x
    for y in range(1,4): # 사실, range가 list를 반환해주는 것도 generator를 활용하는 것임.
        yield y

t1 = generator1()
print(next(t1))
print(next(t1))
print(next(t1))
print(next(t1))
print(next(t1))
# print(next(t1)) # StopIteration 예외 발생

t2 = generator1()
print(list(t2)) # 알아서 next가 호출되어서 list가 생성됨.


# yield from을 활용해보기
def generator2():
    yield from 'AB' 
    yield from range(1,4) 

t3 = generator2()
print(next(t3))
print(next(t3))
print(next(t3))
print(next(t3))
print(next(t3))
# print(next(t3)) # StopIteration 예외 발생
```

<br>

<br>

# References

1. 우리를 위한 프로그래밍: 파이썬 중급 인프런 오리지널
1. "Advanced Python concurrency and parallelism": https://medium.com/fintechexplained/advanced-python-concurrency-and-parallelism-82e378f26ced
1. "Python concurrency - making sense of asyncio": https://learningdaily.dev/python-concurrency-making-sense-of-asyncio-ebf18d722341
