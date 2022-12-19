---
layout: post                          # (require) default post layout
title: "Multithreading"   # (require) a string title
date: 2022-11-30       # (require) a post date
categories: [PythonProgramming]          # (custom) some categories, but make sure these categories already exists inside path of `category/`
tags: [PythonProgramming]                      # (custom) tags only for meta `property="article:tag"`

---

# Multi-threading

## Difference between process and thread

### Process

운영체제에서 할당받는 자원의 단위 (실행중인 프로그램)

CPU동작 시간, 주소 공간 독립적(메모리 독립적)

Code, Data, Stack, Heap -> 각각이 독립적

최소 1개의 main thread 보유

파이프, 파일, 소켓 등을 사용해서 process간 통신 (cost 높음) -> Context Switching cost

### Thread

Process 내에 실행 흐름 단위

Process 자원 사용 - Stack만 별도로 만들어짐. 나머지는 공유(Code, Data, Heap은 공유)

메모리 공유 (변수 공유)

한 thread의 결과가 다른 thread에 영향을 끼침

동기화 문제는 정말 주의해야 함. (디버깅이 어려움.)

### Multi-thread

한 개의 단일 어플리케이션 (e.g., 응용 프로그램) -> 여러 thread로 구성 후 통합적인 작업 처리

시스템 자원 소모가 감소 됨 -> 효율성, 처리량이 증가 됨. (cost 감소)

통신 부담이 감소되나, 디버깅은 어려움, 단일 프로세스에는 효과 미약, 자원 공유 문제 (교착 상태 - deadlock) 프로세스에 영향을 줌.

### Multi-process

한 개의 단일 어플리케이션 (e.g., 응용 프로그램) -> 여러 프로세스 구성 후 작업 처리

한 개의 프로세스 문제 발생은 확산이 없음. (프로세스 kill)

cache change, cost 비용이 내우 높음 (overhead), 복잡한 통신 방식 사용

<br>

## GIL (Global Interpreter Lock)

keywords : CPython, 메모리관리, GIL 사용 이유

내가 작성한 code를  행하기위해 bytecode Python으로 변환하는데, CPython으로 실행 시, 여러 threads가 아닌, 하나의 thread만 사용하도록 제한을 둠. 단일 thread만이 Python object 에 접근하도록 제한하는 mutex를 GIL이라고 부른다.

Cpython 메모리 관리가 취약함. (즉, not thread-safe) GIL을 통해서 race condition을 방지하고 thread safety를 확보할 수 있다.

단일 thread로 충분히 빠름.

Process 사용 가능 (Numpy, Scipy)등 GIL 외부 영역에서 효율적인 코딩 

병렬 처리는 Multiprocessing, Asyncio, 등 선택지가 다양함.

Thread 동시성 완벽 처리를 위해, Jython, IronPython, Stackless Python등이 존재

<br>

## Main & Sub-thread

```python
# main (parent)thread vs. sub (child) thread

import logging
import threading
import time

# thread 실행 함수
def thread_func(name):
    logging.info("Sub-thread: %s: starting", name)
    time.sleep(3)
    logging.info("Sub-thread: %s: finishing", name)

# 메인 영역
# main thread의 시작점을 의미함.
if __name__ == "__main__":
    # Logging format 설정
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    logging.info("Main-thread: before creating thread")

    # 함수 인자 확인
    x = threading.Thread(target=thread_func, args=('First',))

    logging.info("Main-thread: before running thread")

    # sub-thread 시작
    x.start()

    # 주석 전후 결과
    # join()실행 시, sub-thread가 다 끝나야 나머지 main-thread가 실행 됨.
    x.join()

    logging.info("Main-thread: wait for the thread to finish")

    logging.info("Main-thread: ALL DONE.")
```

<br>

## DaemonThread

background에서 실행되는 thread.

main thread 종료 시, 즉시 종료 됨.

Daemon thread는 자신을 생성한 main thread가 종료되면 즉시 종료된다.

일반 thread는 작업 종료시 까지 실행하는것과는 다르게, daemon thread는  주로 background 무한 대기 시, event 발생으로 인한 특정 요청 사항을 실행하는 부분을 담당함. -> JVM의 garbage collection, word processor의 자동 저장, 등과 비슷한 역할

<br>

## ThreadPoolExecutor

'ㅁㅁㅁ'PoolExecutor = ThreadPoolExecutor or ProcessPoolExecutor class를 불러와서 여러 thread를 생성할때에 편하게 구현할 수 있는 방법이다.

고성능을 구현할때에 concurrent.futures package를 사용하는것이 유용함.

```python
import logging
from concurrent.futures import ThreadPoolExecutor
import time

def task(name):
    logging.info('Sub-Thread %s: starting', name)
    result = 0
    for i in range(10001):
        result += i
    logging.info('Sub-Thread %s: finishing result: %d', name, result)
    return result

# 실행방법 -1
def main1():
    # Logging format 설정
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    logging.info('Main-Thread : before creating and running thread')

    # max_workers : 몇명이 일을 나누어서 할지.(운영체제의 상태에 따라 무조건 n빵이 아닐수있음.)  
    # 작업의 개수가 worker수를 넘어가면 직접 worker수를 설정하는 것이 유리 
    executor = ThreadPoolExecutor(max_workers=3)
    task1 = executor.submit(task, ('First',))
    task2 = executor.submit(task, ('Second',))

    # thread의 결과 값 반환받아 출력
    print(task1.result())
    print(task2.result())

# 실행방법 -2
def main2():
    # Logging format 설정
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    logging.info('Main-Thread : before creating and running thread')

    # map 사용
    with ThreadPoolExecutor(max_workers=3) as executor:
        tasks = executor.map(task, ['First', 'Second', 'Third', 'Fourth'])
        # 결과 확인
        print(list(tasks))


if __name__ == "__main__":
    main2()
```

<br>

## Thread Synchronization

세마포어(semaphore): 프로세스간 공유된 자원에 접근 시, 문제 발생 가능성 있음. -> 한개의 프로세스만 접근 처리 고안 (경쟁 상태 예방)

뮤텍스(mutex): 공유된 자원의 데이터를 여러 스레드가 접근한는 것을 막는 것 -> 경쟁상태 예방

Lock: 상호 배제를 위한 잠금(lock)처리 -> 데이터 경쟁

Deadlock: 프로세스가 자원을 획득하지 못해 다음 처리를 못하는 무한 대기 상태 (교착 상태, 경쟁 상태이기도 함.) -> e.g., A는 printer 사용중인제 HDD로 넘어가야함, B는 HDD 사용중인데 printer로 넘어가야하는 상태. A는 HDD를 못써서, B는 printer를 못써서 둘다 무한 대기 중.

Thread Synchronization: 스레드 동기화. 동기화를 통해서 스레드가 안정적으로 동작하도록 처리 -> 동기화 메소드, 동기화 블록

semaphore vs.mutex 차이: 

- mutex: 공유된 자원을 여러 스레드가 열쇠(lock)을 가지고 순서대로 사용 (e.g., 1개의 화장실 1개의 열쇠, 여러 사람들)

- semaphore: 공유된 여러 자원을 여서 스레드가 각 자원의 열쇠(lock)을 가지고 함께 사용 (e.g., 여러개의 화장실, 각 화장실의 열쇠, 여러 사람들)

- semaphore는 mutex가 될 수 있지만, but vice versa is not valid

- semaphore와 mutex 개체는 모두 병렬 프로그래밍 환경에서 상호배제를 위해 사용

- mutex 개체를 단일 스레드가 resource 또는 중요 섹션을 소비 허용

- semaphore는 resource에 대한 제한된 수의 동시 access를 허용

```python
import logging
from concurrent.futures import ThreadPoolExecutor
import time

class FakeDataStore:
    # 공유변수 (value)
    def __init__(self):
        # 이 value는 Data, heap 영역에서 공유되는 것임.
        # thread의 stack은 별도 생성되고 
        # 왜 별도 stack? : 각 thread로 함수를 수행할때의 값(시작, 끝, 전달, 반환 값)들이 stack에 보관되어야함.
        # code는 여기 작성된 내용이 공유되는 것임.
        self.value = 0 
    
    # 변수 업데이트 함수
    def update(self, n):
        logging.info('Thread %s: starting update', n)

        # mutex & lock 동기화 (thread synchronization 필요)
        local_copy = self.value 
        local_copy += 1
        time.sleep(0.1)
        self.value = local_copy

        logging.info('Thread %s: finishing update', n)
    

if __name__ == "__main__":
    # Logging format 설정
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    # class instance화
    store = FakeDataStore()
    logging.info('Testing update. Starting value is %d', store.value)

    # with context
    with ThreadPoolExecutor(max_workers=2) as executor:
        for n in ['First', 'Second', 'Third']:
            executor.submit(store.update, n)

    logging.info('Testing update. Finishing value is %d', store.value)
```

output:

```
19:51:50: Testing update. Starting value is 0
19:51:50: Thread First: starting update
19:51:50: Thread Second: starting update
19:51:50: Thread First: finishing update
19:51:50: Thread Second: finishing update
19:51:50: Thread Third: starting update
19:51:50: Thread Third: finishing update
19:51:50: Testing update. Finishing value is 2
```

위와 같은 경우에는 예상과는 다르게 마지막 value로 3이 나오지 않았다.  value에 local copy가 담기기전에 실행되어버릴 수 있어서 value가 1 또는 2 무엇일지 정확하지 않기 때문이다. 

Thread synchronization을 다음과 같이 구현해서 공유 자원이 제어될 수 있다.

```python
class FakeDataStore:
    # 공유변수 (value)
    def __init__(self):
        self.value = 0 
        self._lock = threading.Lock()
    
    # 변수 업데이트 함수-1
    def update1(self, n):
        logging.info('Thread %s: starting update', n)

        # mutex & lock 동기화 (thread synchronization)
        # lock 획득 방법-1
        self._lock.acquire()
        logging.info('Thread %s has the lock', n)

        local_copy = self.value 
        local_copy += 1
        time.sleep(0.1)
        self.value = local_copy

        logging.info('Thread %s release the lock', n)

        # Lock 반환
        self._lock.release()

        logging.info('Thread %s: finishing update', n)

    # 변수 업데이트 함수-2
    def update2(self, n):
        logging.info('Thread %s: starting update', n)

        # mutex & lock 동기화 (thread synchronization)
        # lock 획득 방법-2
        with self._lock:
            logging.info('Thread %s has the lock', n)
            local_copy = self.value 
            local_copy += 1
            time.sleep(0.1)
            self.value = local_copy
            logging.info('Thread %s release the lock', n) 

        logging.info('Thread %s: finishing update', n)
    

if __name__ == "__main__":
    # Logging format 설정
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    # class instance화
    store = FakeDataStore()
    logging.info('Testing update. Starting value is %d', store.value)

    # with context
    with ThreadPoolExecutor(max_workers=2) as executor:
        for n in ['First', 'Second', 'Third']:
            executor.submit(store.update2, n)

    logging.info('Testing update. Finishing value is %d', store.value)
```

``` 
19:56:32: Testing update. Starting value is 0
19:56:32: Thread First: starting update
19:56:32: Thread First has the lock
19:56:32: Thread Second: starting update
19:56:32: Thread First release the lock
19:56:32: Thread First: finishing update
19:56:32: Thread Second has the lock
19:56:32: Thread Third: starting update
19:56:32: Thread Second release the lock
19:56:32: Thread Second: finishing update
19:56:32: Thread Third has the lock
19:56:32: Thread Third release the lock
19:56:32: Thread Third: finishing update
19:56:32: Testing update. Finishing value is 3
```

<br>

## Prod and Cons using Queue

생산자 & 소비자 패턴 (Producer-Consumer Pattern)은 Multiprocessing design pattern의 정석임으로 참고하면 좋다.

Server측 프로그래밍의 핵심 (주로 허리 역할. 매우 중요함.)

```python
import concurrent.futures
import logging
import queue
import random
import threading
import time

# 생산자
def producer(queue, event):
    """network 대기 상태라고 가정(서버)"""
    while not event.is_set():
        message = random.randint(1,11)
        logging.info('Producer made message: %s', message)
        queue.put(message)

    logging.info('Producer received event. Exiting.')

# 소비자
def consumer(queue, event):
    """응답 받고 소비하는 것으로 가정 or DB 저장"""
    while not event.is_set() or not queue.empty():
        message = queue.get()
        logging.info('Consumer storing message: %s (size=%d)', message, queue.qsize())

    logging.info('Consumer received event. Exiting.')


if __name__ == "__main__":
    # Logging format 설정
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    # 사이즈 중요 : 환경에 적합한 queue 사이즈가 설정되어야함. 병목현상 예방하고 원활한 흐름을 구현하기 위함.
    pipeline = queue.Queue(maxsize=10)

    # Event flag 초기값 0
    event = threading.Event()

    # with Context
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(producer, pipeline, event)
        executor.submit(consumer, pipeline, event)
        # 실행시간 조정
        time.sleep(0.1)
        # 일반적으로, 다음과 같이 while문으로 특정 상태에 도달할 때까지 지속 실행하도록 구현한다.
        # while True:
        #     pass
        #     break

        logging.info('Main: about to set event')

        # 프로그램 종료
        event.set()
```

output:

```
20:45:07: Producer made message: 10
20:45:07: Producer made message: 7
20:45:07: Consumer storing message: 10 (size=0)
20:45:07: Producer made message: 6
20:45:07: Consumer storing message: 7 (size=0) 
20:45:07: Consumer storing message: 6 (size=0) 
20:45:07: Producer made message: 6
20:45:07: Producer made message: 7
20:45:07: Consumer storing message: 6 (size=0) 
20:45:07: Producer made message: 3
20:45:07: Consumer storing message: 7 (size=0) 
20:45:07: Producer made message: 4
20:45:07: Consumer storing message: 3 (size=0)
20:45:07: Producer made message: 10
20:45:07: Consumer storing message: 4 (size=0)
20:45:07: Producer made message: 6
20:45:07: Consumer storing message: 10 (size=0)
20:45:07: Producer made message: 3
20:45:07: Consumer storing message: 6 (size=0)
20:45:07: Producer made message: 8
20:45:07: Consumer storing message: 3 (size=0)
20:45:07: Producer made message: 8
20:45:07: Consumer storing message: 8 (size=0)
20:45:07: Producer made message: 2
20:45:07: Consumer storing message: 8 (size=0)
20:45:07: Producer made message: 7
20:45:07: Consumer storing message: 2 (size=0)
20:45:07: Consumer storing message: 7 (size=0)
20:45:07: Producer made message: 10
20:45:07: Producer made message: 5
20:45:07: Consumer storing message: 10 (size=0)
20:45:07: Producer made message: 3
20:45:07: Producer made message: 7
20:45:07: Producer made message: 2
20:45:07: Main: about to set event
20:45:07: Consumer storing message: 5 (size=0)
20:45:07: Producer made message: 2
20:45:07: Consumer storing message: 3 (size=2)
20:45:07: Producer received event. Exiting.
20:45:07: Consumer storing message: 7 (size=2)
20:45:07: Consumer storing message: 2 (size=1)
20:45:07: Consumer storing message: 2 (size=0)
20:45:07: Consumer received event. Exiting.
```

<br>

<br>

# References

1. 프로그래밍-파이썬-완성-인프런-오리지널
