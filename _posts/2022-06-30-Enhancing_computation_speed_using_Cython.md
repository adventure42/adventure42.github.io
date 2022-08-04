---
layout: post                          # (require) default post layout
title: "Enhancing computation speed using Cython"   # (require) a string title
date: 2022-06-30       # (require) a post date
categories: [RCAClassification]          # (custom) some categories, but make sure these categories already exists inside path of `category/`
tags: [RCAClassification]                      # (custom) tags only for meta `property="article:tag"`

---

<br>

# Cython

DBA와 같이 반복적으로 많은 량의 computation을 수행해야하는 경우, Python과 C를 함께 사용해서 연산의 속도를 향상 시킬 수 있다. (Pycharm IDE 사용시, professional version에서만 Cython이 인식되는 점 주의! )

Cython은 다음과 같이 쉽게 현재 사용하고있는 environment에 install될 수 있다.

```python
pip install Cython
```

Python code로 구현 할 수 있는 연산을 Cython을 통해서 C로 구현하여 더 빠른 연산 속도를 확보할 수 있다. 구현 순서는 대략적으로 다음과 같다:

1. pyx file 생성

   Cython에서는 C와 동일하게 data type declaration을 사용한다. int, float, double,등과 같다. Cython의 syntax에 관한 더 상세한 내용은 [여기](https://nyu-cds.github.io/python-cython/01-syntax/)를 참고.

   ```python
   # variable declaration
   cdef int i = 10
   
   # function declaration
   cdef int square(int x):
       return x**2
   ```

   Python code에 import해서 사용 할 수 있도록 visible하게 만들기위해서는 cdef가 아닌 cpdef를 사용해야 한다고 한다. 

   

2. setup file 생성

   다음과 같은 내용을 담은 setup.py 파일을 생성해쟈지만 Cython이 C로 translate될 수 있다.

   ```python
   from distutils.core import setup
   from Cython.Build import cythonize
   setup(ext_modules = cythonize('my_pyx_file.pyx'))
   ```

   

3. compile

   다음과 같이 --inplace를 포함하여 compilation을 수행하면, 현재 working directory안에 shared object file이 생성된다.

   ```python
   python3 setup.py build_ext --inplace
   ```

   

4. run

   compilation이 완료된 후, pyx파일을 일반적인 Python module과 같이 Python code에 import해서 사용할 수 있다. 

   참고했던 [Medium 블로그 포스트](https://betterprogramming.pub/make-your-python-code-dramatically-faster-with-cython-2a307253234b) 에서는 factorial을 계산하는 간단한 코드를 구현해서 순수 Python과 C를 활용한 Python 실행 결과를 비교했다.  20의 factorial을 계산한 결과, Cython 활용 시, 88배의 더 빠른 연산 속도가 확인되었다.

​	

<br>

<br>

# References

1. Make Your Python Code Dramatically Faster With Cython by Halil Yildirim https://medium.com/better-programming/make-your-python-code-dramatically-faster-with-cython-2a307253234b

1. NYU intro to Cython: https://nyu-cds.github.io/python-cython/

1. Basic Cython Tutorial: https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html
