# Vector 의미 & 연산

### 1. Vector 의미

**벡터는 여러개의 숫자들로 이루어진 배열이다. **
숫자의 개수만큼 N차원 벡터라고 말할 수 있으며, 배열에서 얘기하는 차원의 개념과는 다르다.                                                   배열의 차원은 행렬에서 다룰 것이니 지금 차원의 개념이 확실하지 않아도 된다.

```python
[2,3]  ------------>  2차원 벡터
[1,2,3] ----------->  3차원 벡터
[1,2,3,4] --------->  4차원 벡터
[1,2,3,4,5] ------->  5차원 벡터
[1,2,3,4,5,6,7] --->  6차원 벡터
```

벡터를 코드로 표현하면 아래와 같다.

```python
#1,2,3,4로 이루어진 벡터를 만들어서 변수 x에 저장하기 using NumPy
x = np.array([1, 2, 3, 4])

=========================================== 참고 ============================================

# 벡터는 배열인데 단순히 list로 표현하지 않고 numpy를 이용하는 이유 :
list를 통해서는 벡터 연산을 할 수 없기 때문이다. 그래서 필요시에 데이터를 더하고, 곱하고, 빼는 등 다양한 연산을 하기 위해서 NumPy가 필요하다.
```

```python
#x를 출력해보기
print(x)
```

array([1, 2, 3, 4])로 결과가 나온다. 																																		그럼 파이썬에서 벡터는 어떤 shape으로 인식할까? shape함수를 사용하여 나타내면 아래와 같다.

```python
#x에 저장된 벡터의 shape을 알아보기
x.shape
```

(4,  ) 이라는 결과가 나온다. (4,  )는 1차원이며 총 4라는 크기를 가지고 있음을 알 수 있다. 이처럼 shape함수를 통해 몇 개의 데이터가 있는지, 몇 차원으로 존재하는지 등을 확인할 수 있다.

> #### ※ 참고
>
> **벡터는 배열인데 단순히 list로 표현하지 않고 numpy를 이용하는 이유**
>
> --> list를 통해서는 벡터 연산을 할 수 없기 때문이다. 그래서 필요시에 데이터를 더하고, 곱하고, 빼는 등 다양한 연산을 하기 위해서 NumPy가 필요하다.

------

### 2. Vector 연산(더하기, 빼기, SUM, 곱하기)

#### 2-1 더하기 & 빼기

만약에 두 벡터의 덧셈과 뺄셈을 할 때 반드시 두 벡터의 형태가 같아야 한다. 그 이유는 덧셈과 뺄셈을 할 때 각각의 벡터 상에서 같은 위치에 있는 성분들끼리 연산이 되기 때문이다. 

<img src="/Users/youngjunyoon/Desktop/Github/img/스크린샷 2020-02-01 오후 11.31.12.png" style="zoom:200%;" />

먼저 벡터의 덧셈을 함수로 만들어보자. (Data Science from Scratch 53쪽). 																		      벡터의 덧셈은 zip을 사용해서 두 벡터를 묶은 뒤, 두 배열의 각 성분끼리 더하는 리스트로 값을 받으면 된다.

```python
def vector_add(vector1, vector2): 
    return [vector1_i + vector2_i for vector1_i, vector2_i in zip(vector1,vector2)]
```

이제 벡터를 더하는 함수를 만들었으니 한 번 테스트를 해보자!

```python
#벡터 2개를 먼저 만들어주었다
vector1 = np.array([1,2,3,4,5,6])
vector2 = np.array([1,2,3,4,5,6])
```

```python
#만든 함수에 2개의 벡터를 인자로 넣어주었다
vector_add(vector1,vector2) 
```

**출력**

```
[2, 4, 6, 8, 10, 12]
```



이제 벡터의 빼기 함수를 만들어보자(Data Science from Scratch 53쪽)																				덧셈의 함수에서 vector1_i + vector2_i 을 vector1_i - vector2_i 로 (-)빼기 기호로만 바꿔주면 된다.

```python
def vector_add(vector1, vector2): 
    return [vector1_i - vector2_i for vector1_i, vector2_i in zip(vector1,vector2)]
```



#### 2-2 SUM

모든 벡터의 각 성분을 더할 때는 어떻게 할까? 

<img src="/Users/youngjunyoon/Desktop/Github/img/스크린샷 2020-02-02 오전 2.41.25.png" style="zoom:200%;" />



모든 벡터의 각 성분의 합을 구하려면 두 벡터의 합에 합을 또 그 합에 합을 더하기로 누적시켜서 합계를 구해야 한다.

<img src="/Users/youngjunyoon/Desktop/Github/img/스크린샷 2020-02-02 오전 2.41.20.png" style="zoom:200%;" />



이것을 코드로 구현하면 아래와 같다.(Data Science from Scratch 53쪽)

```	python
def vectors_sum(vectors):
    result = vectors[0] 
  #1. 첫번째 벡터부터 시작하여 N개의 벡터를 더해야 하기 때문에 미리 최종 결과값에 첫번째 벡터를 입력하였다.
    for vector in vectors[1:]: 
        result = vector_add(result, vector)
  #2. 첫번째 벡터에서 두번째벡터를 더해야하니 두번째 벡터부터(vector[1:]) for문을 시작해야한다. 
    return result
  #3. 첫번째 벡터부터 N번째 벡터까지 전부 더한 값을 받는다.   
```

주석을 제거하고 보면 이렇다.

```python
#주석제거ver.
def vectors_sum(vectors):
    result = vectors[0] 
    for vector in vectors[1:]: 
        result = vector_add(result, vector)
    return result
```



이제 테스트를 한 번 해보자.

먼저 여러개의 벡터가 묶여 있는 벡터(vectors)가 필요하다. 

```python
#여래가의 벡터들이 있다.
vector1 = np.array([1,2,3,4,5,6])
vector2 = np.array([1,2,3,4,5,6])
vector3 = np.array([1,2,3,4,5,6])
vector4 = np.array([1,2,3,4,5,6])
```

벡터들을 하나로 묶어보자.

```python
vectors = np.array([vector1,vector2,vector3,vector4])

'''
이렇게 나온다.
array([[1, 2, 3, 4, 5, 6],
       [1, 2, 3, 4, 5, 6],
       [1, 2, 3, 4, 5, 6],
       [1, 2, 3, 4, 5, 6]])
'''
```

함수에 적용시켜보자.

예상 결과값은 [4, 8, 12, 16, 20, 24] 일 것이다.

```python
vectors_sum(vectors)

'''
[4, 8, 12, 16, 20, 24]
'''
```



### 2-3 곱하기

벡터의 곱하기는 스칼라값을 벡터에 각 요소마다 곱해주는 것이다. 

![](/Users/youngjunyoon/Desktop/Github/img/스크린샷 2020-02-02 오후 1.35.36.png)

> ※ 스칼라 : 숫자 하나 

```python
#n은 스칼라 값
def vector_multiply(vector, n):
		return [v_i * n for v_i in vector]
```

함수에 적용시켜보자.

```python
#1. 벡터 하나를 준비했다.
vector1 = np.array([1,2,3,4,5,6])
```

```python
#2. 스칼라는 4를 적용시켜보자
vector_multiply(vector1, 4)

'''
결과값이 이렇게 나온다.
[4, 8, 12, 16, 20, 24]
'''
```