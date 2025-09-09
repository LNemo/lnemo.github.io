---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 6: Vector, Matrix, Tensor"
date: 2025-09-08 19:00:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, 부스트캠프, numpy, vector, matrix, tensor]
description: "AI Math에 대해서 배우자."
keywords: [numpy, ai math, tensor, colab, 행렬, 벡터, 텐서, einsum, einops,EVD, SVD, PCA, 특이값분해, 고유값분해, eigenvector, eigenvalue, ridge, lasso, norm, determinant, rank, inverse, pseudoinverse]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Vector, Matrix, Tensor

AI를 위한 기초 수학 지식을 배웁니다. 이번에는 선형대수를 배우겠습니다.

## Vector

벡터는 숫자가 element인 data type입니다. **배열**이라고 불립니다. 벡터의 차원은 벡터의 원소의 개수를 말합니다. 

벡터는 공간에서 한 점을 표시하고, 상대적 위치를 표현합니다. 벡터에 scalar를 곱한다면 방향은 변하지 않고 길이만 변합니다.

벡터는 벡터의 차원이 서로 같다면 성분곱, 합, 차가 가능합니다.
```python
import numpy as np

x = np.array([1, 7, 2])
y = np.array([5, 2, 1])
# x + y # 벡터의 합
# x * y	# 벡터의 성분곱
# x @ y	# 벡터의 곱
```

두 벡터의 각도를 구하는 방법은 이전에도 나오듯이 벡터의 내적을 각각의 벡터의 L2-Norm을 곱한 값으로 나누어주면 코사인 값을 구할 수 있습니다.

**내적**은 정사영된 벡터의 길이(Scalar)입니다.

## Matrix

행렬은 벡터를 원소로 가지는 2차원 배열입니다. 행렬은 행과 열의 index를 가집니다. 행렬의 특정 행이나 열을 고정하면 행벡터 또는 열벡터라고 부릅니다. Numpy에서는 기본으로 행벡터를 사용합니다.

행렬은 같은 모양끼리 덧셈 뺄셈 성분곱이 가능합니다. 

행렬의 곱셈은 덧셈, 뺄셈과 다릅니다. **행렬의 곱셈**은 i번째 **행벡터**와 j번째 **열벡터** 사이의 내적을 성분으로 갖는 행렬을 계산합니다. ``n x m``, ``m x l``크기를 가진 행렬끼리 행렬 곱셈을 한다고 하면 결과 행렬은 ``n x l``크기의 행렬을 가지게 됩니다.

```python
# Numpy의 np.inner는 행벡터와 행벡터 간의 내적을 성분으로 가지는 행렬을 계산!
X = np.array([[1, 2, 3,],
              [3, 2, 1],
              [2, 3, 1]])
Y = np.array([[0, -1, 1,],
              [-1, 0, 1]])

print(np.inner(X, Y))
print(X @ np.transpose(Y))  # @ 처럼 행렬곱을 사용하면 행과 열 간의 내적을 계산
print(np.matmul(X, np.transpose(Y)))    # @과 동일한 기능
print(np.multiply(X, np.transpose(Y)))  # *와 동일한 기능
```

### Inverse Matrix
역행렬은 어떤 행렬의 연산을 거꾸로 되돌리는 행렬입니다. 역행렬은 행과 열이 같은 정사각행렬이어야 하고, determinant가 0이 아니어야 합니다. 연립방정식을 역행렬을 활용해서 해를 구할 수 있습니다.

```python
X @ np.linalg.inv(X)	# 행렬을 역행렬과 곱함! 결과는 I가 나옴
```

### Determinant

**행렬식**(Determinant)은 행렬을 구성하는 벡터로 만들어낸 다포체의 부피의 크기와 같습니다. 행렬 A를 구성하는 벡터 중 선형독립인 벡터의 개수가 n이면 A의 determinant가 0이 아니라고 할 수 있습니다. 이때 행렬 A는 역행렬을 가집니다.

### Pseudo-inverse Matrix

**유사역행렬**(Pseudo-inverse Matrix) 또는 무어-펜로즈(Moore-Penrose) 역행렬은 역행렬을 계산할 수 없을때 사용할 수 있습니다(n!=m 또는 det(A)=0 등…). 유사역행렬은 연립방정식에서 해가 무수히 많을 경우(부정)에 사용됩니다. 
```python
# 해가 무수히 많을 경우 -> 부정
# 해가 없을 경우 -> 불능
# 부정형 연립방정식 풀기에 pinv 사용 가능
A = np.array([[0, -1, 1],
              [1, 1, 1]])
b = np.array([[3],
              [4]])
np.linalg.pinv(A) @ b
```
```
```python
np.linalg.inv(A.transpose() @ A) @ A.transpose()    # 유사역행렬
np.linalg.pinv(A)   # 이것 또한 유사역행렬
```

선형회귀분석에서도 ``np.linalg.pinv()``를 이용해서 선형회귀식을 찾을 수 있습니다.

```python
# 유사역행렬 계산
X_ = np.array(np.append(x,[1]) for x in X)	# 1을 넣어주는 이유는 y절편 때문에
beta = np.linalg.pinv(X_) @ y
y_test = np.append(x, [1]) @ beta
```

### Matrix Decomposition

행렬분해(Matrix Decomposition)는 행렬을 여러 행렬의 곱으로 표현하는 것을 의미합니다. 행렬을 분해하는 방법은 여러가지가 있고, 여러 목적에 따라 쓰입니다.

#### Eigenvalue Decomposition

행렬에 어떤 벡터를 곱했을 때에 그 벡터가 상수배가 되는 경우, 해당 벡터를 **고유벡터**(eigenvector)라고 하고 그 상수를 **고유값**(eigenvalue)이라고 합니다. 

고유값 분해(Eigenvalue Decomposition)는 행렬을 고유벡터로 이루어진 벡터와 고유값으로 이루어진 대각행렬로 분해하는 것입니다.

![이미지](/assets/img/posts/boostcamp/day6/eigendecomposition.svg)

해당 표현은 A 연산이 Q^(-1)로 rotate 한 후 람다만큼 scale하고 다시 돌아오는 연산이라는 것을 풀어쓴 것입니다.
#### Singular value Decomposition

고유값 분해는 고유값을 가질수 있는 정사각행렬만 가능하기 때문에 직사각행렬에 대해서는 사용할 수 없습니다. **특이값 분해**(SVD)는 행렬에 일반적으로 사용할 수 있는 방법입니다. (더 일반적이고 더 강력한 도구입니다!)

![이미지](/assets/img/posts/boostcamp/day6/singulardecomposition.svg)

유사 역행렬이 min{n, m} 일때만 유사역행렬을 구할 수 있었지만, Full Rank가 아니더라도 SVD를 이용한다면 유사역행렬을 구할 수 있습니다.

SVD를 선형회귀분석에도 적용할 수 있습니다. 실제 역행렬을 계산하는 것보다 훨씬 효율적입니다. (사실 Numpy의 pinv도 SVD를 이용해서 유사역행렬을 구한다고 합니다.)

## Tensor

텐서는 사실 첫번째 주에 학습했기에 훨씬 익숙합니다. 여기서는 Tensor의 einsum, einops 연산을 익혀보겠습니다.

### ``einsum``
``einsum``은 아인슈타인 표기법(Einstein summation convention)에서 유래하였습니다. 아인슈타인 표기법은 인덱스를 활용해서 텐서곱을 정의합니다.

- ``np.einsum('ik, kj -> ij', X, Y)``: 행렬곱
- ``np.einsum('ij -> ji', X)``: Transpose
- ``np.einsum('ik, jk -> ij', X, Y)``: Transpose 후 행렬곱
- ``np.einsum('bik, pkj ￼-> bipj', X, Y)``: 4차원 텐서 (``dot(X,Y)``)
- ``np.einsum('bik, pkj ￼-> ipj', X, Y)``: 3차원 텐서 (``matmul(X,Y)``)
- ``np.einsum('bii -> b', X)``: 3차원 텐서의 각 성분 별로 trace 값 계산
- ``einops.rearrange(X, 'b i j k -> b (i j k)')``: b 인덱스 제외 모두 묶어줌 (``reshape()``)
- ``np.einsum('bi, bi -> b', X, Y)``: 행끼리 내적 계산
- ``einops.reduce(z, 'b c -> b', 'mean')``: b 인덱스별로 평균값 구하기

(``einops.reduce()``사용할때는 data type 명시하는 것이 좋음)

---
**피어세션을 통해 알아간 것**
- 그 고유벡터 방향으로 정사영하면 정보 손실이 최소화된다.
  - 따라서 분산을 최대로 하는 고유벡터가 정보 손실이 적은 이유는 해당 방향으로 데이터가 많이 퍼져 있을 수록 해당 데이터를 더 잘 표현하기 때문이다.
- eigenvector와 eigenvalue로 Matrix A를 표현하는 것은 rotate -> scale -> rotate(다시 되돌리기)의 과정과 같다.
