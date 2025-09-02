---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 1: PyTorch 기초 (1)"
date: 2025-09-01 19:00:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, pytorch, 부스트캠프]
description: "딥러닝 프레임워크인 PyTorch를 통해 텐서 연산의 기초를 배우자."
keywords: [pytorch, torch, tensor, colab, scalar, vector, matrix, data type, cuda, 연속균등분포, 표준정규분포]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# PyTorch 기초

PyTorch는 **간편한 딥러닝 API를 제공하며, 머신러닝 알고리즘을 구현하고 실행하기 위한 확장성이 뛰어난 멀티플랫폼 프로그래밍 인터페이스(Raschka, Liu & Mirjalili, 2022)**입니다.

API는 Application Programming Interface로 **서로 다른 소프트웨어 프로그램이 통신하고 데이터를 교환하며 기능을 사용할 수 있도록 정의된 규칙이나 약속, 인터페이스**를 의미합니다.

우리는 왜 PyTorch를 배워야 할까요?
PyTorch는 사용자 친화적이며 현재 산업계의 기술 개발(Tesla, Uber, Hugging Face 등)에도 널리 쓰이고 있습니다. 또한 동적 계산 그래프를 사용하기 때문에 연산을 평가하고, 계산을 실행하고, 구체적인 값을 즉시 반환하는 명령형 프로그래밍 환경을 제공합니다.
GPU를 사용할 수 있어 성능과 효율도 좋고, 다양한 모델 구현체도 있습니다.

이러한 PyTorch를 사용하기 위해서는 우리는 기본적으로 Tensor를 알 필요가 있습니다.
## Tensor

Tensor는 PyTorch의 핵심 데이터 구조입니다. Numpy의 다차원배열처럼 데이터를 표현합니다. 

### 0-D Tensor

0-D Tensor는 **Scalar**입니다. 하나의 수로 표현된다면 0-D Tensor로 표현할 수 있습니다.

예시로 사람의 체온, 음료수의 수 등을 말할 수 있습니다.

```python
>>> import torch
>>> a = torch.tensor(36.5)
>>> print('a =', a)
```
```
a = tensor(36.5000)
```

### 1-D Tensor

1-D Tensor는 **Vector**입니다. 순서가 정해진 여러 수가 일렬로 나열된다면 1-D Tensor로 표현할 수 있습니다.

예시로 사람의 신체정보(나이, 키, 몸무게, 시력)이 있습니다.

```python
>>> b = torch.tensor([175, 60, 81, 0.8, 0.9])
>>> print('b =', b)
```
```plaintext
b = tensor([175.0000, 60.0000, 81.0000, 0.8000, 0.9000])
```

### 2-D Tensor

2-D Tensor는 **Matrix**입니다. 1-D의 정보가 여러개 모여있다고 생각하면 됩니다.

예시로 사람의 신체정보를 모아둔 의료용 명부를 생각하면 됩니다. 그레이 스케일 이미지도 여기에 해당됩니다.

```python
>>> c = torch.tensor([[77, 114, 140, 191],
                      [39, 56, 46, 119],
                      [61, 29, 20, 33]])
>>> print('c =', c)
```
```plaintext
c = tensor([[77, 114, 140, 191],
       [39, 56, 46, 119],
       [61, 29, 20, 33]])
```

### 3-D Tensor

3-D Tensor는 2-D Tensor들이 여러개 쌓여 형성된 **입체적인 배열 구조**입니다. 2-D에서는 그레이 스케일 이미지가 있었다면 3-D Tensor에서는 컬러 이미지를 예시로 들 수 있습니다.

```python
>>> red_channel = torch.tensor([[255, 0],
                                [0, 255]])
>>> green_channel = torch.tensor([[0, 255],
                                  [0, 255]])
>>> blue_channel = torch.tensor([[0, 0],
                                 [255, 0]])
>>> d = torch.stack((red_channel, green_channel, blue_channel), dim=2)
>>> print('d = ', d)
```
```plaintext
d =  tensor([[[255,   0,   0],
         [  0, 255,   0]],

        [[  0,   0, 255],
         [255, 255,   0]]])
```

### N-D Tensor

4-D Tensor는 3-D Tensor를 한쪽 방향으로 또 쌓으면 됩니다. 5-D Tensor는 4-D Tensor를 또 한쪽 방향으로 쌓으면 됩니다. 6-D Tensor는 또…

이렇게 동일한 크기의 (N-1)-D Tensor들이 여러 개 쌓으면 N-D Tensor를 만들 수 있습니다.

## PyTorch에서 Data Type

PyTorch에서의 데이터 타입(dtype)은 Tensor가 가지는 값의 데이터 타입을 의미합니다.

### 정수형 데이터 타입

비트가 클수록 더 큰 수를 저장할 수 있습니다. 

부호가 있는 정수는 가장 왼쪽 자리를 부호로 사용합니다. (0: 양수, 1: 음수)

8bit: 0000 0000

16bit: 0000 0000 0000 0000

- 8비트 부호 없는 정수: ``dtype = torch.uint8``
- 8비트 부호 있는 정수: ``dtype = torch.int8``
- 16비트 부호 있는 정수``dtype = torch.int16`` 또는 ``dtype = torch.short``
- 32비트 부호 있는 정수``dtype = torch.int32`` 또는 ``dtype = torch.int``
- 64비트 부호 있는 정수``dtype = torch.int64`` 또는 ``dtype = torch.long``

### 실수형 데이터 타입

컴퓨터에서 실수를 저장하는 방법에는 **고정 소수점**과 **부동 소수점**이 있습니다.

고정 소수점에서는 비트를 부호, 정수부, 소수부로 나눕니다. 하지만 이 방식은 소수점 아래 자리수가 커질수록 소수부가 그만큼 커져야 하기 때문에 컴퓨터가 메모리를 감당하기 어렵습니다.

부동 소수점에서는 이런 문제를 해결하기 위해서 부호를 부호, 지수부, 가수부로 나눕니다. 실수를 지수부와 가수부로 바꾸는 정규화 과정을 거치면 더 넓은 범위의 실수를 표시할 수 있습니다.

- 32비트 부동 소수점 실수: ``dtype = torch.float32`` 또는 ``dtype = torch.float``
- 64비트 부동 소수점 실수: ``dtype = torch.float64`` 또는 ``dtype = torch.double``

### 타입 캐스팅

PyTorch에서 타입 캐스팅은 한 데이터 타입을 다른 데이터 타입으로 변환하는 것을 의미합니다.

```python
>>> i = torch.tensor([2, 3, 4], dtype=torch.int8)    # 데이터 타입이 torch.int8
>>> j = i.float()    # 데이터 타입을 torch.float로 변환
>>> k = i.double()   # 데이터 타입을 torch.double로 변환
```

## Tensor의 기초 함수 및 메서드

PyTorch에서는 Tensor에서 계산을 편리하게 하기 위한 함수와 메서드가 존재합니다.

### Tensor의 요소를 반환하거나 계산
- ``torch.min(tensor)``: 가장 작은 요소
- ``torch.max(tensor)``: 가장 큰 요소
- ``torch.sum(tensor)``: 모든 요소들의 합
- ``torch.prod(tensor)``: 모든 요소들의 곱
- ``torch.mean(tensor)``: 모든 요소들의 평균
- ``torch.var(tensor)``: 모든 요소들의 표본분산
- ``torch.std(tensor)``: 모든 요소들의 표본표준편차

### Tensor의 특성을 확인하는 메서드

``t = torch.tensor([12, 13])`` 일 때,
- ``t.dim()``: 차원의 수
- ``t.size()`` 또는 ``t.shape``: 텐서의 크기, 모양 (shape는 메서드가 아닌 속성임에 유의)
- ``t.numel()``: 요소의 총 개수 (num of elements)

## Tensor의 생성

Tensor를 생성할 때에는 ``torch.tensor([1, 2, 3])``처럼 직접 리스트를 넣어서 만들 수도 있지만 다양한 방법이 있습니다.

### 특정한 값으로 초기화된 Tensor 생성

``torch.zeros()``와 ``torch.ones()``는 각각 0과 1로 채워진 tensor를 생성합니다.

파라미터로 정수를 넣는다면 1차원 tensor를 만들 수 있습니다.
```python
>>> a = torch.zeros(3)
>>> print(a)
```
```plaintext
tensor([0., 0., 0.])
```

정수 대신 리스트를 넣는다면 해당 차원으로 tensor를 만듭니다.
```python
>>> b = torch.ones([2, 3])
>>> print(b)
```
```plaintext
tensor([[1., 1., 1.],
        [1., 1., 1.]])
```


```python
>>> c = torch.zeros([3, 2, 4])
>>> print(c)
```
```plaintext
tensor([[[0., 0., 0., 0.],
         [0., 0., 0., 0.]],

        [[0., 0., 0., 0.],
         [0., 0., 0., 0.]],

        [[0., 0., 0., 0.],
         [0., 0., 0., 0.]]])
```
위 코드는 2행 4열을 3개 쌓은 형태의 tensor입니다.

다음의 코드는 인자의 tensor와 같은 구조의 tensor를 만들고 0이나 1을 채웁니다.
- ``torch.zeros_like(a)``: a와 같은 구조의 0으로 채워진 tensor
- ``torch.ones_like(b)``: b와 같은 구조의 1으로 채워진 tensor

### 난수로 초기화된 Tensor 생성

무작위 값의 tensor가 필요할 경우 다음의 코드를 사용합니다. 
- ``torch.rand(3)``
- ``torch.rand([2, 3])``
- ``torch.randn(3)``
- ``torch.randn([2, 3])``

``torch.rand()``는 **연속균등분포**를 따르고, ``torch.randn()``은 평균이 0이고 분산과 표준편차가 1인 **표준정규분포**를 따릅니다.

### 지정된 범위 내에서 초기화된 Tensor 생성

``torch.arange(start, end, step)``는 start 이상, end 미만의 수를 step 간격으로 가져와 초기화합니다.

```python
>>> d = torch.arange(start = 1, end = 11, step = 2)    # array range
>>> print(d)
```
```plaintext
tensor([1, 3, 5, 7, 9])
```

### 초기화 되지 않은 Tensor 생성

Tensor는 초기화 하지 않을 수도 있습니다. 어차피 다른 수로 덮어씌워야 하는 Tensor라면 초기화를 할 필요가 없습니다. (초기화하지 않는다면 임의의 값이 채워져 있을 수도 있습니다.)

- ``torch.empty(5)``
- ``torch.empty([1,2])``

만약에 초기화 하지 않은 Tensor를 3.0으로 채우고 싶다면 ``t.fill_(3.0)``을 실행합니다. 메모리 주소는 변경되지 않고 해당 주소의 텐서의 요소들을 변경합니다.

### Numpy 데이터로부터 Tensor 생성

리스트형은 ``torch.tensor()``로 Tensor를 만들 수 있지만, Numpy의 array는 다른 메서드가 필요합니다.

```python
import numpy as np
u = np.array([[0, 1], [2, 3]])
v = torch.from_numpy(u)   # Numpy로 생성된 Tensor는 기본적으로 정수형
v = torch.from_numpy(u).float()    # 따라서 타입 캐스팅이 필요
```

### Tensor 복제

Tensor를 복제하여 또다른 Tensor에 저장할 수 있습니다.

- ``y = x.clone()``: x를 y로 복제(계산그래프까지)
- ``z = x.detach()``: 계산그래프에서 분리해서 복제

### CPU Tensor 생성
- ``torch.ByteTensor()``: 8비트 부호 없는 정수형
- ``torch.CharTensor()``: 8비트 부호 있는 정수형
- ``torch.ShortTensor()``: 16비트 부호 있는 정수형
- ``torch.IntTensor()``: 32비트 부호 있는 정수형
- ``torch.LongTensor()``: 64비트 부호 있는 정수형
- ``torch.FloatTensor()``: 32비트 부호 있는 실수형
- ``torch.DoubleTensor()``: 64비트 부호 있는 실수형

### CUDA Tensor 생성과 변환

GPU는 그래픽 처리 장치이지만, 현재는 AI 연구 및 개발에 있어 대규모 데이터 처리와 복잡한 계산을 위해 사용되고 있습니다. GPU는 아주 많은 작은 코어를 가지고 있어 **병렬 데이터 처리에 효율적**입니다. 따라서 대량의 연산을 동시에 할 수 있게 하여 AI 모델의 훈련과 추론 속도를 높일 수 있습니다!

CUDA는 엔비디아의 GPU를 활용하여 고도의 계산처리를 할 수 있도록 합니다. 이를 통해 우리는 효율적으로 Tensor 계산을 할 수 있습니다.

```python
# tensor가 현재 어떤 디바이스에 있는지 확인
>>> a = torch.tensor([1, 2, 3])
>>> print(a.device)
```
```plaintext
device(type='cpu')
```

아래의 메서드들로 CUDA 관련 정보를 확인할 수 있습니다.
```python
# CUDA를 사용할 수 있는 환경인지 확인
>>> torch.cuda.is_available()
True
# CUDA 디바이스 이름 확인
>>> torch.cuda.get_device_name(device=0)
'Tesla T4'
# 사용 가능한 GPU 개수 확인
>>> torch.cuda.device_count()
1
```

Tensor를 CUDA에 할당할 때에는 Tensor에 ``.to(device=‘cuda’)``를 붙입니다.
```python
# Tensor를 GPU에 할당
>>> b = torch.tensor([1, 2, 3, 4, 5]).to(device='cuda')    # 또는 .cuda()
>>> print('b_gpu =', b)
```
```plaintext
b_gpu = tensor([1, 2, 3, 4, 5], device='cuda:0')
```

CUDA Tensor를 다시 CPU에 할당할 때에는 Tensor에 ``.to(device=‘cpu’)``를 붙입니다.
```python
# GPU에 할당된 Tensor를 CPU Tensor로 변환
>>> c = b.to(device='cpu')    # 또는 .cpu()
>>> print('c =', c)
```
```plaintext
c = tensor([1, 2, 3, 4, 5])
```