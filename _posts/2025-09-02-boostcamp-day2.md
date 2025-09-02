---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 2: PyTorch 기초 (2)"
date: 2025-09-02 19:00:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, pytorch, 부스트캠프]
description: "딥러닝 프레임워크인 PyTorch를 통해 텐서 연산의 기초를 배우자."
keywords: [pytorch, torch, tensor, colab, scalar, vector, matrix, data type, cuda, norm, 노름, l1 norm, l2 norm]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# PyTorch 기초 (2)

## Tensor의 Indexing & Slicing
PyTorch에서도 Numpy처럼 **indexing**과 **slicing**을 할 수 있습니다. PyTorch의 indexing, slicing은 직관적으로 데이터를 조작을 가능하게 합니다.

### Indexing

각각의 요소값을 참조하기 위해 index를 사용합니다.

```python
a = torch.tensor([10, 20, 30, 40, 50, 60])

print('a[0] =', a[0])
```
```
tensor(10)
```

Index는 Python과 동일하게 음수값으로도 사용할 수 있습니다. 음수값으로 인덱싱할 경우에는 오른쪽 값부터 가져옵니다.

```python
print('a[-1] =', a[-1])
```
```
tensor(60)
```

### Slicing

PyTorch에서는 여러 개의 요소를 가져오기 위해 slicing을 사용합니다.

```python
print('a[1:4] =', a[1:4])
```
```
tensor([20, 30, 40])
```

Slicing에는 다음의 표현도 사용할 수 있습니다:
- ``a[:5]``: 0부터 4까지
- ``a[1:]``: 1부터 끝까지
- ``a[:]``: 전부
- ``a[0:5:2]``: 0부터 5전까지 2씩 더하면서

### 2-D Tensor의 Indexing

PyTorch에서는 2-D Tensor의 Indexing을 좌표값처럼 튜플로 가져올 수 있습니다!

```python
b = torch.tensor([[10, 20, 30],
                  [40, 50, 60]])

# 0번 row, 1번 column element
print('b[0,1]', b[0,1])	# 튜플로 가능하다!
# b[-2,-2]도 같은 요소를 출력한다
# b[0][1]은 당장은 같은 요소를 출력하지만 슬라이싱에서 예상과 다른 출력을 할 수 있음
```
```
tensor(20)
```

### 2-D Tensor의 Slicing

2-D Tensor의 indexing처럼 튜플로 표현할 수 있습니다. 그리고 콜론(:)을 사용하여 slicing 할 수 있습니다. 모든 elements를 가져오고 싶을 때에는 콜론만 입력하거나 점 세 개(...)를 입력하면 됩니다.

```python
print('b[0, 1:] =', b[0, 1:])
```
```
tensor([20, 30])
```

...을 사용한 예시입니다.

```python
print('b[1, ...] =', b[1, ...])	# b[1,:], b[-1, ...], b[-1,:]도 가능하다
```
```
tensor([40, 50, 60])
```


## Tensor의 모양 변경
PyTorch에서는 Tensor에 데이터를 저장할 때에 해당 Tensor의 데이터 타입과 차원 정보에 기반하여 컴퓨터 메모리에서 충분한 공간을 할당받아 저장합니다. 이 때, 데이터는 메모리에 순차적으로 쓰이게 되는데, 이것을 contiguous 하다고 합니다.

```python
d = torch.tensor([[0, 1, 2],
                  [3, 4, 5]])
```

위의 텐서가 메모리에 저장된다면 아래와 같이 저장됩니다.

| Memory  | Address | Element |
|:-------:|:-------:|:-------:|
| d[1, 2] | 0x14    | 5       |
| d[1, 1] | 0x10    | 4       |
| d[1, 0] | 0x0C    | 3       |
| d[0, 2] | 0x08    | 2       |
| d[0, 1] | 0x04    | 1       |
| d[0, 0] | 0x00    | 0       |

이것이 바로 **contiguous** 입니다. Tensor ``t``에 대해서 ``t.is_coniguous()``를 실행하면 메모리가 연속적으로 할당되었는지 비연속적으로 할당되었는지 알 수 있습니다.

### view()

Tensor가 contiguous 하다면 ``torch.view()``를 사용할 수 있습니다. ``torch.view()``는 parameter로 tensor의 shape를 받습니다.

```python
f = torch.arange(12)
print(f)
print(f.shape)
print(f.is_contiguous())
```
```
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
torch.Size([12])
True
```
``f`` Tensor가 연속성을 가지고 있기 때문에 ``f.view()`` 메서드를 사용할 수 있습니다.
```python
# 4행 3열로 모양 변경
g = f.view(4, 3)	# f.view(4, -1) -1은 잘 모르는값에 넣으면 자동으로 계산해줌. 한번만 가능
print(g)
print(g.shape)
```
```
tensor([[ 0 , 1 , 2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]])
torch.Size([4, 3])
```

3-D Tensor에 대해서도 가능합니다.

```python
h = f.view(3, 2, 2)	# or f.view(3, -1, 2)
print(h)
print(h.shape)
```
```
tensor([[[ 0,  1],
         [ 2,  3]],

        [[ 4,  5],
         [ 6,  7]],

        [[ 8,  9],
         [10, 11]]])
torch.Size([3, 2, 2])
```

### flatten()
``flatten()``은 Tensor를 1-D Tensor로 평탄화하기 위한 함수/메서드입니다. 다차원의 데이터를 전처리 하기위해 많이 활용됩니다.

```python
i = torch.randn(3, 3)
print(i)
j = torch.flatten(i) # or j = i.flatten()
print(j)
```
```
tensor([[-0.7592,  0.6516,  0.2902],
        [ 1.0063, -1.2858, -2.3647],
        [ 1.4522,  0.9851, -0.2593]])
tensor([-0.7592,  0.6516,  0.2902,  1.0063, -1.2858, -2.3647,  1.4522,  0.9851, -0.2593])
```

``torch.flatten()``은 parameter로 평탄화 할 차원을 지정해 줄 수 있습니다.

``torch.flatten(t, 0)``은 ``torch.flatten(t, 0)``와 같습니다.

```python
l = torch.flatten(k, 0)
print(l)
print(l.shape)
```
```
tensor([ 0.5106,  0.4206,  0.5766,  1.5839, -0.9223, -0.5679,  1.2922, -0.7799,  0.4539,  0.1716, -0.0611,  0.9202])
torch.Size([12])
```

``torch.flatten(t, n)`` 은 n차원부터 마지막 차원까지 평탄화합니다.

```python
l = torch.flatten(k, 1)
print(l)
print(l.shape)
```
```
tensor([[ 0.5106,  0.4206,  0.5766,  1.5839],
        [-0.9223, -0.5679,  1.2922, -0.7799],
        [ 0.4539,  0.1716, -0.0611,  0.9202]])
torch.Size([3, 4])
```

``torch.flatten(t, n, m)``은 n차원부터 m차원까지 평탄화합니다.

```python
l = torch.flatten(k, 0, 1)
print(l)
print(l.shape)
```
```
tensor([[ 0.5106,  0.4206],
        [ 0.5766,  1.5839],
        [-0.9223, -0.5679],
        [ 1.2922, -0.7799],
        [ 0.4539,  0.1716],
        [-0.0611,  0.9202]])
torch.Size([6, 2])
```

### reshape()
``view()`` 말고도 ``reshape()``를 사용해서 tensor의 모양을 변경할 수 있습니다. ``reshape()``는 tensor가 연속성이 없어도 사용 가능하지만, 성능이 저하된다는 단점이 있습니다.

```python
n = torch.arange(12)
o = n.reshape(4, 3)	# or n.reshape(4, -1)
```

### transpose()
``transpose()``는 특정한 두 차원의 축을 서로 바꾸는 메서드입니다.
```python
q = torch.tensor([[0, 1, 2],
                  [3, 4, 5]])
r = q.transpose(0, 1)	# 0차원과 1차원간 축을 변경!
```
```
tensor([[0, 3],
        [1, 4],
        [2, 5]])
```

아래는 1차원과 2차원 간 변경하는 또 다른 예시입니다.
```python
s = torch.tensor([[[0, 1],
                   [2, 3],
                   [4, 5]],
                   
                  [[6, 7],
                   [8, 9],
                   [10, 11]],
                  
                  [[12, 13],
                   [14, 15],
                   [16, 17]]])

t = s.transpose(1, 2)	# 1차원과 2차원간 변경!
print(t)
```
```
tensor([[[ 0,  2,  4],
         [ 1,  3,  5]],

        [[ 6,  8, 10],
         [ 7,  9, 11]],

        [[12, 14, 16],
         [13, 15, 17]]])
```

### squeeze()
``squeeze()`` 함수는 tensor에서 차원의 크기가 1인 차원을 축소합니다. Parameter를 지정해주지 않으면 모든 1인 차원을 축소합니다.
```python
u = torch.randn(1, 3, 4)
v = torch.squeeze(u)	# 차원이 1이면 축소시켜줌 shape가 [1, 1, 4]인 경우 [4]가 됨
print(v)
print(v.shape)
```
```
tensor([[ 0.0997, -0.7637,  1.7196, -0.2055],
        [ 0.6536,  1.1291,  0.7916, -1.0196],
        [ 1.1963,  0.6175,  1.0531,  0.4534]])
torch.Size([3, 4])
```
``dim``을 지정해서 특정 차원만 축소할 수도 있습니다.
```python
w = torch.randn(1, 1, 4)
x = torch.squeeze(w, dim = 1)	# dim에 해당하는 차원을 축소 (dim이 0이면 depth, 1이면 row)
```
### unsqueeze()
``unsqueeze()``는 ``squeeze()``와 정반대 함수입니다. ``unsqueeze()``는 차원을 확장합니다. ``dim``을 지정해서 해당 차원에 크기가 1인 차원을 추가합니다.

- ``dim=0``
```python
y = torch.randn(3, 4)
z = torch.unsqueeze(y, dim = 0)	# depth 차원 확장
print(z)
print(z.shape)
```
```

tensor([[[-0.8067, -1.1391, -1.9774, -0.3196],
         [-0.2963, -0.3446,  0.2757, -0.6229],
         [ 1.7075,  0.7363,  1.2011, -0.2810]]])
torch.Size([1, 3, 4])
```

- ``dim=1``
```python
y = torch.randn(3, 4)
z = torch.unsqueeze(y, dim = 1)	# row 차원 확장
print(z)
print(z.shape)
```
```
tensor([[[-1.8428, -1.0983,  2.0327, -1.1115]],

        [[ 0.7548, -1.0482, -0.2081, -0.3656]],

        [[ 0.8281, -1.3230,  0.3488, -0.2097]]])
torch.Size([3, 1, 4])
```
기존 row 차원이 depth 차원이 됩니다.
- ``dim=2``
```python
y = torch.randn(3, 4)
z = torch.unsqueeze(y, dim = 2)	# column 차원 확장
print(z)
print(z.shape)
```
```
tensor([[[-0.9036],
         [-0.3267],
         [ 1.6371],
         [-0.5565]],

        [[ 2.7228],
         [-0.4201],
         [-0.6133],
         [-0.4849]],

        [[-0.2018],
         [ 1.7400],
         [-0.7246],
         [ 0.8479]]])
torch.Size([3, 4, 1])
```
기존 row 차원은 depth 차원, 기존 column 차원은 row 차원이 됩니다.

### stack()
일전에 [색상이 있는 이미지를 표현할 때](https://lnemo.github.io/posts/boostcamp-day1/#3-d-tensor) ``stack()`` 함수를 사용한 적이 있습니다. 여러 채널의 2-D Tensor를 쌓아서 ``stack()`` 함수로 하나의 3-D Tensor를 만들었습니다.

``dim=0``: 각 Tensor를 쌓는 느낌입니다.
```python
red_channel = torch.tensor([[255, 0],
                            [0, 255]])
green_channel = torch.tensor([[0, 255],
                              [0, 255]])
blue_channel = torch.tensor([[0, 0],
                             [255, 0]])
a = torch.stack((red_channel, green_channel, blue_channel))	# default dim=0
print(a)
```
```
tensor([[[255,   0],
         [  0, 255]],

        [[  0, 255],
         [  0, 255]],

        [[  0,   0],
         [255,   0]]])
```
``dim=1``: 각 Tensor의 행끼리 모아서 Tensor를 만듭니다.
```python
a = torch.stack((red_channel, green_channel, blue_channel), dim=1)
print(a)
```
```
tensor([[[255,   0],
         [  0, 255],
         [  0,   0]],

        [[  0, 255],
         [  0, 255],
         [255,   0]]])
```
``dim=2``: 각 Tensor의 요소들을 모아서 Tensor를 만듭니다.
```python
a = torch.stack((red_channel, green_channel, blue_channel), dim=2)
print(a)
```
```
tensor([[[255,   0,   0],
         [  0, 255,   0]],

        [[  0,   0, 255],
         [255, 255,   0]]])
```

### cat()
``cat()``은 concat을 생각하면 될 것 같습니다. Tensor를 이어서 붙인다는 느낌입니다. 따라서 ``stack()``과 달리 차원은 유지하면서 Tensor를 연결합니다. 같은 차원을 가져야 합니다.
```python
a = torch.tensor([[0, 1],
                  [2, 3]])
b = torch.tensor([[4, 5]])	# 2-D Tensor. cat은 같은 차원을 가져야 함

c = torch.cat((a,b))	# concat! default dim=0
print(c)
```
```
tensor([[0, 1],
        [2, 3],
        [4, 5]])
```
아래의 경우 column으로 결합하려고 하지만 행의 개수가 다르기 때문에 불가능합니다. ``b.reshape(2,1)``로 shape을 변경한다면 결합할 수 있습니다.
```python
d = torch.cat((a, b), 1)	# dim=1에 결합 -> 에러발생! 행의 개수가 달라서 불가능
```

### expand()
``expand()``는 **주어진 차원의 크기가 1일 때, 해당 차원의 크기를 확장**합니다.
```python
f = torch.tensor([1, 2, 3])

g = f.expand(4, 3)	# f를 사용하여 4행 3열의 tensor를 생성
print(g)
```
```
tensor([[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]])
```
``expand()``는 차원 중 일부의 크기가 1이어야 합니다.

### repeat()
``repeat()``는 Tensor의 요소들을 반복해서 크기를 확장하는데 사용합니다. 또한 ``expand()``와 다르게 차원의 크기가 1이어야 한다는 제약이 없습니다. 하지만 ``repeat()``는 추가 메모리를 할당하기 때문에 ``expand()``에 비해 메모리 효율성이 크게 떨어집니다.
```python
h = torch.tensor([[1, 2],
                  [3, 4]])
i = h.repeat(2, 3)	# dim=0으로는 2번, dim=1으로는 3번 반복하여 확장
print(i)
```
```
tensor([[1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4],
        [1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4]])
```

## Tensor의 기초 연산
Tensor 간에 더하기, 빼기, 곱하기, 나누기 같은 기본적인 연산이 가능합니다. 함수 또는 메서드로 계산할 수 있습니다. Tensor 간 요소끼리 계산할 수 있고, shape이 다른 경우에 자동으로 expand 하여 계산됩니다. 메서드로 계산시 언더스코어(_)를 붙여서 연산하면 in-place 연산이 가능합니다. *ex)``t.add_(s)``*
### add()
- Tensor의 shape이 같은 경우
```python
# add (a+b, 요소합)
a = torch.tensor([[1, 2],
                  [3, 4]])
b = torch.tensor([[1, 3],
                  [5, 7]])
c = torch.add(a, b)	# or a.add(b)
print(c)
```
```
tensor([[ 2,  5],
        [ 8, 11]])
```
- Tensor의 shape이 다른 경우
```python
# add (a+b, shape이 다른 경우)
a = torch.tensor([[1, 2],
                  [3, 4]])
b = torch.tensor([1, 3])	# 한쪽은 사이즈가 같아야 함
c = torch.add(a, b)	# or a.add(b)
print(c)
```
```
tensor([[ 2,  5],
        [ 8, 11]])
```

### sub()
- Tensor의 shape이 같은 경우
```python
# sub (d-e, 요소차)
d = torch.tensor([[2, 5],
                  [8, 11]])
e = torch.tensor([[1, 3],
                  [5, 7]])
f = torch.sub(d, e)	# or d.sub(e)
print(f)
```
```
tensor([[1, 2],
        [3, 4]])
```
- Tensor의 shape이 다른 경우
```python
# sub (d-e, shape이 다른 경우)
d = torch.tensor([[2, 5],
                  [8, 11]])
e = torch.tensor([[1],
                  [5]])	# 한쪽은 사이즈가 같아야 함. 자동으로 expand되어 빼짐
f = torch.sub(d, e)	# or d.sub(e)
print(f)
```
```
tensor([[1, 4],
        [3, 6]])
```
### mul()
- Tensor에 Scalar를 곱하는 경우
```python
# mul (g*h, 스칼라곱)
g = 2
h = torch.tensor([[1, 2],
                  [3, 4]])
i = torch.mul(g, h)	# or g.mul(h)
print(i)
```
```
tensor([[2, 4],
        [6, 8]])
```
- Tensor의 shape이 같은 경우
```python
# mul (g*h, 요소곱)
g = torch.tensor([[1, 2],
                  [1, 3]])
h = torch.tensor([[2, 3],
                  [1, 4]])
i = torch.mul(g, h)	# or g.mul(h)
print(i)
```
```
tensor([[ 2,  6],
        [ 1, 12]])
```
- Tensor의 shape이 다른 경우
```python
# mul (g*h, shape이 다른 경우)
g = torch.tensor([[1, 4],
                  [7, 8]])
h = torch.tensor([2, 3])
i = torch.mul(g, h)	# or g.mul(h)
print(i)
```
```
tensor([[ 2, 12],
        [14, 24]])
```

### div()
**div_는 dtype에러에 유의하여야 합니다.** in-place 연산되는 Tensor의 dtype이 실수형인지 확인해야 합니다.

- Tensor의 shape이 같은 경우
```python
# div (j/k, 요소별 나누기)
j = torch.tensor([[18, 9],
                  [10, 4]]) 
k = torch.tensor([[6, 3],
                  [5, 2]])
l = torch.div(j, k)	# or j.div(k)
print(l)
```
```
tensor([[3., 3.],
        [2., 2.]])
```
- Tensor의 shape이 다른 경우
```python
# div (j/k, shape이 다른 경우)
j = torch.tensor([[12, 9],
                  [9, 4]]) 
k = torch.tensor([3, 2])
l = torch.div(j, k)	# or j.div(k)
print(l)
```
```
tensor([[4.0000, 4.5000],
        [3.0000, 2.0000]])
```

### pow()
- Tensor에 scalar 제곱
```python
# pow (m**n, 스칼라)
m = torch.tensor([[1, 2],
                  [3, 4]])
n = 2
o = torch.pow(m, n)	# or m.pow(n)
print(o)
```
```
tensor([[ 1,  4],
        [ 9, 16]])
```
- Tensor의 shape이 같은 경우
```python
# pow (m**n, 요소별 거듭제곱)
m = torch.tensor([[5, 4],
                  [3, 2]])
n = torch.tensor([[1, 2],
                  [3, 4]])
o = torch.pow(m, n)	# or m.pow(n) 분수로 거듭제곱하여 제곱근도 가능!
print(o)
```
```
tensor([[ 5, 16],
        [27, 16]])
```

### 비교연산
Tensor의 요소간 비교연산도 가능합니다! 각 Tensor에서 같은 index에 위치한 요소끼리 비교합니다. 비교하는 Tensor 둘은 shape이 같아야 합니다.
```python
p = torch.tensor([1, 3, 5, 7])
q = torch.tensor([2, 3, 5, 7])
r = torch.eq(p, q)	# or p.eq(q)
print(r)
```
```
tensor([False,  True,  True,  True])
```

- ``torch.eq()``: equal
- ``torch.ne()``: not equal
- ``torch.gt()``: greater than
- ``torch.ge()``: greater or equal
- ``torch.lt()``: less than
- ``torch.le()``: less or equal

### 논리연산
PyTorch에서 Tensor 간 논리 연산도 가능합니다. 

- ``torch.logical_and()``: AND
- ``torch.logical_or()``: OR
- ``torch.logical_xor()``: XOR

## Tensor의 Norm
``[1, 2, 3, 4, 5]``의 값을 가지는 1-D Tensor가 있습니다. 이 Tensor의 크기는 요소의 개수가 5개이기 때문에 5입니다. 그렇다면 이것으로 Tensor 간 크기를 비교할 수 있을까요? 그럴 수 없습니다. 요소의 개수가 다르다는 것은 차원의 크기가 다르다는 것입니다. 차원의 크기가 다른 것끼리 크기를 비교할 수는 없습니다. 그렇다면 Tensor를 어떤 방법으로 비교해야 할까요?

### Norm
노름은 Vector의 길이를 측정하는 방법으로 사용됩니다. 노름은 1-D Tensor의 노름에는 L1 노름, L2 노름, L∞ 노름 등 여러가지 유형의 노름이 있습니다. 

- L1 Norm(맨해튼 노름): 각 요소의 절대값의 합
- L2 Norm(유클리드 노름): 각 요소의 제곱합의 제곱근
- L∞ Norm: 요소의 절대값 중 최대값

PyTorch에서는 각각
- L1 Norm: ``torch.norm(a, p=1)``
- L2 Norm: ``torch.norm(a, p=2)``
- L∞ Norm: ``torch.norm(a, p=float(‘inf’))``
으로 계산할 수 있습니다.

### 유사도 (Similarity)
유사도(Similarity)는 두 1-D Tensor가 얼마나 유사한지에 대한 측정값을 의미합니다. 유사도는 Clustering에서 데이터들이 얼마나 유사한지를 판단하는 중요한 기준으로 사용됩니다. 유사도를 계산하는 방법은 여러가지가 있습니다.

### 맨해튼 유사도
맨해튼 유사도는 두 1-D Tensor의 **맨해튼 거리**를 역수로 변환하여 계산한 값입니다. 유사도가 1에 가까울수록 비슷한 값입니다.
```python
b = torch.tensor([1, 0, 2], dtype=torch.float32)
c = torch.tensor([1, 0, 2], dtype=torch.float32)

# 맨해튼 거리 계산 (L1 노름)
manhattan_distance = torch.norm(b-c, p=1)

# 맨해튼 유사도
manhattan_similarity = 1 / (1 + manhattan_distance)
```

맨해튼 거리에 1을 더해주는 이유는 분모가 0이 되는 것을 방지하기 위해서입니다.

### 유클리드 유사도
유클리드 유사도는 맨해튼 유사도와 비슷합니다. 맨해튼 거리 대신 **유클리드 거리**를 역수로 변환하여 계산합니다. 유클리드 유사도도 1에 가까울수록 비슷한 값입니다.
```python
# 유클리드 거리 계산 (L2 노름)
euclidean_distance = torch.norm(b-c, p=2)

# 맨해튼 유사도
euclidean_similarity = 1 / (1 + euclidean_distance)
```

### 코사인 유사도
코사인 유사도는 두 1-D Tensor 사이의 **각도**를 측정하여 계산합니다. 두 Tensor 사이의 각도 측정은 내적을 활용합니다. 
```python
cosine_similarity = torch.dot(b,c) / (torch.norm(b, p=2) * torch.norm(c, p=2))
```

## 2-D Tensor 곱셈 연산
2-D Tensor의 행렬 곱셈은 두 행렬을 결합하여 새로운 행렬을 생성합니다. 행렬 곱셈은 **신경망 구현에 핵심이 되는 연산**입니다.

행렬곱은 왼쪽 행렬의 행과 오른쪽 행렬의 열을 내적하여 새로운 행렬을 만듭니다. 행렬곱은 PyTorch로 다음과 같이 표현할 수 있습니다.
```python
D = torch.tensor([[1, 1, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
E = torch.tensor([[1, 0],
                  [1, -1],
                  [2, 1]])
F = D.matmul(E)
# 또는 F = D.mm(E)
# 또는 F = D @ E
```

행렬곱을 통해서 행렬의 대칭이동도 가능합니다.

```python
G = torch.tensor([[255, 114, 140],
                  [39, 255, 46],
                  [61, 29, 255]])
H = torch.tensor([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]])
I = G @ H	# 좌우 대칭 이동
print(I)
```
```
tensor([[140, 114, 255],
        [ 46, 255,  39],
        [255,  29,  61]])
```

그렇다면 행렬을 상하로 대칭 이동 하기 위해서는 어떤 행렬을 곱해야 할까요? 어떤 행렬을 곱해야할지 생각해봅시다.