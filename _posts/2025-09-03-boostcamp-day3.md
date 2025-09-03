---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 3: Linear Regression"
date: 2025-09-03 19:00:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, pytorch, 부스트캠프, linear regression, 선형회귀]
description: "선형 회귀 분석이 무엇인지 이해하고 직접 모델을 적용해보자."
keywords: [boostcamp, 부스트캠프, ai, pytorch, torch, tensor, colab, linear regression, 선형회귀, gradient]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Linear Regression

Linear Regression(선형 회귀)는 주어진 트레이닝 데이터를 사용하여 **특징 변수**와 **목표 변수**간의 선형 관계를 분석하고, 이를 바탕으로 모델을 학습시켜 트레이닝 데이터에 포함되지 않은 새로운 데이터의 겨로가를 연속적인 숫자 값으로 예측하는 과정입니다.

## 선형 회귀 모델
### 데이터셋
여기에서는 Kaggle의 데이터셋을 사용하였습니다. 구글 드라이브에 kaggle.json 파일을 넣고 다음을 실행하면 데이터셋을 다운로드 받을 수 있습니다.
```python
from google.colab import drive
drive.mount('/content/drive')	# 드라이브를 마운트 하지 않고, files.upload()로 업로드 하는 방법도 있음

!mkdir -p ~/.kaggle
!cp '/content/drive/MyDrive/<드라이브의 kaggle.json의 경로>' ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d abhishek14398/salary-dataset-simple-linear-regression
!unzip salary-dataset-simple-linear-regression.zip
```

다운로드받은 데이터를 데이터를 불러오는 코드는 다음과 같습니다.

```python
import pandas as pd	

data = pd.read_csv("Salary_dataset.csv", sep=',', header=0) # 0번째행 header
print(data)
```
```
    Unnamed: 0  YearsExperience    Salary
0            0              1.2   39344.0
1            1              1.4   46206.0
2            2              1.6   37732.0
3            3              2.1   43526.0
4            4              2.3   39892.0
5            5              3.0   56643.0
6            6              3.1   60151.0
7            7              3.3   54446.0
8            8              3.3   64446.0
9            9              3.8   57190.0
10          10              4.0   63219.0
11          11              4.1   55795.0
12          12              4.1   56958.0
13          13              4.2   57082.0
14          14              4.6   61112.0
15          15              5.0   67939.0
16          16              5.2   66030.0
17          17              5.4   83089.0
18          18              6.0   81364.0
19          19              6.1   93941.0
20          20              6.9   91739.0
21          21              7.2   98274.0
22          22              8.0  101303.0
23          23              8.3  113813.0
24          24              8.8  109432.0
25          25              9.1  105583.0
26          26              9.6  116970.0
27          27              9.7  112636.0
28          28             10.4  122392.0
29          29             10.6  121873.0
```
### 
선형 회귀 모델에 사용하기 위해 데이터에서 특징 변수와 목적 변수를 분리해줍니다.

```python
x = data.iloc[:,1].values	# 특징 변수. 두 번째 열 (YearExperience)
t = data.iloc[:,2].values	# 목적 변수. 세 번째 열 (Salary)
print(x)
print(t)
```
```
[ 1.2  1.4  1.6  2.1  2.3  3.   3.1  3.3  3.3  3.8  4.   4.1  4.1  4.2
  4.6  5.   5.2  5.4  6.   6.1  6.9  7.2  8.   8.3  8.8  9.1  9.6  9.7
 10.4 10.6]
[ 39344.  46206.  37732.  43526.  39892.  56643.  60151.  54446.  64446.
  57190.  63219.  55795.  56958.  57082.  61112.  67939.  66030.  83089.
  81364.  93941.  91739.  98274. 101303. 113813. 109432. 105583. 116970.
 112636. 122392. 121873.]
```

### 상관 관계 분석

특징 변수와 목표 변수 간의 선형 관계를 파악하기 위해 상관 관계를 분석합니다. 상관 관계를 분석하면
1. 두 번수 간에 선형 관계를 파악할 수 있고
2. 그 관계가 양의 관계인지 또는 음의 관계인지 알 수 있으며
3. 높은 상관 관계를 가지는 특징 변수들을 파악할 수 있습니다.

상관 관계를 분석하는 수식 표현은 다음과 같습니다.
![이미지](/assets/img/posts/boostcamp/day3/coef.png)
_표본 상관 계수_

위의 식을 코드로 계산한다면,
```python
import numpy as np

correlation_matrix = np.corrcoef(x, t)
correlation_coefficient = correlation_matrix[0, 1]

print(correlation_matrix)

print('Correlation Coefficient between YearsExperience and Salary :', correlation_coefficient)
```
```
[[1.         0.97824162]
 [0.97824162 1.        ]]
Correlation Coefficient between YearsExperience and Salary : 0.9782416184887599
```

2x2의 상관 관계 행렬이 나옵니다. [0, 0], [1, 1]은 자기 자신과의 상관 관계를 나타내므로 1로 표현됩니다. [0, 1]은 x와 t의 상관 계수를 나타내기 때문에 여기에서 YearExperience와 Salary의 상관 계수는 약 0.9782 라는 것을 알 수 있습니다. 

상관 관계의 정도는 절대값이 1에 가까울수록 높고, 0에 가까울수록 낮다고 할 수 있습니다. YearExperience와 Salary의 상관 관계는 높다고 볼 수 있겠네요.

상관 관계를 시각화 해서 확인할 수도 있습니다. 아래의 코드는 x와 t의 관계를 산점도 그래프로 시각화합니다.

```python
import matplotlib.pyplot as plt

plt.scatter(x, t)
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('YearsExperience vs Salary')
plt.grid(True)
plt.show()
```
![이미지](/assets/img/posts/boostcamp/day3/graph.png)
_출력 그래프_

### 선형 회귀 모델에서의 학습

우선, 선형 회귀 모델을 학습하기 전에 PyTorch에 데이터를 사용할 것이기 때문에 Numpy 배열을 Tensor로 변환합니다. 이 때 Tensor를 2-D Tensor로 생성합니다. 나중에 다중 선형 회귀 모델 구축에서는 특징 변수가 여러개일 수도 있기 때문에 2차원 구조로 생성합니다.

```python
x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)	# 2차원으로 변경. 1열(행은 -1로 알아서 계산)
t_tensor = torch.tensor(t, dtype=torch.float32).view(-1, 1)

print('x_tensor =', x_tensor[:5,:])
print('t_tensor =', t_tensor[:5,:])
```
```
x_tensor = tensor([[1.2000],
        [1.4000],
        [1.6000],
        [2.1000],
        [2.3000]])
t_tensor = tensor([[39344.],
        [46206.],
        [37732.],
        [43526.],
        [39892.]])
```

선형 회귀 모델에서 학습은 주어진 트레이닝 데이터의 특성을 가장 잘 표현할 수 있는 직선 y = wx + b의 기울기(가중치) w와 y절편(bias) b를 찾는 과정을 의미합니다. 

신경망 관점에서는 선형 회귀 모델이 특징 변수가 출력층의 예측 변수로 mapping(사상)되는 과정이라고 할 수 있습니다. 입력층에 있는 특징 변수가 각각 하나의 뉴런에 대응하고, 각 뉴런들은 가중치와 바이어스를 통해 출력층과 연결되게 됩니다.
![이미지](/assets/img/posts/boostcamp/day3/nn.png)
_신경망 관점에서의 선형 회귀 모델_

이제 PyTorch에서 선형 회귀 모델 구축을 해보겠습니다. PyTorch에서 선형 회귀 모델을 구축할 때에 ``nn.Module``에서 클래스를 상속받아 생성할 수 있습니다. ``nn.Module``은 신경망의 모든 계층을 정의하기 위해 사용되는 기본 클래스입니다. 해당 클래스를 사용함으로써 일관성과 모듈화 등등의 여러 장점들이 있습니다. 

```python
import torch.nn as nn	# neural network

class LinearRegressionModel(nn.Module):	# nn.Module을 상속
    def __init__(self):
		# 생성자
        super(LinearRegressionModel, self).__init__()	# nn.Module의 생성자 실행
        self.linear = nn.Linear(1, 1)	# 입력과 출력이 모두 1인 선형 회귀 모델

    def forward(self, x_tensor):
		# 순전파
        y = self.linear(x_tensor)	# 입력 데이터를 선형 계층을 통해 예측값 계산
        return y
```

``nn.Module``을 활용해 ``LinearRegressionModel`` 클래스를 작성했습니다. 이제 해당 클래스로 인스턴스를 하나 만들고, GPU를 지원하기 위한 코드를 작성하겠습니다.

```python
model = LinearRegressionModel()	# 인스턴스 생성

# GPU 지원
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
x_tensor = x_tensor.to(device)
t_tensor = t_tensor.to(device)
```

여기까지 만들었다면 이제 모델을 학습을 시킬 차례입니다. 선형 회귀 모델에서 학습은 w(가중치)와 b(바이어스)를 찾는 과정입니다.

예측 선형 그래프와 데이터 값이 있을때 예측값과 실제 데이터와의 차가 크면 제대로 예측하고 있다고 할 수 없습니다. 우리는 이 간격을 줄여나가야 합니다. 그리고 이 간격을 오차(편차)라고 합니다. 따라서 **오차의 총합이 최소**가 되도록 하는 y=wx+b를 찾아야 하는 것입니다.

이 문제를 해결하기 위해 모든 오차를 제곱하여 더한 값 또는 절대값하여 더한 값을 사용합니다. 이 값들은 오차가 클수록 더 큰 값들을 가지게 될 것입니다. 또한 제곱한 값들은 오차가 클수록 훨씬 큰 값이 더해지게 되니 학습이 더 수월해질 수도 있습니다. 이 값들을 사용하여 손실함수(Loss Function)를 구할 수 있습니다.

오차를 제곱하여 더한 후 평균을 구하는 손실 함수는 **평균 제곱 오차**(Mean Squared Error, MSE)라고 합니다. 수식으로는 다음과 같습니다.
![이미지](/assets/img/posts/boostcamp/day3/mse.png)

코드로는 다음과 같습니다.
```python
loss_function = nn.MSELoss()
```

손실 함수의 값은 다음의 의미를 가집니다.
- 손실 함수의 값이 크다 -> 목표 변수와 예측 변수의 평균 오차가 크다
- 손실 함수의 값이 작다 -> 목표 변수와 예측 변수의 평균 오차가 작다

따라서 우리는 이 손실함수가 **최소**가 되도록 하는 w와 b를 구한다면 가장 최적의 직선을 구할 수 있습니다. 그렇다면 어떻게 손실함수가 최소가 되는 지점을 구할 수 있을까요?

### 경사하강법

경사하강법(Gradient Descent Algorithm, GDA)은 머신러닝의 최적화(Optimization) 알고리즘 중 하나입니다. 경사하강법을 통해 모델의 가중치와 바이어스의 최적값을 찾을 수 있습니다.

경사하강법에서 가중치의 값을 찾아나가는 과정은 쉽게 설명 가능합니다. 손실함수의 그래프에서 최소값은 0의 기울기를 가지고 있습니다. 따라서 가중치에 기울기만큼을 빼준다면 최소값에 더 가까워질 수 있습니다. (기울기가 양수일 경우, 왼쪽으로 가야 최소값에 가까워짐, 기울기가 음수일 경우는 오른쪽으로 가야 최소값에 가까워짐)

w에 대한 loss의 기울기는 다음과 같이 구할 수 있습니다.
![이미지](/assets/img/posts/boostcamp/day3/grad.png)

바이어스도 기울기와 동일하게 loss를 b로 편미분하여 계산할 수 있습니다.

학습률은 머신러닝과 신경망에서 매우 중요한 하이퍼 파라미터입니다. 학습률에 의해서 가중치가 기울기에 대해 얼마나 업데이트 되는지를 정해줍니다. 조정해가며 최적의 학습률을 찾는 것도 중요합니다.

```python
import torch.optim as optim

loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 학습률을 적절히 설정
```

경사하강법은 단점도 있습니다. 경사하강법은 전체 데이터셋을 사용하여 가중치와 바이어스를 구하기 때문에 데이터셋의 크기가 커질수록 계산 비용도 커지게 됩니다. 또한 최소값이 아닌 극소값에 빠지는 경우도 있습니다(local minima). 이런 문제점의 대안으로 **확률적 경사하강법**이 등장했습니다.

### 확률적 경사하강법
확률적 경사하강법은 모든 데이터를 사용하기보다 각각의 데이터 포인트마다 오차를 계산하여 가중치와 바이어스를 계산합니다. 전체를 계산하지 않기 때문에 큰 데이터셋에 효율적이고 극소값을 탈출하고 최소값에 도달할 수 있습니다. 

### 에폭
에폭은 모델이 전체 데이터셋을 한 번 완전히 학습하는 과정을 의미합니다. 데이터셋에 30개의 데이터가 있을 경우 에폭이 1이라면 30개의 데이터를 한 번만 학습합니다. 에폭의 수가 많을수록 더 많은 학습을 해서 모델의 성능이 향상될 수 있지만 과적합의 우려도 있습니다.

```python
num_epochs = 1000
loss_list = []

for epoch in range(num_epochs):
    y = model(x_tensor)
    loss = loss_function(y, t_tensor)

    # 확률적 경사하강법
    optimizer.zero_grad()	# 이전 단계의 기울기를 0으로 초기화
    loss.backward()			# 현재 loss에 대한 기울기 계산(역전파)
    optimizer.step()		# 계산된 기울기를 사용하여 가중치 업데이트

    # 손실 값을 저장
    loss_list.append(loss.item())

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

        # 디버깅 정보 출력
        for name, param in model.named_parameters():
            print(f'{name}: {param.data}')
```
```
Epoch [100/1000], Loss: 80263432.0
linear.weight: tensor([[11682.8672]], device='cuda:0')
linear.bias: tensor([9634.2715], device='cuda:0')
Epoch [200/1000], Loss: 52769136.0
linear.weight: tensor([[10929.0928]], device='cuda:0')
linear.bias: tensor([14770.1240], device='cuda:0')
.
.
.
linear.weight: tensor([[9532.7461]], device='cuda:0')
linear.bias: tensor([24284.1523], device='cuda:0')
Epoch [1000/1000], Loss: 31300502.0
linear.weight: tensor([[9504.8008]], device='cuda:0')
linear.bias: tensor([24474.5586], device='cuda:0')
```


### 데이터 표준화
위에서 학습을 진행하였을 때 loss가 너무 크다는 것을 확인할 수 있습니다.(학습이 완료되었을때 loss가 31300502…) 이 loss의 크기를 줄이기 위해 데이터를 표준화하여 전처리할 수 있습니다. 특징 변수와 목표 변수 값의 차이가 클 때 두 변수의 평균을 0, 분산을 1로 맞추어 표준화합니다.

```python
from sklearn.preprocessing import StandardScaler

scaler_x = StandardScaler()
x_scaled = scaler_x.fit_transform(x.reshape(-1, 1))

scaler_t = StandardScaler()
t_scaled = scaler_t.fit_transform(t.reshape(-1, 1))
```

위처럼 표준화 한 후에 학습을 진행한다면 loss가 소수점 아래로 줄어든 것을 볼 수 있습니다.


--- 

**피어세션을 통해 알아간 것**
- PyTorch의 역전파 과정에서 loss과 기울기 전달이 어떻게 되는지?
  -> ``loss.backward()``가 기울기를 계산하면 **각 파라미터마다** 기울기가 저장됨. optimizer는 그것을 업데이트 하는 역할
- 데이터 평탄화할 때 ``reshape()``를 사용해도 되는지?
  -> ``reshape()``는 ``view()``를 사용할 수 있는지 확인하고 ``view()``가 가능하다면 ``view()``로 보여준다. 때문에 새로 복사하여 생성하는 ``flatten()``보다 효율이 좋다.
- ``nn.Module``을 상속받은 클래스의 생성자에 ``super(LinearRegressionModel,self).__init__``을 사용하는 이유?
  -> 부모의 생성자도 사용하려고
- scaling을 전체로 진행할 경우, 왜 데이터 누수인지?
  -> 전체 데이터로 스케일링을 하면 테스트 데이터의 정보가 같이 학습에 들어가게 되므로
- 경사하강법에서 loss function의 최소치만 찾으면 되는데 scaler를 쓰는 이점이 불확실한 것 같다. 사용하는 이유는 무엇인지?
  -> 와 y의 스케일 차이가 크면 loss function의 그래프가 급격한데 반해, scaler를 쓰면 loss function의 그래프 형태가 고르고 완만해져서 안정적인 기울기를 가지고 학습이 가능