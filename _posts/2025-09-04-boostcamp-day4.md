---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 4: Binary Classification"
date: 2025-09-04 19:00:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, pytorch, 부스트캠프, classification, preprocessing]
description: "이진 분류 모델이 무엇인지 이해하고 직접 모델을 적용해보자."
keywords: [pytorch, torch, tensor, colab, linear regression, 선형회귀, gradient, dataloader, sigmoid, classification, BCE, cross entropy loss, 조건부 확률, 최대 가능도 추정, MLE, 딥러닝, 머신러닝]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Binary Classification

이전에 만들었던 선형 회귀 모델(Linear Regression)의 테스트를 진행하고 다음으로 넘어가보도록 하겠습니다.

테스트는 training data에 포함되지 않은 새로운 데이터의 결과를 연속적인 숫자 값으로 예측하는 과정입니다.

```python
import numpy as np

# 테스트 데이터 예측
def predict_test_data(test_data):
    # 테스트 데이터 표준화
    test_scaled = scaler_x.transform(test_data.reshape(-1, 1))  # 표준화
    test_tensor = torch.tensor(test_scaled, dtype=torch.float32).view(-1, 1).to(device) # 텐서로 전환

    # 모델을 사용하여 예측
    model.eval()    # 평가모드로 전환. model.train()과 반대
    with torch.no_grad():   # gradient를 계산하지 않음
        predictions_scaled = model(test_tensor)

    # 표준화 해제
    predictions = scaler_t.inverse_transform(predictions_scaled.cpu().numpy())
    return predictions

test_years_experience = np.array([1.0, 2.0, 7.0])   # 1년, 2년, 7년 경력
predicted_salaries = predict_test_data(test_years_experience)

# 결과
for YearsExperience, salary in zip(test_years_experience, predicted_salaries):
    print(f'YearsExperience: {YearsExperience}, Predicted Salary: {salary[0]:.0f}')	
```
```
YearsExperience: 1.0, Predicted Salary: 34298
YearsExperience: 2.0, Predicted Salary: 43748
YearsExperience: 7.0, Predicted Salary: 90998
```

이제 Binary Classification으로 넘어가보도록 하겠습니다.

---
## 이진 분류 모델

이진 분류란 특징 변수와 목표 변수 사이의 관계를 학습하여 트레이닝 데이터에 포함되지 않은 새로운 데이터를 사전에 정의된 두 가지 범주 중 하나로 분류하는 것입니다.

이진 분류의 예는 다음과 같습니다:
- 붓꽃(iris)의 종류 분류 (Iris-versicolor/Iris-setosa)
- 이메일 스팸 분류 (Spam/Ham)
- 금융 사기 탐지 (사기 거래/정상거래)
- 의료 진단 (암 조직/정상 조직)

이제 이진 분류 모델의 학습 과정을 순차적으로 알아보겠습니다.

### 이진 분류 모델의 트레이닝 데이터

이번에도 Kaggle에서 데이터를 가져오도록 하겠습니다.

```shell
kaggle datasets download -d uciml/iris
unzip iris.zip
```
```python
df = pd.read_csv('Iris.csv', sep=',', header=0)
```

가져온 Iris 데이터에서 ``['PetalLengthCm', Species]`` 열만 따로 저장해줍니다. 각각 특징 변수와 목표 변수로 사용할 것입니다.

```python
df = pd.read_csv('Iris.csv', sep=',', header=0)[['PetalLengthCm', 'Species']]
```

Iris의 종류 중에 Iris-setona와 Iris-versicolor 만 가져와서 두 꽃의 분류를 학습할 것입니다. 

```python
filtered_data = df[df['Species'].isin(['Iris-setosa', 'Iris-versicolor'])]

filtered_df = filtered_data
print(filtered_df)
```
```

    PetalLengthCm          Species
0             1.4      Iris-setosa
1             1.4      Iris-setosa
2             1.3      Iris-setosa
3             1.5      Iris-setosa
4             1.4      Iris-setosa
..            ...              ...
95            4.2  Iris-versicolor
96            4.2  Iris-versicolor
97            4.3  Iris-versicolor
98            3.0  Iris-versicolor
99            4.1  Iris-versicolor

[100 rows x 2 columns]
```

학습을 위해 임의로 Iris_setosa는 0, Iris-versicolor는 1로 지정해줍니다. 그리고 특징 변수는 표준화할 때에 2차원 배열을 요구하기 때문에 2차원 배열로 바꾼 다음 데이터를 나누고 표준화하도록 하겠습니다.

```python
filtered_df.loc[:,'Species'] = filtered_df['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1})

x = filtered_df[['PetalLengthCm']].values   # 2차원 배열로 변환 (표준화할때 2차원 배열을 요구하기 때문)
t = filtered_df['Species'].values.astype(int)

from sklearn.model_selection import train_test_split

x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.2, random_state=42)	# 8:2로 train과 test 데이터를 나눔

from sklearn.preprocessing import StandardScaler
# sclaer로 표준화
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # x_train에 fit하고 transform
x_test = scaler.transform(x_test)		# fit된 scaler로 transform

```

이제 학습을 위해 데이터들을 Tensor로 바꿔줍니다. 이때 목표 변수의 Tensor를 2차원으로 확장하기 위해 ``unsqueeze(-1)``을 사용합니다. 다음의 이유로 2차원 Tensor를 사용하기 때문입니다.
- 특징 변수가 이미 2차원
- 배치 처리를 위해서
- 손실 함수도 2차원 Tensor 형태를 기

```python
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
t_train = torch.tensor(t_train, dtype=torch.float32).unsqueeze(-1)  
t_test = torch.tensor(t_test, dtype=torch.float32).unsqueeze(-1)
```

### Dataset & DataLoader 클래스

배치(Batch)란, 머신러닝과 딥러닝에서 데이터를 처리하는 묶음 단위를 의미합니다. 이러한 배치는 미니 배치 경사하강법에서 사용 가능합니다. 미니 배치 경사하강법은 기존의 확률적 경사하강법보다 노이즈를 줄일 수 있고 전체 데이터셋을 한 번에 사용하는 경사하강법보다 계산 속도가 빠릅니다.

Pytorch에서는 데이터의 전처리와 배치 처리를 용이하게 할 수 있도록 Dataset과 DataLoader 클래스를 사용합니다. Dataset 클래스는 데이터셋을 정의하는 기본 클래스로서, Dataset을 상속받아 사용자 정의 데이터셋을 만들 수 있습니다. Dataset 클래스는 ``__init__``, ``__len__``, ``__getitem__`` 메서드로 구성되어 있습니다. CustomDataset은 아래와 같이 구성할 수 있습니다. (간단한 Dataset은 TensorDataset을 사용하여도 됩니다.)

```python
from torch.utils.data import Dataset, DataLoader

class IrisDataset(Dataset): # CustomDataset
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
```

DataLoader 클래스는 Dataset의 인스턴스를 감싸서 배치 단위로 데이터를 로드하고 데이터 셋을 섞는 작업을 수행합니다. 모델 훈련 시에는 데이터 순서에 따른 편향을 줄이기 위해 데이터를 섞고, 모델 성능 평가 시에는 데이터 순서를 유지하는 것이 일반적이므로 데이터를 섞지 않습니다.

```python
train_dataset = IrisDataset(x_train, t_train)
test_dataset = IrisDataset(x_test, t_test)

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)	# train_loader는 데이터를 섞음
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)	# test_loader는 섞지 않음
```

### 로지스틱 회귀

로지스틱 회귀란 트레이닝 데이터의 특성과 분포를 바탕으로 데이터를 잘 구분할 수 있는 **최적의 결정 경계**를 찾아 **시그모이드**(Sigmoid) 함수를 통해 이 경계를 기준으로 데이터를 이진 분류하는 알고리즘입니다. 이러한 로지스틱 회귀는 이진 분류 알고리즘 중에서도 성능이 뛰어난 것으로 알려져 있으며, 딥러닝의 기본적인 구성요소로 널리 사용됩니다.

위에서 말하는 최적의 결정 경계를 구하기 위해서는 선형 결정 경계를 구해야 합니다. 해당 선형 결정 경계를 시그모이드 함수를 통해 0과 1사이의 값으로 반환합니다. 따라서 해당 과정을 식으로 나타내면 ``y=sigmoid(wx+b)``가 됩니다.

아래는 해당 과정을 정의한 BinaryClassificationModel 클래스입니다. 해당 분류 모델의 인스턴스를 model로 만들어주겠습니다.

```python
import torch.nn as nn

class BinaryClassificationModel(nn.Module):
    def __init__(self):
        super(BinaryClassificationModel, self).__init__()
        self.layer_1 = nn.Linear(1, 1)	# 입력차원 1, 출력차원 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.layer_1(x)
        y = self.sigmoid(z)
        return y

model = BinaryClassificationModel()
```

### 이진 교차 엔트로피 (Binary Cross Entropy)

Linear Regression과 마찬가지로 Binary Classsification 모델도 Loss를 최소로 만드는 역전파의 과정이 있어야 모델을 학습할 수 있습니다. Sigmoid 함수가 추가된 이진 분류 모델은 선형 회귀 모델과는 다른 Loss 함수를 가질 것입니다.

**이진 교차 엔트로피**(Binary Cross Entropy)는 이진 분류 문제에서 모델의 예측 변수와 목표 변수 간의 차이를 측정하기 위해 사용되는 손실 함수입니다. 이진 교차 엔트로피를 어떻게 구해냈는지 차근차근 알아봅시다.

**최대 가능도 추정**(Maximum Likelihood Estimation)은 주어진 데이터셋에 대해 모수를 추정하는 방법론 또는 절차입니다. 이는 데이터를 잘 설명하는 모수를 찾기위해 likelihood 함수를 최대화 하는 과정입니다. Estimation은 관찰된 데이터를 사용하여 모수의 값을 추정하는 과정입니다. Likelihood는 주어진 데이터가 특정 모수 값 하에서 관찰될 확률을 의미합니다. 따라서 우리는 “관찰치가 가정한 확률 분포에 따라 특정 모수 값으로 설명될 **가능도를 최대화**” 하는 방법을 사용합니다.

가능도 함수 L(θ; x=1)는 P(x=1 \| θ)로 나타낼 수 있습니다. 따라서 여러 데이터에 대한 가능도 함수의 수식표현은 L(θ;X) = Π P(xi \| θ) 로 표현합니다. 

입력값 x에 대하여 출력이 1일 확률은 다음과 같이 나타낼 수 있습니다 

**P(T=1 \| x) = y = sigmoid(Wx+b)**

또한, 출력이 0일 확률은 전체 확률에서 1일 확률을 뺀것과 같으니 다음과 같이 나타낼 수 있습니다. 

**P(T=0 \| x) = 1 - P(T=1 \| x) = 1 - y**

이 두 수식을 하나로 표현한다면 다음과 같이 나타낼 수 있습니다.

**P(T=t \| x) = y^t * (1-y)^(1-t)**

위 식에서 t가 1일 경우에는 (1-y)^(1-t) 부분이 1이 되기 때문에 y만 남고, t가 0일 경우에는 y^t부분이 1이 되기 때문에 앞서 나온 두 식과 동일하다고 볼 수 있습니다. 따라서 각각의 데이터는 위의 수식을 만족하기 때문에 모든 데이터에 대하여 곱한 Π P(T=ti \| xi)가 바로 가능도 함수입니다. 

이 가능도 함수에 log와 음의 부호를 붙이게 되면 이것이 바로 이진 교차 엔트로피가 됩니다. (log는 곱을 합으로 바꾸기 위해, 음의 부호는 최대를 구하는 것을 최소를 구하는 것으로 바꾸기 위해)

![이미지](/assets/img/posts/boostcamp/day4/mle.png)
_Loss_MLE 식_

```python
import torch.optim as optim

loss_function = nn.BCELoss()	# 이진 교차 엔트로피를 loss_function으로 설정
optimizer = optim.SGD(model.parameters(), lr=0.01)	# 미니 배치 GD로 진행
```

loss_function과 optimizer 까지 설정해주었으니, 이제 에폭을 설정하고 학습을 진행합니다.

```python
num_epochs = 500
loss_list = []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()
        output = model(batch_features)
        loss = loss_function(output, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    loss_list.append(epoch_loss / len(train_loader))

    if(epoch+1)%100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
```

아래 코드로 loss의 경향을 파악할 수 있습니다.

```python
import matplotlib.pyplot as plt

plt.figure()
plt.plot(loss_list, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.title('Loss Trend')
plt.show()
```
![이미지](/assets/img/posts/boostcamp/day4/loss_trend.png)

### 테스트

이제 훈련에 포함되지 않았던 새로운 데이터를 모델에 넣어서 사전에 정의된 두 가지의 범주 중 하나로 분류가 되는지 확인해봅니다.

```python
model.eval()	# model을 eval 모드로 변경
with torch.no_grad():
    predictions = model(x_test)
    predicted_labels = (predictions > 0.5).float()

actual_labels = t_test.numpy()
predicted_labels = predicted_labels.numpy()

print("Predictions:", predicted_labels.flatten())
print("Actual Labels:", actual_labels.flatten())
```
```
Predictions: [1. 1. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0. 1. 0. 1. 0. 1. 1. 0. 0.]
Actual Labels: [1. 1. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0. 1. 0. 1. 0. 1. 1. 0. 0.]
```

테스트한 데이터에서는 모두 일치하는 것을 볼 수 있습니다. 아래 코드를 작성해 그래프로도 확인해 볼 수 있습니다.

```python
plt.figure()
plt.scatter(range(len(actual_labels)), actual_labels, color='blue', label='Actual Labels')
plt.scatter(range(len(predicted_labels)), predicted_labels, color='red', marker='x', label='Predicted Labels')
plt.xlabel('Sample Index')
plt.ylabel('Label')
plt.legend()    # 범례
plt.title('Actual vs Predicted Labels')
plt.show()
```

--- 

**피어세션을 통해 알아간 것**
- Tensor의 학습에 대해서 gradient는 어디에 존재하고 어떻게 미분하는지
  - 기본적으로 Tensor에는 .grad_fn .grad 속성이 있고, .grad_fn는 계산그래프, .grad가 gradient이다. 초기 생성단계에서는 이 두 속성은 비어있는 상태.
  1. forward: Tensor가 계산될때마다 .grad_fn에 해당 연산을 미분하는 계산식이 쌓임
  2. backward: .grad_fn의 계산그래프대로 계산을 수행하고. 해당 계산의 결과가 .grad에 저장
  3. backward가 실행되고 나면 .grad_fn의 계산그래프는 연결이 끊김 (메모리 해제)
  4. no_grad()로 실행한다면 .grad_fn에 계산 그래프 자체를 남기지 않는것
- PyTorch에서 optimizer로 존재하는 SGD는 실제로 SGD가 아님. 많은 딥러닝 프레임워크에서 편의상 MBGD를 SGD라고 부르는 관행이 있다고 함
- 랜덤 시드로 관행적으로 사용하는 24는 SF소설에서 유래됐다고..