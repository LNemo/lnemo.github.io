---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 12: ML LifeCycle, 기초 신경망 이론"
date: 2025-09-16 19:21:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, ml, 머신러닝, neural networks]
description: "머신러닝의 생애주기에 대해 배우자."
keywords: [numpy, colab, ML LifeCycle, regression, linear classifier, neural networks, 데이터, 전처리]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# 기초 신경망 이론

## Neural Networks

### Background
Linear Classifier는 강력하지 않습니다. 클래스 당 하나의 template만 학습할 수 있고, 기하학적으로 “직선”의 decision boundary만 그릴 수 있습니다. 이 때문에 복잡한 관계의 클래스는 분류할 수 없습니다.

이러한 문제를 해결하는 방법은 단순합니다. 복잡한 데이터 형태를 선형으로 만들어주면 됩니다. 다시 말하자면 Original Space를 우리가 선형으로 분리할 수 있는 Feature Space로 매핑합니다. (특징 추출, Featurization)

고전적인 특징 추출의 방법은 실제로 입력 데이터에서 여러 정보를 뽑아서 Rule-based로 해결하는 것입니다. 예를 들어 비둘기의 특징을 추출한다고 한다면 색상, 픽셀마다의 그래디언트의 방향 등을 뽑아냅니다. 결국에 input은 이 features가 되는 것입니다.


### Neural Network

고전적인 방식 말고 end-to-end의 모델을 구성하는 방법은 어떨까요? 하지만 어떻게 해야할까요?

#### Perceptron

**퍼셉트론**(Perceptron)은 인공신경망의 한 종류입니다. 하나의 퍼셉트론의 구조는 다음과 같이 이루어집니다.

1. x(input)
2. W(weight)와 X의 곱
3. Wx를 Activation Function에 통과 -> y(output)

이 구조에서 2번까지는 여전히 선형적입니다. 하지만 3번에서 활성화 함수를 통과시켜주면서 비선형으로 만들어주게 됩니다. 이러한 Perceptron이 여러 층으로 이루어져 있는 것을 다층 퍼셉트론(Multilayer Perceptron, MLP)이라고 합니다.

### Backpropagation

Gradient Descent를 위해서는 각 가중치에 대한 Loss의 Gradient가 필요합니다. 해당 Loss의 미분값은 Chain Rule로 구할 수 있습니다. 이렇게 예측 값부터 시작해서 미분 값을 구하는 과정을 역전파(Backpropagation)이라고 합니다.  
(예측 값을 구하는 과정은 정방향이기 때문에 순전파(Forward Pass)라고 부릅니다)

![이미지](/assets/img/posts/boostcamp/day12/backpropagation.png)

위는 순전파와 역전파의 간단한 예제입니다. 실제로는 여러 층이 있고 활성화 함수가 있으므로 더 복잡한 구조를 가지고 있습니다.

역전파를 따라가면서 직접 계산하다보면 어떤 규칙이 있는 것을 알 수 있습니다. 아래는 Gradient Flow의 규칙을 정리하였습니다.

![이미지](/assets/img/posts/boostcamp/day12/patterns.png)

* add: 그대로
* mul: 서로 바꿔서 곱
* copy: 갈라진 gradient 더하기
* max: 큰쪽에만 전달, 다른곳은 0

### Activation Function

활성화 함수(Activation Function)은 nonlinear한 함수를 어떻게 활성화 할 것인지 결정하는 과정에서 나온 개념입니다. 활성화 함수의 역할은 선형함수를 비선형으로 만드는 것입니다. 다양한 종류의 활성화 함수가 있습니다.

#### Sigmoid

Sigmoid Function은 앞서 확률로 변환하는 과정에서 사용한 적이 있습니다. Sigmoid는 예전에 활성화 함수로 가장 많이 사용되었습니다. 하지만 input이 너무 크거나 작다면 기울기가 0에 가까워지는 Vanishing Gradient 문제가 있습니다. 또한 zero-centered 하지 않으며 exp()의 연산이 비싸다는 단점이 있습니다. 이러한 문제 때문에 지금은 거의 사용되지 않습니다.

![이미지](/assets/img/posts/boostcamp/day12/sigmoid.png)

**Not Zero-centered Output이 왜 문제가 되는가?**

입력이 모두 양수라고 가정할 경우 모든 가중치에 대한 Upstream Gradient의 부호가 변하지 않습니다. 따라서 모든 Gradient는 모두 양수거나 모두 음수입니다. 이 경우에 업데이트를 비효율적인 방향으로 할 수 있습니다.(지그재그로 최적화)

#### Tanh

Tanh Function은 Sigmoid의 Not Zero-centered 문제를 해결한 함수입니다. [-1, 1]의 output을 가집니다. 하지만 여전히 input이 너무 크거나 작을 때 생기는 문제인 Vanishing Gradient가 여전히 발생합니다.

![이미지](/assets/img/posts/boostcamp/day12/tanh.png)

#### ReLU

ReLU(Rectified Linear Unit) 함수는 많은 장점을 가지고 있습니다. ``ReLU(x) = max(0,x)``로 표현합니다. 연산이 단순하므로 효율적이므로 Sigmoid나 Tanh보다 더 빨리 수렴합니다. 또한 양수일 때에는 input이 커져도 0이 되지 않습니다.

하지만 여전히 zero-centered 하지 않으며 음수 값을 완전히 무시해버리는 문제가 있습니다. 또한 x가 0일 때에 미분이 불가능합니다.

![이미지](/assets/img/posts/boostcamp/day12/relu.png)

#### Leaky ReLU

Leaky ReLU는 ReLU의 단점을 개선하기 위해 등장했습니다. Leaky ReLU의 식은 ReLU에서 음수 부분의 문제를 해결하기 위해 ``Leaky ReLU = max(0.01x, x)``으로 수정되었습니다. 따라서 ReLU의 장점은 가져가면서 Gradient Vanishing 문제까지 해결하였습니다.

![이미지](/assets/img/posts/boostcamp/day12/leakyrelu.png)

#### ELU

ELU(Exponential Linear Unit)은 ReLU에서 음수 부분을 exp()을 포함한 함수를 넣어서 보다 자연스럽게 그래프를 이어주었습니다. 이 함수는 ReLU의 모든 장점을 가지면서 Saturated된 음수 지역에 견고성을 더합니다. 하지만 exp()의 연산이 비싸다는 단점이 있습니다.

![이미지](/assets/img/posts/boostcamp/day12/elu.png)

### Weight Initialization

가중치는 어차피 학습되는 값이라고 생각하여 초기값을 중요하게 생각하지 않을 수도 있습니다. 하지만 가중치를 얼마나 잘 초기화하느냐도 얼마나 모델을 효율적으로 학습시킬 수 있는지를 판가름합니다.

#### Small Gaussian Random
- 표준 정규 분포를 작은 상수로 곱합니다.
- 얕은 신경망에서는 좋은 결과를 내며 널리 사용됩니다.
- x가 0에 가까워지면 모든 기울기가 0에 가까워져 학습이 되지 않을 것입니다.

#### Large Gaussian Random
- 표준 정규 분포를 큰 상수로 곱합니다.
- 너무 커지면 모든 기울기가 0에 가까워져서 학습되지 않을 것입니다.

#### Xavier Initialization
- 표준 정규 분포를 ``sqrt(d_in)``으로 나눠준다면 모든 레이어로부터 activation이 적당한 크기를 가집니다!

### Learning Rate Scheduling

Gradient를 원래 가중치에서 빼주기 전에 Learning Rate를 곱하여 빼주게 됩니다. 이 때에 학습률이 너무 높다면 오히려 Loss가 증가할 수 있고 학습률이 너무 낮다면 Loss가 매우 천천히 줄어들 것입니다. 따라서 적절한 Learning Rate의 설정도 필요합니다.

#### Learning Rate Decay

Learning Rate Decay는 처음에는 큰 학습률을 사용하다가 학습을 진행함에 따라 학습률을 점점 낮추는 방법입니다. 처음에는 최적해에서 멀리 떨어져 있을 것이기 때문에 큰 보폭으로 움직일 필요가 있고 학습할수록 최적해에 가까워지므로 정밀하게 찾기 위해 작은 보폭으로 움직이는 것입니다.

다음은 Initial Warmup과 Step을 합한 Learning Rate Decay 방법입니다.
- 학습률을 선형으로 증가시키다가 고정하고 50% 구간과 75% 구간에서 학습률에 0.1을 곱하여 줄입니다.

### Data Preprocessing

모델을 학습하기 전에 데이터를 전처리하는 것도 중요합니다. 데이터를 잘 정리하지 않을 경우 모델을 학습하는 과정에서 결과가 좋지 않은 문제가 생길 수 있습니다. 

**Zero-centering**은 모든 데이터에 대해서 평균만큼을 빼줍니다. 이 과정으로 데이터의 평균은 0을 가지게 됩니다. Zero-centering을 한다면 가중치의 작은 변화에 덜 민감하기 때문에 최적화가 쉬워집니다.

**Normalization**은 모든 데이터에 대해서 표준편차를 나누어줘서 1로 만듭니다. Neural Network는 기본적으로 정규분포를 가정하기 때문에 정규화를 하지 않으면 학습이 제대로 되지 않습니다.

**PCA**(Principal Component Analysis, 주성분 분석)는 가장 분산이 긴 축을 중심으로 zero-centered와 동시에 축을 정렬합니다. 

**Whitening**은 분산을 1로 만듭니다.

### Data Argumentation

Data Argumentation(데이터 증강)은 데이터가 많지 않을 때에 shift, box, tint 등의 데이터의 의미에 영향을 주지 않는 선에서 데이터를 수정하며 데이터의 수를 늘립니다. 

**Horizontal Flips**는 데이터를 좌우로 뒤집습니다. 강아지 사진의 경우 좌우로 뒤집어도 똑같이 강아지이기 때문에 두 경우가 모두 강아지라고 훈련시킬 수 있습니다. 하지만 수직으로 뒤집는 것은 의미가 왜곡될 수 있으므로 주의합니다.

**Random Crops**는 데이터의 일부만 크롭해서 사용합니다. 강아지 사진의 일부(다리, 얼굴, 몸통 등)만 보더라도 강아지라고 훈련시킵니다.

**Scaling**은 이미지 크기는 그대로인 상태로 사물의 크기를 다르게 합니다. 이미지의 크기에 관계 없이 물체가 인식되어야 하기 때문에 모든 경우를 훈련시킵니다.

**Color Jitter**는 색상을 일부 변형시킵니다. 빛이나 다른 요인들로 인해 색깔이 다르게 보일 수도 있기 때문에 이런 변화에 따라서도 모델이 영향받지 않는 것을 목표로 훈련시킵니다. 

