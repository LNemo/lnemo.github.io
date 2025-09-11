---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 9: AI Math, Futher Question"
date: 2025-09-11 19:00:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, numpy, 확률론, 통계학]
description: "AI Math에 대해서 배우자."
keywords: [numpy, ai math, tensor, colab, L1, L2, norm, loss, lasso, ridge, 선형변환, 비선형변환, regularization]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# AI Math, Futher Question

오늘은 AI Math와 관련해 더 깊게 탐구해보겠습니다.
## Norm

### L1-Norm과 L2-Norm

- L1-노름과 L2-노름의 기하학적 차이를 이해하면서 L1-Loss와 L2-Loss의 차이를 살펴보자.
  * L1-Loss(MAE)를 사용하면 그래디언트가 오차의 크기와 상관없이 항상 +1, -1로 일정한 반면에 L2-Loss는 오차의 크기에 따라 그래디언트의 크기가 달라짐(절댓값 함수와 이차함수의 모양 생각!).
  * 따라서 L1-Loss는 최적점 근처에서 ‘진동 현상’이 발견될 수 있음. * 이 때문에 학습이 진행될 때마다 학습률을 줄여주는 기법을 함께 사용함.

### Lasso? Ridge?

- Lasso 회귀와 Ridge 회귀가 뭘까?
  * Lasso 회귀(L1 Regularization)
    * **가중치의 절댓값 합** (``λ * Σ|w|``)을 Loss에 더해줌
    * 이 페널티 방식은 중요하지 않은 가중치를 아예 **0으로 만들어** 특정 특성을 모델에서 제외시키는 **특성 선택** 효과가 있음(마름모이기 때문에)
  * Ridge 회귀(L2 Regularization)
    * **가중치의 제곱합** (``λ * Σw²``)을 Loss에 더해줌
    * 이 페널티는 가중치를 0에 가깝게 만들지만 완전히 0으로 만들지는 않습니다. 모든 특성을 유지하되 그 영향력을 줄이는 역할(원이기 때문에)
  * L1, L2 회귀는 w에 대한 Norm 을 구하는 것. (Loss는 실제값과 예측값의 차의 Norm)
  * 가중치의 크기를 너무 크게하지 않도록 제한하는 역할(Overfitting 예방)

## 선형 회귀

### 선형 변환, 비선형 변환
* 선형 변환은 무엇이고 비선형 변환은 무엇인가?
  * 선형 변환은 공간을 휘거나 접지 않고, 늘이거나 줄이거나 회전시키는 변환
  * 비선형 변환은 공간을 휘게 하거나 접거나 특정 영역을 압축 팽창시킬 수 있음
  * 선형 변환은 예측 가능한 방식으로 공간 전체를 균일하게 바꾸지만 비선형 변환은 예측 불가능한 방식으로 공간을 국소적으로 왜곡 → 복잡한 데이터의 패턴을 학습 가능
  * 선형 예) Y=WX+b
  * 비선형 예) ReLU, Sigmoid, Tanh 등 (Activation Function)


### 표준 선형 회귀 + Ridge

* Ridge 회귀의 해가 ![이미지](/assets/img/posts/boostcamp/day9/sik.png)  형태로 유도되는 과정에서 규제(Ridge Regularization)와 있을 때와 없을 때를 비교한다면 수학적으로 보았을 때와 기하학적으로 보았을 때 어떤 차이가 있을까?
* 수학적으로는
  * 항상 역행렬이 존재하는 안정적인 행렬로 만들어줌
  * 다중공선성이 매우 높은 데이터에 대해 **XᵀX**의 determinant는 0에 매우 가까운 값이 나오고 0에 가깝다면 역행렬의 값들이 비정상적으로 커져 매우 불안정한 상태가 됨
  * Ridge의 경우에는 λ를 더해주므로 역행렬을 구할 수 있는 안정적인 행렬로 바꿀 수 있음
* 기하학적으로는
  * 해당 규제는 최소값의 경계를 정해주는 역할. Ridge 경계 안에서의 최소값을 구한다는 의미

---
