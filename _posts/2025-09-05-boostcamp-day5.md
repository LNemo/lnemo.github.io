---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 5: PyTorch 과제와 보완할 점"
date: 2025-09-05 19:00:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, pytorch, 부스트캠프, classification, preprocessing]
description: "1주차 과제 점검과 위클리미션 리뷰"
keywords: [pytorch, torch, tensor, colab, linear regression, 선형회귀, gradient, dataloader, sigmoid, classification, BCE, cross entropy loss, 조건부 확률, 최대 가능도 추정, MLE, 딥러닝, 머신러닝]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Week 1 Review 

오늘은 과제를 진행하면서 보완이 필요하거나 학습이 더 필요한 부분을 리뷰합니다.

## 과제

### 기본-1
- ``torch.fill(t, 1)``과 ``torch.ones_like(t)``의 차이는?
  - 기능적으로 거의 동일
  - 참고로 ``torch.fill()``은 PyTorch 공식문서에 없지만 사용할 수 있다.
- Tensor를 GPU에 올리고 싶다면``torch.Tensor.to(device=‘cuda’)``, ``torch.Tensor.cuda()``
- ``reshape()``는 ``reshape(1,2)`` 말고도 ``reshape([1,2])``도 된다.
### 기본-2
- ``model.parameters()``은 generator로 return하기 때문에 디버깅하기 위해서는 ``list()``로 씌워주는 작업 필요.
- ``model.parameters()``의 반환값은 업데이트 되는 w, b
### 기본-3
- dataframe에서 해당하는 열을 없앨 때에는 ``.drop([list], axis)``
- 한 열만 가져올때는 ``y = data.Purchased`` 처럼 가능
- 이진 분류에서 0.5보다 큰 예측값은 1로 변환하기 위해 ``y_pred>0.5`` 를 사용 -> 그 다음 예측 class와 목표 class 를 비교하여 ``sum().item()``
  - ``prediction = (y_pred>0.5).sum().item()``
- ``.item()``은 텐서를 꺼내는 용도

### 심화-1
- ``relu`` 의 다른 식: ``x.clamp(min=0)``, ``torch.where(x > 0, x, 0)``
- ``x.clamp()``와 ``x.clip()``은 동일
- Cross Entropy Loss에서 ``torch.gather(log_y_hat, dim=1, index=target)``은 ``log_y_hat``에서 ``target``의 element를 인덱스로 사용하여 가져옴
  - 수식에서는 결과값이랑 예측값이 곱해지는 건데 왜 결과 인덱스에 해당되는 값만 가져오는거지? 싶었는데 수식에서 표현하는 결과값은 one-hot 벡터로 표현되어 있어서... (네 종류 중 두번째 class로 분류된다 하면 ``[0, 1, 0, 0]``, 코드에서는 ``[1]``)
  - 그래서 ``log_y_hat = [-3.91, -0.48, -2.04, -1.47]`` 일 때 target의 값이 ``[3]`` 이라면 ``[-1.47]``을 가져옴

![이미지](/assets/img/posts/boostcamp/day5/cross_entropy1.png)

![이미지](/assets/img/posts/boostcamp/day5/cross_entropy2.png)

## Weekly Mission

### Feedback

- Dataset + Dataset 가능하다: 하나의 데이터셋으로
- 불리언인덱싱이란게 있다..
  ```python
  A=torch.tensor([[1,2],[3,4],[5,6],[7,8]])
  print(A[ torch.tensor([[False,True],[False,False],[False,False],[False,False]]) ]) 
                  # tensor([2]) 똑같은 크기의 Boolean Tensor로 True에 해당하는 것만 가져옴
  print(A[A==2])	# tensor([2]) A==2에 해당하는 것만
  print(A[ [True,False,False,False], [False,True] ])	# tensor([2]) 0행 1열
  ```
