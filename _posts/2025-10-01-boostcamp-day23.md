---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 23: NLP 이론, Embedding 시각화"
date: 2025-10-01 18:33:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, nlp]
description: "Natural Language Processing에 대해 배우자."
keywords: [colab, tokenization, embedding, pca, SNE, t-SNE]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Embedding 시각화

오늘은 과제 중 embedding을 시각화하는 PCA와 t-SNE에 대해 궁금한 점을 탐구하였습니다.
## PCA
- PCA 주성분 선택할 때에
  - PC1 (첫번째 축): 데이터 분산이 가장 큰 방향을 갖는 1차원 축
  - PC2 (두번째 축): PC1에서 직교하는 방향 중 두 번째로 분산이 큰 방향을 갖는 1차원 축
  - PC1을 가로축, PC2를 세로축으로 하는 평면에 Projection

## t-SNE
- SNE를 t-분포에 적용한 것
- 유사도를 조건부 확률로 바꿔서, 유사도를 나타내겠다는 SNE를 개선한 방법
- 고차원에서 데이터 간 유사도 분포와 저차원에서 유사도 분포 간의 차이를 최소화
- Cost function을 저차원 데이터의 좌표값으로 미분하여 gradient 계산 -> 최적화
- [KL-Divergence](https://lnemo.github.io/posts/boostcamp-day8/#kl-divergence)를 Cost Function으로 사용

### 순서
1. 한 데이터 포인트에서 다른 모든 점들과의 유클리드 거리를 계산
2. 두 데이터 포인트 사이의 거리가 d일 확률을 다음의 조건부 확률로 나타낸다  
   ![이미지](/assets/img/posts/boostcamp/day23/t-sne2.png)
3. `p(i|j)`와 `p(j|i)`는 서로 다르기 때문에 대칭성 확보를 위해 두 데이터 포인트 사이의 거리를 다음과 같이 정의한다  
   ![이미지](/assets/img/posts/boostcamp/day23/t-sne3.png)
4. 저차원에서 분포를 t-분포로 가정한다. 따라서 데이터 포인트 간 거리를 다음과 같이 나타낸다. (SNE는 가우시안 분포로 가정)  
   ![이미지](/assets/img/posts/boostcamp/day23/t-sne4.png)
5. 고차원에서 데이터 포인트 간의 거리를 나타내는 분포 p를 제일 잘 표현하는 저차원 분포 q를 찾는 것이기 때문에 p와 q 분포의 KL-Divergence가 작아지도록 학습(Gradient Descent)  
   ![이미지](/assets/img/posts/boostcamp/day23/t-sne5.png)
   1. 한 포인트(A)에서 다른 포인트(B)에 대해 실제 조건부 확률과 예측 조건부 확률의 차이를 계산
   2. 두 포인트 사이(A, B)의 거리 벡터에 해당 차이를 곱
   3. 위의 과정을 다른 모든 포인트들(C, D, …)에 시행하고 더해줌 -> 해당 포인트(A)가 이동해야할 거리 벡터
   4. 모든 포인트의 경우를 다 구한 다음 한번에 이동 <- 한 단계 완료