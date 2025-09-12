---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 10: AI Math 과제와 보완할 점"
date: 2025-09-12 18:25:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, numpy, 확률론, 통계학]
description: "2주차 과제 점검과 위클리미션 리뷰"
keywords: [numpy, ai math, tensor, colab, L1, L2, norm, loss, lasso, ridge, 선형변환, 비선형변환, regularization]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Week 2 Review 

오늘은 과제를 진행하면서 보완이 필요하거나 학습이 더 필요한 부분을 리뷰합니다.

## 과제

### 기본-1
- 고유값, 고유벡터를 가져오려면
  - ``eigenvalue, eigenvector = np.linalg.eig(number_matrix)``

### 기본-2
- einsum
  - ``np.einsum('ii->', array_2d)``: 대각 성분 모두 더할 때
- einops
  - ``einops.rearrange(array_4d, 'b i j k -> (b i)(j k)')``
- np.einsum의 규칙: "문자" 기반 + 위치 매핑 ('ij -> ji' )
- einops의 규칙: "이름" 기반 + 띄어쓰기 구분 (batch ch1 ch2 -> batch (ch1 ch2))

### 기본-3
- ``x = np.random.uniform(low, high, sample_size)``: 균일 분포 난수 생성

### 심화-1
- ``error = error * pred * (1 - pred)``: sigmoid 편미분하는 부분
## Weekly Mission

- ``X_poly = np.concat((x**2, x), axis=1)``: x^2 계수와 x계수
  - ``X_poly = np.c_[x**2, x] ``와 같음
- 만약 위에서 ``x**2, x``의 순서를 바꿀지라도 학습에는 문제가 없음 (다만 순서가 달라짐)
