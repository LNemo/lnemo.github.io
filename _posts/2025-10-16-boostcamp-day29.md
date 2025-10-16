---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 29: ML for RecSys, 변분추론"
date: 2025-10-16 18:59:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, recsys, vae, vi]
description: "Recommander System에 대해 배우자."
keywords: [colab, vae, gaussian mixture model, kl divergence, variational autoencoder]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Variational Inference

VI(Variational Inference)는 변분추론으로, 계산할 수 없는 분포를 다루기 쉬운 분포로 근사하는 방법입니다. 오늘은 EM 알고리즘과 VI에 대해 정리하였습니다.

## EM 알고리즘

EM(Expectation-Maximization) 알고리즘은 이름 그대로 Expectation과 Maximization을 반복하는 알고리즘입니다. 우리가 만약 잠재 변수 Z의 값을 안다면 모델의 파라미터 θ를 추정하기 훨씬 쉽습니다. 반대로 파라미터 θ를 알면 Z의 분포를 쉽게 추정할 수 있습니다. 

1. E-step (Expectation)
   - 현재의 파라미터 추정치 θ^t를 사용해 잠재변수 Z의 사후 분포 p(Z\|X, θt)에 대한 log-likelihood의 기댓값을 계산합니다. 
2. M-step (Maximization)
   - E-step에서 계산한 기댓값을 최대화하는 θ^(t+1)를 찾습니다. 데이터가 완전히 주어진 상황에서 MLE를 수행하는 것과 같습니다.

이 두 과정을 반복하면 log-likelihood가 단조 증가함이 보장됩니다. 따라서 local optimum으로 수렴합니다.


## VI
VI는 베이지안 추론에서 계산하기 어려운 실제 사후 분포 p(Z|X)를 다루기 쉬운 간단한 분포로 근사하는 최적화 기법입니다. VI에서는 실제 사후 분포 p(Z|X)와 근사 분포 q(Z) 사이의 KL-Divergence를 최소화 하는 것을 목표로 합니다. KL 발산은 두 확률 분포가 얼마나 다른 정도를 나타내기 때문에 이 값을 줄이면 q(Z)와 p(Z|X)가 비슷하게 됩니다.

하지만 p(Z\|X)를 직접 다룰 수 없기 때문에 KL-Divergence를 최소화하는 대신 ELBO(Evidence Lower Bound)를 최대화 합니다. 목적식에서의 ELBO를 최대화 하는 것은 KL-Divergence를 최소화 하는 것과 동일한 효과를 가집니다.

![이미지](/assets/img/posts/boostcamp/day29/vi1.png)  
![이미지](/assets/img/posts/boostcamp/day29/vi2.png)  
![이미지](/assets/img/posts/boostcamp/day29/vi3.png)  
![이미지](/assets/img/posts/boostcamp/day29/vi4.png)  
![이미지](/assets/img/posts/boostcamp/day29/vi5.png)  

KL(q(x) \|\| p(z\|x))은 항상 0보다 크거나 같습니다. log p(x)는 상수이므로 결국 KL(q(x) \|\| p(z\|x))을 최소화하는 것은 뒷부분인 ELBO를 최대화 하는 것과 같습니다.

## MFVI

Mean-Field Variational Inference(MFVI)는 평균장 변분 추론입니다. MFVI는 VI의 한 종류로 **근사 분포 q(Z)를 구성하는 모든 잠재 변수가 서로 독립**이라는 가정(mean-field 가정)을 추가하여 문제를 더 단순하게 만듭니다. 이러한 가정은 다변수 최적화 문제가 여러 개의 단일 변수 최적화 문제로 만드는 장점이 있지만 잠재 변수 간의 상관관계를 모델링하지는 못한다는 단점이 존재합니다.

## GMM vs. BGMM?

Gaussian Mixture Model은 데이터를 여러 개의 가우시안 분포가 섞인 형태라고 보고 각 데이터가 어떤 분포에 속하는지를 찾아내는 모델입니다. EM 알고리즘으로 각 클러스터의 최적 파라미터라는 **하나의 고정된 값**을 찾습니다.

Bayesian Gaussian Mixture Model은 GMM에 Prior라는 개념을 도한 것으로 파라미터를 하나의 정답으로 보지 않고 **파라미터가 확률 분포를 따른다**고 봅니다. 즉, 각 그룹의 평균이나 분산 등을 하나으 ㅣ고정된 값으로 찾지 않고 이 값일 확률을 추정하는 방식입니다.

BGMM은 GMM과 달리 π, μ, Σ에 Prior를 주입합니다.

- 혼합 비율(π)
  - 디리클레 분포
  - GMM에서 π_k을 모두 더하면 1이 되어야 합니다. 디리클레 분포에서 샘플링하게 되면 π의 합이 1이 되는 것이 보장됩니다.
- 공분산 행렬(Σ)
  - 위샤트 분포
  - 공분산 행렬은 항상 Positive definite matrix이어야 합니다. 위샤트 분포는 항상 positive definite matrix를 보장합니다.
- 평균(μ)
  - 가우시안 분포
  - 각 그룹의 평균이 대략 어느 위치에 있을 것이라는 prior knowledge를 주입합니다.

따라서 GMM은 파라미터의 최적값 하나를 찾아가는 것이 목표이고 BGMM은 파라미터의 확률 분포를 찾아가는 것이 목표이기 때문에 각 모델은 아래와 같이 알고리즘을 사용하게 됩니다.

- GMM — EM
- BGMM — VI