---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 26: ML for RecSys, RecSys 동향과 통계학 기본"
date: 2025-10-13 18:15:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, recsys, distribution, likelihood]
description: "Recommander System에 대해 배우자."
keywords: [colab, binomial distribution, uniform distribution, normal distribution, beta distribution, clt, central limit theorem, mle, maximum likelihood estimation]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# RecSys를 들어가기 전에

## 최신 RecSys 동향

이커머스, SNS, 검색엔진, 네비게이션, 헬스케어 등 추천시스템은 굉장히 여러 분야에 걸쳐 사용되고 있습니다. 실생활에서 가장 많이 활용되고 있는 머신러닝 알고리즘입니다.

### 발전 방향
추천시스템은 아래에 나오는 모델의 순서로 발전하였습니다.

#### Shallow Model

Shallow Model은 행렬 분해를 통해 추천 시스템을 만듭니다. 사용자의 데이터가 행렬로 주어졌을 경우 어떻게 행렬을 분해해야 하는지 최적화합니다.

#### Deep Model

Neural Network를 통한 추천시스템입니다. Autoencoder 구조로 input 데이터를 output 데이터로 온전히 복원할 수 있도록 학습합니다. 이 과정에서 모델은 입력 데이터의 핵심적인 특징을 잠재 표현에 담아내도록 학습됩니다.

#### Large-scale Generative Models

최근에는 Large-scale 기반의 모델이 많이 사용됩니다. **P5** 모델은 여러가지 Task를 하나의 모델로 할 수 있는 통합적인 모델입니다. 다음의 Task가 가능합니다.
- 4100 -> 4459 -> 4332 순서로 아이템을 구매했을 경우 다음에 구매할 아이템은 어떤 것일까요?
- 4321번 아이템에 해당 유저가 몇 점의 평가를 줄 것 같나요?
- 특정 유저가 5142번 아이템을 구매한 이유가 무엇일까요?
- 등등…

텍스트 기반의 모델뿐만 아니라 텍스트와 이미지 모두를 다룰 수 있는 멀티모달 기반으로도 활용되기도 합니다. 만약 아이템을 구매할때 텍스트뿐만 아니라 아이템의 디자인이나 색상 등도 중요하게 작용할 것이기 때문입니다.

이미지 생성 모델이 추천시스템에 쓰이는 경우도 있습니다. Diffusion 모델은 데이터에 노이즈를 점점 주었을 때 노이즈에서 원래 데이터로 다시 생성할 수 있게 하는 모델입니다. 이미지를 생성할 때에 노이즈에서 시작해서 노이즈를 계속 빼주어 원하는 이미지로 도달할 수 있도록 합니다. 이러한 알고리즘이 추천시스템에서도 많이 활용되고 있습니다. 

그리고 현재 추천시스템에서 중요하게 다뤄지는 task는 다음과 같습니다:
- Explainability(설명성) — 우리 모델이 output을 그렇게 도출한(추천하는) 이유는 무엇일까?
- Debiasing and Causality — 영상을 클릭했을 때에 유명해서 클릭했을지, 취향이라서 클릭했는지를 구분
  * 편향을 줄이고 인과관계를 높임


## 통계학 기본

통계학 부분은 키워드 중심으로 정리하도록 하겠습니다.

* Random variable: 확률변수
  * input: sample space — 가능한 모든 경우의 수
* Distribution: 분포
  * Discrete 이산
  * Binomial 이항 — Bernoulli는 한번만 시행, Binomial은 n번 시행
  * Uniform 균등
  * Gaussian/Normal 정규
  * Beta 베타
    * 0에서 1까지를 확률변수로 가짐
    * alpha, beta가 1이면 uniform이 됨
* Central Limit Theorem, CLT (중심 극한 정리)
  * "동일한 확률분포를 가진 확률변수 n개의 평균의 분포"에서 n이 충분히 크면 -> 정규분포에 가까워진다
* likelihood: 가능도/우도
* prior: 사전 확률
  * mean이나 covar가 이쯤 있을 것이라는 믿음
* posterior: 사후 확률
  * 어떤 데이터가 관측되었을 때 사전 확률의 믿음도 같이 결합된 것

