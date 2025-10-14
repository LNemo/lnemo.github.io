---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 27: ML for RecSys, 생성모델"
date: 2025-10-14 18:15:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, recsys, generative model, vae]
description: "Recommander System에 대해 배우자."
keywords: [colab, vae, gaussian mixture model, kl divergence, variational autoencoder]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Generative Model

## VAE 
VAE(Variational AutoEncoders)는 대표적인 생성 모델 중 하나입니다. VAE는 잠재 공간의 확률 분포가 '표준정규분포에 가깝도록' 강제하는 구조를 가지고 있습니다. 데이터를 가장 잘 표현하는 잠재 공간의 확률 분포(평균과 분산)를 학습하되, 이 분포가 표준정규분포라는 사전 분포(prior distribution)를 벗어나지 않도록 규제합니다.


### 목적식

![이미지](/assets/img/posts/boostcamp/day27/pass.png)

위 목적식에 마이너스를 붙인 것이 Loss Function입니다.
- Reconstruction Loss
  - KL-Divergence 앞부분으로, 인코더를 통해 압축된 잠재 변수(z)로부터 디코더가 원본 데이터(x)와 최대한 유사한 데이터(x′)를 생성하도록 만듭니다.
- KL-Divergence
  - 인코더가 만들어내는 잠재 변수의 확률 분포가 우리가 미리 가정한 사전 분포와 최대한 비슷해지도록 만듭니다. 보통 이 사전 분포로 표준정규분포를 사용합니다.

VAE는 이 두 가지 목표를 동시에 최적화함으로써 데이터를 잘 복원하면서도, 새로운 데이터를 생성할 수 있는 의미 있는 잠재 공간을 학습하게 됩니다.

