---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 15: ML LifeCycle 과제와 보완할 점"
date: 2025-09-19 18:41:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, attention, transformer]
description: "3주차 과제 점검과 위클리미션 리뷰"
keywords: [numpy, colab, ML LifeCycle, regression, linear classifier, neural networks, 데이터, 전처리, rnn, lstm, seq2seq, attention, transformer, encoder, decoder, 인코더, 디코더, 트랜스포머]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Week 3 Review 

오늘은 과제를 진행하면서 보완이 필요하거나 학습이 더 필요한 부분을 리뷰합니다.

## 과제

### 기본-2

- Sigmoid에는 Xaiver initialization으로 초기화
- ReLU에는 He initialization으로 초기화
- ``np.where(ff_dict['z1']>0,1,0.01)``: ReLU의 미분

### 기본-3
- colab에서 라인당 주석 확인
- ``Tensor.permute(0, 2, 1, 3)``: 차원 변경

## Weekly Mission
- ``Tensor.masked_fill(mask, float('-inf'))``: mask 텐서에서 True인 것을 텐서에서 ``-inf``로 마스킹
