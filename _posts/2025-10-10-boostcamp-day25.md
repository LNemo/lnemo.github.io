---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 25: NLP 이론 과제와 보완할 것"
date: 2025-10-10 18:27:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, nlp]
description: "5주차 과제 점검과 위클리미션 리뷰"
keywords: [colab, tokenization, embedding, rnn, lstm]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Week 5 Review

오늘은 과제를 진행하면서 보완이 필요하거나 학습이 더 필요한 부분을 리뷰합니다.

## 과제

### 기본-2
- `chain.from_iterable()`을 사용해서 토큰화 하는 방법 익히기

### 심화-1
- 문장의 첫 부분과 끝 부분은 window 사이즈 안에 단어가 없을 수도 있는데 이 때는 어떻게 처리하는지
  - CBOW의 원리 관점에서 보았을 때에 `<PAD>`는 무의미한 값을 전달할 수도 있기 때문에 잘못된 문맥이 학습될 수도 있다.
  - 그래서 제외하는게 좋을 것 같다

## Weekly Mission
- `torch.Tensor`로도 Tensor를 만들 수 있다
- `nn.Parameter()` 안에 Tensor를 넣어서 학습 가능한 변수로 만들 수 있다