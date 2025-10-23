---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 34: 경진대회 준비, 대회에서 배울 것"
date: 2025-10-23 19:24:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, nlp, competition, hyperparameter, wandb, finetuning, pre-trained model]
description: "자연어처리 모델 학습 파이프라인을 이해하자."
keywords: [colab, nlp, data, tuning, ensemble, hyperparameter, overfitting, underfitting]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# 대회에서 얻어가기 위해
AI 경진대회에서 리더보드의 순위가 중요하지 않다고도 말 못하지만 그보다 중요한 것들이 있습니다. 아래는 대회를 진행하며 얻어가야 할 것들을 정리하였습니다.
## 대회 진행 중
### 캐글 공유 정신
* 참가자들은 적극적으로 본인들의 코드와 아이디어 공유
* 같이 공유하고 고민하며 새로운 아이디어를 만들 수 있도록 하는 문화

#### 코드 및 아이디어 공유하기
* 내 코드를 공유하는 것은 부끄러울 수 있지만 의견을 공유하는 것이 도움이 됨
* 하지만 내가 잘못 알고 있었던 경우? 그걸 방치한다면 나중에 더 큰 문제가 됨
* 코드와 내 생각을 공유할 때에는 내용의 성격에 따라 맥락이 잘 설명되어야 함

#### EDA 코드 공유
* 명확하고 직관적인 시각화
* 데이터 인사이트의 해석
* 인사이트 요약 및 활용 방안

#### Feature Engineering 코드 공유
* 변수 생성 과정의 상세 설명
* 결측값 처리 과정의 상세 설명
* 변수 재현에 필요한 정보

#### Baseline 코드 공유
* 베이스라인 코드의 개요
* 각 단계별 명확한 설명
* 가독성 향상 시키기

### 검증 데이터셋
Validation Dataset이 잘못 선택될 경우 제대로 학습하지 못함
#### 좋은 데이터셋
* 대표성 - 검증 데이터셋이 훈련 및 실제 데이터 분포를 잘 반영하는가?
* 클래스 비중 - 클래스 불균형이 있는 경우 검증 데이터셋도 이를 반영하는가?
* 독립성 - 훈련 데이터셋의 정보가 검증 데이터셋에 포함되지 않는가?
* 적절한 전처리 - 훈련 데이터와 동일한 전처리 과정을 거쳤는가?
* 충분한 크기 - 검증 데이터셋의 크기는 동일한가?

## 최종 제출 선택
Make **general function**(Model) to obtain goal(Minimize cost/loss)
우리가 사용할 수 있는 지표는 Public Score, Validation Score
## 대회 마무리하기
### 베이스라인 뜯어보기
* 베이스라인 코드를 디테일하게 뜯어보자 → **직접 구현할 수 있을 때까지**
* 놓쳤던 실험 시도하기
* 상위 랭커의 솔루션 재현해보기
* 발표해보기
