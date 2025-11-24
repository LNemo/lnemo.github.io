---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 51: RecSys, MCMC & Data Attribution"
date: 2025-11-24 18:53:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, recsys, mcmc, data attribution]
description: "Recommander System에 대해 배우자."
keywords: [MCMC, Rejection Sampling, Importance Sampling, Markov Chain, Monte Carlo, Metropolis-Hastings, Probability Distribution, Sampling, Gibbs Sampling, HMC, Hamiltonian Monte Carlo, Diffusion Models, Langevin Dynamics, Data Attribution, Feature Attribution, Explainability, Model Diagnosis, RAG, Influence Function, Hessian Matrix, Data Shapley, Game Theory, DVRL, Reinforcement Learning, Policy Gradient, Data-OOB, Bagging, Out-of-Bag]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# MCMC & Data Attribution

오늘은 MCMC와 데이터의 가치 측정 방법과 그 해석에 대해 학습하였습니다. 아래에 키워드 흐름 별로 정리하였습니다.

## MCMC

### MCMC 이전의 방식

- Rejection Sampling (기각 샘플링)
  - 목표 분포 p(z)를 덮는 단순한 분포 q(z)를 만들고, q(z)에서 뽑은 샘플 중 p(z)보다 높은 위치에 있는(회색 영역) 샘플은 버립니다(Reject)
- Importance Sampling (중요도 샘플링)
  - 샘플을 버리지 말고, q(z)에서 뽑되 실제 분포 p(z)와의 비율만큼 가중치(Weight)를 줘서 계산하자.
- 차원이 커질수록 두 방법의 효율이 좋지 않음

### MCMC가 뭐지?

- Markov Chain Monte Carlo
- 정확한 확률 분포를 구할 수가 없으니까 대신에 **그 분포를 따르는 샘플을 아주 많이 뽑아서 그 샘플로 분포를 추정하자**
- Monte Carlo: 무작위 샘플링을 통해 값을 추정
- Markov Chain: **‘마르코프 성질‘**을 가진 **‘이산시간 확률과정’**
  - 마르코프 성질: 과거와 현재 상태가 주어졌을 때의 미래 상태의 조건부 확률 분포가 과거 상태와는 독립적으로 현재 상태에 의해서만 결정
  - 이산시간 확률과정: 이산적인 시간의 변화에 따라 확률이 변화하는 과정
- MCMC의 핵심 아이디어: Metropolis-Hastings 알고리즘
  - P(x)의 분포를 구하고자 할 때에
  - 다음의 4단계를 반복
    1. 현재 위치(`x_old`)의 확률을 구함
    2. 무작위로 근처 위치를 찍음(`x_new`)
    3. `x_new`에서의 확률을 구해서 현재 위치와 2에서 찍은 위치의 비율(`P(x_new)/P(x_old)`)을 구함
       - 만약 비율이 1보다 크다면 새로운 곳이 더 확률이 높으므로 `x_new`로 이동
       - 비율이 1보다 작다면 이동하지 않는 것이 아니라 **확률적으로 이동** (새로운 위치의 확률이 70%라면 70%의 확률로 이동)
    4. 확정 및 기록
       - Accept: 이동한다면 그곳으로 위치를 옮기고 그 위치를 점으로 표시
       - Reject: 제자리에 그대로 있다면 그 자리에 점을 한번 더 찍음
  - 이 과정을 반복하면 확률이 높은 곳에는 많은 점이 찍히고 낮은 곳에는 드문드문 점이 찍힘
- MCMC는 최적화 방식이 아니라 **샘플링 방식**

### 최적화 방식을 거치지 않는 MCMC가 AI라고 말할 수 있을까?

- VI는 레이어를 통과시켜 분포의 평균과 표준편차를 구하는 방식
- 반면에 MCMC는 샘플링 해서 분포를 추정
- 현대 AI 주류가 딥러닝인것은 맞지만 추론과 확률을 통해 결과를 내는 것은 머신러닝의 정의에 부합

### MCMC를 어디에 사용하는 거지?

- Diffusion 모델
  - Langevin Dynamics라는 MCMC의 특수한 방식
  - Stable Diffusion
  - Midjourney
- 강화 학습
  - AlphaGo

### MH 알고리즘을 제외한 MCMC의 다른 알고리즘

#### Gibbs Sampling?
- 변수가 여러개일때 하나씩만 업데이트 — 변수가 `x`, `y`, `z`라면 `y`, `z`를 고정하고 `x`만 샘플링, 다음은 `x`, `z` 고정 후 `y`만 샘플링
- 계산이 훨씬 쉽다
- 샘플링한 점이 항상 채택됨
- 조건부 확률 분포를 수식으로 알아야 한다는 단점

#### HMC
- Hamiltonian Monte Carlo
- 물리학의 원리(에너지 보존)를 빌림
- MH 알고리즘은 랜덤워크이기때문에 모든 분포를 구하는데 시간이 오래 걸림
- HMC는 확률 분포를 뒤집어서 최솟값을 찾아가는 문제로 변환 -> 미분값으로 최저점으로의 더 빠른 이동
- 고차원 효율성이 높음

## 데이터의 가치 측정 및 해석

### Feature Attribution vs. Data Attribution

- Feature Attribution
  - 모델이 특정 예측을 할 때 입력 데이터(Test data)의 어떤 부분(Feature)에 집중했는지 평가
- Data Attribution
  - 모델이 특정 예측 결과를 낼 때 학습 데이터셋에서 어떤 데이터가 가장 크게 작용했는지 평가

### RecSys, CV, NLP에서 Data Attribution 기술이 어떻게 활용될 수 있을까?
- Explainability(설명가능성)
  - 추천시스템이 사용자에게 어떤 아이템을 추천했을 때, 유사한 다른 유저의 데이터가 얼마나 기여했는지 알 수 있음
- Model Diagnosis(모델 진단)
  - Loss를 줄이는데 방해가 되는 데이터를 식별하여 제거하거나 수정해서 모델을 개선할 수 있음
- RAG(Retrieval Augmented Generation)
  - LLM이 외부 지식을 참조할 때에 정확한 지식을 DB에서 추출하기 위해 데이터 기여도 활용

### Data Attribution 방법
- 특정 데이터를 학습에서 제외할 때 모델의 변화를 어떻게 계산할 수 있을까?
- 가장 직관적인 방법: 해당 데이터를 제거 후 재학습 -> 매우 비효율적

#### Influence Function
- 재학습 없이 파라미터를 근사하기 위해 특정 데이터`z`의 가중치`ε`를 아주 조금 변화시켰을 때 파라미터의 변화를 미분을 통해 계산
- 테일러 급수 전개와 헤시안 행렬 활용
- 용례
  - 추천시스템의 ‘특정 사용자 기록 삭제’ 과정에서 재학습 없이 파라미터 변화를 근사할 수 있음
  - 데이터 증강에서 변형된 데이터의 가치 평가

#### Data Shapley

- 게임 이론의 섀플리 값(Shapley Value)을 머신러닝 데이터 가치 측정에 활용
- 단순히 데이터를 빼고 넣는 것은 데이터 간의 상호작용을 무시하는 것이라고 생각
- 데이터 `i`를 제외한 **모든 가능한 부분집합(Subset)** `S`에 대해 `i`가 추가되었을 때의 기여도를 계산하고 평균을 계산
  - 모든 가능한 부분집합에서 계산하기 때문에 계산량이 매우 많음
  - 이를 해결하기 위해 Monte Carlo Approximation(TMC-Shapley)을 사용하여 근사값을 계산

#### DVRL
- Data Valuation using Reinforcement Learning
- 강화학습을 활용해 **모델 학습**과 **데이터의 가치**를 동시에 측정
- Validation Set의 성능을 Reward로 설정하고 이 보상을 최대화하도록 훈련 데이터를 선택하는 Policy를 학습
- 구조
  - Data Value Estimator: 데이터를 선택할 확률을 출력
  - Predictor(모델): 선택된 데이터로 학습
- Data Value Estimator의 샘플링은 미분이 불가능 -> Reinforce 알고리즘(Policy Gradient)을 사용하여 학습
- 적은 양의 데이터로도 높은 성능을 내거나 노이즈가 섞인 데이터 셋에서 성능을 저해하는 데이터를 효과적으로 걸러낼 수 있음

#### Data-OOB?
- Out-of-Bag
- DVRL은 클린한 Validation Set이 필요하다는 제약
- Data-OOB는 별도의 Validation Set이 필요 없고 계산이 효율적인 Bagging 기반의 데이터 가치 측정 방법
- 과정
  - Bagging 기법을 활용하여 여러 개의 약한 학습기(Weak Learner)를 만듦
  - 데이터가 학습에 포함되지 않았을 때(Out-of-Bag, OOB)의 예측 에러를 측정
  - 만약 특정 데이터가 OOB일 때(즉, 그 데이터를 보지 않고 학습했을 때) 모델의 에러가 크다면, 그 데이터는 다른 데이터로는 설명되지 않는 독특하거나 중요한 패턴을 가진 데이터라고 간주
- 다른 방법들에 비해 계산 시간이 매우 짧고 Validation Set이 필요하지 않다는 장점
- 복잡한 데이터셋에도 적용가능한 scalable한 방법론