---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 14: ML LifeCycle, Transformer 논문 정리"
date: 2025-09-18 18:29:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, attention, transformer]
description: "머신러닝의 생애주기에 대해 배우자."
keywords: [numpy, colab, ML LifeCycle, regression, linear classifier, neural networks, 데이터, 전처리, rnn, lstm, seq2seq, attention, transformer, encoder, decoder, 인코더, 디코더, 트랜스포머]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Attention Is All You Need


오늘은 Transformer 논문을 정리 요약하였습니다.

---
* **논문 제목:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* **저자:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin

---
⠀
## 1. Introduction
이전에는 LSTM, GRU가 최고 수준이었다.

Recurrent 모델은 위치를 따라 계산하기 때문에 데이터 내 병렬화가 불가능하다.
최근에는 factorization trick과 conditional computation으로 계산 효율을 향상시켰지만 여전히 sequential한 계산이었다

attention 매커니즘은 input이나 output의 거리에 상관없이 의존성 모델링을 해주지만 RNN과 사용된다.

여기에서는 recurrence를 제외하고 어텐션에만 의존하는 아키텍쳐인 트랜스포머를 제안한다. 트랜스포머는 높은 수준의 병렬화와 최고수준의 번역품질을 보여준다.

## 2. Background

순차적 계산을 줄이려는 목표는 CNN을 사용하는 모델들에 기초가 된다.
CNN에서는 위치표현을 병렬적으로 처리하는데 연산 횟수가 위치 간의 거리에 따라 증가한다. -> 멀리 떨어진 위치 관계를 학습하기 어려움
트랜스포머에서는 O(1)로 줄였다 effective resolution이 감소하는 비용발생은 multi-head attention으로 해결

self-attention은 단일 시퀀스에서 서로 다른 위치들을 연관시켜 시퀀스의 표현을 계산하는 어텐션 메커니즘

end-to-end memory networks는 a recurrent attention mechanism에 기반하는데 좋은 성능을 보였다.

트랜스포머는 RNN이나 합성곱없이 self-attention만 사용하는 최초의 변환 모델이다.

## 3. Model Architecture

트랜스포머는 stacked self-attention, point-wise, fully-connected layers를 사용하여 architecture를 구성한다.

![이미지](/assets/img/posts/boostcamp/day14/architecture.png)


### 3.1. Encoder and Decoder Stacks

#### 인코더
* 6개의 동일한 레이어를 쌓음
* 레이어는 두 개의 서브 레이어를 가짐
  * multi-head self-attention
  * position-wise fully connected feed-forward network
* residual connection, layer normalization
* 각 서브 레이어의 출력 LayerNorm(x + Sublayer(x))
* 잔차연결의 용이성을 위해 서브레이어와 임베딩레이어는 512차원의 출력을 생성

#### 디코더
* 6개의 동일한 레이어를 쌓음
* 두 개의 서브 레이어 외에 디코더는 세번째 서브 레이어를 추가
  * 인코더의 출력에 대해 multi-head attention 수행
* residual connection, layer normalization
* 디코더의 self-attention에서 현재 위치가 다음 위치를 참조하지 못하도록 마스킹

### 3.2. Attention

어텐션은 하나의 Query와 여러개의 Key-Value 쌍을 하나의 output으로 매핑하는 것
output은 Value의 가중합으로 계산, value에 대한 가중치는 해당 query와 key의 유사도 함수에 의해 계산

#### 3.2.1. Scaled Dot-Product Attention

Query와 모든 Key를 내적하고 d_k의 제곱근으로 나눈 뒤 softmax를 적용해 Value에 대한 가중치를 얻음  
실제로는 여러 Query에 대한 어텐션 함수를 동시에 계산  
일반적으로 additive attention과 dot-product attention이 사용되는데  
논문에서의 attention이 다른 건 sqrt dk로 나누어 줬다는 것  
d_k가 클 때 스케일링을 하지 않으면 내적이 너무 커져서 softmax가 기울기가 극도로 작은 영역으로 들어가버림

#### 3.2.2. Multi-head Attention

Multi-head Attention은 모델이 서로 다른 위치의 다른 표현 subspace로 온 정보를 집중할 수 있도록 한다
단일로 사용하면 평균이 이걸 방해한다
여기에서는 h=8개의 병렬 헤드를 사용하였다
각 헤드에 대해 d_k=d_v=d_model/h=64를 사용
각 헤드의 차원이 줄어들기 때문에 총 계산 비용은 전체 차원을 가진 단일 헤드 어텐션과 비슷함

#### 3.2.3. Applications of Attention in our Model

* 인코더-디코더 어텐션 레이어: Query가 이전 디코더 레이어에서, K, V가 인코더의 출력에서
* 인코더의 셀프-어텐션 레이어: 인코더에서 Q, K, V
* 디코더의 셀프-어텐션 레이어: 미래 토큰 참조를 막고 디코더에서 Q, K, V

### 3.3. Position-wise Feed-Forward Networks
* 두개의 선형 변형과 사이의 ReLU함수
* FFN(x) = max(0, xW1​ + b1​)W2 ​+ b2
* d_model은 512, d_ff은 2048

### 3.4. Embedding and Softmax
* 학습된 임베딩 사용해서 입력토큰 출력토큰을 d_model 차원의 벡터로 변화
* 디코더의 출력을 다음 토큰의 예측 확률로 변환하기 위해 학습된 선형 변환과 softmax 함수 사용
* 해당 선형 변환에 동일한 가중치 행렬을 공유
* 임베딩 레이어에서는 이 가중치들에 sqrt(d_model)을 곱함

### 3.5. Positional Encoding
* recurrence, convolution을 사용하지 않기 때문에 위치에 대한 정보 주입을 해주어야 함
* 입력 임베딩에 positional encodings를 더해줌
* positional encoding은 임베딩과 동일차원(d_model)을 가짐
* 이 모델에서 학습된 positional embedding 대신에 sine 버전을 사용한 이유는 모델이 더 긴 시퀀스에도 일반화 할 수 있도록 하였기 때문

## 4. Why Self-Attention
* Self-Attention을 사용하는 이유에 대해 설명
* 세가지를 생각함
  * 레이어당 총 계산 복잡도
  * 병렬화 할 수 있는 계산의 양 -> 필요한 최소 순차 연산 수
  * 장거리 의존성
* self- attention은 O(1)의 순차 실행 연산으로 모든 위치를 연결함, 반면에 Recurrent 레이어는 O(n)의 연산이 필요
* 계산 복잡도 측면에서 시퀀스 길이 n이 표현 차원 d보다 작을 때 self-attention이 recurrent보다 빨랐다 (최신 모델들이 사용하는 문장 표현에 대부분에 해당)
* convolution layor는 비싸다
* self-attention은 더 해석하기 쉬운 모델
  * 개별 어텐션 헤드들이 서로 다른 작업을 수행하도록 명확하게 학습할 뿐만 아니라, 많은 헤드들이 문장의 구문적(syntactic), 의미적(semantic) 구조와 관련된 행동을 보임

## 5. Training

### 5.1. Training Data
* 영어-독일어 데이터셋으로 훈련
  * Byte-Pair Encoding
  * 37,000개의 토큰 어휘 사전
* 영어-프랑스어 데이터셋
  * 3,600만 개 문장
  * 32,000개의 word-piece
* 훈련 배치는 약 25,000개 소스토큰과 25,000개 타겟토큰을 포함하는 문장 쌍들의 집합을 포함

### 5.2. Hardware & Schedule
* 8개의 NVIDIA P100 GPU 장착된 머신 하나로 모델 훈련
* base model은 훈련 스텝에 약 0.4초
  * 100,000 step 훈련 (12시간)
* big model은 훈련 스텝 1.0초
  * 300,000 step 훈련 (3.5일)
⠀
### 5.3. Optimizer
* beta1 = 0.9, beta2=0.98, eps=10^-9d인 Adam Optimizer 사용
* lr 은 (3)에 따라 변화
  * warmup에는 선형적으로 증가하다가 warmup 후에는 step 수의 역제곱근에 비례하여 감소
  * warmup_steps=4000 을 사용
⠀
### 5.4. Regularization
* Residual Dropout
  * 서브-레이어의 출력에 드롭아웃
  * 임베딩과 포지셔널 인코딩의 합에도 드롭아웃
* label Smoothing

## 6. Result

### 6.1. 기계 번역
* WMT 2014 영어-독일어 번역 과제
  * 빅모델
    * 28.4라는 새로운 최고 수준(state-of-the-art)의 BLEU 점수
    * 훈련은 3.5일
  * 기본모델 기준으로도 이전 모든 모델과 앙상블을 능가함
* WMT 2014 영어-프랑스어 번역 과제
  * 빅모델
    * 41.0의 BLEU 점수
    * 이전 최고 모델 훈련 비용의 1/4 미만으로 달성한 결과
    * 드롭아웃 비율 P를 0.3 대신 0.1을 사용

⠀
* beam_size가 4, length penalty가 0.6인 beam search를 사용
* 표 2에 결과 요약

![이미지](/assets/img/posts/boostcamp/day14/result.png)

### 6.2. Model Variations

![이미지](/assets/img/posts/boostcamp/day14/variations.png)

* 표 3에서 모델 변형하면서 실행해봄
  * (A)는 계산량을 유지하면서 어텐션 헤드수와 어텐션 K/V 차원을 변경 -> 단일 헤드 0.9 BLEU가 낮았는데 너무 많아도 품질 떨어짐
  * (B)는 어텐션 Key 크기를 줄여봄 -> 모델 품질 저하. -> 유사도 결정이 쉽지 않고 내적보다 더 정교한 유사도 함수가 유익할 수 있음을 시사
  * (C), (D)는 더 큰 모델이 더 좋고, dropout이 overfitting을 피하는데 도움이 되는것을 관찰
  * (E) Positional encoding을 사인에서 학습된 포지셔널 임베딩으로 교체 -> 거의 동일한 결과

### 6.3. English Constituency Parsing
* 트랜스포머가 다른 과제에도 일반화 가능한지 영어 구문 분석에 대한 실험 진행
  * 출력은 구조적 제한/입력보다 훨씬 긺
  * RNN의 seq-to-seq 모델은 데이터가 적을때 최고수준의 결과를 달성하지 못함
  * Wall Street Journal 부분에서 d_model=1024인 4-layer 트랜스포머를 훈련함
    * semi-supervised 환경에서도 훈련
    * WSJ만 사용 - 16,000개 토큰 어휘 사전, semi-supervised - 32,000개 토큰 어휘 사전
  * 결과는 과제에 특화된 튜닝이 부족함에도 모델이 잘 작동하고 RNNG를 제외한 모든 모델보다 더 나은 결과를 냄

## 7. Conclusion
* 이 연구에서 Recurrent layer를 Multi-head self-attention으로 대체하여 전적으로 attention에만 기반한 최초의 시퀀스 변환 모델인 Transformer를 제시
* 순환 또는 합성곱 레이어에 기반한 아키텍처보다 훨씬 더 빠르게 훈련될 수 있고 최고 수준의 성능을 달성
* 다른 과제들에도 적용할 계획
  * 텍스트 이외의 이미지, 오디오, 비디오 같은 입출력을 처리하기 위해 local, restricted attention mechanisms을 연구할 계획
  * 생성 과정을 덜 순차적으로 만드는 것도 연구 목표

---
**피어세션을 통해 알아간 것**
-  Scaled Dot-Product Attention에서 나눠주는 값이 sqrt(dk)인 이유는?
  - q, k가 평균0, 분산이1인 독립적인 확률변수라고 가정하면 sqrt(dk)이 표준편차이기 때문에
- 단일 어텐션에서 해상도가 감소한다는 것의 의미는?
  - 입력 시퀀스의 모든 위치에 대한 정보를 가중합하여 하나의 출력을 생성할 때, 특정 단어에 대한 정보가 손실될 수 있다는 것
