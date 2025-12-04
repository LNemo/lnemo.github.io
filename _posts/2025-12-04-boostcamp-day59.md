---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 59: RecSys 기초 프로젝트, Item2Vec & ANN & DL"
date: 2025-12-04 18:28:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, recsys]
description: "Recommander System에 대해 배우자."
keywords: [Recommendation System, Deep Learning, Word2Vec, Item2Vec, ANN, HNSW, Faiss, Neural Collaborative Filtering, YouTube RecSys, Autoencoder, AutoRec, GNN, NGCF, LightGCN, RNN, GRU4Rec, Session-based Recommendation]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Item2Vec & ANN & DL

오늘은 추천 시스템에 적용할 수 있는 여러 방법들을 공부하였습니다. Item2Vec, ANN, 추천시스템에 사용되는 DL 모델들을 알아보았습니다.

## Word2Vec

- 뉴럴 네트워크 기반
- 압축된 형태의 많은 의미를 갖는 dense vector로 표현

### Embedding

- 주어진 데이터를 낮은 차원의 벡터로 만들어서 표현하는 방법
- Sparse Representation 대신 Dense Representation으로 표현
  - Sparse Representation: one-hot encoding 또는 multi-hot encoding 등
  - Dense Representation: 아이템의 전체 가짓수보다 훨씬 작은 차원으로 표현. 0, 1이 아닌 실수값으로 표현
- Word Embedding은 비슷한 단어는 서로 가까운 위치에 분포할 것이기 때문에 단어간 의미적인 유사도를 구할 수 있음

### 학습 방법

- CBOW
  - Continuous Bag of Words (CBOW)
  - 주변 단어를 가지고 센터에 있는 단어를 예측하는 방법
  - window의 크기를 정해주어서 해당 window에 포함된 단어들로 학습
- Skip-Gram
  - CBOW의 입력층과 출력층이 반대로 구성된 모델
  - 센터의 단어로 주변 단어를 예측하는 방법
  - 일반적으로 CBOW보다 Skip-Gram의 성능이 좋다고 알려져 있음
- Skip-Gram with Negative Sampling (SGNS)
  - Item2Vec에서 사용되는 방법
  - Skip-Gram은 입력을 넣어서 4개의 라벨이 나오도록 하지만 SGNS는 입력1과 입력2를 넣어서 해당 단어 둘이 주변 단어인지 예측 (Multi Classification → Binary Classification 문제 변경)
  - 위의 두 방법은 주변의 단어만 학습하면 되지만 SGNS는 **주변에 있지 않은 단어를 샘플링 해서 0이라고(negative) 학습**해야 함
  - 방법
    1. 중심 단어를 기준으로 주변 단어들과의 내적의 sigmoid를 예측값으로 하여 0과 1을 분류
    2. 역전파를 통해 각 임베딩이 업데이트
    3. 최종 생성된 워드 임베딩이 2개 (중심단어, 주변단어)
       - 하나만 사용하거나 평균을 사용 (성능은 비슷함)

## Item2Vec

- SGNS에서 영감을 받아 아이템 기반 CF에 Word2Vec을 적용
- Word2Vec에서 가져온 Item2Vec의 개념
  - 문장 → 유저가 소비한 아이템 리스트
  - 단어 → 아이템
  - SGNS 기반으로 아이템을 벡터화 하는 것이 목표
- Word2Vec과 다른 Item2Vec의 개념
  - 시퀀스를 집합으로 바꾸면서 집합 안의 아이템은 서로 유사하다고 가정 (시간/공간적 정보 삭제)
  - 공간적 정보를 무시하므로 동일한 아이템 집합 내 아이템 쌍들은 모두 SGNS의 Positive Sample이 됨

### 적용

- 학습된 아이템 벡터를 t-SNE로 임베딩하여 시각화
- 비슷한 카테고리에 대해 MF보다 Item2Vec의 아이템 벡터 임베딩의 클러스터링 결과가 더 우수

## ANN

- NN: Nearest Neighbor
  - Vector Space Model에서 내가 원하는 Query Vector와 가장 유사한 Vector를 찾는 알고리즘
- ANN: Approximate Nearest Neighbor
- 추천 모델은 모델 학습을 통해 구한 유저/아이템의 Vector가 주어질 때 주어진 Qery Vector의 인접한 이웃을 찾아주는 것 → Nearest Neighbor Search
- NN을 정확하게 구하기 위해서는 모든 Vector와의 유사도 비교를 해야함 → 계산 비용이 큼
- 그러면 정확도를 조금 포기하고 빠른 속도로 주어진 Vector의 근접 이웃을 찾자 → ANN
  - 200ms 걸려서 100% 정확도로 구할 것인가
  - 1ms 걸려서 90% 정확도로 구할 것인가
  - 2~3ms 걸려서 99% 정확도로 구할 것인가

### ANNOY

- Spotify에서 개발한 tree-based ANN
- 주어진 벡터들을 여러개의 subset으로 나누어 tree 형태의 자료 구조로 구성하고 이를 활용하여 탐색
- 방법
  1. Vector Space에서 임의의 두 점을 선택한 뒤 두 점 사이의 hyperplane으로 Vector Space를 나눔
  2. Subspace에 있는 점들의 개수를 node로 하여 binary tree 생성 또는 갱신
  3. Subspace 내의 점이 K개 초과로 존재한다면 해당 Subspace에 대해 1과 2를 진행  
        
     → ANN을 구하기 위해서는 현재 점을 binary tree에서 검색한 뒤 해당 subspace에서 NN을 search
        
- 문제점
  - 가장 근접한 점이 다른 subspace에 포함될 경우에는 해당 점은 후보에 포함되지 못함
    - 가장 가까운 다른 subspace도 포함해서 탐색해서 해결할 수 있음
    - binary tree를 여러개를 만들어 병렬적으로 탐색
  - 기존 생성된 binary tree에 새로운 데이터 추가할 수 없음

### HNSW

- Hierarchical Navigable Small World Graphs
- 벡터를 그래프의 node로 표현하고 인접한 벡터를 edge로 연결
- Layer를 여러 개 만들어 계층적 탐색을 진행함
  - 상위 레이어로 갈수록 개수가 적음 (랜덤 샘플링)
- 방법
  1. 최상위 Layer에서 임의의 노드에서 시작
  2. 현재 Layer에서 타겟 노드와 가장 가까운 노드로 이동
  3. 현재 Layer에서 더 가까워 질 수 없다면 하위 Layer로 이동
  4. 타겟 노드에 도착할 때까지 2, 3을 반복
  5. 2~4를 진행할 때 방문한 노드만 후보로 하여 NN Search
- 라이브러리
  - nmslib, faiss

### IVF

- Inverted File Index
- 주어진 벡터를 클러스터링을 통해 n개의 클러스터로 나눠서 저장
- 벡터의 인덱스를 클러스터 별 inverted list로 저장
- Query vector가 주어졌을 때 해당 클러스터 안에서만 NN Search 진행
- 가장 가까운 벡터가 다른 클러스터에 있는 경우 탐색해야하는 cluster를 증가시킬 수 있지만 속도가 저하되는 trade-off

### Product Quantization - Compression

- 방법
  1. 기존 벡터를 n개의 sub-vector로 나눔
  2. 각 sub-vector 군에 대해 k-means clustering을 통해 centroid를 구함
  3. 기존의 모든 vector를 n개의 centroid로 압축해서 표현
- 두 벡터의 유사도를 구하는 연산이 거의 요구되지 않음
  - 미리 구한 centroid 사이의 유사도를 활용

- PQ와 IVF를 동시에 사용하면 더 빠르고 효율적인 ANN이 가능 (faiss에서 제공함)

## 추천시스템과 딥러닝

- 추천시스템에 딥러닝을 사용하는 이유
  - Nonlinear Transformation — 복잡한 user-item interaction pattern을 효과적으로 모델링 가능
  - Representation Learning — 복잡한 featrue를 사람이 직접 feature design 하지 않아도 됨
  - Sequence Modeling — DNN이 sequential modeling task에 성공적으로 적용된 것을 추천시스템의 next-item prediction, session-based recommendation에 사용
  - Flexibility — Tensorflow, PyTorch등 다양한 DL 프레임워크

## MLP

- Multi-Layer Perceptron
- Perceptron으로 이루어진 여러 레이어를 순차적으로 쌓아놓는 방법
- user와 item 사이 복잡한 관계가 있을 때 MF(Matrix Factorization)은 선형 조합이기 때문에 표현에 한계가 있기 때문에 nonlinear하게 표현할 수 있는 MLP로 해결 가능

### Neural Collaborative Filtering

- 구조
  - MLP Layer
    - Input Layer
    - Embedding Layer
    - Neural CF Layers
    - Output Layer
  - GMF Layer
    - MF 레이어
- MLP, GMF를 앙상블해서 사용
  - 둘은 서로 다른 embedding layer 사용
  - 각 결과를 concat 하여 최종 결과

- 기존 MF나 MLP 모델보다 성능이 좋음

### YouTube Recommendation

- 유튜브 추천 문제 특징
  - 유튜브는 엄청 많은 유저와 아이템이 있기 때문에 효율적인 서빙과 이에 특화된 추천 알고리즘이 필요
  - 잘 학습된 컨텐츠와 새로 업로드 된 컨텐츠를 실시간으로 적절히 조합해야 함
  - 높은 Sparsity, 다양한 외부 요인으로 유저 행동 예측이 힘듦
- 구조 (2단계 추천)
  - Candidate Generation — High Recall 목표. 후보를 millions에서 hundreds로 줄여줌
  - Ranking — 유저, 비디오 피처를 풍부하게 사용하여 스코어를 구하고 최종 추천 리스트 제공

#### Candidate Generation

- 특정 시간(t)에 유저 U가 C라는 context를 가지고 있을 때, 비디오(i)를 각각 볼 확률을 계산
- 과정
  1. 여러 피처들을 하나로 concat해줌
     - Watch Vector와 Search Vector를 각각 임베딩하여 각각 평균냄
     - Demographic & Geographic features 피처로 포함
     - Example Age feature 피처로 포함
       - Example Age feature: 시청 로그가 학습 시점으로부터 경과한 정도
  2. concat한 벡터를 n개의 dense layer를 거치게 하여 user vector를 구함
  3. Serving — 유저벡터를 input으로 하여 상위 N개 비디오를 추출
     - 유저 벡터와 모든 비디오 벡터의 내적을 계산하여 ANN을 사용하여 빠르게 서빙

#### Ranking

- CG 단계에서 생성한 비디오 후보들을 input으로 하여 최종 추천될 비디오들의 순위를 매기는 문제
- Logistic 회귀를 사용
- loss function에 시청시간을 가중치로 사용
- 과정
  1. User actions feature 사용
     - 특정 채널에서 얼마나 많은 영상을 봤는지
     - 특정 토픽의 영상을 본지 얼마나 지났는지
     - 영상의 과거 시청 여부 등
     - DL 구조보다 도메인 전문가의 역량이 좌우
  2. Dense layer를 통과시켜 비디오가 실제로 시청될 확률로 매핑
- Loss function
  - 단순 binary 아닌 weighted cross-entropy loss 사용
  - 비디오 시청 시간으로 가중치

#### 결과

- 딥러닝 기반 2단계 추천을 처음으로 제안
- Candidate Generation — 기존 CF 아이디어 기반으로 다양한 피처를 사용해 추천 성능 향상
- Ranking — 과거에 많이 사용된 선형/트리 기반 모델보다 딥러닝 모델이 더 뛰어난 성능을 보여줌

## Autoencoder

- 입력 데이터를 출력으로 복원하는 비지도 학습 모델
- DAE
  - Denoising Autoencoder
  - 입력 데이터에 random noise나 dropout을 추가하여 학습
  - noisy input을 더 잘 복원하는 robust한 모델이 학습되어 전체적인 성능 향상

### AutoRec

- AE를 CF에 적용하여 CF 모델에 비해 Representation과 Complexity 측면에서 뛰어남을 보임
- Rating Vector를 입력과 출력으로 해서 Encoder & Decoder 과정을 수행
  - 저차원의 latent feature로 나타내 이를 사용해 평점 예측
- MF도 저차원으로 나타내지만 linear, low-order interaction을 통한 representation,  
AutoRec은 non-linear activation function을 사용하므로 더 복잡한 interaction 표현 가능
- 기존의 rating과 reconstructed rating의 RMSE를 최소화하는 방향으로 학습
  - 관측된 데이터에 대해서만 역전파 및 파라미터 업데이트 진행

- 무비렌즈와 넷플릭스 데이터셋에서  RBM, MF 등의 모델보다 좋은 성능을 보임

### CDAE

- Collaborative Denoising Auto-Encoders for Top-N Recommender Systems
- Denoising Autoencoder를 CF에 적용하여 top-N 추천에 활용
- 문제 단순화를 위해 유저-아이템 상호작용 정보를 binary로 바꿔서 학습 데이터로 사용
- 과정
  - 인코더
    - 노이즈가 추가된 유저-아이템 상호작용 정보들과 유저노드(V)를 인코더에 함께 넣음
    - 유저에 따른 특징을 해당 파라미터가 학습하고 Top N 추천에 사용
  - 디코더
    - latent representation으로 다시 원본 유저-아이템 상호작용 정보들로 복원할 수 있도록 함

## GNN

- Graph Neural Network
- Graph는 Node들을 Edge로 모아 구성한 자료구조
- Graph의 사용 이유
  - 관계, 상호작용 같은 추상적인 개념을 다루기 적합
  - Non-Euclidean Space의 표현 및 학습이 가능

### NGCF

#### GCN

- Graph Convolution Network
- local connectivity, shared weights, multi-layer를 이용하여 convolution 효과를 만듦
  - 연산량을 줄이면서 깊은 네트워크로 간접적인 관계 특징까지 추출 가능
  - 해당 노드의 n개의 엣지 연결만 보겠다는 뜻

#### NGCF

- 유저-아이템 상호작용을 GNN으로 임베딩 과정에서 인코딩하는 접근법을 제시한 논문
- 신경망을 적용한 기존 CF모델들은 유저-아이템 상호작용을 임베딩 단계에서 접근하지 못함
- 기본 아이디어
  - Collaborative Signal — 유저-아이템 상호작용이 임베딩 단에서부터 학습될 수 있도록 접근
  - 유저, 아이템 개수가 많아질수록 모든 상호작용을 표현하기엔 한계가 존재
- 구조
  1. 임베딩 레이어 — 유저-아이템의 초기 임베딩 제공
     - 유저 u에 대한 임베딩과 아이템 i에 대한 임베딩이 바로 사용되지 않고 Graph Convolution Layer로 전파시켜 refine하여 사용 → Collaborative Signal을 명시적으로 임베딩 레이어에 주입하기 위한 과정
  2. **임베딩 전파 레이어** — high-order connectivity 학습
     - 생성된 임베딩을 전파시키는 레이어
     - 유저-아이템의 Collaborative signal을 담을 message를 구성하고 결합
     - Message Construction: 유저-아이템 간 affinity를 고려할 수 있도록 메시지 구성
     - Message Aggregation: u의 이웃 노드로부터 전파된 message들을 결합
     - l개의 임베딩 전파 레이어를 쌓으면 유저 노트는 l-hop 까지 전파된 메시지를 이용할 수 있음
  3. 유저-아이템 선호도 예측 레이어 — 서로 다른 전파 레이어에서 refine된 임베딩 concat
     - L차 까지의 임베딩 벡터를 concat하여 최종 임베딩 벡터 계산 (유저 벡터, 아이템 벡터)
     - 유저-아이템 벡터를 내적하여 최종 선호도 예측값 계산

- 결과
  - MF보다 더 빠르게 수렴하고 recall도 높음
  - MF와 비교하여 유저-아이템이 임베딩 공간에서 더 명확하게 구분
    - representation power가 좋다 → NGCF가 유저-아이템 관계를 더 잘 표현

### LightGCN

- LightGCN의 아이디어
  - LightGCN은 이웃 노드의 임베딩을 가중합 하는 것이 전부 → 학습 파라미터와 연산량 감소
  - 레이어가 깊어질수록 강도가 약해질 것이라는 아이디어를 적용해서 모델을 단순화

- 구조
  1. 임베딩 전파 레이어 
     - NGCF의 방식에서 feature transformation이나 nonlinear activation을 제거
     - 가중합으로 GCN 적용
     - 연결된 노드만 사용하기 때문에 self-connection이 없음
     - 파라미터는 0번째 임베딩 레이어에만 존재
  2. 예측 레이어
     - NGCF는 concat해서 유저 임베딩과 아이템 임베딩을 구했지만 LightGCN은 가중합으로 구함
     - 가중합의 파라미터는 (K+1)^(-1) 사용
       - 레이어가 깊어질수록 가중치가 작아짐
    
- 결과
  - Training loss와 추천 성능이 모두 NGCF보다 뛰어남

## RNN

- Recurrent Neural Network
- 시퀀스 데이터의 처리와 이해에 좋은 성능을 보이는 신경망 구조
- 현재의 상태가 다음의 상태에 영향을 미치도록 루프 구조

### RNN family

#### LSTM

- Long-Short Term Memory
- 시퀀스가 길어지는 경우 학습 능력이 현저히 저하되는 RNN의 단점을 극복하기 위해 고안된 모델
- cell state 구조를 추가

#### GRU

- Gated Recurrent Unit
- LSTM의 변형으로 출력 게이트가 따로 없어 파라미터와 연산량이 더 적은 모델
- LSTM과 성능차이가 별로 없음에도 훨씬 가벼운 모델

### GRU4Rec

- 고객의 선호는 고정된 것이 아님 → ‘지금’ 고객이 원하는 것이 무엇인지?
- Session이라는 시퀀스를 GRU 레이어에 입력해서 바로 다음에 올 확률이 가장 높은 아이템을 추천

- 구조
  1. 입력
     - one-hot 인코딩 된 session (임베딩 레이어를 사용하지 않는 것이 더 높은 성능을 보임)
     - Session의 길이는 짧을수도 길수도 있음 → 길이가 짧은 session이 단독으로 사용되어 idle하지 않도록 세션을 병렬적으로 구성해서 미니 배치 학습
     - 상호작용하지 않은 아이템에 대해 아이템의 인기가 높은데 상호작용이 없으면 사용자가 관심 없는 아이템이라 가정함
  2. GRU 레이어
     - 시퀀스 상 모든 아이템들에 대한 맥락적 관계 학습
  3. 출력
     - 다음에 골라질 아이템에 대한 선호도 스코어

- 결과
  - RSC15와 VIDEO Dataset에서 가장 좋은 성능을 보인 item-KNN 모델 대비 약 20% 높은 추천 성능
  - GRU 레이어의 hidden unit이 클 때 더 좋은 추천 성능을 보임