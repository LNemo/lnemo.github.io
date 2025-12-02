---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 56: RecSys 기초 프로젝트, 추천 시스템과 CF"
date: 2025-12-02 18:28:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, recsys]
description: "Recommander System에 대해 배우자."
keywords: [Recommendation System, Collaborative Filtering, Matrix Factorization, Content-based Filtering, TF-IDF, Association Rule, Apriori, ALS, BPR, Evaluation Metrics]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# 추천 시스템과 CF

오늘은 추천 시스템 프로젝트를 시작하기 위해 알아야 할 기본 지식들과 Collaborative Filtering에 대해 공부하였습니다.

## 추천시스템이란?

- 과거에는 유저가 접할 수 있는 상품과 컨텐츠가 제한적이었지만 현재는 다양한 상품, 컨텐츠가 등장하면서 정보를 찾는데 시간이 오래 걸림
- Long Tail Phenomenon
  - 상품이 많아짐에 따라 사람들이 유명한 것들만 찾게됨
  - 유명하지 않은 상품들이 훨씬 많음에도 불구하고 소수의 유명한 상품들만 더 유명해짐
  - 여기서 유명하지 않은 것들을 Long-Tail이라고 부름

### 추천시스템에서 사용하는 데이터

- 유저 관련 정보
  - 식별자
  - 데모그래픽 정보 (성별, 연령, 지역 등)
  - 유저 행동 정보
- 아이템 관련 정보
  - 아이템 ID
  - 아이템 고유 정보
- 유저-아이템 상호작용 정보
  - Feedback
    - Explicit Feedback — 별점 등
    - Implicit Feedback — 유저가 아이템을 클릭하거나 구매

### 문제 정의

- 랭킹: 유저에게 적합한 아이템 Top K개를 추천
- 예측: 유저가 아이템을 가질 선호도를 정확하게 예측

## 추천 시스템의 평가 지표

추천 시스템은 다른 AI의 평가와 다르게 적용해야 할 것 같은데 어떻게 적용하여야 할까

### Offline Test

- train/valid/test로 나누어 평가
- Offline Test에서 좋은 성능을 보여야 Online 서빙에 투입되지만 실제 서비스에서는 다양한 양상을 보일 수 있음
- 성능 지표
  - 랭킹
    - Precision@K — 우리가 추천한 아이템 K개 중 실제 유저가 관심있는 아이템의 비율
    - Recall@K — 유저가 관심있는 아이템 중 우리가 추천한 아이템의 비율
    - AP@K — Precision@1부터 Precision@K 까지의 평균값 (관련 아이템의 순위를 높게 추천할수록 점수 상승)
    - MAP@K — 모든 유저에 대한 Average Precision 값의 평균
    - NDCG@K
      - 가장 많이 사용되는 지표 중 하나
      - 원래는 검색에서 등장한 지표
      - 추천의 순서에 가중치를 더 많이 두어 평가하며 1에 가까울수록 좋음
  - 예측
    - RMSE, MAE

### Online Test

- Offline Test에서 검증된 가설이나 모델을 이용해 실제 추천 결과를 서빙
- Online A/B Test

## 인기도 기반 추천

- 인기도 기반 추천은 서비스 초기에 데이터가 부족할 때에 사용할 수 있는 방법으로 가장 인기있는 아이템 추천해주는 것
- 인기도의 척도는 조회수, 평균 평점, 리뷰 개수, 좋아요/싫어요 수

### Most Popular

- 가장 많이 조회된 것도 중요하지만 최선성도 중요
  - 2025년에 2015년도 뉴스 기사를 가져온다면?
- Hacker News Formula
  - score = (pageviews - 1)/ (age+2)^gravity
  - 시간에 따라 줄어드는 것을 조절하기 위해 gravity라는 상수 사용
- Redit Formula
  - score = log_10(ups-downs) + (sign(ups - downs)*seconds / 45000)
  - 첫번째 term은 popularity, 두번째 term은 포스팅 된 절대 시간
  - ups-downs를 로그취함 = 첫 vote에 대해서 가장 높은 가치를 부여

### Highly Rated

- 가장 높은 평점을 받은 것을 추천할 때 중요한 것은 신뢰할 수 있는 평점인지 중요
- Steam Rating Formula
  - score = avg_rating - (avg_rating - 0.5) * 2^(-log(# of reviews))
  - avg_rating = # of positive reviews / # of reviews
  - review 개수에 따라 rating을 보정
  - review가 많을 경우에 score가 평균 rating과 유사해짐
  - 0.5 부분을 다른 평가 점수의 중앙값을 넣어서 다른 rating에 사용할 수도 있음

### EDA 팁

- 데이터 이해 및 구조 파악
  - 데이터셋 구성 확인
  - 각 피처 의미와 데이터 타입 파악
  - 결측값 확인
- 데이터 분포 시각화
  - 평점(예측값) 분포 확인
  - 사용자 및 책의 빈도 분석
  - 상관관계 분석 (피처-피처, 피처-예측값)
- 범주형 데이터 처리
  - 카테고리 분포 및 연관성 분석
  - 데이터에 맞는 적절한 인코딩 적용
  - 연속형 데이터를 범주형으로 사용
- 데이터 정제
  - 이상치 필터링
  - 중복데이터 제거
  - 결측값 처리

## 연관 분석

- 연관 분석은 흔히 장바구니 분석 혹은 서열 분석이라고도 불림
- 상품의 구매, 조회 등 하나의 연속된 거래들 사이의 규칙을 발견하기 위해 적용
- 하나의 transaction에 대해 하나의 상품이 등장했을 때 다른 상품이 같이 등장하는 규칙을 찾는 것

### 연관 규칙

- 규칙: IF (condition) THEN (result)
- 연관 규칙: IF (antecedent) THEN (consequent)
  - 특정 사건이 발생했을 때 함께 빈번하게 발생하는 또 다른 사건의 규칙을 의미
- antecedent와 consequent는 disjoint(서로소)를 만족
  - ex) antecedent: {빵, 버터}, consequent: {우유}
- Itemset: antecedent와 consequent 각각을 구성하는 상품들의 집합
- Frequent Itemset(빈발 집합)
  - 유저가 지정한 minimum support 값 이상의 itemset을 의미
  - support: itemset이 전체 transaction data에서 등장하는 비율
  - Frequent Itemset 사이의 연관 규칙을 만들기 위해서는 measurement가 필요

#### 척도

- support
  - 두 itemset X, Y를 모두 포함하는 transaction의 비율 ( s(X→Y) )
- confidence
  - X가 포함된 transaction 가운데 Y도 포함하는 transaction 비율
  - confidence가 높을수록 유용한 규칙임을 뜻함
- lift
  - (X가 포함된 transaction 가운데 Y가 등장할 확률) / (Y가 등장할 확률)
  - 1을 기준으로 판단
    - lift = 1 → X, Y는 독립
    - lift > 1 → X, Y는 양의 상관관계를 가짐
    - lift < 1 → X, Y는 음의 상관관계를 가짐

- Item 수가 많아지면 가능한 itemset이 많아짐
- Itemset이 많아지면 가능한 rule 수도 많아짐
- 이 중에서 유의미한 rule만 사용하도록 해야함

#### 사용법

1. minimum support, minimum confidence로 의미없는 rule을 screen out
   - 전체 transaction 중에서 너무 적게 등장하거나, 조건부 확률이 아주 낮은 rule을 필터링 하기 위함
2. lift값으로 내림차순 정렬을 해서 의미있는 rule을 평가
   - lift가 크다는 것은 rule을 구성하는 antecedent와 consequent가 연관성이 높고 유의미 하다는 뜻이기 때문

### 연관 규칙 탐색

- Mining Association Rules
  - 주어진 트랜잭션 가운데 아래 조건을 만족하는 가능한 모든 연관규칙을 찾아야함
    - support ≥ minimum support
    - confidence ≥ minimum confidence
- Brute-force approach
  - 무식하게 다 찾아버리면 되지
  - 모든 연관 규칙에 대해 개별 support와 confidence를 계산해서 조건을 만족하는 rule만 남기고 모두 pruning
  - 당연히 엄청나게 많은 계산량을 요구
- minimum support 이상의 모든 itemset을 생성하는 계산비용이 크기 때문에 이부분을 해결해야함
  - 가능한 후보 itemset의 개수를 줄인다 — Apriori 알고리즘 *
  - 탐색하는 transaction의 개수를 줄인다 — DHP(Direct Hashing & Pruning) 알고리즘
  - 탐색 횟수를 줄인다 — FP-Growth 알고리즘

## TF-IDF 활용 컨텐츠 기반 추천

- 컨텐츠 기반 추천은 유저가 과거에 선호한 아이템과 비슷한 아이템을 추천하는 방법
- 장점
  - 다른 유저의 데이터 없이 해당 유저의 데이터로만 추천 가능
  - 새로운 아이템 혹은 인기도 낮은 아이템 추천 가능
  - 추천 아이템 설명 가능
- 단점
  - 아이템의 적합한 피처를 찾는 것이 어려움
  - 한 분야/장르의 추천 결과만 계속 나올 수 있음
  - 다른 유저의 데이터 활용 불가

### Item Profile

- 추천 대산이 되는 아이템의 feature로 구성된 item profile을 만들어야 함
  - 벡터 형태로 표현하는 것이 일반적
- 문서의 경우에는 중요한 단어들의 집합으로 표현 가능
- TF-IDF (Term Frequency - Inverse Document Frequency)
  - 문서 d에 등장하는 단어 w에 대해서
  - 단어 w가 문서 d에 많이 등장하면서 (TF)
  - 단어 w가 전체 문서(D)에서는 적게 등장하는 단어라면 (IDF)
  - 단어 w는 문서 d를 설명하는 중요한 feature로 TF-IDF 값이 높음

### TF-IDF Formula

- TF-IDF(w, d) = TF(w, d) * IDF(w)
- TF(w, d)는 단어 w가 문서 d에 등장하는 횟수
- IDF(w)는 전체 문서 가운데 단어 w가 등장한 문서 비율의 역수 (smoothing을 위해 log)

### User Profile

- TF-IDF로 아이템(d)마다 벡터를 만들었다면 기반으로 User Profile을 만들 수 있음
- ex) 사용자가 d1, d3를 선호한다면 단순히 d1의 벡터와 d3의 벡터를 평균내는 방법

### User Profile 기반 추천

- Cosine Similarity: 두 벡터의 각도를 이용하여 구할 수 있는 유사도
  - 두 벡터의 차원이 같아야함
  - 직관적으로 두 벡터가 가리키는 방향이 얼마나 유사한지를 나타냄
- cos(u,i): 유저 벡터와 아이템 벡터 간의 거리를 계산
  - 유사도가 클수록 해당 아이템이 유저에게 관련성이 높음
- 이렇게 각각에 아이템에 대해 계산한 다음 가장 유사한 아이템에 대해 유저에게 추천 → Ranking

### Rating 예측

- 유저가 선호하는 아이템을 기반으로 새로운 아이템에 대해 평점을 예측
- sim(i, i’) = cos(v_i, v_i’)
- sim(i, i’)를 가중치로 사용하여 i’의 평점을 추론
  - 각 가중치를 각 평점과 곱한 값을 모두 더한 것을 각 가중치들만 모두 더한 것으로 나눔

## Collaborative Filtering

- 협업 필터링(Collaborative Filtering, CF)은 많은 유저들로부터 얻은 기호 정보를 이용해 유저의 관심사를 자동으로 예측하는 방법
- 더 많은 유저/아이템 데이터가 축적될수록 협업의 효과는 커지고 추천은 정확해질 것이란 가정에서 출발
- 최종 목적은 유저(u)가 아이템(i)에 부여할 평점을 예측
- 방법
  1. 주어진 데이터로 유저-아이템 행렬 생성
  2. 유사도 기준을 정하고 유저 혹은 아이템 간의 유사도를 구한다
  3. 주어진 평점과 유사도를 활용하여 행렬의 비어있는 값(평점)을 예측
- **아이템이 가진 특성을 하나도 활용하지 않으**면서 좋은 추천 성능을 보임

### Neighborhood-based CF

- 이웃기반 협업 필터링. Memory-based CF라고도 함
- User-based CF, Item-based CF가 있음

#### User-based CF

- 두 유저가 얼마나 유사한 아이템을 선호하는가?
- 유저 간 유사도를 구한 뒤 타겟 유저와 유사도가 높은 유저들이 선호하는 아이템을 추천

#### Item-based CF

- 두 아이템이 유저들로부터 얼마나 유사한 평점을 받았는가?
- 아이템 간 유사도를 구한 뒤 타겟 아이템과 유사도가 높은 아이템 중 선호도가 큰 아이템을 추천

#### 특징

- 구현이 간단하고 이해가 쉽다
- 하지만 아이템이나 유저가 늘어날 경우 확장성이 떨어짐 (Scalability)
- 주어진 평점/선호도 데이터가 적을 경우에 성능이 떨어짐 (Sparsity)

#### Sparsity

- 주어진 데이터를 활용해 유저-아이템 행렬을 만들때 대부분 원소가 비어 있음 → sparse matrix 희소행렬
- NBCF를 적용하려면 적어도 sparsity ratio가 99.5%를 넘지 않는 것이 좋음
  - 넘는다면 모델 기반 CF를 사용

### KNN CF

- K-Nearest Neighbors CF(KNN CF)
- 유저와 가장 가까운(유사한) 유저를 뽑는 방법

#### Similarity Measure

- 유사도 측정법
- 얼마나 가까운지를 어떻게 알건데?
  - 거리의 역수 개념을 사용
  - 거리를 측정하는 방법에 따라 유사도 측정 방법이 달라짐
- MSDS
  - **M**ean **S**quared **D**ifference **S**imilarity
  - 유저-아이템 rating에 대해 비교하고 싶은 유저의 각 아이템의 차를 구해서 차의 제곱을 모두 더해서 rating 개수로 나눠줌 ← msd(u,v)
  - msd_sim(u, v) = 1 / (msd(u, v)+1)
  - 역수를 취할 때 분모가 0이 되는 것을 방지하고자 1을 더해줌(smoothing)
- Cosine Similarity
  - 두 벡터가 가리키는 방향이 얼마나 유사한지를 의미
  - cos(X, Y) = X·Y / \|X\|\|Y\|
- Pearson Similarity
  - 각 벡터를 표본평균으로 정규화한 뒤에 코사인 유사도를 구한 값
  - 직관적으로 (X와 Y가 함께 변하는 정도) / (X와 Y가 따로 변하는 정도)
  - 1에 가까우면 양의 상관관계
  - 0은 독립
  - -1에 가까울수록 음의 상관관계
- Jaccard Similarity
  - 집합의 개념을 사용한 유사도
  - J(A, B) = \|A∩B\| / \|A∪B\|
  - 차원이 달라도 이론적으로 계산 가능
  - 두 집합이 같은 아이템을 얼마나 공유하고 있는지를 나타냄

### Rating Prediction

#### Absolute Rating

- Average
  - 다른 유저들의 해당 아이템에 대한 rating을 평균을 내어 예측
  - 이렇게 모든 유저의 rating을 동일한 비율로 반영하는게 적절할까? → Weighted
- Weighted Average
  - 유저 간의 유사도 값을 가중치로 설정하여 rating을 weighted average 예측
- 한계
  - 유저가 평점을 주는 기준이 제각각 다름
    - 긍정적인 유저는 3점이 박한 점수, 부정적인 유저는 3점이 후한 점수일수도 있음

#### Relative Rating

- 유저의 평균 평점에서 얼마나 높은지 혹은 낮은지 편차를 사용
- Absolute Rating에서 평점을 구하는데 사용했던 rating 대신 deviation으로 바꾼 다음에 계산
- 결과도 편차로 나오기 때문에 마지막 단계에 해당 유저의 평균 rating을 더해주어야 함

위 prediction 방법들은 당연히 User-based, Item-based 모두 가능함

### MBCF

- Model Based Collaborative Filtering
- NBCF는 Sparsity 문제와 Scalability 문제가 있음
- MBCF에서는 항목 간 유사성을 단순 비교에서 벗어나 데이터에 내재한 패턴을 이용해 추천하기 때문에 Sparsity 문제와 Scalability 문제도 개선 가능

#### NBCF와 비교했을 때 장점

- 모델 학습시에만 데이터가 사용되고 학습된 모델은 압축된 형태로 저장도기 때문에 서빙 속도가 빠름
- Sparsity / Scalability 문제가 개선됨
- Overfitting이 방지됨 (전체 데이터의 패턴을 학습하도록 작동하기 때문)
- Limited Coverage 극복 (NBCF는 유사도 값이 정확하지 않은 경우 이웃의 효과를 보기 어려움)

#### Latent Factor Model

- 유저와 아이템 관계를 잠재적 요인으로 표현할 수 있다고 보는 모델
- 유저-아이템 행렬을 저차원의 행렬로 분해
- 같은 벡터 공간에 유저와 아이템 벡터가 놓일 경우 유저와 아이템의 유사한 정도를 확인할 수 있음
  - 비슷한 위치에 놓이면 유사하다고 생각할 수 있음

### SVD

- Singular Value Decomposition
- 선형대수학에서 차원 축소 기법 중 하나
- Rating Matrix(R)에 대해 유저와 아이템의 잠재 요인을 포함할 수 있는 행렬로 분해

#### Full SVD

- R = U∑Vᵀ
- R은 Users X Items
- U는 Latent Factor의 관계
- V는 아이템과 Latent Factor의 관계
- ∑는 Latent Factor의 중요도

#### Truncated SVD

- Truncated SVD는 대표값으로 사용될 k개의 특이치만 사용
  - (m x n)을 (m x k) (k x k) (k x n)으로 분해
  - (k x k)에서 대각선의 대표 특이치만 사용

#### 한계점

- 분해하려는 행렬의 knowledge가 불완전할 때 정의되지 않음
  - 실제 데이터는 대부분 Sparse Matrix → 결측치가 매우 많음
  - 결측된 entry 모두 채우는 Imputation을 통해 dense matrix를 만들어 SVD 수행
    - Imputation은 entry를 0 또는 유저/아이템의 평균 평점으로 채움
    - 하지만 데이터 양이 늘어나므로 Computation 비용이 높아짐
  - 정확하지 않은 Imputation은 데이터를 왜곡시키고 예측 성능을 떨어뜨림
    - 행렬의 entry가 매우 적으면 SVD를 적용하면 과적합이 쉬움

→ SVD 원리를 차용하되 다른 접근 방법은 없을까? “MF”

### MF

- Matrix Factorization
- User-Item 행렬을 저차원의 User와 Item의 latent factor 행렬의 곱으로 분해하는 방법
- SVD의 개념과 유사하나 관측된 선호도만 모델링에 활용하여 관측되지 않은 선호도를 예측하는 일반적인 모델을 만드는 것이 목표
- Rating Matrix를 P와 Q로 분해하여 R과 최대한 유사하게 R’을 추론
  - R ≈ P × Qᵀ = R’

#### 기본 MF 모델

- 실제 rating과 p와 q를 내적한 값의 차이를 제곱해서 모두 더한것을 목적 함수로 정의하고 최소화
  - 그렇다면 R ≈ R’ 에 가까워질 것
- Penalty term은 L2 Norm을 적용함으로 학습 데이터에 과적합되는 것을 방지
- **관측된 데이터만을 사용**하여 모델을 학습

![MF](/assets/img/posts/boostcamp/day56/mf.png)

- 확률적 경사하강법(SGD)으로 학습

#### MF + ⍺

- Adding Biases
  - 어떤 유저는 모든 영화에 대해 평점을 짜게 줄 수 있음 → 유저에 생기는 편향
  - 어떤 아이템에도 편향이 생길 수 있음
  - 기존 목적 함수에 Bias를 추가하여 예측 성능을 높임

![Adding Biases](/assets/img/posts/boostcamp/day56/mf_adding_biases.png)

- Adding Confidence Level
  - 모든 평점이 동일한 신뢰도를 갖지 않음
  - 신뢰도를 의미하는 c 추가

![Adding Confidence Level](/assets/img/posts/boostcamp/day56/mf_adding_c.png)

- Adding Temporal Dynamics
  - 시간에 따라 변화는 유저, 아이템의 특성을 반영하고 싶음
  - 시간을 반영한 평점 예측

![Adding Temporal Dynamics](/assets/img/posts/boostcamp/day56/mf_adding_t.png)

### MF for Implicit Feedback

- Implicit Feedback 데이터에 적합하도록 MF 기반 모델을 설계하여 성능을 향상

#### Alternative Least Square (ALS)

- Basic Concpet
  - 유저와 아이템 매트릭스를 번갈아가면서 업데이트
  - 두 매트릭스 중 하나를 상수로 두고 나머지 매트릭스를 업데이트
  - pᵤ, qᵢ 가운데 하나를 고정하고 다른 하나로 leat-sqaure 문제를 푸는 것
    - 상수로 고정할 경우 목적함수가 이차함수가 됨 → convex(볼록)
  - Sparse한 데이터에 대해 SGD보다 더 Robust
  - 대용량 데이터를 병렬 처리 가능

- Implicit Feedback 고려
  - Preference — 유저 u가 아이템 i를 선호하는지 여부를 binary로 표현 (0, 1)
  - Confidence — 유저 u가 아이템 i를 선호하는 정도를 나타내는 increasing function
    - c = 1 + ⍺·rᵤᵢ  
![ALS Implicit Feedback](/assets/img/posts/boostcamp/day56/als_implicit.png)


- 기본 해의 형태  
![pq](/assets/img/posts/boostcamp/day56/pq_before.png)

- Confidence / Preference 고려한 해의 형태  
![pq (Confidence, Preference)](/assets/img/posts/boostcamp/day56/pq_after.png)

### Bayesian Personalized Ranking

- Implicit Feedback 데이터를 활용해 MF를 학습할 수 있는 새로운 관점을 제시
- 베이지안 추론에 기반하여 서로 다른 아이템에 대한 유저의 선호도를 반영하도록 모델링
- Personalized Ranking을 반영한 최적화
  - 관측되지 않은 데이터에 대해 아래를 고려
    - 유저가 아이템에 관심이 없는지
    - 유저가 관심이 있지만 아직 모르는지
  - 유저의 아이템에 대한 선호도 랭킹을 생성하여 이를 MF의 학습 데이터로 사용

#### 요약

- Implicit Feedback 데이터만을 활용해 아이템 간의 선호도 도출
- Maximum A Posterior 방법을 통해 파라미터를 최적화
- LEARNBPR이라는 Bootstrap 기반의 SGD를 활용해 파라미터 업데이트
- Matrix Factorization에 BPR Optimization을 적용한 결과 성능 우수