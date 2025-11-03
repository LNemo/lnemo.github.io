---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 41: Generative AI, Intro와 LLM"
date: 2025-11-03 17:42:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, generative ai, nlp, llm]
description: "생성형 AI를 이해하자."
keywords: [generative ai, llm, statistical lm, neural lm, pretrained lm, llm, peft]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Generative AI Intro
오늘은 Generative AI에 대해 공부하였습니다. Generative AI의 발전 과정과 LLM의 발전 과정을 알아보겠습니다.

## Generative AI 개요

### LLM은?
- LLM(Large Languaage Model): 텍스트를 입력으로 받아 적절한 출력을 산출하는 언어 모델
- 대량의 텍스트 데이터로 사전학습
- Billion Scale의 파라미터 보유

- InstructGPT/ChatGPT 이후 활발한 연구와 적용이 이뤄지고 있음
  - 학습 방법론 관련 연구(Corpus 정제, Instruction Tuning), 활용 연구(특정 도메인, 서비스 적용 디자인), 최적화 연구(추론 속도/메모리 사용량 최적화, 입력 문장 길이 확장)

#### NLP 모델 발전
- Statistical LM: 통계 및 어휘 빈도 기반 방법론
  - 단어를 단순히 카운팅만 하기때문에 단어의 의미를 반영하지 못함
  - 특정 시스템에 적용하기 위해서는 많은 엔지니어링 비용 소모
  - 시스템 별 별도 모델 구축곽 학습이 필요하여 시스템 간 전이 불가
- Neural LM: 딥러닝 기술 자연어 처리 분야 적용
  - 단어의 의미를 고정된 크기의 벡터에 표현(Word2Vec 등)
  - 일부 태스크 적용 가능(감성 분석 등)
  - 사전 학습된 언어 벡터 활용 가능
  - 하지만 시스템 별 별도 모델 구축곽 학습이 필요하여 시스템 간 전이 불가
  - 순서를 고려하지 않기 때문에 단어 맥락을 반영하지 못함
- Pretrained LM: 사전학습된 언어모델 개발 및 활용
  - 대량의 코퍼스로 사전학습된 언어모델 사전 학습 및 Finetune(BERT, T5, GPT-1,2)
  - 다양한 태스크 적용 가능
  - 각 태스크 별 Finetune 데이터 구축 필요
  - 일부 시스템에는 적용할 수 없음(챗봇, 창의적 글쓰기, 코드 생성)
- LLM: 대형 언어 모델
  - 대량의 코퍼스를 많은 파라미터의 모델에 대해 사전학습 수행
  - 일련의 Finetune 과정으로 최종적인 모델 학습 종료
  - 매우 다양한 태스크에 적용 가능
  - 태스크 별 Finetune 데이터 구축할 필요가 없음 → 프롬프트로 각 태스크에 적용 가능

#### LLM 활용 필요성

- LLM은 비용이 크기 때문에 다음의 상황에서 사용
  - 인간 행동 모사가 필요할 경우
  - 태스크가 매우 어려운 경우
  - 데이터가 매우 제한적인 경우
  - 사실 정보를 기반으로 생성해야 하는 경우

### Vision Generative AI

- 생성형 이미지 모델은 특정 데이터의 분포를 기반으로 새로운 이미지를 생성하는 모델
- 생성형 이미지 모델의 학습 목표는 특정 데이터를 생성할 확률인 likelihood를 최대화하는 것

#### 대표적인 생성형 이미지 모델

- **GAN**: 진짜 이미지와 생성된 이미지를 구별하지 못하도록 학습
- **VAE**: encoder로 이미지를 벡터로 만들면 잠재변수 z 벡터를 다시 decoder를 통해 이미지를 생성해낼 수 있도록 학습
- **Flow-based models**: 이미지를 flow 함수를 통해 잠재공간으로 만들면 그것의 역함수로 이미지를 생성하도록 학습
- **Diffusion models**: 이미지를 여러 단계를 거치도록 해서 잠재공간을 만들었다가 다시 역방향으로 돌아오는 구조로 학습

#### 활용분야

- Style transfer: 이미지 스타일을 다른 이미지에 적용
- Inpainting: 이미지의 손상, 누락된 부분을 복원
- Image editing: 이미지를 변경하거나 개선하는 방법
- Super-resolution: 저해상도 이미지를 고해상도 이미지로 변환

#### Multi-modal 생성형 이미지 모델

- Text-to-Image
- Text-to-Video
- Image-to-Video

## LLM

- 사전학습 데이터 및 파라미터 수가 매우 큰 모델의 종합적 지칭
- 사전학습 데이터는 온라인 상 수집 가능한 최대한의 텍스트 데이터
- 파리미터 수는 하드웨어 상 학습 가능한 최대한의 파라미터 수
  - LLaMA 학습 데이터는 4TB, 파라미터 수는 7B ~ 65B

- Pretrained LM은 Task에 맞게 Finetune을 통한 목적 별 모델 구축이 목표이고,
- LLM은 Finetune을 통해 범용 목적인 모델을 구축하는 것이 목표
- Pretrained LM: GPT-1/2, BERT 등
- LLM: GPT-3/4, ChatGPT, LLaMA, Mistral 등

### Zero/Few-Shot Learning

- LLM의 범용 목적 모델 동작 원리
- 모델 추가 학습 X
- Zero Shot: 모델이 프롬프트만드로 태스크를 이해하고 수행
- Few Shot Learning: 모델이 Prompt와 Demonstration을 통해 태스크를 이해하여 수행

### Prompt

- Zero/Few Shot Learning이 가능한 LLM의 입력 구성 방식
- 구성 요소
  - Task Description: Task에 대한 설명
  - Demonstration: Task 예시
  - Input: 실제 Task를 수행할 입력 데이터

### 모델 아키텍처

- 두가지 모델 구조 — Transformer 구조 변형
  - Encoder - Decoder 구조
  - Decoder Only 구조
- Encoder - Decoder 구조에서는 입력 이해(Encoder)와 문장 생성(Decoder) 모델을 분리하여 처리
- Decoder Only는 단일 모델을 통해서 이해 생성을 동시에 함
- 모델 구조 별 사전 학습 태스크가 다름
  - Encoder - Decoder: Span Corruption (T5에서 제안된 Pretrain Task, 입력 문장의 일부를 복원하는 Task)
  - Decoder Only: Language Modeling (입력된 토큰을 기반으로 다음 토큰 예측)

- 최근 모델의 대부분은 Causal Decoder 구조, Next Token Prediction으로 사전학습
- 모델 크기는 GPT-3 이후로 모델 크기가 점차 확장

### Corpus

- 코퍼스는 사전학습을 위한 대량의 텍스트 데이터 집합
- 코퍼스는 원시데이터로부터 수집
- 온라인 상에서 최대한 많은 데이터를 가져올 수 있지마 학습 불필요 데이터가 존재할 수 있음
  - 욕설 및 혐오 표현, 중복 데이터, 개인정보 등 → 데이터 정제 필요
- LLM이 코퍼스 내의 데이터를 암기할 수 있음 → Memorization in LLM
  - 모델이 클수록 암기 능력이 향상되고 코퍼스 내 중복하여 등장한 데이터를 쉽게 암기함
  - 따라서 데이터 정제를 수행하지 않는다면 부적절한 데이터를 도출할 가능성이 있음

#### 원시 데이터

- 원시 데이터에서 정제가 필요한 경우
  - 무의미한 문장의 중복 (`……..`, `????????`, `!!!!!!!`)
  - 개인정보

### Instruction Tuning

#### Safety & Helpfulness

- 대형 코퍼스로 학습된 LLM은 다양한 문장 생성 능력을 보유
- 대형 코퍼스에는 혐오/차별/위험 표현이 포함될 수 있음  
→ LLM 학습에 반영된다면 해당 표현이 생성될 수 있음
- **Safety**
  - LLM이 생성한 표현이 사회 통념상 혐오/위험/차별적 표현이 아니어야 함
  - 특정 질병에 관해 잘못된 조언 및 진단을 생성하면 안됨
  - 특정 입력에 대해 답변을 거부하거나 우회할 수 있어야 함
- **Helpfulness**
  - LLM은 사용자의 다양한 입력에 적절한 답변을 생성할 수 있어야 함
  - 6살 꼬마에게 달 착륙 과정을 설명해달라고 하였을 때 어려운 이론을 말하는 대신 쉽게 설명할 수 있어야 함

#### Instruction Tuning

- 사전 학습은 이전 단어를 바탕으로 다음 단어를 예측하도록 학습하는 것이고
- Instruction Tuning은 사용자의 광범위한 입력에 대해(instruction) Safety & Helpfulness 답변을 하도록 fine-tune하는 과정

- 3단계로 구성(RLHF 논문 기준)
  1. SFT(Supervised Fine-Tuning): 광범위한 사용자 입력에 대해 정해진 문장을 생성하도록 Fine-tune
  2. Reward Modeling: LLM의 생성문에 대한 선호도를 판별하도록 Fine-tune
  3. RLHF(Reinforcement Learning with Human Feedback): 광범위한 사용자 입력에 대해 인간이 선호하는 답변을 출력하도록 강화학습

#### 효과

- SFT 및 RLHF 학습 시에 사전학습 LLM보다 다양한 지표에서 개선
  - Appropriate for customer assistant 상승
  - Hallucinations 감소
- LLM은 모델의 크기도 중요하지만 **Instruction Tuning 방법론도 중요**

### LLM 발전 과정

- 2020년 이후 General-purpose를 위한 LLM 들이 제안되어 왔음
- 학습 데이터와 파라미터 사이즈가 커질수록 성능이 더욱 높아짐

#### 기존 학습 방법론

**1. Finetuning**

- 언어 모델을 task에 맞추어 adapting 하거나 finetuning
- 3가지 유형
  - Feature-based Approach — 사전학습 모델로부터 embedding을 추출하고 classifier 학습
  - Finetuning I — 사전학습 모델은 그대로 두고 Output layer를 업데이트
  - Finetuning II — 사전학습 모델을 포함한 모든 layer들을 업데이트
- 모든 파라미터를 학습하는 경우(Finetuning II)에 가장 높은 성능을 기록

**2. In-context Learning (ICL)**  

- GPT3 발표 이후에는 finetuning 없이도 언어모델을 쉽게 활용할 수 있게 됨
- Task에 대해 몇가지 예시를 모델에 입력해주게 될 경우 모델을 튜닝하지 않고 쉽게 문제를 풀 수 있게 되었음
- 하지만..
  - 성능 측면에서  
    - 모델을 지속적으로 추가 학습하는 과정에서 언어 모델이 기존에 학습한 정보를 잊는 현상  
    - 모델의 모든 파라미터를 새로운 데이터에 대해 학습하는 것이 항상 정답은 아님  
  - 자원 측면에서  
    - 모델의 크기가 매우 커져서 전체 파라미터를 학습하는 것이 어려움  
    - 각 task마다 독립적으로 학습된 모델을 저장하고 배포할 때에 막대한 시간과 컴퓨팅 자원이 필요  
  - 신뢰성 측면에서  
    - Task의 예시에서 몇몇 경우에 random한 label을 넣어주더라도 문제를 잘 해결한다는 연구 결과 존재  
    → ICL의 결과물을 항상 신뢰하기 어려움  

#### PEFT

- 모델의 크기가 커짐에 따라 전체 파라미터를 학습하는 것이 어려워지고, 각 task마다 독립적으로 학습된 모델을 저장하고 배포할때 막대한 자원이 필요  
→ 파라미터 수가 많은 LLM도 효율적으로 학습할 수 있는 방법은 없을까?
- 모델의 모든 parameter를 학습하지 않고 일부 파라미터만 Finetuning하는 방법
- 대표적인 방법
  - Adapter Tuning
  - Prefix Tuning
  - Prompt Tuning
  - Low-Rank Adapation

##### Adapter

- 이미 학습이 완료된 모델에 각 레이어에 학습 가능한 FFN을 삽입하는 구조
- Adapter layer의 과정 (bottleneck architecture)
  1. Transformer의 vector를 더 작은 차원으로 압축
  2. 비선형 변환
  3. 다시 원래 차원으로 복원
- Adapter는 finetuning 단계에서 특정 task에 대해 최적화
- 이때 다른 transformer 레이어는 고정
- Adapter를 제안한 논문에서는 매우 작은 학습 파라미터로도 finetuning에 근접한 성능을 기록함

##### Prefix Tuning

- Transformer의 각 레이어에 prefix라는 훈련 가능한 vector를 추가하는 방법
- prefix는 가상의 embedding으로 간주될 수 있음
- 각 task를 더욱 잘 해결하기 위한 벡터를 최적화하여 기존 모델과 병합할 수 있음

##### Prompt Tuning

- Prefix tuning과 달리 모델의 입력 레이어에 훈련 가능한 prompt vector를 통합하는 방법
- embedding layer를 최적화하는 방법론
- 문장에 직접적인 자연어 prompt를 덧붙이는 것과 다른 개념
- Prompt tuning도 target task에 최적화 가능

##### **LoRA**

- PEFT 중 가장 널리 쓰이는 방법론
- 사전 학습된 모델의 파라미터를 고정하고 학습 가능한 rank decomposition 행렬을 삽입하는 방법
- 행렬의 차원을 rank 만큼 줄이는 행렬과 다시 원래 차원 크기로 되돌려주는 행렬고 구성 (Low rank decomposition)
- 레이어마다 hidden states에 lora parameter를 더하면 tuning
- LoRA는 새롭게 학습한 파라미터를 기존 모델에 합쳐 줌으로써 추가 연산이 필요하지 않음
- 기존 방법들 대비 월등히 높은 성능을 보임