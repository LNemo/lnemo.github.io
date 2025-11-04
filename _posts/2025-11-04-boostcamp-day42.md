---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 42: Generative AI, Text Generation"
date: 2025-11-04 18:35:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, generative ai, nlp, llm]
description: "생성형 AI를 이해하자."
keywords: [generative ai, llm, open source llm, llama, evaluate]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Text Generation

오늘은 오픈 소스 LLM과 LLM의 평가 방법에 대해 공부하였습니다. 아래에 키워드 중심으로 정리하였습니다.

## Open-Source LLM

### Open-Source License

- 오픈소스 라이선스: 소프트웨어의 코드 공개를 통한 복제·배포·수정에 대한 명시적 권한
- Linux는 GNU GPL 라이선스 → 자유롭게 수정 및 배포 가능
- 개발 목족 및 향후 활용 방안에 따라 활용 소프트웨어 라이선스 검토 필요

- 종류
  - MIT License: 자유로운 복사 및 배포 가능, 유료화 불가능
  - CC-BY-SA 4.0: 자유로운 복사, 배포 및 유료화 가능

- 머신러닝과 딥러닝은 "학습 데이터 + 학습/추론 코드 + 사전학습 모델"로 구성되기 때문에 각 요소마다의 라이선스 및 저작권 검토가 필요

### Open-Source LLM vs. Closed LLM

- 오픈소스 LLM은 학습, 배포, 상업화에 자유로움
- 기업 별 요구사항에 맞추어 Finetune 가능
- 민감 정보를 외부 Closed LLM(GPT API 등)에 활용할 수 없음

### LLaMA

- Meta에서 공개한 연구 목적 활용이 가능한 Open-Source LLM
  - LLaMA1: 연구목적 활용만 가능
  - LLaMA2: 월간 700억 건까지 상업 활용 가능
- 학습 코드 및 모델 공개
- 모델 크기: 7B ~ 70B
- 기존 Open/Closed LLM에 비해 높은 성능을 달성
- 모델 크기에 비해서도 높은 성능
- 어떻게 더 좋은 성능을 보였는지?
  - 더 많은 데이터로 더 오래 학습했기 때문

#### Chinchilla Scaling Law

- 기존 LLM은 제한된 사전학습 자원 내에서 최적의 학습 조건 사용
- Chinchiilla Scaling Law
  - 동일 자원에서 모델 성능을 가장 높이는 학습 데이터 수와 모델 크기 관계식
        
    → 정해진 사전학습 예산 존재 시 모델 크기와 학습 데이터는 반비례 관계
        
  - 제한된 자원 내에서 최적의 성능을 도달할 수 있는 (학습 데이터 수, 모델 파라미터 수)의 조합을 찾음
- LLaMA는 Chinchilla Scaling Law 이상으로 학습
  - 학습 예산 최적화가 중요하지 않고 **추론 시 비용 최소화가 중요하다고 생각**
        
    → 작은 모델을 더 오래 학습시키는 것이 모델 배포 관점에서 효율적이라 생각
        

### Self-Instruct

- LLM의 실제 서비스 활용을 위해서는 Pretrain → SFT → RLHF의 과정이 필요
- SFT, RLHF 학습 데이터 구축 비용이 막대한 문제

#### Demonstration Data

- Demonstration Data의 필수 요건
  - 다양성 — Prompt는 사용자들의 다양한 요청 사항을 담고 있어야 함
  - 적절성 — 답변은 Prompt에 대응하는 적절한 내용을 포함해야 함
  - 안전성 — 답변은 ChatBot으로서 혐오·차별·위현 표현을 담지 않아야 함
- 데이터 크기는 1만건 이상으로 양질의 데이터를 충분히 확보하는 것이 중요

#### Self-Instruct

- 고품질 Demonstration Data를 확보할 수 있는 **자동화된 데이터 구축 방법론**
- GPT API를 이용하여 데이터 구축
- Human Annotator 수준의 데이터 구축, 적은 비용 소모

- 단계
  1. Prompt Pool
     - 데이터 수집을 위한 초기 Prompt Pool 확보
     - 다양한 Task에 대해 Prompt-Answer Pair 구축
     - Human Annotation을 통해서 175개 확보
  2. Instruction Generation
     - 기존의 Pool 내 Prompt(Instruction + Input) 8개를 샘플링하여 In-Context Learning에 활용
     - 각 Task에 적합한 Instruction 구조 사용
  3. Classification Task Identification
     - 생성된 Instruction가 분류문제인지 판단
     - 고정된 In-Context Learning 이용
  4. Instance Generation
     - 생성된 Instruction에 부합하는 답변을 생성
     - 고정된 In-Context Learning Sample 이용
     - 앞 단계에서 Classification인지 확인했을때
       - 분류문제라면 → Output 먼저 생성
       - 분류문제가 아니면 → Input 먼저 생성
  5. Filtering and Post Processing
     - 기존 Task Pool 내 데이터와 일정 유사도 이하인 데이터만 Task Pool 추가
     - 텍스트로 해결할 수 없는 Task 제거
  6. Supervised Fine-Tuning
     - Self-Instruct를 통해 생성한 데이터를 이용한 SFT 학습

#### Alpaca

- 2023년 Stanford의 LLM SFT 학습 프로젝트
- 초기 175개 데이터로 52,000개의 SFT 학습 데이터를 생성
- Alpaca는 GPT API를 이용한 SFT 데이터 생성 및 학습 프레임워크
- API가 수행하지 못하는 Task는 SFT 학습으로 성능 개선에 한계가 있음
- 하지만 API가 잘 이해하는 영역에서는 성능 개선 기대 가능
- Alpaca 이후에 오픈소스 LLM 성능이 Closed LLM과 근접해짐

## LLM Evaluation

- LLM의 평가는 기존 Task 수행 능력 평가와 상이함
- LLM은 범용 Task 수행능력을 평가

### LLM 평가 데이터셋

#### MMLU

- Massive Multitask Language Understanding
- 범용 Task 수행 능력 평가용 데이터셋
- 57개 Task(생물, 정치, 수학, 물리학, 역사, 지리, 해부학)
- 객관식 형태로 평가

#### HellaSwag

- 일반 상식 능력 평가
- 주어진 문장에 이어질 자연스러운 문장 선택
- 객관식 형태로 평가

#### HumanEval

- 코드 생성 능력 평가
- 함수명 및 docstring 입력
- LLM이 생성한 코드의 실제 실행 결과물을 이용하여 평가 진행
- 실행 겨로가물이 실제값과 일치하면 맞춘 것으로 간주

### llm-evaluation-harness

- 자동화된 LLM 평가 프레임워크
- 다양한 Benchmark 데이터를 이용한 평가 가능
- 평가 데이터셋 구성 요소
  - (optional) 고정된 Few-Shot Example
  - Instruction: 해당 task에 대한 설명
  - Choices: 보기 문장
  - Correct Answer: 정답 문장

### G-Eval

- LLM은 대부분 정답이 존재하지 않는 Task 수행 (자기소개서 첨삭, 광고 문구 생성 등)
- G-Eval은 GPT-4를 이용한 생성문 평가 방법론
- 방법
  1. 평가 방식에 대한 Instruction 구성
  2. 평가 기준 제시
  3. 평가 단계 생성
  4. 1~3의 문장을 Prompt로 사용하여 각 요약문에 대한 평가 진행
- 예시
  1. “다음 문장은 뉴스 기사에 대한 요약문입니다. 당신은 요약문에 대해 하나의 점수로 평가해야 합니다. …”
  2. “일관성(1점 ~ 5점) - 문장 전반의 품질. 모든 문장은 잘 구조화 되어 있고 유기적으로 연결되어야 합니다. …”
  3. “1. 뉴스 원문을 주의깊게 읽고… 2. 요약문을 읽고 원문과 비교합니다. … 3. 일관성 점수를 부여합니다. …”

- Human Evaluation Score와 Correlation을 측정하였는데 기존 평가 방법론 대비 높은 Correlation 보유
- 주의사항
  - 평가 기준에 따라 모델이 평가를 진행
  - 모델의 성능에 따라 평가 결과물의 신뢰도 결정
  - 평가 점수 신뢰도 확보를 위한 일부 데이터 검수 필요