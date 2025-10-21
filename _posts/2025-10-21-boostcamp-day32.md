---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 32: 경진대회 준비, 모델 학습 파이프라인 & 실험관리"
date: 2025-10-21 18:28:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, nlp, competition, machine learning pipeline, k-fold, wandb]
description: "자연어처리 모델 학습 파이프라인을 이해하자."
keywords: [colab, nlp, data, train data, validation data, test data, 훈련 데이터, 검증 데이터, 테스트 데이터, under sampling, over sampling, K-Fold, Machine Learning Experiment Management]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# 모델 학습 파이프라인 & 실험 관리(feat. WandB)

오늘은 데이터를 나누어서 학습하는 이유, 모델 학습 파이프라인과 머신러닝 실험 관리를 공부해보았습니다. 아래에는 키워드 중심으로 정리하였습니다.

## 데이터 분류
머신러닝에서 학습을 시작하기 전에 데이터를 우선적으로 **학습 데이터, 검증 데이터, 테스트 데이터**로 분리합니다.
* 훈련 데이터(Train Data): 모델을 학습시키기 위해 사용하는 데이터
* 검증 데이터(Validation Data): 모델의 과적합 여부 판단 및 하이퍼파라미터 튜닝에 사용하는 데이터
* 테스트 데이터(Test Data): 모델의 최종 성능을 평가하기 위해 사용하는 데이터

### 검증 데이터의 필요성
* 검증 데이터가 없다면 훈련 데이터로 모델 학습 후 테스트 데이터로 모델 평가 및 튜닝  
  → 테스트 데이터에서 가장 성능이 좋은 모델을 선택  
  → 테스트 데이터에서 모델 성능 확인
  * 데이터 스누핑 편향
  * 해당 과정에서 테스트 데이터만 잘 맞추고 일반화되지 않는 모델이 구현됨
  * 따라서 검증 데이터로 모델 평가 및 튜닝을 진행

### 분할 데이터의 비율

#### 데이터 Label의 비율
데이터를 분할할 때에 데이터의 수가 많은 경우에는 **큰수의 법칙**에 의해 무작위로 분할해도 레이블 비율이 유지되지만,  
그렇지 않은 경우(**데이터가 적은 경우**)에는 **비율이 깨질 수 있기 때문**에 무작위로 분할하면 안됩니다.  
→ `train_test_split()` 할 때에 `stratify=y`인자를 추가하여 클래스 비율을 동일하게 유지할 수 있도록 합니다.  

#### 나눈 데이터의 비율
비율은 보통 8:1:1을 사용하지만 반드시 그렇지만은 않습니다. 데이터가 많은 경우에는 98:1:1, 데이터가 적은 경우에는 6:2:2 처럼 유동적으로 모델이 잘 학습할 수 있도록 분할합니다.

### 데이터 누수
* **Data Leak**
* 모델을 학습하는 과정에서 예측하려는 타겟 정보가 직/간접적으로 노출
* 모델 성능이 과장되어 나오는 반면에 실제 환경에서는 좋은 성능이 나오지 않음
* 이 경우에 모두 해당
  * 예측 시점 이후의 정보 포함 (오후 2시의 교통량 예측에 **평균 온도를 사용**)
  * 타겟 변수의 직접적인 정보를 제공 (질병에 대한 환자의 진단 모델에 **복용중인 약**을 변수로 사용)
  * 전처리 과정에서 검증/테스트 데이터의 정보가 사용 (**전체 데이터셋의 통계**를 사용하여 전처리를 진행)
  * 똑같은 샘플이 훈련 데이터와 검증/테스트 데이터에 존재

### 데이터 불균형
* 분류 문제에서 클래스 간에 비율 차이가 많이 나는 경우를 의미
* 모델이 다수 클래스에 맞추어 학습되므로 소수 클래스 예측 성능이 떨어질 수 있음
* 문제점
  * 평가 지표의 왜곡 - 오프라인
    * 클래스 A가 95%인 경우 모든 경우에 클래스 A를 예측해도 정확도가 95%
  * 모델의 일반화 성능 저하 - 온라인
    * 실제 환경에서의 모델의 신뢰성 저하

#### 해결법
분류 문제에서는 데이터 불균형을 다음과 같이 해결합니다.

##### 언더 샘플링: 적은 데이터셋에 맞추는 것
* Random Under Sampling: 랜덤이기 때문에 처리 속도가 빠르지만 샘플링마다 다른 결과가 나올 수 있음
* Near Miss: 텍스트를 임베딩으로 변환하고 벡터 공간에서 소수 클래스 데이터 포인트에 가장 가까운 다수 클래스 데이터를 선택
* Tomek Links: 서로 다른 클래스에 속하는 두 데이터가 서로에게 가장 가까운 경우(토멕 링크), 다수 클래스에 속하는 데이터를 제거
  
##### 오버 샘플링: 많은 데이터셋에 맞추는 것 (데이터 증강이 필요)
* EDA
  * 동의어 대체
  * 무작위 삽입
  * 무작위 삭제
  * 무작위 교환
* AEDA
  * 특수 기호를 무작위 위치에 삽입 (이 영화 정말 재미있어 → 이 영화, 정말 재미있어!)
* Back Translation
  * 원본 문장을 중간 언어로 번역한 뒤 다시 원본 언어로 번역
* LLM 기반 증강
  * LLM을 이용하여 기존 텍스트의 맥락을 유지한 채 새로운 문장 생성

##### 회귀 문제에도 데이터 불균형이 있을까?
* 타겟 변수가 한쪽으로 치우친 경우가 존재할 수 있음
  → 종모양의 분포로 변환해서 학습
* 추론할 때에는 다시 원래 분포로 변환

## 모델 학습 파이프라인

### 모델의 평가?
모델의 평가는 온라인 성능과 오프라인 성능으로 나눌 수 있습니다.
온라인 성능(평가)은 **실제 모델 배포 단계에서 평가하는 모델의 성능**,  
오프라인 성능(평가)은 **모델 학습 단계에서 검증 데이터로 평가하는 모델의 성능**입니다.

### 오프라인 평가 지표
* Accuracy(정확도): 전체 예측에서 올바르게 예측된 비율
* Precision(정밀도): 예측된 클래스에 대한 실제 클래스의 비율
* Recall(재현율): 실제 클래스에 대해 올바르게 예측된 클래스의 비율
* F1 Score: 정밀도와 재현율의 조화 평균
* ROC-AUC(Area Under the ROC Curve): 다양한 임계값에서의 TPR과 FPR의 관계를 곡선으로 나타냈을 때 아래 면적

#### Accuracy
* 가장 직관적이고 이해하기 쉽다
* 하지만 클래스 분포가 불균형일때는 사용에 주의
* `Accuracy = (TP + TN) / (TP + TN + FP + FN)`

#### Precision
* False Positive가 치명적인 경우 사용 (정상 메일(TN)을 스팸 메일(FP)로 판단)
* `Precision = TP / (TP + FP)`

#### Recall
* False Negative가 치명적인 경우 사용 (암 진단에서 양성(TP)을 음성(FN)으로 판단)
* `Recall = TP / (TP + FN)`

#### F1 Score
* 정밀도와 재현율을 모두 만족시켜야 할 때 사용
* `F1 = 2 * (Precision * Recall)/(Precision + Recall)`

#### ROC-AUC
* 여러 임계값을 고려한 모델의 전반적인 분류 성능을 평가
* `TPR = TP / (TP + FN)`
* `FPR = FP / (FP + TN)`
* 이 두 관계를 곡선으로 나타냈을 때의 아래의 면적

## Cross-Validation
- 교차 검증 - K-Fold
  - K-Fold는 K만큼 등분내어 돌아가며 Validation Data의 역할을 합니다.

## 머신러닝 실험 관리
Machine Learning Experiment Management는 모든 실험의 과정과 결과를 체계적으로 관리하는 활동입니다.

아래의 내용을 기록합니다.
* Code: 어떤 코드로 실험했는가 (코드 버전)
* Hyperparameters: 어떤 하이퍼파라미터 조합을 사용했는가
* Data: 어떤 데이터로 학습하고 검증했는가 (데이터셋 버전)
* Environment: 어떤 환경에서 실행했는가 (라이브러리, 하드웨어)
* Metrics & Models: 결과로 어떤 성능(평가 메트릭)을 보였고 어떤 모델(모델 버전)이 만들어졌는가

### 실험 관리가 중요한 이유
* 모델 성능 향상 및 최적화 용이
  * 실험 관리가 없다면 수많은 실험을 하더라도 어떤 요인이 성능 향상에 기여했는지 파악하기 어려움
  * 실험 관리를 통해 각 실험의 메타데이터를 비교분석하여 모델 성능에 영향을 미치는 주요 요인을 식별할 수 있음
* 실험 재현성 확보 - 과거 실험 추적 가능
* 협업 및 커뮤니케이션 촉진

### WandB
WandB는 Weights & Biases로 간편하게 실험 관리를 할 수 있습니다.
* PyTorch, TensorFlow, Keras 등 모든 프레임워크와 간편한 통합
* CPU, GPU 사용량 등 리소스 실시간 모니터링
* 웹브라우저를 통한 직관적인 결과 확인 및 팀원과 손쉬운 공유 등등
- shell에 아래를 입력하여 wandb에 로그인
  ```shell
  export WANDB_API_KEY=<api_key>
  
  pip install wandb
  wandb login
  ```

#### 대시보드
* Projects & Runs
  * Runs는 한 번의 실험에 대한 모든 기록 저장
  * Projects는 Runs를 담는 상위 폴더
* Workspace
  * 다양한 차트와 테이블을 조합하여 여러 Run을 한눈에 비교하고 분석 가능
  * Panel을 추가하고 레이아웃을 변경할 수 있음
* Panel
  * Charts — Loss, Acc 등 wandb.log()로 기록된 지표들이 시각화되는 기본 단위
  * Table — 예측 결과, 이미지, 텍스트 등 복잡한 데이터를 표 형태로 기록하고 정렬하거나 쿼리하며 상세히 분석하는 강력한 기능
  * System — 실험이 실행되는 동안의 CPU/GPU 사용량, 메모리, 네트워크 상태 등을 실시간으로 보여주는 패널
* Python 코드에 연동하는 방법
  * wandb.init : 실험 추적을 시작하고 초기화
  * wandb.config : 하이퍼파라미터처럼 실험 시작 전에 고정되는 설정값을 저장
  * wandb.log : 학습 과정에서 변화하는 Loss, Accuracy 같은 값들을 기록
* ```python
  # 새로운 run 설정
  wandb.init(
  	project="basic-intro",
  	name="experiment_1",
  	config={
  	  	"learning_rate": 0.02,
  		"architecture": "CNN",
  	  	"dataset": "CIFAR-100",
  		"epochs": 10,
  	},
  )
  ```
  ```python
  # epoch 하나마다
  wandb.log({"epoch": epoch + 1, "accuracy": acc, "loss": loss})
  
  ```
  ```python
  # 실험 추적 종료
  wandb.finish()
  ```

#### W&B Artifacts
* WandB Run에 사용된 입력과 출력을 추적하고 버전을 관리하는 기능
* 실험의 재현성을 완벽하게 보장할 수 있음
* 실험에 사용된 파일 자체를 기록 (데이터셋, 모델 가중치, 결과 파일)

```python
# run 초기화
run = wandb.init(project="preprocess-pipeline")

# 'reviews-processed' 이름의 첫 아티팩트 생성
# 여기의 v0은 전처리되지 않은 원본 데이터를 담음
artifact = wandb.Artifact("reviews-processed", type="dataset")
artifact.add_file('movie_reviews.csv')
run.log_artifact(artifact)
run.finish()
# v0을 불러오고 전처리하여 v1에 담는 방법
run = wandb.init(project="preprocess-pipeline")

artifact = run.use_artifact("reviews-processed:v0")
artifact_dir = artifact.download()
df = pd.read_csv(f"{artifact_dir}/movie_reviews.csv")

# df 전처리 (구두점 제거, 소문자 변환)
df["text"] = df["text"].str.replace("[^\\w\\s]", "", regex=True).str.lower()
df.to_csv("reviews_processed.csv", index=False)

# v0을 생성할 때와 동일한 이름으로 새 아티팩트 생성
new_artifact = wandb.Artifact("reviews-processed", type="dataset")
new_artifact.add_file("reviews_processed.csv")

# W&B에 로깅하면 자동으로 v1이 생성
run.log_artifact(new_artifact)
run.finish()
```

주요 프레임워크와 통합 — Hugging Face, Pytorch Lightning 등 주요 라이브러리와 긴밀한 통합을 제공
```python
from transformers import Trainer, TrainingArguments
from wandb.integration.huggingface import WandbCallback

training_args = TrainingArguments(
    output_dir="./results",
    report_to="wandb",
    ...
)

...

trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		callbacks=[WandbCallback(log_model="checkpoint")],    # 체크포인트마다 모델을 Artifact로 자동 저장
)

trainer.train() # 이제 자동으로 Artifact에 모델이 저장됨
```

#### Reports 기능
* 실험 결과(차트, 표)와 분석 내용(마크다운 텍스트)을 결합하여 하나의 인터랙티브 문서로 만드는 기능
* “왜 이 실험을 했는지”, “결과는 어떠했는지”, “다음 스텝은 뭔지”를 팀원들과 쉽게 공유하고 논의할 수 있음

#### 추가로 WandB는…
* 실험 추적뿐만 아니라 MLOps 파이프라인 전반을 지원하는 기능을 제공
  * Sweeps — 여러 하이퍼파라미터 조합을 자동으로 탐색하여 최적의 조합을 찾아줌
  * Model Registry — 훈련된 모델의 버전을 관리하고 Staging, Production 등 배포 단계를 체계화하는 중앙 저장소
  * Automations — “정확도 95% 달성시 슬랙 알림 보내기” 등 특정 조건에 따라 후속 작업을 자동화

