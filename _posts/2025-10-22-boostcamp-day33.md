---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 33: 경진대회 준비, 모델 튜닝 기법"
date: 2025-10-22 18:58:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, nlp, competition, hyperparameter, wandb, finetuning, pre-trained model]
description: "자연어처리 모델 학습 파이프라인을 이해하자."
keywords: [colab, nlp, data, tuning, ensemble, hyperparameter, overfitting, underfitting]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# 더 좋은 성능을 위한 모델 튜닝

오늘은 모델 튜닝을 위한 여러 방법들에 대해 공부해보았습니다. 아래에는 키워드 중심으로 정리하였습니다.

## 사전 학습 모델
### 전이 학습
* 대규모 데이터셋으로 미리 학습된 모델(사전 학습 모델)의 지식(가중치)을 사용
* 새로운 태스크에 맞게 미세조정(파인 튜닝)하는 학습 방식

#### 과정
1. 사전 훈련된 모델 선택
2. 사전 훈련된 모델 구성
3. 대상 도메인에 대한 모델 훈련

#### NLP에서 사전 학습 모델의 효과
* 성능 향상 — 대규모 텍스트를 학습하여 언어의 패턴, 문법, 의미론적 관계를 학습한 상태이기 때문에 특정 태스크에서 처음부터 학습하는 모델보다 뛰어난 성능을 보임
* 학습 리소스 단축 — 초기 가중치가 이미 있는 상태이므로 적은 데이터와 시간으로도 목표한 성능 달성이 용이

#### 어떤 사전 학습 모델을 선택해야 할까?
* 사전 학습 모델의 핵심적 역할 중 하나는 **효율적이고 의미론적인 임베딩 생성에 있음**
* 정적 임베딩
  * 단어 하나의 임베딩 벡터가 항상 동일하게 유지
  * 어떤 문맥에서 사용되든지 모든 단어의 벡터는 불변
  * 학습 이후 벡터 값이 고정되어 도메인 변화에 적응 불가능
  * 희귀어 또는 신조어 처리가 힘듦 (형태소나 서브워드 수준 표현 부족)
  * Word2Vec, GloVe 등
* 동적 임베딩
  * 단어의 임베딩 벡터가 문맥에 따라 동적으로 변경
  * Seq2Seq, Transformer 등

### 사전 학습 모델
#### GPT(Generative Pre-trained Transformer)
* Auto-regressive(자기회귀) 생성 방식
* 별도의 파인 튜닝 과정 없이 프롬프트에 몇 개의 예시만으로 다양한 태스크 수행 가능

#### BERT
* Masked Language Model(MLM)
  * 문장 내 일부 단어를 Mask 토큰으로 가린 뒤 가려진 단어를 추측
* Next Sentence Prediction(NSP)
  * 두 문장이 주어졌을 때 원문에서 이어지는 문장인지 아닌지 예측
* 위 두가지 방식으로 모든 단어가 앞뒤 문맥을 동시에 고려하여 학습하도록 구성

#### GPT vs. BERT
* GPT는 Transformer의 Decoder 부분을 사용하여 생성 능력에 특화
* GPT-2는 오픈소스, GPT-3와 GPT-4는 가중치가 비공개이고 API만 사용 가능
* BERT는 Encoder 부분을 사용해서 문장 이해 능력에 특화
* BERT는 라이브러리를 통한 쉬운 접근 및 활용이 가능

#### RoBERTa(Robustly Optimized BERT Pretraining Approach)
* BERT의 사전 학습 방식을 최적화하고 더 많은 데이터로 더 긴 학습
* NSP 태스크를 제거하고 MLM 태스크에 집중
* BERT는 사전 학습 단계에서 한 번 마스킹하지만, RoBERTa는 학습 데이터가 모델에 주입될 때마다 다른 마스킹 패턴을 적용

#### ALBERT
* 파라미터를 줄여 학습속도를 올림
  * 임베딩 행렬을 두 행렬로 분해, 히든레이어와 단어 임베딩을 분리하여 파라미터 수 감소
  * Transformer의 각 레이어 간에 같은 파라미터를 공유 → 모델 크기 및 메모리 사용량, 학습 시간을 효율화
  * 문장 사이의 순서를 학습하여 문장 간의 일관성을 효율적으로 학습 (SOP)

#### DistilBERT
* 지식을 증류(Distilation)하여 경량화
* BERT 모델의 출력을 Student 모델인 DistilBERT가 모방하도록 학습
* BERT-base 모델에 비해 파라미터 수가 40% 적어 추론 속도가 약 60% 더 빠름

#### LLaMA
* GPT와 유사한 Auto-regressive 생성 방식
* 7B, 13B, 30B, 175B 등 버전에 따라 자원 요구량이 다양
* 가중치가 연구 목적으로 공개되어있음 (연구 라이선스기 때문에 연구 목적 외 상업적 사용은 제한)

### Hugging Face Transformers
Hugging Gace Trannsformers 라이브러리는 많은 Transformer 계열의 모델들을 쉽게 사용할 수 있도록 다양한 기능을 제공합니다.
* 다양한 코드와 사전 학습 모델을 쉽게 다운로드하고 파인튜닝 할 수 있도록 지원
* 주요 LLM은 Hugging Face와 같은 플랫폼을 통해 사전 학습 모델의 형태로 배포
* 파인 튜닝은 문장 분류, 질의응답, 개체명 인식 등에 맞추어 마지막 레이어를 추가하여 학습

#### Pipeline
* 사전 훈련된 모델 추론 수행
* NLP, CV, Audio, Multimodal 등 다양한 태스크를 기본적으로 지원

```python
pipe = pipeline(model="openai-community/gpt2", device=0)  # 0번 GPU 사용. CPU 사용시에는 -1
result = pipe(
	"I can do this all day.",   
	truncation=True,            # 입력 길이 제한
	num_return_sequences=1,     # 생성 결과 수 제한
)[0]["generated_text"]          # 리스트의 형태로 반환. 0번 인덱스의 generated_text를 사용
pprint(result)
```
```python
classifier = pipeline(model="facebook/bart-large-mnli")
result = classifier(
	"I can do this all day.",
	candidate_labels=["possible", "impossible"],    # Zero-shot 분류
)
pprint(result)
```
- 미리 정의된 레이블(candidate_labels)에 대해 학습 없이 즉시 다중 클래스로 분류

#### AutoClass
* 사전 학습 모델의 아키텍처와 관련된 토크나이저를 자동으로 가져와 로드하는 바로가기 역할
* 대형 모델을 직접 개발하지 않고 쉽게 로드하여 사용할 수 있고 맞춤형 ML을 쉽게 채택할 수 있음
* LLM을 포함한 다양한 Transformer 계열 모델의 접근성과 활용성을 크게 향상

* **AutoTokenizer**
  * 텍스트를 모델 입력에 필요한 숫자 배열 형태로 전처리
  * 토큰화 규칙들이 포함되어 있음
  * 모델마다 다른 토크나이저 활용
  ```python
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
  ```

* **AutoModel**
  * 사전 훈련된 인스턴스를 간단하게 로드
  * 텍스트 또는 시퀀스 분류를 위한 모델: AutoModelForSequenceClassification 처럼 로드  
    ```python
	model_name = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    inputs = tokenizer("I can do this all day.", return_tensors="pt")
    
    with torch.no_grad():
    	outputs = model(**inputs)
    	logits = outputs.logits
    	probs = F.softmax(logits, dim=-1)
    
    print(logits)
    print(probs)
    ```
* distilbert/distilbert-base-uncased은 별도의 파인튜닝이 필요 (위에서 출력된 값은 의미 없는 값)

#### Trainer
* 훈련 루프를 효율적으로 관리하고 대규모 언어모델을 파인튜닝 하는데 필수적인 기능 제공
* 분산 학습 같은 고급 기능을 제공하여 자원 효율성을 높여줌
* Loss function, Optimizer, Scheduler 변경 가능
* Callbacks으로 훈련 루프를 바꾸지 않으면서 다른 라이브러리와 통합/훈련 상황 보고/훈련 조기 종료 등이 가능
  ```python
  dataset = load_dataset("yelp_reivew_full")
  tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
  
  def tokenize_function(examples):
          return tokenizer(examples["text"], padding="max_length", truncation=True)
  	
  tokenized_dataset = dataset.map(tokenize_function, batched=True)
  small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
  small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
  
  model = AutoModelForSequenceClassification.from_pretrained(
          "google-bert/bert-base-cased", num_labels=5, torch_dtype="auto"
  )
  
  metric = evaluate.load("accuracy")
  
  def compute_metrics(eval_pred):
          logits, labels = eval_pred
          predictions = np.argmax(logits, axis=-1)
          return metric.compute(predictions=predictions, references=labels)
  	
  training_args = TrainingArguments(
          output_dir="test_trainer",
          eval_strategy="epoch"
  )
  
  trainer = Trainer(
          model=model,
          args=training_args,
          train_dataset=small_train_dataset,
          evel_dataset=small_eval_dataset,
          compute_metrics=compute_metrics,
  )
  trainer.train()
  
  ```

## Hyperparameter
처음 AI를 접했을 때에 코딩을 하면서 parameter란 말은 많이 들어봤지만 hyperparameter는 처음 들어봤기 때문에 인터넷 검색을 하며 허우적댔던 기억이 있습니다.

하이퍼파라미터는 모델 학습 과정의 디자인에 반영되는 값으로 **학습 시작 전에 미리 세팅**되는 Learning rate, Loss function, 레이어 수, 배치사이즈 등이 있다. 이 값들은 우리가 직접 조정하는 값으로 모델 내부에서 학습으로 결정되는 변수인 parameter와는 차이가 있습니다.

### Hyperparameter Optimization의 필요성
* 모델 성능
* 학습 시간
* 모델 일반화 (overfitting과 underfitting을 줄일 수 있음)

- 하지만 해당 hyperparameter로 직접 학습시켜야 최적의 값을 찾을 수 있기 때문에 그만큼 시간과 계산 비용이 발생합니다.

### Huggingface 사전학습모델의 하이퍼파라미터
| **항목** | **설명** | **BERT-base** | **BERT-mini** |
|:-:|:-:|:-:|:-:|
| num_layers | Transformer 인코더 층 수 | 12 | 4 |
| hidden_size | 히든 상태 벡터 차원 | 768 | 256 |
| num_attention_heads | self-attention head 수 | 12 | 4 |
| intermediate_size | FFN 내부 차원 | 3072 | 1024 |
| vocab_size | tokenizer vocab 크기 | 30,522 | 30,522 |
| max_position_embeddings | 입력 시퀀스 최대 길이 | 512 | 512 |
| activation_function | 활성화 함수 | GELU | GELU |
| dropout | 드롭아웃 비율 | 0.1 | 0.1 |
|  | 전체 모델 파라미터 개수 | 110M | 22M |

### 훈련 하이퍼파라미터
* num_epochs : epoch 횟수
* early_stopping_patience : 해당 횟수만큼 모델이 개선되지 않을 경우 조기 종료
* batch_size : 미니배치 크기를 결정
* learning_rate : 학습률
* loss_function : BCE, MSE, MAE 등
* optimizer : AdamW 등
* max_seq_length : 최대 squence 길이

### 하이퍼파라미터 튜닝
- 직접 찾아내야 함
1. Grid Search
   * 최적화하고 싶은 하이퍼파라미터 값을 그리드로 설정 후 가능한 모든 경우의 수를 탐색
   * 쉬운 방법이지만 아주 작은 실수일 경우 찾기 어려운 문제가 있음
2. Random Search
   * 튜닝할 하이퍼파라미터의 범위를 지정해서 랜덤으로 값을 설정해서 탐색
   * Grid Search 보다 실수를 찾는데 유리할 수 있지만 여전히 비효율적인 문제
3. Bayesian Search
   * 하이퍼파라미터와 모델 성능 사이에 관계성이 있을 것을 고려하여 최적의 조합을 찾는 것
   * 순서
     * 랜덤 하이퍼파라미터 조합으로 모델 학습 몇 개 시도  
       → 얻은 정보로 예측 모델을 만듦(Gaussian Process)  
       → 모델 성능이 높을 것 같은 곳과 불확실성이 높은 곳을 고려하여 다음 하이퍼파라미터를 선정  
       → 성능을 확인하고 해당 과정 반복

### 하이퍼파라미터 최적화 도구
#### WandB
* Sweep 기능으로 가능
* sweep config
```python
sweep_config = {
	"method": "bayes",    # 하이퍼파라미터 튜닝에 사용할 메서드 설정
	"parameters": {       # sweep으로 서치할 파라미터 정의
		"learning_rate": {
			"values": [1e-5, 5e-5, 1e-4, 5e-4]
		},
		"batch_size": {
				"values": [16, 32, 64]
		},
		"dropout_rate": {
			"distribution": "uniform",    # 서치 범위를 정의
			"min": 0.1,
			"max": 0.5
			},
		"d_model": {
			"values": [64, 128, 256]
		}
	},
	"metric": {           # 성능메트릭을 설정하고 maximize할 지 minimize할지 설정
		"name": "best_auprc_score",
		"goal": "maximize"
	}
}
```
* Sweep 준비
```python
sweep_id = wandb.sweep(sweep_config, project="wandb demo")
```
```python
def wandb_training_function():
	with wandb.init() as run:
	    params = wandb.config
	    model_name = params["model_name"]
	    tokenizer = setup_tokenizer(model_name)
		    
		X_train, X_val, y_train, y_val = train_test_split(
			X.values, y.values, test_size=0.2, random_state=42, stratify=y
		)
		    
		train_dataset = SMSDataset(X_train, y_train, tokenizer, max_length=params["max_length"])
		val_dataset = SMSDataset(X_val, y_val, tokenizer, max_length=params["max_length"])
		train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
		val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)
```
* wandb.init()이 실행되면 이후 실행되는 실험이 wandb 웹에 로깅되기 시작
* params 는 sweep config에 따라 매번 달라짐
* 모델을 초기부터 scratch로 학습하는 경우와 사전학습 모델을 파인튜닝하는 경우를 비교할 수 있음
```python
	if model_name is not None:
	    print("Hugging Face 모델 로딩:", model_name)
	    # 사전학습 모델을 가져와 쓰는 경우
	    # 사전학습 모델은 모델의 임베딩 크기, 레이어 수 등이 미리 정해져 있음
	    # dropout_rate만 조정
	    model = TransformerClassifier(
		    model_name=model_name,
		    num_classes=1,
		    dropout_rate=params["dropout_rate"]
	    ).to(device)
	else:
		print("Scratch 모델 생성")
		# 사전학습 모델을 사용하지 않는 경우
		# 파라미터가 랜덤하게 초기화된 트랜스포머를 학습
		# d_model과 dropout_rate를 조정하며 학습
		model = TransformerClassifier(
			model_name=None,
			vocab_size=30522,
			d_model=params["d_model"],
			nhead=8,
			num_layers=6,
			num_classes=1,
			dropout_rate=1,
			dropout_rate=params["dropout_rate"]
		).to(device)
```
* 학습
```python
	best_val_auprc = train_model(        # 미리 정한 train_model 함수를 통해 모델 학습
		model=model,
		train_loader=train_loader,
		val_loader=val_loader,
		num_epochs=params["num_epochs"], # num_epochs, learning_rate 하이퍼파라미터 사용
		learning_rate=params["learning_rate"],
		device=device
	)
	# 메트릭으로 AUPRC를 사용
	final_train_auprc, _ = evaluate_model(model, train_loader, device)
	final_val_auprc, _ = evaluate_model(model, val_loader, device)
	wandb.log({    # 모델 성능 로깅, sweep에서 최적화 실행
		"best_auprc_score": best_val_auprc,
		"final_train_auprc": final_train_auprc,
		"final_val_auprc": final_val_auprc,
		"total_params" sum(p.numel() for p in model. parameters()),
	})
```
* Sweep 준비가 완료되었다면 wandb.agent() 를 통해 sweep을 진행
```python
wandb.agent(sweep.id, function=wandb_training_function, count=5)
```

#### Optuna
* 머신러닝 모델의 하이퍼파라미터 최적화를 위해 만들어진 오픈 소스 프레임워크
* WandB의 Sweep과 거의 같은 기능 제공
* 웹보다는 notebook 상에서 visualization을 하는데 초점
* Hyperparameter뿐만 아니라 모델도 같이 optimization 대상으로 설정 가능
* Optuna는 trial 함수 안에 sweep할 config와 training function을 모두 넣음
```python
def objective(trial):
	model_name = trial.suggest_categorical("model_name", [None, "prajjwal1/bert-mini", "distilbert-base-uncased"])
	max_length = trial.suggest_categorical("max_length", [64, 128, 256, 512])
	learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 5e-5, 1e-4, 5e-4])
	batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
	num_epochs = trial.suggest_categorical("num_epochs", [2, 3, 4, 5])
	dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
	d_model = trial.suggest_categorical("d_model", [64, 128, 256])
	tokenizer = setup_tokenizer(model_name)
```
```python
	study = optuna.create_study(    # 실험을 관리할 스터디를 만들고 메트릭의 최적화 방향 설정
		direction="maximize",
		pruner=pruner,
		sampler=optuna.samplers.TPESampler()
	)
	study.optimize(objective, n_trials=20)    # 20번 하이퍼파라미터 탐색
	
	trial = study.best_trial
	print(f"Sampler: {study.sampler.__class__.__name__}")    # TPE(Tree-structured Parzen Estimator)
	print(f"Pruner: {study.pruner.__class__.__name__}")
	
	print(f"\\nBest AUPRC Score: {trial.value:.4f}")
	print("Best hyperparameters:")
	for key,value in trial.params.items():
		print(f"	{key}: {value}")
```
* 대시보드로 시각화 가능

## 앙상블
단일 모델을 사용하는 것보다 여러 모델을 같이 사용하는 것이 더 좋은 예측 성능을 보일 수 있습니다. 앙상블은 여러 모델을 같이 사용하여 예측 성능을 높이는 방법입니다.

### Bagging (voting)
* 가지고 있는 데이터셋을 여러 분할을 나누어 여러 모델들에 학습
* 나누어 학습된 모델들의 예측을 하나로 합침 → Voting
* Hard voting
  * 3 Apple / 1 Hotdog → Apple
* Soft voting
  * 0.5, 0.8, 0.7, 0.3 Apple / 0.3, 0.1, 0.2, 0.6 Hotdog → Hotdog
  * 이 방식이 일반적

### Boosting
* 내부적으로 여러개의 모델을 생성
* 모델이 순차적으로 진행(모델1 → 모델2 → 모델3 → 예측)
* 이전 모델에서 예측하지 못했다면 다음 모델이 집중적으로 학습할 수 있도록 함(맞추지 못한 데이터를 다음 모델 학습 데이터에 붙여 넣음)

### 모델의 분산과 편향
* 편향: 얼마나 예측이 실제와 다른지
* 분산: 모델의 예측이 같은 샘플에 대해 얼마나 달라질 수 있는지
* Underfitting: high bias, low variance
* Overfitting: low bias, high variance
* 적정 모델: low bias, low variance

#### 앙상블을 사용하면 편향과 분산이 낮아지는 이유
* 단일모델을 사용하면 overfitting으로 분산이 높아짐→ bagging 앙상블로 분산을 줄일 수 있음
* 높은 bias는 boosting으로 편향을 줄일 수 있음

### 다른 앙상블 기법
#### 교차 검증 앙상블
* k-fold에서 fold 별 각 모델들을 앙상블하여 예측
* 장점
  * 모델의 일반화 성능 평가
  * 과적합 방지
  * 안정된 성능 평가
* 단점
  * 시간과 계산 비용
  * 복잡성 증가
  * 데이터 요구 사항

#### 스태킹 앙상블
* 모델마다 가중치를 주어 앙상블
* 가중치를 어떻게 셋업할 것인지 학습하는 모델을 따로 둠 → **메타 모델**
* 장점
  * 일반화 성능 향상
  * 복잡한 패턴 학습
  * 유연성
* 단점
  * 시간과 계산 비용
  * 복잡성 증가
  * 모델 선택의 어려움

#### Test-Time Augmentation (TTA)
* **샘플을 augmentation**하여 모델에 넣어 예측한 후 평균내어 예측

#### Stochastic Weight Average (SWA)
* 여러 모델의 **가중치**를 평균내어 가중치로 사용

#### Mixture of Expert (MOE)
* 서로 다른 Expert 모델 중 일부만 선택적으로 활성화
* 각 전문가의 특화된 예측을 게이트 모델이 조합하여 전체 예측 성능을 높임
