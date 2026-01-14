---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 84: Product Serving, 배포와 모델 관리·평가"
date: 2026-01-14 17:48:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, recsys, serving, cloud]
description: "Recommander System에 대해 배우자."
keywords: [Cloud Computing, AWS, GCP, Serverless, Infrastructure as Code, CI/CD, Github Actions, Docker, MLOps, MLflow, Model Management, Experiment Tracking, Model Evaluation, A/B Testing, Canary Deployment]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Product Serving (2)

오늘은 AI Serving 관점에서 학습하였습니다. 아래는 Cloud와 배포, 모델 관리와 평가에 대한 강의를 듣고 정리해보았습니다.

## 클라우드 서비스

- local로 서비스를 만들고 공유할 수 있지만 local 컴퓨터가 종료되면 해당 웹, 앱 서비스를 이용할 수 없음
- 또한 로컬로 제공하는 서비스에 트래픽이 급증한다면 즉각적으로 확장할 수 없음
- 클라우드 서비스는 이런 문제로들로부터 자유로움

### 클라우드 제품

- Computing Service (Server)
- Serverless Computing — AWS()
- Stateless Container
- Object Storage
- Database
- Data Warebouse
- AI Platform

#### Computing Service (Server)

- 우리가 통상적으로 생각하는 서버
- 가상 컴퓨터로 CPU, Memory, GPU를 선택할 수 있음
- AWS - EC2, GCP - Compute Engine

#### Serverless Computing

- Computing Service와 유사하지만 서버 관리를 클라우드쪽에서 진행
- 코드를 클라우드에 제출하면 그 코드를 가지고 서버를 실행해주는 형태
- 요청 부하에 따라 자동으로 확장(Auto Scaling)
- 마이크로서비스로 많이 활용
- AWS - Lambda, GCP - Cloud Function

#### Stateless Container

- Stateless: 컨테이너 외부에 데이터를 저장, 컨테이너는 그 데이터로 동작
- Docker를 사용한 container 기반으로 서버를 실행하는 구조
- Docker Image를 업로드하면 해당 이미지 기반으로 서버를 실행해주는 형태
- Auto Scaling 가능
- AWS - ECS, GCP Cloud Run

#### Object Storage

- 다양한 형태의 데이터를 저장할 수 있음
- API를 사용해 데이터에 접근 가능
- 머신러닝 모델 pkl 파일, csv, log, 이미지 등 저장 가능
- AWS S3, GCP Cloud Storage

#### Cloud DB

- 웹, 앱 서비스와 데이터베이스가 연결되어 있는 경우가 많으며, 대표적으로 MySQL, PostgreSQL 등을 사용할 수 있음
- 사용자 로그 데이터를 DB에 저장할 수도 있고 Object Storage에 저장할 수도 있음
- AWS - RDS, GCP Cloud SQL

#### Data Wareboud

- 보통 Database는 서비스에서 사용하기 위한 데이터를 저장
- Data Warehouse는 데이터 분석에 특화된 DB
- DB에 있는 데이터, Object Storage에 있는 데이터 등을 모두 Data Warehouse에 저장
- AWS Redshift, GCP BigQuery

#### AI Platform

- AI Research, AI Develop 과정을 더 편리하게 해주는 제품
- MLOps 관련 서비스 제공
- GCP에서 TPU 사용 가능
- AWS SageMaker, GCP Vertex AI

### Cloud Network

- VPC: Virtual Private Cloud
- 실제로 같은 네트워크 안에 있지만 논리적으로 분리한 것
- Cloud Computing Service 사이의 연결 복잡도 줄여줌
- 여러 서버를 하나의 네트워크에 있도록 묶는 개념

#### Subnet

- VPC 안에서 여러 망을 쪼갬
- Public Subnet: 외부에서 접근 가능한 망
- Private Subnet: 외부에서 접근이 불가능한 망

#### Routing

- 경로를 설정하고 찾아가는 길
- 경로 지정

## 코드 배포

- 현업에서 개발할 때 Dev, Staging, Production 환경을 따로 관리해야 함
- Dev — 개발 환경
- Staging — Production 환경 배포 전에 운영하거나 보안 성능 측정하는 환경
- Production — 배포
- 그래서 현업에서의 git flow는
  - main → staging → dev → feature/기능

### CI/CD

- CI(Continuous Integration): 지속적 통합
  - 새롭게 작성한 코드 변경사항이 빌드, 테스트 진행한 후 Test Case 통과했는지 확인
  - 지속적 코드 품질 관리
- CD(Continuous Deployment/Delivery): 지속적 배포
  - 작성한 코드가 항상 신뢰 가능한 상태가 되면 자동으로 배포될 수 있도록 하는 과정
  - CI 이후 CD 진행
  - dev/staging/main 브랜치에 merge가 될 경우 코드가 자동으로 서버에 배포
- 결국 CI는 빌드/테스트 자동화, CD는 배포 자동화

### Github Action

- Github에서 출시한 기능으로 소프트웨어 Workflow 자동화 도구
- public repo는 무료

#### Workflow 예시

- Test Code
  - Unit Test, End to End Test
- 배포
  - FTP로 파일 전송할 수도 있고, Docker 이미지를 푸시할 수도 있음
  - Node.js 등 다양한 언어 배포 지원
- 파이썬, 쉘 스크립트 실행
  - 일정 주기로 Github Repo에 저장된 스크립트를 실행
  - crontab의 대용
  - 데이터 수집을 주기적으로 해야할 경우 활용할 수 있음
- Github Tag, Release 자동으로 설정
  - main 브랜치에 Merge 될 경우에 특정 작업 실행
  - 기존 버전에서 버전 up하기
  - 새로운 브랜치 생성시 특정 작업 실행도 가능
- 사용자가 만들어서 workflow 템플릿을 공유하기도 함

#### Github Action 제약 조건

- 하나의 Github Repository 당 workflow는 최대 20개까지 등록 가능
- workflow에 존재하는 job은 최대 6시간 실행할 수 있고 초과시 자동으로 중지
- 동시에 실행할 수 있는 job 제한 존재

#### Github Action 핵심 개념

- **Workflow**
  - 여러 Job으로 구성. Event로 Trigger되는 자동화된 Process
  - .github/workflows 폴더에 yaml 파일로 저장
- **Event**
  - workflow를 trigger하는 특정 활동, 규칙
  - 예시
    - 특정 branch로 push하는 경우
    - 특정 branch로 PR하는 경우
    - 특정 시간대 반복
- **Job**
  - Runner에서 실행되는 Steps의 조합
  - 다른 job이 있는 경우 병렬로 실행, 순차적으로 실행할 수도 있음
  - 다른 job에 의존 관계를 가질 수 있음
- **Step**
  - Job에서 실행되는 개별 작업
  - Action을 실행하거나 쉘 커맨드 실행
  - 하나의 job에선 데이터를 공유할 수 있음
- **Action**
  - workflow의 제일 작은 단위
  - job을 생성하기 위해 여러 step을 묶은 개념
  - 재사용이 가능한 컴포넌트
  - 개인적으로 action을 만들 수도 있고 marketplace의 action을 사용할 수도 있음
- **Runner**
  - workflow가 실행될 서버
  - vCPU2, Memory 7GB, Storage 14GB
  - self-hosted Runner: 직접 서버를 호스팅해서 사용할 수 있음

#### Github Action으로 Docker Image Build, Push 자동화

- Github의 Repository secrets에 Google Cloud Service Account 설정
- SHA(hash) 값을 태그 값으로 설정해서 매 커밋마다 이름이 겹치지 않도록 새로운 이미지 build + push
- CI가 완료되면 CD를 진행하도록 설정
  - `gcloud compute instances update-container <서버이름> --container-images <컨테이너 이미지>` 로 이미지로 만들어진 인스턴스를 빠르게 업데이트 할 수 있음

## 모델 관리

- 지속적으로 AI 모델을 발전시키고 검증하기 위해서는 모델 관리와 평가가 필요

### 모델 관리 기본

- 모델 메타 데이터 — 모델이 언제 만들어졌고 어떤 데이터를 사용해 만들어졌는지 저장
- 모델 아티팩트 — 모델의 학습된 결과물 (pickle, joblib 등)
- Feature / Data —  모델을 위한 feature, data

### MLflow

- 모델을 관리할 수 있는 오픈소스

#### 사용방법

- `pip install mlflow==2.10.0`
- `mlflow server --host 127.0.0.1 --port 8080`
- `localhost:8080`으로 MLflow UI 접속 가능

#### 핵심기능

- Experiment Management & Tracking
  - 머신러닝 관련 실험들을 관리하고 각 실험의 내용들을 기록할 수 있음
  - 여러 사람이 하나의 MLflow 서버 위에서 각자 자기 실험을 만들고 공유 가능
  - 실험을 정의하고 실험을 실행할 수 있음
  - 각 실행에 사용한 소스코드, 하이퍼파라미터, metric, 부산물 등을 저장
- Model Registry
  - MLflow로 실행한 머신러닝 모델을 Model Registry에 등록할 수 있음
  - 모델을 저장할 때마다 버전이 자동으로 올라감
  - Model Registry에 등록된 모델은 다른 사람들에게 쉽게 공유 가능
- Model Serving
  - Model Registry에 등록한 모델을 REST API 형태의 서버로 Serving 가능
  - input과 output은 모델의 input, output
  - 직접 Docker image를 만들지 않아도 생성 가능

#### 핵심 요소

- Tracking
  - 머신러닝 코드 실행, 로깅을 위한 API
  - 파라미터, 코드 버전, metric, artifact 로깅
  - 웹UI 제공
  - MLflow Tracking을 사용해 여러 실험 결과를 쉽게 기록하고 비교할 수 있음
  - 팀에서 다른 사용자의 결과와 비교하며 협업
- Model Registry
  - 모델 버전 관리
  - 태그, 별칭 지정, 버저닝, 계보를 포함한 모델의 전체 수명 주기를 관리
- Project
  - 머신러닝 코드, workflow, artifact 패키징을 표준화
  - 재현이 가능하도록 관련된 내용을 모두 포함

#### Experiment

- MLflow에서 제일 먼저 Experiment를 생성
- 하나의 Experiment는 진행하고 있는 머신러닝 프로젝트 단위로 구성
- 정해진 metric으로 모델을 평가
- 하나의 experiment는 여러 run을 가짐
- `mlflow experiments search`로 experiment 리스트 확인 가능

#### Run

- `mlflow run <프로젝트이름> --experiment-name <실험이름>`
  - `python_env.yaml`에 정의된 가상환경을 생성하고 실행
  - `--env-manager=local`을 추가하면 가상 환경을 추가로 생성하지 않음
- 편하게 자동으로 로깅 가능: `mlflow.autolog()`
- Run에서 로깅하는 것들
  - Source — 실행한 project의 이름
  - Version — 실행 hash
  - Start & end time
  - Parameters
  - Metrics
  - Tags
  - Artifacts — 실행 과정에서 생기는 다양한 파일들

## 모델 평가

### 오프라인 평가

- 데이터를 train set, test set으로 나누어 모델을 훈련하고 평가

#### 방법

- K-fold Cross Validation
  - K개의 부분 집합으로 나누고 각 폴드를 한 번씩 Test Set으로 사용
- Bootstrap resampling
  - 중복을 허용하여 원본 데이터셋에서 샘플을 랜덤하게 추출하여 여러 개의 부분 집합을 생성
  - 모델을 반복적으로 훈련 및 평가하여 일반화 성능을 추정함

### 온라인 평가

- 실제 배포 환경에서 평가

#### 방법

- A/B Test
  - traffic을 반반으로 나눠서 테스트
  - 통계적 유의미성을 얻기까지 시간이 오래 걸려서 Multi-Armed Bandit 같은 최적화 기법을 같이 씀
- Canary Test
  - 새로운 버전의 모델로 트래픽이 들어가도 문제 없는지 체크
  - 새로운 시스템에 적은 트래픽이 통과하게 함
- Shadow Test
  - 프로덕션과 같은 트래픽을 새로운 버전의 모델에 적용
  - 실제 모든 트래픽은 현재 시스템에 전송하고 트래픽을 복제해서 새로운 시스템에 적용해서 평가

#### QA Pattern

- 예측 서버와 모델 평가를 위한 패턴이 디자인 패턴으로 존재
  - Shadow AB-testing pattern
  - Online AB-testing pattern
  - Loading test pattern

### End-to-End

- 오프라인과 온라인 평가를 반복하면서 최적의 모델을 서빙하기 위해 지속적으로 개선해야 함