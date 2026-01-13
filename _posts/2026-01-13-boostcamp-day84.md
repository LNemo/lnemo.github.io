---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 84: Product Serving, Serving과 FastAPI"
date: 2026-01-13 18:28:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, recsys, serving, fastapi]
description: "Recommander System에 대해 배우자."
keywords: [AI Serving, MLOps, Batch Serving, Online Serving, Serving Patterns, FastAPI, Airflow, Poetry, Pydantic, REST API, Python, Web Programming, Microservices, Background Tasks, APIRouter]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Product Serving (1)

오늘은 AI Serving 관점에서 학습하였습니다. 아래는 Serving, 웹 프로그래밍, FastAPI 등 학습 내용을 정리하였습니다.

## Serving

### Serving 종류

- 모델 서빙은 크게 두가지 방식
  - Batch Serving
  - Online(Real Time) Serving
- 어떤 종류를 선택해야 하는지는 정답이 있지는 않음
  - 문제 상황, 문제 정의, 제약 조건, 개발할 인력 수, 데이터 저장 형태, 레거시 유무 등에 따라 결정

### Batch Serving

- 데이터를 일정 묶음 단위로 서빙, 주기적으로 예측 후 DB에 저장
- 어떤 상황에 사용하는지?
  - 실시간 응답이 중요하지 않은 경우
  - 대량의 데이터를 처리할 때
  - 정기적인 일정으로 수행할 때
- 예시
  - DoorDash 레스토랑 추천
  - Netflix 추천(2021년)

### Online Serving

- 클라이언트가 요청할 때 서빙, 요청할 때 데이터를 같이 제공
- 어떤 상황에 사용하는지?
  - 실시간 응답이 중요한 경우
  - 개별 요청에 대한 맞춤 처리가 중요할 때
  - 동적인 데이터에 대응할 때
- 예시
  - 유튜브 추천 시스템
  - 번역
  - 은행 사기 탐지 시스템

## Serving 패턴

### 단순한 Serving 패턴

- 소프트웨어처럼 ML 서빙에도 디자인 패턴이 존재
- 머신러닝의 특수성으로 별도의 디자인 패턴이 생김
  - 대용량 모델 로드
  - 모델 관리
  - 데이터 대량 전처리
  - 데이터 통계적 확인 후 이상치 제외
  - 예측 요청 후 반응 시간이 오래 소요될 수 있음

### 패턴 종류

- Batch Serving
  - Batch 패턴
- Online Serving
  - Web Single 패턴
  - Synchronous 패턴
  - Asynchronous 패턴

### Batch Serving 패턴

- Batch 패턴
  - 과정
    1. 주기적으로 이 추천모델에 사용자의 행동 데이터를 input data로 넣어서 예측
    2. 예측한 output을 DB에 저장
    3. 추천 결과를 활용하는 서버(서비스 서버)에서는 이 DB에 주기적으로 접근해 추천 결과 노출
  - 주로 Airflow를 사용해서 특정 시간에 주기적으로 Batch Job을 실행

### Online Serving 패턴

- Web Single 패턴
  - API 서버 코드에 모델을 포함시킨 뒤 배포
  - 예측이 필요한 곳에서 직접 Request 요청
  - 장점
    - 보통 하나의 프로그래밍 언어로 진행
    - 아키텍처의 단순함
  - 단점
    - 구성 요소 하나가 바뀌면 전체 업데이트가 필요
    - 모델이 크면 로드하는데 시간이 오래 걸릴 수 있음 → 오래 걸리면 서버에 부하 걸림
- Synchronous 패턴
  - Web Single 패턴을 동기적으로 서빙 (기본적으로 대부분 REST API는 동기적으로 서빙함)
  - 장점
    - 아키텍처가 단순
  - 단점
    - 예측의 병목 현상으로 timeout 될 수 있음
    - 예측 지연으로 사용자 경험 악화
- Asynchronous 패턴
  - 하나의 작업을 시작하고 결과를 기다리는 동안 다른 작업을 할 수 있음
  - 클라이언트와 예측 서버 사이에 메시지 큐를 추가 (Kafka)
  - 장점
    - 클라이언트와 예측 프로세스가 분리되어 관계가 의존적이지 않음
    - 클라이언트가 예측을 기다릴 필요가 없음
  - 단점
    - 메시지 큐 시스템을 만들어야 함
    - 전체적으로 복잡한 구조
    - 완전한 실시간 예측에 적절하지 않음 (메시지 가져갈 때 시간 소요될 수 있음)

### Anti Serving 패턴

- 하면 안되는 패턴
- Online Bigsize 패턴
  - 실시간 대응이 필요한데 예측이 오래 걸리는 모델을 사용하는 것은 적절하지 않음
  - 배치로 변경하는게 가능한지 검토
- All-in-one 패턴
  - 하나의 서버에 여러 예측 모델을 띄우는 경우
  - 라이브러리 선택 제한이 존재
  - 장애가 발생할 경우에 시스템이 마비 (Single Point Of Failure)
  - 모델 별로 서버를 분리 (Microservice 패턴)

## Batch Serving

- 일정 기간 데이터 수집 후 일괄 학습 및 결과 제공
- 대량의 데이터 처리

### Batch Serving 예시

- 스포티파이 예측 알고리즘 — Discover Weekly
- 수요 예측
- 이미지 예측 — S3에 저장된 이미지 사용해 예측
- 자연어 예측 — DB나 데이터 웨어하우스에 저장된 자연어 데이터를 활용해서 사용

### Airflow

- Crontab을 대체하는 스케줄링 도구
- Crontab은 정해진 때마다 자동으로 파일을 실행할 수 있지만 다음의 문제가 있음
  - 파일을 실행하다 오류가 발생한 경우 별도의 처리를 하지 못함 (알림을 받을 수 없음)
  - 과거 실행 이력 및 실행 로그를 보기 어려움
  - 여러 파일을 실행하거나 복잡한 파이프라인을 만들기 힘듦
- Airflow는 워크플로우 관리 도구
- 코드로 작성된 데이터 파이프라인 흐름을 스케줄링하고 모니터링하는 목적

#### Airflow 기능

- 파이썬으로 스케줄링 및 파이프라인 작성
- 스케줄링 및 파이프라인 목록을 볼 수 있는 웹 UI 제공
- Task 실패시 Slack 메시지 전송 가능

## Online Serving

- 실시간으로 데이터를 처리하고 즉각적인 결과 반환
- 구현 방법
  - 직접 웹서버 개발 (Flask, FastAPI 등)
  - 클라우드 서비스 활용 (AWS SageMake, GCP VertexAI 등)
  - 오픈소스 활용 (BentoML)

## 웹 프로그래밍 지식

### 서버 아키텍처

- 모놀리스 아키텍처 — 하나로 관리
- 마이크로서비스 아키텍처 — 작은 여러개의 서비스로 개발

### URI & URL

- URI: Uniform Resource Identifier
- URL: Uniform Resource Locator
- URI가 URL보다 더 큰 범위로 URL이 URI에 속함

### API

- Application Programming Interface
- 특정 소프트웨어에서 다른 소프트웨어를 사용할 때의 인터페이스
- HTTP
  - Hyper Text Transfer Protocol
  - 정보를 주고 받을 때 지켜야 하는 통신 프로토콜

### Web API 종류

- REST(Representational State Transfer)
- GraphQL
- RPC(Remote Procedure Call)

### REST API

- 자원을 표현하고 상태를 전송하는 것에 중점을 둔 API
- REST라고 부르는 아키텍처 스타일로 HTTP 통신
- 가장 대중적이고 현대의 대부분 서버들이 이 API 방법을 사용함
- RESTful하지 않은 API를 만들면 API가 무엇을 의미하는지 알기 어려움

### Status Code

- 클라이언트 요청에 따라 서버가 어떻게 반응하는지 알려주는 코드
  - 1XX(정보): 요청을 받았고 프로세스를 계속 진행함
  - 2xx(성공): 요청을 성공적으로 받았고 실행함
  - 3xx(리다이렉션): 요청 완료를 위한 추가 작업이 필요
  - 4xx(클라이언트 오류): 요청 문법이 잘못되었거나 요청을 처리할 수 없음
  - 5xx(서버 오류): 서버가 요청에 대해 실패

## FastAPI

- 대표적인 Python Web Framework
- Node.js, Go와 대등한 성능
- Flask와 비슷한 구조, 마이크로서비스에 적합
- Swagger 자동 생성, Pydantic을 이용한 Serialization

### Poetry

- pip를 대체하는 패키지 매니저
- Dependency Resolver로 복잡한 의존성들의 버전 충돌을 방지
- Viortualenv를 생성하여 격리된 환경에서 빠르게 개발 가능
- 기존 파이썬 패키지 관리 도구에서 지원하지 않는 build, publish가 가능
- pyproject.toml을 기준으로 여러 툴들의 config를 명시적으로 관리

#### Poetry 사용 흐름

1. 프로젝트 init
   - `poetry init`
   - 대화 형식으로 패키지 설치 가능
   - 개발용/프로덕션용 패키지를 분리할 수 있음
   - pyproject.toml에 설정 저장됨
2. Poetry Shell 활성화
   - `poetry shell` 으로 가상환경 진입
3. Poetry Install
   - pyproject.toml에 저장된 내용에 기반해 명시된 의존성 라이브러리 설치
4. Poetry Add
   - `poetry add pandas` 처럼 필요한 패키지를 추가할 수 있음
   - poetry.lock은 Writing lock file에서 생성되는 파일
     - 이 파일이 존재하면 작성하고 있는 프로젝트 의존성과 동일한 의존성을 가질 수 있음
     - Github Repository에 꼭 커밋
5. Poetry Remove
   - `poetry remove pandas`

### 기초적인 웹 서버 실행

```python
# 01_simple_webserver.py
from fastapi import FastAPI
# FastAPI 객체 생성
app = FastAPI()
@app.get("/")
def read_root():
    return {"Hello": "World"}
```

```bash
uvicorn 01_simple_webserver:app --reload
```

- uvicorn: ASGI(Asynchronous Server Gateway Interface). 비동기 코드를 처리할 수 있는 Python 웹 서버, 프레임워크 간의 표준 인터페이스
- 터미널에서 uvicorn을 작성하기 싫다면 코드 내에 uvicorn.run을 추가하면 됨 (물론 패키지 추가해야함)

```python
# 01_simple_webserver.py
from fastapi import FastAPI
import uvicorn
# FastAPI 객체 생성
app = FastAPI()
@app.get("/")
def read_root():
    return {"Hello": "World"}
    
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- localhost:8000에 접근하면 GET 결과를 볼 수 있음
- localhost:8000/docs로 이동하면 swagger 문서를 확인할 수 있음
- Swagger 기능
  - API 디자인
  - API 빌드
  - API 문서화
  - API 테스팅
    

### URL Parameters

- Path parameter
  - Resource를 식별해야 하는 경우 더 적합
- Query parameter
  - 정렬, 필터링을 해야 하는 경우 더 적합

### Request Body

- 클라이언트에서 API 요청을 보낼 때 Request Body(=Payload)를 사용함
- Request Body에 데이터를 보내고 싶다면 POST method를 사용
- POST는 body의 데이터가 어떤 타입인지 설명하는 content-type이란 헤더 필드에 타입을 명시해야함

### Response Body

- 서버가 요청을 받아 response를 보내는 것은 Response Body

```python
from typing import Optional
from fastapi import FastAPI
import uvicorn

from pydantic import BaseModel

class ItemIn(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None
    
class ItemOut(BaseModel):
    name: str
    price: float
    tax: Optional[float] = None
    
app = FastAPI()

@app.post("/items/", response_model=ItemOut)   # response는 request와 다르게 지정 가능
def create_item(item: ItemIn):
    return item
    
if __name__ == '__main__':
    uvicorn.run(app, host"0.0.0.0", port=8000)
```

### Form과 File

- Form 제출과 File 제출을 원할 경우에 python-multipart를 설치하여 사용
- python-multipart는 직접 import하지 않지만 FastAPI가 내부적으로 사용

```python
@app.post("/login/")
def login(username: str = Form(...), password: str = Form(...)):
    return{"username": username}

@app.post("/files/")
def create_files(files: List[bytes] = File(...)):
    return {"file_sizes": [len(file) for file in files]}
    
@app.post("/uploadfiles/")
def create_upload_files(files: List[UploadFile] = File(...)):
    return {"filenames": [file.filename for file in files]}
```

### Pydantic

- FastAPI의 Data Validation, Settings Management 라이브러리
- Type Hint를 런타임에서 강제해 안전하게 데이터 핸들링
- 파이썬 기본 타입 + List, Dict, Tuple에 대한 Validation 지원
- 기존 Validation 라이브러리보다 빠름
- 머신러닝 Feature Data Validation으로도 활용 가능

#### 사용법

- 일반 python class를 활용
    
  ```python
  class ModelInput01:
      url: str
      rate: int
      target_dir: str
      
      def __init__(self, url: str, rate: int, target_dir: str):
          self.url = url
          self.rate = rate
          self.target_dir = target_dir
      def validate(self) -> bool:
          validation_results = ...
          return all(validation_results)
          
  valid_python_class_model_input = ModelInput(**VALID_INPUT)
  assert valid_python_class_model_input.validate() is True
  ```
    
  - 코드가 복잡해짐
- Dataclass를 활용
    
  ```python
  from dataclasses import dataclass
  
  @dataclass # dataclass decorator로 init method를 따로 작성할 필요가 없음
  class ModelInput02:
      url: str
      rate: int
      target_dir: str
      
      def __post_int__(self):
          if not self.validate():    # 하지만 여전히 validate method를 만들어야 함
              raise ValidationException()
              
  # post init method 사용으로 validate 메서드를 호출하지 않아도 생성 시점에서 validation 
  ModelInput02(**INVALID_INPUT)  
  ```
    
  - 인스턴스 생성 시점에 validation 수행이 자동으로 진행
  - 하지만 validation 로직을 직접 작성해야
- Pydantic 활용
    
  ```python
  from pydantic import BaseModel, HttpUrl, Field, DirectoryPath
  
  class ModelInput03(BaseModel):
      url: HttpUrl                   # http URL인지 검증
      rate: int = Field(ge=1, le=10) # 크기 검증
      target_dir: DirectoryPath      # 존재하는 디렉토리인지 검증
  ```
    
  - 훨씬 코드가 간결해짐
  - 어디서 에러가 발생했는지
    - location, type, message 등을 알려줌

#### config

- 앱을 실행하기 위해 사용자가 설정해야 하는 일련의 정보
- 관리할 때 사용하는 방법
  - 코드 내 상수로 관리
    - 가장 간단하지만 코드에 secret 정보들이 노출될 수 있고
    - 개발/운영 환경에 따라 값을 다르게 줄 수 없음
  - yaml 등의 파일로 관리
    - 보안정보가 여전히 파일에 노출됨
  - 환경변수로 관리
    - pydantic은 BaseSettings를 상속한 클래스에서 Type Hint로 주입된 설정 데이터를 검증할 수 있음

### FastAPI 확장 기능

#### Lifespan function

- FastAPI 앱을 실행할 때와 종료할 때 로직을 넣고 싶은 경우
  - FastAPI 앱이 처음 실행될 때 머신러닝 모델을 Load하고 종료할 때 연결해두었던 DB Connection을 정리

```python
@asynccontextmanger
async def lifespan(app: FastAPI):
    ml_models["answer_to_everything"] = fake_answer_to_everything_ml_model
    yield    # yield를 기준으로 앞 코드들을 FastAPI가 켜질때 실행, 뒤 코드를 종료될때 실행함
    ml_models.clear()
    
app = FastAPI(lifespan=lifespan)
```

#### API Router

- API Router는 큰 애플리케이션들에서 많이 사용되는 기능
- Mini FastAPI로 여러 API를 연결해서 활용

```python
user_router = APIRouter(prefix="/users", tags=["users"])

@user_router.get("/")
def read_users():
    return [{"username": "Rick"}, {"username": "Morty"}]
```

```python
order_router = APIRouter(prefix="/orders", tags=["orders"])

@order_router.get("/")
def read_orders():
    return [{"order": "Taco"}, {"order": "Burritto"}]
```

```python
from routers import user, order  # 만들어둔 라우터 임포트
app = FastAPI()
if __name__ == '__main__':
    app.include_router(user.user_router)
    app.include_router(order.order_router)
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### Error Handler

- 어떤 에러가 발생했는지 잘 알 수 있어야 함
- 예외처리를 제대로 해두지 않으면 500 ERROR가 발생 (일반적으로 발생하면 안되는 에러)
- 각 상황에 맞게 예외처리를 해두어야 함 (잘못된 request를 받았을 때 4xx)

#### Background Tasks

- Background Tasks를 사용하지 않는 작업들은 작업 시간만큼 응답을 기다림
- Background Tasks를 사용하면 기다리지 않고 바로 응답을 주고 작업은 background에서 실행