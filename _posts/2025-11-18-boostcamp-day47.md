---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 47: AI 개발 기초 (2)"
date: 2025-11-18 18:53:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, streamlit, python]
description: "AI 개발을 위한 기초를 이해하자."
keywords: [Streamlit, Prototyping, Dashboard, Data Visualization, UI Components, Session State, Caching, Python, Pyenv, venv, pip, Dependency Management, Debugging, Troubleshooting, Error Log]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# AI 개발 기초 (2)

오늘은 AI 개발을 하기 위한 기초적인 지식들을 익혔습니다. Streamlit과 파이썬 환경설정을 정리하였습니다.

## Streamlit

### 프로토타입

- 완벽한 제품이 나오기 전에 확인할 수 있는 샘플
- 주로 웹페이지로 제작
- AI 모델의 input ⇒ output을 확인할 수 있도록 설정
- 다른 조직의 도움 없이 빠르게 웹 서비스를 만드는 방법이 없을까  
→ 이런 문제를 해결하기 위해 등장한 것이 Streamlit

### Streamlit을 사용할 때의 장점

- 엔지니어는 노트북(.ipynb)을 활용해 쉽게 프로토타입 구현 가능 (대시보드처럼 레이아웃을 잡는것은 어려움)
- 기존 코드를 조금만 수정해서 효율적으로 웹서비스 개발

### Streamlit 대안

- R의 Shiny
- Flask, FastAPI
- Dash: 제일 기능이 풍부한 Python 대시보드 라이브러리
- Gradio: Streamlit과 비슷한 컨셉의 도구

### Hello World

```python
# pip3 install streamlit==1.36.0
# 01-streamlit-hello.py
import streamlit as st
st.title("Hello Streamlit")
st.write("Hello World")
```

아래 쉘 명령어로 실행

```bash
streamlit run 01-streamlit-hello.py
```

### Streamlit 개발 흐름

- AI/ML 모델링 → Streamlit 설계 → Streamlit 개발 → 테스트 및 배포
1. AI/ML 모델링
   - 파이썬 스크립트 형태
2. Streamlit 설계
   - 목적, 기능 정의 — 인퍼런스, 결과노출, 파라미터 선택
   - UI 레이아웃 정의
   - 사용할 Component 결정
3. Streamlit 개발
   - AI/ML 모델링의 스크립트 약간 수정
   - UI 컴포넌트 추가(input/output)
   - 데이터 처리(전처리 시각화), 상호작용 로직 개발
4. 테스트 및 배포
   - 배포
   - Use Case 확인

### Streamlit UI Component

#### Text

- `st.title("Title")`
- `st.header("Header")`
- `st.subheader("subheader")`
- `st.write("write something")`

#### Input

- `st.text_input("텍스트를 입력해주세요")` , `st.text_input("암호를 입력해주세요", type="password")`
- `st.number_input("숫자를 입력해주세요")`
- `st.date_input("날짜를 입력하세요")`
- `st.time_input("시간을 입력하세요")`

#### File Uploader

- `st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])`

#### Select

- `st.radio("Radio Part", ("A", "B", "C"))`
- `st.selectbox("Please select in selectbox!", ("kyle", "seongyun", "zzsza"))`
- `st.multiselect("Please select somethings in multi selectbox!", ["A", "B", "C", "D"])`
- `st.checkbox('체크박스 버튼', value=True)` : True/False

#### Slider

- `st.slider("Select a range", 0.0, 100.0, (25.0, 75.0))`

#### Button

- `if st.button("버튼"): st.write("클릭 후 보이는 메시지")`

#### Form

```python
with st.form(key="입력 form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    st.form_submit_button("login")
```

#### 여러 컴포넌트

- Pandas Dataframe, Markdown

```python
df = pd.Dataframe({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})

st.markdown("=======")

st.write(df)     # 문자, 숫자, Dataframe, 차트 등을 표시
st.dataframe(df) # Interactive한 Dataframe, zㅓㄹ럼 클릭/정렬 가능
st.table(df)     # Static 한 Dataframe
st.table(df.style.highlight_max(axis=0)) # 가장 큰 값 하이라이트
```

- Metric, Json

```python
st.metric("My metric", 42, 2)
st.json(df.to_json())
```

- Line Chart

```python
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c']
)
st.line_chart(chart_data)
```

- Map Chart

```python
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4], 
    columns=['lat', 'lon']
)
st.map(map_data)
```

- Caption, Code, LaTex

```python
st.caption("caption")
st.code("code")
st.latex("latex")
```

#### Spinner

```python
with st.spinner("Please wait.."):    # 연산 도중 메시지를 보여주기 (st.status도 존재)
    time.sleep(5)
```

#### Status Box

- 여러 상황에서 메시지 표현. 각각 박스의 색이 다름

```python
st.success("Success")
st.info("Info")
st.warning("Warning")
st.error("Error")
```

#### Sidebar

- 사이드바에 파라미터를 지정하거나 암호 설정 가능
- `st.sidebar.button("hi")`

#### Columns

- 여러 칸으로 나누어 component를 추가할 때 사용
- `col1, col2, col3, col4 = st.columns(4)`
  - `col1.write("this is col1")`

#### 더많은 함수

- https://cheat-sheet.streamlit.app/

### Session State

- Streamlit은 무언가 업데이트 되면 전체 코드를 다시 실행함
- 예상과 다른 작동을 할 때가 있음
- Session State로 Global Variable처럼 공유할 수 있는 변수를 만들고 저장

```python
if 'count' not in st.session_state:
    st.session_state.count = 0
```

### Streamlit Caching

- 전체 코드를 다시 실행하기 때문에 데이터를 불러오는 코드도 다시 읽어올 수 있고 객체도 계속 생성함
- 이를 해결하기 위해 캐싱 데코레이터를 활용
- 캐시 두가지 방법
  - `@st.cache_data`: 데이터, API Request의 Response
  - `@st.cache_resource`: DB연결, ML모델 Load

```python
@st.cache_data
def fetch_large_dataset():
    df = pd.read_csv("https://example.com/large_dataset.csv")
    return df
```

## 파이썬 환경설정

### 파이썬 버전

- 파이썬은 SemVer 정책으로 버전을 관리
- 내가 사용하려는 라이브러리와 호환되는 버전을 선택할 것

### Pyenv

- Python 설치 환경 도구
- 설치 방법
  - Mac — `brew install pyenv`
  - Linux(Ubuntu)
    - `sudo apt-get update`
    - `sudo apt-get install make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev`
    - `curl https://pyenv.run | bash`
- 환경설정
  - `vim /.bashrc`
  - 마지막 줄에 다음을 추가
    
  ```
  export PATH="~/.pyenv/bin:$PATH"
  eval "$(pyenv init -)"
  eval "$(pyenv virtualenv-init -)"
  ```
    
  - `source ~/.bashrc`로 쉘 설정 업데이트

####  명령어

- `pyenv install —list`: 설치 가능한 파이썬 버전 확인
- `pyenv install 3.11.0`: 3.11.0 버전의 파이썬 설치
- `pyenv shell 3.11.0`: 현재 shell에 파이썬 버전 활성화
- `pyenv versions`: pyenv로 설치된 파이썬 확인 가능
- `pyenv global 3.11.0`: 기본으로 사용할 파이썬 버전을 설정

### venv

- 파이썬 가상 환경 구축에 많이 사용
- 파이썬 내장 모듈 (별도 설치 필요 X)
- 보통 프로젝트 최상위 경로에서 .venv로 만드는 것이 관습
- `python -m venv "가상 환경 폴더를 만들 경로"`
- `source ".venv/bin/activate"`
- 설치된 패키지는 여기에 위치 — `.venv/lib/python3.11/site-packages/`

### pip

- 파이썬 패키지 설치 도구
- 최신 버전의 pip를 사용하는 것이 좋음
  - `pip install —upgrade pip`
- `pip list`로 설치된 패키지 확인 가능
- `pip list —not-required —format=freeze` 를 입력하면 꼭 필요한 패키지만 보여줌
  - `pip freeze > requirements.txt`로 필요한 패키지 목록을 저장
  - requirements.txt의 패키지 목록을 `pip install -r requirements.txt`로 설치 가능

## 디버깅

### 버그가 생기는 이유

- 사람의 실수
- 실행 환경
- 의존성
- 복잡성
- 커뮤니케이션 과정의 실수

### 디버깅 프로세스

- 문제 발생 → 문제 인식 → 해결책 찾기

#### 문제 인식

- 무엇이 문제인지 아는 것이 중요
- 재현이 가능해야 누군가에게 질문 가능

#### 해결책 찾기

- 과거에 경험한 문제인지 확인
  - 맞다면 이전에 해결책을 확인 (오답노트 확인)
  - 아니라면 검색 시작
- 검색
  - 구글링 — stackoverflow, 오픈 소스 공식 문서, 오픈 소스 Github Issue
  - ChatGPT (100% 정답을 내진 않는 것을 인지)
- 질문
  - 예의 (다른 사람이 답변할 의무는 없음)
  - 상황공유 (Github Gist, Github Repo)
  - 시도했던 것 공유

#### 오답노트에 들어갈 부분

- 문제 상황
- 오류 메시지
- 해결 방법