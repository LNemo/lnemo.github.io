---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 46: AI 개발 기초 (1)"
date: 2025-11-17 18:37:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, linux, shell]
description: "AI 개발을 위한 기초를 이해하자."
keywords: [Software Engineering, AI Engineering, SDLC, Cohesion, Coupling, Modularity, Unit Testing, Linux, Shell Script, CLI, Bash, Vim, SSH, SCP, Grep, Awk, Curl, Server Management, DevOps, MLOps]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# AI 개발 기초 (1)

오늘은 AI 개발을 하기 위한 기초적인 지식들을 익혔습니다. 소프트웨어 엔지니어링와 AI 엔지니어링의 차이와 쉘 명령어를 정리하였습니다.

## 엔지니어링

### 소프트웨어 엔지니어링

소프트웨어를 개발하는 과정에서 체계적이고 효율적인 방법을 사용하여 소프트웨어의 품질과 유지 보수성을 보장하는 학문 분야

#### 소프트웨어의 분야

소프트웨어에 다양한 분야가 있음

- 프론트엔드
- 백엔드
- 풀스택
- 머신러닝, AI
- 데이터
- 모바일 앱
- 게임
- DevOps, 클라우드
- 보안

#### 소프트웨어 개발 라이프사이클 

- Software Development Lifecycle
1. Planning 계획
2. Analysis 요구 조건 분석
3. Design 설계
4. Implementation 구현
5. Testing & Integration 테스트
6. Maintenance 유지보수

위 과정의 반복

#### 좋은 소프트웨어 설계

- 모듈성
- 응집도
- 결합도
- 테스팅
- 문서화

#### 모듈성

- 큰 프로그램을 작고 독립적인 부분으로 나누는 것
- 한 부분이 깨지거나 변경해야 하는 경우 나머지 부분에 영향을 주지 않고 쉽게 변경할 수 있음

#### 응집도

- 낮은 응집도 = 하나의 모듈이 여러가지 일을 담당
  - User Class가 user print, email 전송 등 모든 역할을 지고 있음
- 높은 응집도 = 각 모듈이 하나의 역할을 담당
  - 위의 User를 User, UserInformation, EmailSender로 분리
- 높은 응집도를 가진 것이 좋은 설계

#### 결합도

- 모듈간 상호 의존성의 정도
- 낮은 결합도를 가진 것이 좋은 설계

#### 테스팅

- Unit Test: 개별 단위 테스트
- Integration Test: 구성요소 동작 테스트
- End to End Test: 처음부터 끝까지 모두 테스트
- Performance Test: 성능, 부하 테스트

#### 문서화

- 소프트웨어를 이용하기 위한 README, API문서, 아키텍처 문서
- 좋은 소프트웨어는 좋은 문서가 같이 존재

### AI 엔지니어링

- 웹, 앱 서비스에서 AI 모델이 동작할 수 있도록 하는 것

#### SW 엔지니어링과 AI 엔지니어링의 차이

- SW엔지니어 — 코드는 변경에 용이하고 읽기도 좋아야 해
- AI엔지니어 — 모델의 퍼포먼스와 데이터셋, 실험이 중요해

#### AI 엔지니어링의 사례

- 우버
  - 도착 예정 시간(ETA)을 머신러닝을 사용해서 보정
    - Tabular 데이터에 Transformer를 적용
  - 사기, 프로모션 남용, GPS 조작 등의 이상 탐지
    - Anomaly Detection을 통해 탐지
- 도어대시
  - 개인화 추천에 그래프 알고리즘 활용
  - ETA 모델(Multi Task)
- 듀오링고
  - 학습자의 학습 개선을 위해 BirdBrain 제작
    - 연습 문제를 완료할수록 언어에 대해 얼마나 아는지 연습 문제가 얼마나 어려운지를 알게 됨
- 구글
  - Google Lens로 이미지로 검색
- 그 외에도..
  - 인스타그램, 유튜브 추천 시스템
  - 번역
  - SNOW AI
  - Copilot
  - ChatGPT / Claude / Gemini

## Linux 쉘 스크립트

### Linux

- 리눅스는 서버에서 자주 사용하는 OS이기 때문에 알아야 함
- CLI & GUI
  - Command Line Interface
  - Graphic User Interface

#### Linux 배포판

- Debian
- Ubuntu
- Redhat
- CentOS

### Linux 쉘 스크립트 학습 가이드

- 최초에는 자주 사용하는 쉘 커맨드, 쉘 스크립트 위주로 학습
- 필요한 코드가 있을때마다 검색해서 찾기
- 해당 코드에서 나오는 새로운 커맨드 학습해서 정리
- 왜 이렇게 되는가 생각하면 배경지식이 필요한 경우 Linux, OS 학습(아마도 커널)

### 쉘 종류

- sh: 최초의 쉘
- bash: Linux 표준 쉘
- zsh: Mac 카탈리나 OS 기본 쉘

### 기본 쉘 커맨드

- `man`
  - manual
  - 쉘 커맨드의 메뉴얼 문서를 보고 싶은 경우
  - ex) `man python3`
- `mkdir`
  - make directory
  - 폴더 생성
  - ex) `mkdir linux-test`
- `ls`
  - list segments
  - 현재 접근한 폴더의 파일 확인
  - 옵션: `-a`, `-l`, `-h` (human-readable)
  - ex) `ls -al` , `ls -lh`
- `pwd`
  - print working directory
  - 현재 폴더의 절대 경로를 print
- `cd`
  - change directory
  - 폴더 변경, 폴더 이동
  - ex) `cd linux-test`
- `echo`
  - 터미널에 텍스트 출력
  - `echo `쉘 커맨드`` 입력시 쉘 커맨드의 결과를 출력
- `cp`
  - copy
  - 파일 또는 폴더 복사
  - 옵션: `-r` 디렉토리를 복사할 때 디렉토리 안에 파일이 있으면 모두 복사(recursive), `-f`(force)
  - ex) `cp vi-test.sh vi-test2.sh`
- `vi`
  - vim 편집기로 파일 생성
  - i를 눌러서 INSERT 모드로 변경
  - ESC를 누른 후 :wq(저장하고 나가기, write and quit)
    - :wq! (강제로 저장하고 나가기)
    - :q (저장하지 않고 나가기)
    - ![vi 모드](### vi)
    - [여기](#vi)에 더 자세한 명령어
  - ex) `vi vi-test.sh`
- `bash`
  - bash로 쉘 스크립트 실행
  - ex) `bash vi-tesh.sh`
- `sudo`
  - superuser do
  - 관리자 권한으로 실행
- `mv`
  - move
  - 파일, 폴더 이동 또는 이름 변경
  - ex) `mv vi-test.sh vi-test3.sh`
- `cat`
  - concatenate
  - 여러 파일을 인자로 주면 concat해서 print
  - 저장하고 싶은 경우에 `>` , 추가하고 싶은 경우 `>>`
  - ex) `cat vi-test.sh` , 
  `cat vi-test2.sh vi-test3.sh > new_test.sh` , 
  `cat vi-test2.sh vi-test3.sh >> new_test.sh`
- `clear`
  - 터미널 청소
- `history`
  - 최근에 입력한 쉘 커맨드 히스토리 출력
  - history 결과에서 느낌표와 숫자 입력하면 그 커맨드를 다시 사용 가능
  - ex) `history`  `!30`
- `find`
  - 파일 및 디렉토리를 검색할 때 사용
  - ex) `find . -name "file"`
- `export`
  - 환경변수 설정
  - 터미널이 꺼지면 설정한 환경변수가 사라짐
  - 쉘을 실행할때마다 환경변수를 저장하고 싶으면 .bashrc, .zshrc에 저장 (`source ~/.bashrc`로 즉시 적용)
  - ex) `export water="물"`  `echo $water`
- `alias`
  - 기본 명령어를 간단히 줄이는 것(별칭)
  - `ll` 은 `ls -l` 로 별칭 지정되어 있음
  - ex) `alias ll2='ls -l'`

- `tree`
  - 폴더의 하위 구조를 계층적으로 표현
  - 프로젝트 소개할 때 구조 설명할 때 유용
  - 설치가 안되어 있다면 설치 필요
    - `apt-get install tree`
  - ex) `tree -L 1` (1단계까지 보여주기)
- `head`, `tail`
  - 파일의 앞/뒤 n행 출력
  - ex) `head -n 3 vi-test.sh`
- `sort`
  - 행 단위 정렬
  - 옵션: `-r` 내림차순으로 정렬, `-n` Numeric Sort
  - ex) `cat fruits.txt | sort`
- `uniq`
  - 중복된 행이 **연속**으로 있는 경우 중복 제거
  - sort와 함께 사용
  - 옵션: `-c` 중복 행의 개수 출력
  - ex) `cat fruits.txt | sort | uniq | wc -l` (`wc -l` 은 요소가 몇줄인지 출력)
- `grep`
  - 파일에 주어진 패턴 목록과 매칭되는 라인 검색
  - 파이프와 같이 사용
  - 옵션
    - `-i` 대소문자 구분 없이 찾기 (insensitively)
    - `-w` 정확히 그 단어만 찾기
    - `-v` 특정 패턴 제외한 결과 출력
    - `-o` 해당 단어만 출력
    - `-E` 정규 표현식 사용
      - ^단어: 단어로 시작하는 것
      - 단어$: 단어로 끝나는 것
      - . : 하나의 문자 매칭
  - ex) `grep -i "ky" grep_file`
- `cut`
  - 파일에서 특정 필드 추출
  - 옵션: `-f` 잘라낼 필드 지정, `-d` 필드를 구분하는 구분자(default는 \t)
  - ex) `cat cut_file | cut -d : -f 1,7` (구분자가 :인것에서 1번째 7번째 값을 가져옴)
- `awk`
  - 텍스트 처리 도구
  - `awk 'pattern {action}' input_file`
    - pattern: 특정 패턴을 지정
    - action: 선택된 줄에 대해 수행할 동작
    - $1, $2, $NF: 첫번째 필드, 두번째 필드, 마지막 필드
  - 옵션: `-F` 구분자 설정, `-v` .sh 스크립트에서 변수 사용할때 사용
  - ex) `awk -F: '{print $1}' cut_file`

### 서버에서 자주 사용하는 쉘 커맨드

- `ps`
  - process status
  - 현재 실행되고 있는 프로세스 출력. PID를 찾기 위해 활용
  - 옵션: `-e` 모든 프로세스, `-f` full format
- `curl`
  - client url
  - 웹 서버를 작성한 후 요청이 제대로 실행되는지 확인
  - ex) `curl -X localhost:5000/ {data}`
- `df`
  - disk free
  - 현재 사용중인 디스크 용량 확인
  - 옵션: `-h` 사람이 읽기 쉬운 형태로 출력(human-readable)
- `ssh`
  - 안전하게 원격으로 컴퓨터에 접속하고 명령을 실행
  - 22 포트 사용
  - ex) `ssh -i key.pem root@hostname(ip) -p 포트번호`
- `scp`
  - secure copy
  - SSH를 이용해 네트워크로 연결된 호스트 간 파일을 주고 받는 명령어
  - 옵션: `-r` 재귀적으로 복사, `-P` ssh 포트 지정, `-i` ssh 설정을 활용해 실행
  - ex) `scp local_path user@ip:remote_directory` , 
  `scp user@ip:remote_directory local_path` ,
  `scp user@ip:remote_directory user2@ip2:target_remote_directory`
- `nohup`
  - 터미널 종료 후에도 계속 작업이 유지하도록 실행
  - nohup으로 실행될 파일은 권한이 755이어야 함
  - 종료는 pid 찾은 후 `kill -9 pid`
  - 로그는 nohup.out에 저장
  - nohup 외에도 screen이란 도구도 있음
- `chmod`
  - change mode
  - 파일이나 디렉토리의 시스템 모드를 변경
  - 소유자 권한 / 그룹의 권한 / 기타 사용자 권한
- `python`
  - 빌트인 모듈을 쉘커맨드처럼 사용 가능
  - ex) `echo ‘hello’ | python -m base64`

### 쉘 스크립트

- .sh 파일을 생성하고 안에 쉘 커맨드를 추가
- 여러 하이퍼파라미터를 가지고 학습을 실행시킬때 유용
- 모델 학습 후 로그 파일 저장 후 로그 분석할 때도 활용 가능

- 카카오톡 그룹 채팅방에서 대화 내보내기로 csv 저장 후 쉘커맨드 1줄로 특정 연도에 제일 메시지를 많이 보낸 top 3명 추출하기

### Vi

- Command Mode
  - 실행시 기본 모드
  - 방향키로 커서 이동 가능
  - dd: 현재 위치한 라인 삭제
  - i: INSERT 모드로 변경
  - yy: 현재 라인을 복사
  - p: 현재 커서가 있는 줄 아래에 붙여넣기
  - k: 커서 위로
  - j: 커서 아래로
  - l: 커서 오른쪽으로
  - h: 커서 왼쪽으로
- Insert Mode
  - 파일을 수정할 수 있는 모드
  - Command Mode로 이동하고 싶다면 ESC
- Last Line Mode
  - ESC 입력 후 ‘:’(콜론)을 입력하면 나오는 모드
  - :w
  - :q
  - :q!
  - :wq
  - :/(문자)
    - 문자 탐색, 탐색 후 n을 입력하면 계속 탐색 실행
  - :set nu
    - vi 라인 번호 출력
