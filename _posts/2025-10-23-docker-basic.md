---
layout: post
title: "[Docker] 도커 사용법"
date: 2025-10-23 23:24:00+0900
categories: [IT]
tags: [docker, docker-compose]
description: "Docker의 기본적인 사용법을 알아보자."
keywords: [docker, image, container, command]
image:
  path: /assets/img/posts/docker/docker.png
  alt: "Docker"
comment: true
---

# Docker

프로젝트를 진행할 때마다 Dockerfile을 받아서 그 환경을 그대로 가져다 사용한 적이 있습니다. 하지만 스스로 직접 Dockerfile을 만들고 Docker 환경을 만들어 본 적은 없기에 기본적인 커맨드와 파일 작성법을 알아보았습니다.

Docker는 일반적으로 다음과 같은 과정을 가집니다.
- Dockerfile 작성 → Dockerfile로 이미지 생성 → 이미지로 컨테이너 생성

## Image & Container

이미지는 Dockerfile로 생성되는 템플릿입니다. 특정 애플리케이션을 실행하기 위해 필요한 코드, 런타임, 라이브러리, 환경 변수, 설정 파일 등을 모두 포함하고 있습니다.

이미지는 다음의 특징을 갖습니다.
- 정적(Static)이고 Read-only
- 실행파일이 아님
- 계층(Layer)구조 (OS → 자바 설치 → 내 앱 코드)

쉽게, 이미지는 컨테이너를 만들기 위한 틀입니다.

컨테이너는 이미지로부터 생성된 인스턴스입니다. 컨테이너는 다음의 특징을 가집니다.
- 동적(Dynamic)이고 실행가능
- 격리된 환경 (호스트OS와 다른 컨테이너와 완전히 분리)
- 가볍고 빠름 (VM과 달리)

## Docker Commands

### Docker General

- `docker --version` : 버전 확인
- `docker info` : Systemized Information

### Image Management

- `docker images` : 로컬 머신의 이미지를 보여줌
- `docker pull [image-name]` : Hub 또는 on-demand로 연결된 레지스트리에서 이미지를 가져옴
- `docker build -t [image-name] .` : 현재 디렉토리에 있는 Dockerfile을 기준으로 이미지를 빌드함
- `docker tag [source-image] [target-image:tag]` : 이미지에 여러개의 태그를 붙일 수 있음
- `docker push [image-name]` : Hub 또는 레지스트리에 push
- `docker rmi [image-id]` : 이미지 삭제

### Container Management

- `docker ps` : 현재 실행되고 있는 컨테이너 출력
- `docker ps -a` : 모든 컨테이너 출력
- `docker run [options] [image-name]` : 로컬머신에 있는 이미지 실행 (로컬에 있지 않다면 pull해서 실행함)
  - `-d` : 백그라운드 실행
  - `-p [host-port]:[container-port]` : 포트 연결
  - `--name [container-name]` : 이름
  - `-v [host-path]:[container-path]` : 볼륨 연결(mount)
- `docker rm [container-id/name]` : 컨테이너 삭제
- `docker rm -f [container-id/name]` : 실행중인 컨테이너도 force 삭제
- `docker stop` : 실행중인 컨테이너 stop
- `docker start` : stop된 컨테이너 시작
- `docker restart`: 컨테이너 재시작

(`stop`, `start`, `restart` 는 docker compose로 쉽게 가능하기 때문에 자주 사용하지 않음)

### Container Interaction

- `docker exec [container-id/name] [command]` : (excuse) 실행중인 컨테이너에 command 실행
  - `docker exec -it [container-id] /bin/bash` 로 bash shell로 접속 가능
- `docker logs [options] [container-id/name]` : 컨테이너의 로그를 보여줌(standard out)
  - `--follow` : stream logs in real-time
- `docker attach [container-id/name]` : 현재 실행중인 컨테이너를 터미널에 붙임
- `docker kill [options] [container-id/name]` : 컨테이너 강제종료. SIGKILL을 보냄

### Volume

- `docker volume create my_volume` : 볼륨 생성
- `docker volume ls` : 볼륨 리스트
- `docker volume inspect my_volume` : 볼륨에 대한 inspect
- `docker volume rm my_volume` : 볼륨 삭제
- `docker volume prune` : 사용하지 않는 볼륨 삭제

### Debugging Commands

- `docker inspect [object-id/name]` : 컨테이너를 inspect해서 json 포맷으로 반환
- `docker stats [container-id/name]` : 컨테이너에 대한 real-time resource usage 확인
- `docker top [container-id/name]` : 컨테이너에서 실행되고 있는 프로세스 확인
- `docker events` : Docker daemon의 event 확인

## Dockerfile

```docker
### Dockerfile ###
# Step 1: Base image 설정
FROM python:3.13-slim

# Step 2: Work Directory 설정
WORKDIR /app

# Step 3: "."(현재 디렉토리)에 있는 모든 것을 docker 이미지 안에 "/app"안에 copy함
COPY . /app

# Step 4: 앱 depecdencies 설치
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: application이 listening하는 포트를 오픈
# 로컬 포트랑 docker 이미지의 포트랑 연결
# 컨테이너가 내부에서 사용할 포트 번호
# 주의: EXPOSE를 쓴다고 해서 그 포트가 절대로 호스트 PC나 외부 네트워크에 자동으로 열리지 않는다고 함
EXPOSE 5000

# Step 6: application 실행 후 기본 실행 명령어
# 마무리 명령어
# work directory 안에서 python 명령어 실행 후 app.py를 실행하라는 의미
CMD ["python", "app.py"]
```

## Docker Compose

- Docker Run은 애플리케이션을 하나하나 시작해야 하는데 이걸 Docker Compose로 여러 개를 한꺼번에 실행할 수 있음

```yaml
### compose.yml ###
services:
  app1:  # 서비스 이름
    build:  # Dockerfile 위치를 지정해주기 위함
      context: my-first-docker  # directory 위치. 현재 디렉토리 "."도 가능
    ports:
      - "5001:5000"  # port mapping
    depends_on:
      - db   # depends_on의 서비스를 먼저 실행하도록 함
    environment:  # 
      - MYSQL_HOST=db
      - MYSQL_USER=root
      - MYSQL_PASSWORD=password
      - MYSQL_DATABASE=exampledb
  db:   # 서비스 이름
    image: mysql:8.0   # official MySQL 이미지를 사용
    environment:
    MYSQL_ROOT_PASSWORD: password
    MYSQL_DATABASE: exampledb
    ports:
      - "3306:3306"
    volumes:
      - db_data:/var/lib/mysql
    healthcheck:   # 데이터베이스 시작할 때에 아래의 테스트로 DB가 시작되었는지 확인
      test: ["CMD", "mysqladmin", "ping", "--silent"]
      interval: 5s
      timeout: 10s
      retries: 3
    build:
      context: my-first-docker
    ports:
      - "5001:5000"

volumes:
  db_data:  # 볼륨 이름
  redis_data:   # 여러개도 가능
```

등록된 환경변수는 아래처럼 사용 (Python 기준)

```python
import os

DB_CONFIG = {
    'host': os.getenv("MYSQL_HOST"),
    'user': os.getenv("MYSQL_USER"),
    'password': os.getenv("MYSQL_PASSWORD"),
    'database': os.getenv("MYSQL_DATABASE"),
}
```