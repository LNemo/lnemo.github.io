---
layout: post
title: "[Spring] Spring WebFlux"
date: 2026-01-17 20:07:00+0900
categories: [Spring boot]
tags: [spring, spring boot, backend, java, async, webflux]
description: "Spring WebFlux를 익히자."
keywords: [Spring Webflux, Reactive Programming, Asynchronous Non-blocking I/O, Reactive Streams, Project Reactor, Mono, Flux, Netty, Event Loop, Backpressure, Publisher, Subscriber, Subscription, Lazy Evaluation]
image:
  path: /assets/img/posts/spring-boot/spring.png
  alt: "Spring"
comment: true
---

# Spring WebFlux

프로젝트를 진행하면서 AI 챗봇 서비스를 만들 때 비동기로 서버를 구성해야 했습니다. Spring WebFlux와 Netty로 비동기 이벤트 기반 서버를 구성할 수 있어서 위 요구사항에 맞는다고 생각하여 프로젝트에 적용하기로 생각하고 학습하였습니다.

너무 깊고 구체적인 내용보다는 학습을 하며 Webflux를 이해하기 위해 필요한 개념과 궁금한 점을 찾아보면서 정리해보았습니다.

## Webflux를 알기 전에

### 동기와 비동기?
- 동기(Synchronous): 요청한 작업에 대해 완료 여부에 따라 순차적으로 처리
- 비동기(Asynchronous): 요청한 작업에 대해 완료 여부를 따지지 않고 다음 작업을 처리
- 동기와 비동기를 확실히 구분할 수 있는 예시가 동시에 작업을 처리할 때
  ![이미지](/assets/img/posts/spring-boot/webflux-async.png)
  - 동기 방식은 작업을 수행할 때 다른 작업을 처리할 수 없음
  - 비동기 방식은 작업(Task 1)을 수행할 때 외부 Task을 기다리는 동안 다른 작업(Task 2)을 처리할 수 있음

### Blocking I/O, Non-blocking I/O
- 전통적인 서블릿 기반 Spring MVC 모델은 하나의 request당 하나의 thread를 할당해서 작업을 처리
- 해당 스레드에서 DB 조회나 외부 API 호출 같은 I/O 작업을 할 때에 끝날때까지 해당 thread는 아무 일도 할 수 없음 -> **Blocking I/O**
- Webflux는 Event Loop 모델을 기반으로 하기 때문에 DB 조회나 외부 API 호출 같은 I/O 작업을 할 때에 기다리지 않고 바로 다음 작업을 진행함 -> Non-blocking I/O

### Reactive Programming
- Spring [공식 문서](https://docs.spring.io/spring-framework/reference/web/webflux/new-framework.html#webflux-why-reactive)에서 정의하기를 반응형(Reactive)은 특정 이벤트에 대해 반응하는 프로그래밍 모델이라고 정의합니다.
- 어떤 값이 변하면 그 변화를 관찰하고 있다가 연관된 작업들이 자동으로 실행되게 만드는 방식
- 파이프라인을 미리 설계하고 이벤트가 발생했을 때 데이터가 해당 파이프라인을 따라가도록 구조화

### Reactive Streams
- 비동기 데이터 스트림을 적절한 속도로 처리하기 위한 표준 약속
- 여러 라이브러리간 호환될수록 만든 공통 인터페이스
- 과거에는 비동기 데이터를 처리하는 방식이 제각각이었지만 여러 기업이 모여서 표준을 정의함
- 핵심 4대 인터페이스

  | 인터페이스            | 역할                   | 핵심 메서드                            |
  |------------------|----------------------|-----------------------------------|
  | **Publisher**    | 데이터를 생성하고 발행함 (발행자)  | `subscribe(Subscriber)`           |
  | **Subscriber**   | 데이터를 받아서 처리함 (구독자)   | `onNext`, `onError`, `onComplete` |
  | **Subscription** | 발행자와 구독자 사이의 연결      | `request(n)`, `cancel()`          |
  | **Processor**    | 발행자와 구독자의 역할을 동시에 수행 | (두 인터페이스를 모두 상속)                  |

![이미지](/assets/img/posts/spring-boot/webflux-reactive-streams.png)
- 동작 흐름
  1. Subscribe: Subscriber가 Publisher에게 구독 신청
  2. OnSubscribe: Publisher가 Subscriber에게 Subscription 객체를 전달
  3. Request(n): Subscriber는 본인이 처리할 수 있는 양만큼만 데이터를 달라고 요청
  4. OnNext: Publisher는 요청받은 수만큼 데이터를 보냄
  5. OnComplete / OnError: 데이터 송신이 끝나거나 에러가 발생하면 종료
- Backpressure?
  - 기존 방식은 push 방식으로 서버가 초당 1,000개의 데이터를 보내면 클라이언트는 그냥 다 받아내야만 함
  - 리액티브 스트림 방식은 클라이언트가 n개만 요청하면 n개만 주고 다 처리하면 다시 다음 n개 달라고 요청할 수 있음
  - **데이터를 한번에 다 전송하지 않고 조금씩 여러번 보내는 방식**

### Project Reactor
- 리액티브 스트림 규격을 구현한 Java용 리액티브 라이브러리
- Spring 프레임워크를 만드는 Pivotal에서 주도하여 개발
- Spring Webflux의 핵심 엔진


- 특징
  - 논블로킹(Non-blocking) & 효율성: blocking을 없애고 데이터가 준비되었을 때만 CPU가 일하게 함
  - Operators: 데이터를 가공하기 위한 많은 연산자 지원
  - Cold Sequence와 Hot Sequence 지원
    - Cold Sequence: subscribe를 하는 순간 데이터가 흐르기 시작
    - Hot Sequence: subscribe에 상관 없이 데이터가 실시간으로 흐르는 방식

#### Publisher
- Mono
  - 0, 1개 항목에 대한 비동기 시퀀스를 나타내는 Publisher 구현체
- Flux
  - n개 항목에 대한 비동기 시퀀스를 나타내는 Publisher 구현체

&nbsp;

> **Flux로 1개도 해결하면 될 것 같은데 왜 Mono가 필요할까?**
  1. 프로그래밍 의도를 명확하게 하기 위해서 
    - 이 메서드가 단건 반환을 기대하는지 다건 스트림을 기대하는지 명시
  2. 연산자 차별화
    - Mono 전용 연산자와 Flux 전용 연산자가 존재함

---

## WebFlux는 Tomcat이 아니라 Netty를 사용한다던데?
### Netty
- Netty는 **비동기 이벤트 기반 네트워크 애플리케이션 프레임워크**
- 직접 비동기 호출, 논블로킹 소켓을 구현하기에는 코드가 매우 복잡해지기 때문에 사용
- 이벤트에 따라 로직을 분리할 수 있다
- 소켓연결, 데이터 송수신, 에외, 연결종료 등 다양한 이벤트 제공

### Netty는 프레임워크인데?
- Tomcat은 Web Application Server이고 Netty는 프레임워크
- 왜 프레임워크가 WAS와 비교되는지?
- Netty를 사용해 HTTP 프로토콜을 처리하도록 만들면 웹 서버와 유사한 기능을 수행할 수 있음
- Spring WebFlux가 Netty 위에서 돌아갈 때 Netty가 서블릿 컨테이너를 대신해 HTTP 요청을 받는 런타임 엔진 역할을 함

### 그럼 왜 WebFlux를 Tomcat이 아닌 Netty와?
- Tomcat 위에서도 WebFlux를 돌릴 **수는** 있음
  - Tomcat 3.1부터 스레드풀 방식이지만 Non-blocking I/O처럼 동작할 수 있도록 개선했음
  - 근데도 왜 Netty에서 돌려야 할까?

&nbsp;

- Tomcat의 경우 (스레드풀 방식)
  - 비동기 작업 만나면 스레드 반환하고 db 결과가 왔을때 다시 노는 스레드로 작업 처리 (처음과 나중의 스레드가 다를 수 있음) -> **컨텍스트 스위칭 비용 발생**
- Netty (이벤트 루프 방식) 
  - 동일 connection이면 해당 이벤트 루프(스레드)가 끝까지 책임짐

&nbsp;

- 컨텍스트 스위치 비용 발생 이유
  - 스레드 개수 차이
    - Tomcat은 기본적으로 수백개의 스레드, Netty 스레드가 코어 하나당 하나
  - 캐시 로컬리티
    - Netty는 동일 연결에서는 항상 같은 스레드 연결 -> CPU 캐시(L1, L2)에 해당 작업에 필요한 데이터가 그대로 남아있을 확률이 높음
    - Tomcat은 다른 스레드가 처리할 수도 있음 -> 새로운 스레드의 데이터를 RAM에서 읽어와야함
  - 스케줄러가 바쁨
    - 스레드를 교체할 때마다 현재 스레드의 레지스터 값, 스택 포인터 등을 메모리에 저장하고 다음 스레드의 정보를 로드하는 커널 모드 전환이 발생하는데, 이 작업 자체가 매우 무거운 연산
- 이벤트 루프 방식에서는 이 문제를 해결할 수 있음
- 그래서 비동기 방식을 온전히 사용하기 위해서는 Netty 위에서 Webflux를 사용해야함

---

## Spring WebFlux
- Spring WebFlux는 Spring MVC의 전통적인 동기식 모델과 달리 고성능 애플리케이션을 위한 **Non-blocking**, **비동기**(asynchronous), **이벤트 기반 프로그래밍**을 가능하게 하는 **리액티브 웹 프레임워크**
- Reactor와 Reactive Streams를 사용하며 더 적은 스레드로 많은 동시연결을 효율적으로 처리하게 함

### Spring MVC와 WebFlux의 차이
- 전통적인 Spring MVC는 동기/블로킹 방식입니다.
  - 요청 하나당 하나의 스레드가 할당
  - 작업이 끝날때까지 스레드가 기다림
- Spring WebFlux는 비동기/논블로킹 방식입니다.
  - 적은 스레드로 많은 요청 처리
  - 작업 중 비동기 호출 실행할 경우에 멍하니 기다리지 않고 바로 다음 이벤트 처리
    - 비동기/논블로킹 방식이기 때문에 DB도 **블로킹이 발생하지 않는 DB를 선택**해야함

### Event Loop

![이미지](/assets/img/posts/spring-boot/webflux-event-loop.png)
- 이벤트가 발생하면 이벤트 큐에 이벤트가 쌓임
- 이벤트 루프는 이벤트 큐에 있는 작업을 하나씩 꺼내서 실행
- 실행 중 비동기 호출(DB, 외부 API 등)을 만나면 제어권을 즉시 반환하고 이벤트 큐에서 다음 작업을 꺼내서 실행
  - 비동기 호출에서 응답이 오면 이벤트 큐에 데이터가 도착하면 처리할 작업이 추가됨
- 각 이벤트 루프는 전용 큐를 가짐

### 연결 생명 주기
- API를 호출한다고 할 때
  - HTTP 통신 시작 (데이터를 보낼 파이프 연결)
  - 서버 내부에서 Publisher(Mono, Flux)에 대한 구독 발생
  - 서버에서는 데이터를 계속 보냄 (이때도 backpressure 작동)
  - 서버에서 보낼 데이터가 더이상 없으면 끝났다는 신호를 보내고 HTTP 연결 종료

### Lazy Evaluation
- [Reactive Programming](#reactive-programming)에서 했던 설명처럼 파이프라인을 미리 설계하는 것
- ```java
  Flux.just("apple", "banana", "cherry")
          .map(String::toUpperCase)
          .flatMap(s -> Mono.just("Flux: "+ s))
          .subscribe(System.out::println);
  ```
  
  위 코드를 실행했을 때에 

  ```java
  Flux.just("apple", "banana", "cherry")
          .map(String::toUpperCase)
          .flatMap(s -> Mono.just("Flux: "+ s))
  ```
  
  여기까지는 데이터가 흐르지 않음(설계도)

  ```java
          .subscribe(System.out::println);
  ```
  
  `subscribe()`에서 설계도에 따라 데이터가 흘러들어가면서 실행됨

### Subscribe
- `subscribe()`는 파이프라인을 실행하는 트리거
- 시그널
  - onNext: Publisher가 생성한 데이터를 subscriber에게 하나씩 전달
  - onComplete: 모든 데이터를 전달하였음을 알리고 스트림 종료
  - onError: 에러가 발생함을 알리고 스트림 종료
- 코드 예시
  ```java
  Flux<String> travelFlux = Flux.just("Seoul", "Busan", "Jeju");
  
  travelFlux.subscribe(
      data -> System.out.println("onNext: " + data),   // 1. 데이터 처리 (onNext)
      error -> System.err.println("onError: " + error), // 2. 에러 처리 (onError)
      () -> System.out.println("onComplete!")          // 3. 완료 처리 (onComplete)
  );
  ```