---
layout: post
title: "[Termux] 안드로이드 공기계로 홈 서버 구축해보기 (feat. Discord Bot)"
date: 2025-12-04 18:55:00+0900
categories: [Project, Server]
tags: [termux, android, server, discord bot]
description: "공기계로 서버 열어보기!"
keywords: [Android Server, Termux, Galaxy Note 10, Home Lab, Node.js, Discord Bot, SSH, Port Forwarding, Upcycling]
image:
  path: /assets/img/posts/project/android-server/androidserver.jpg
  alt: "Spring"
comment: true
---

# 스마트폰 공기계로 서버 열어보기

## 과정

- [이 글](https://far.computer/how-to/)을 보고 공기계로 서버를 제공해보겠다 생각했지만 Note10은 postmarketOS 호환기기 목록에 Note10이 없었다
- 그래서 Termux로 시도해 보았다

### 환경

- 기종: Galaxy Note10 (SM-N971N)
  - 안드로이드 버전: 12
- 공유기: ipTIME N604E

### 초기세팅

1. F-Droid에서 Termux 다운로드 후 설치
2. 설정 - 애플리케이션 - 더보기(점 세 개 버튼) - 특별한 접근 - 모든 파일에 대한 접근 - Termux 권한 켜주기
3. Termux에서 `termux-setup-storage` 입력
4. `apt update && apt upgrade` 
5. `apt install openssh`: ssh 접속을 위해 설치
6. `apt install termux_services`: `sv-enable sshd`를 위해 설치
   - You can enbale sshd to autostart
     - `sv-enable sshd`
7. `apt install nodejs`: nodejs를 구동시키기 위해 설치

### 공유기 포트 열어주기

1. DHCP 서버 설정
   - 스마트폰의 IP를 찾아서 등록
2. 포트포워드 설정
   - 포트 열어줌 (Termux는 포트를 22 대신 8022를 사용)
   - 필요한 포트를 열어주면 됨
   - 외부에서도 접속가능해짐
3. DDNS 설정 (선택)
   - 도메인 이름 설정해서 외부에서 접속할 때 외우기 더 편함

- 이제 `ssh (username)@(도메인/IP 주소) -p 8022`로 접속 가능
  - Termux에서 다음을 수행해야 함
    - `whoami`로 유저 확인
    - `passwd` 로 비밀번호 설정
- ssh로 접속해 출력해본 폰 스펙

```bash
~ $ lscpu
Architecture:           aarch64
  CPU op-mode(s):       32-bit, 64-bit
  Byte Order:           Little Endian
CPU(s):                 8
  On-line CPU(s) list:  0-7
Vendor ID:              ARM
  Model name:           Cortex-A55
    Model:              0
    Thread(s) per core: 1
    Core(s) per socket: 4
    Socket(s):          1
    Stepping:           r1p0
    CPU(s) scaling MHz: 54%
    CPU max MHz:        1950.0000
    CPU min MHz:        442.0000
    BogoMIPS:           52.00
    Flags:              fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm dcpop asimddp
  Model name:           Cortex-A75
    Model:              1
    Thread(s) per core: 1
    Core(s) per socket: 2
    Socket(s):          1
    Stepping:           r2p1
    CPU(s) scaling MHz: 63%
    CPU max MHz:        2400.0000
    CPU min MHz:        507.0000
    BogoMIPS:           52.00
    Flags:              fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm dcpop asimddp
  Model name:           exynos-m4
    Model:              0
    Thread(s) per core: 1
    Core(s) per socket: 2
    Socket(s):          1
    Stepping:           0x1
    CPU(s) scaling MHz: 19%
    CPU max MHz:        2730.0000
    CPU min MHz:        520.0000
    BogoMIPS:           52.00
    Flags:              fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm dcpop asimddp
Vulnerabilities:
  L1tf:                 Not affected
  Meltdown:             Mitigation; PTI
  Spec store bypass:    Vulnerable
  Spectre v1:           Mitigation; __user pointer sanitization
  Spectre v2:           Mitigation; Branch predictor hardening, but not BHB
```

### Discord 봇 서버 넣기

- 코드를 github에 올린 다음에 clone하여 실행하였다
- 정상적으로 작동하는 것을 확인했다 (!)

![디스코드 봇 테스트](/assets/img/posts/project/android-server/discordbot.png)

## 생각

- 안드로이드에서 많은 권한을 막아두었기 때문에 루팅 없이는 Docker를 사용하는 것은 불가능하였다
- 스마트폰을 꽤나 괜찮은 서버로 쓸 수 있겠다고 생각했다
- (꽤나 비싸게 샀던) 방치된 자원을 사용한다는 것이 좋다고 생각했다
