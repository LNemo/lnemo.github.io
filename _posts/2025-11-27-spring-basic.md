---
layout: post
title: "[Spring] 스프링 핵심 원리 기본편"
date: 2025-11-27 20:31:00+0900
categories: [Spring boot]
tags: [spring, spring boot, backend, java]
description: "Spring의 기본을 익히자."
keywords: [Spring Framework, IoC, DI, SOLID, 객체지향, 의존관계 주입, Spring Bean, Component Scan, Singleton, Bean Scope]
image:
  path: /assets/img/posts/spring-boot/spring.png
  alt: "Spring"
comment: true
---

# Spring 기본

김영한 님의 [스프링 핵심 원리 - 기본편] 강의를 듣고 주요 내용을 정리하였습니다.

## 개요

### Spring 이전에

- EJB(Enterprise Java Bean): 기업 환경의 시스템을 구현하기 위한 서버 측 컴포넌트 모델, 애플리케이션의 업무 로직을 가지고 있는 서버 애플리케이션
- EJB의 단점
  - 객체지향적이지 않음
  - 복잡한 프로그래밍 모델
  - 특정 환경, 기술에 종속적인 코드
- EJB → Spring, hibernate

### Spring 역사

- 로드 존슨이 EJB의 문제점을 지적하며 등장
- EJB 없이도 고품질 확장 가능한 애플리케이션을 개발할 수 있음을 보임(30,000라인 이상의 기반 기술을 예제 코드로 보여줌)
- 책의 예제 코드를 개발자들이 프로젝트에 사용
- 유겔 휠러, 얀 카로프가 로드 존슨에게 오픈소스 프로젝트를 제안 → Spring

### Spring

- Spring에는 프레임워크 하나만 있는 것이 아니라 여러 기술이 있음
  - Spring Framework, Spring Boot 뿐만 아니라 Spring Data, Spring Session 등..
- Spring Framework
  - 핵심 기술: 스프링 DI 컨테이너, AOP, 이벤트, 기타
  - 웹 기술: 스프링 MVC, 스프링 WebFlux
  - 데이터 접근 기술: 트랜잭션, JDBC, ORM 지원, XML 지원
  - 기술 통합: 캐시, 이메일, 원격접근, 스케줄링
  - 테스트: 스프링 기반 테스트
  - 언어: 코틀린, 구르비
- Spring Boot
  - Spring을 편리하게 사용할 수 있도록 지원
  - Tomcat 같은 웹 서버를 내장 → 웹 서버를 설치하지 않아도 됨
  - 손쉬운 빌드 구성을 위해 starter 종속성 제공
  - 스프링과 써드파티 라이브러리 자동 구성(버전 관리)
  - 메트릭, 상태 확인, 외부 구성 같은 프로덕션 준비 기능 제공
  - 간결한 설정

### 객체지향 프로그래밍

- Spring은 Java의 객체 지향 언어라는 특징을 살리기 위해 나온 프레임워크
- 추상화, 캡슐화, 상속, **다형성**
- **역할**과 **구현**으로 구분하면 구조가 단순해지고 유연해지며 변경도 편해짐
  - **인터페이스**와 **구현체**
- 서버가 변할때  클라이언트가 변하지 않을 수 있어야 함 ⇒ 다형성의 본질
- 스프링에서의 제어의 역전(IoC), 의존관계 주입(DI)은 다형성을 활용해 역할과 구현을 편리하게 다룰 수 있도록 지원

### SOLID

- 단일 책임 원칙(Single Responsibility principle)
  - 한 클래스는 하나의 책임만 가져야 함
  - 변경이 있을 때 파급 효과가 적으면 단일 책임 원칙을 잘 따른 것
- 개방-패쇄 원칙(Open/Closed Principle)
  - 확장에는 열려있으나 변경에는 닫혀있어야 함
  - **다형성**을 활용
  - 객체를 생성하고 연관관계를 맺어주는 별도의 조립, 설정자가 필요
- 리스코프 치환 원칙(Liskov Substitution Principle)
  - 프로그램의 객체는 프로그램의 정확성을 깨뜨리지 않으면서 하위 타입의 인스턴스로 바꿀 수 있어야 함
  - 다형성에서 하위 클래스는 인터페이스의 규약을 다 지켜야 한다는 것 ex) 자동차의 엑셀을 뒤로 가게 구현하면 안됨
- 인터페이스 분리 원칙(Interface Segregation Principle)
  - 특정 클라이언트를 위한 인터페이스가 범용 인터페이스 하나보다 나음
  - ex) 자동차 인터페이스보다는 운전 인터페이스, 정비 인터페이스로
- 의존관계 역전 원칙(Dependency Inversion Principle)
  - 추상화에 의존해야지 구체화에 의존하면 안됨
  - 구현 클래스에 의존하지 말고 인터페이스에 의존하라는 뜻
  - ex) 운전자가 자동차(인터페이스)에 의존해야지, k5나 아반떼(구현 클래스)에 의존하면 안됨

- 근데 다형성만으로는 OCP와 DIP를 지킬 수 없는데?
  - `private MemberRepository memberRepository = new MemoryMemberRepository();`
  - 위 코드는 `MemberRepository` 인터페이스뿐만 아니라 `MemoryMemberRepository` 구현 클래스까지 의존 → DIP 위반
  - `MemoryMemeberRepository`에서 `JdbcMemberRepository`로 변경할 경우 위의 클라이언트 코드도 수정해야함 → OCP 위반
  - 무언가가 더 필요하다 → **스프링이 DI로 다형성+OCP, DIP를 가능하도록 지원**
    - DI가 중요한 개념인 것!
- 

## 예제 만들기

- 의문
  - getter setter 메서드 이유 왜 직접 참조를 꺼리는지, 메서드로 사용해도 접근가능한건 똑같은거 아닌지
  - Map<Long, Member>, HashMap<> 이게 뭔지

### 순수 Java

```java
public class OrderServiceImpl implements OrderService {

    private final MemberRepository memberRepository = new MemoryMemberRepository();
    private final DiscountPolicy discountPolicy = new FixDiscountPolicy();

    
    @Override
    public Order createOrder(Long memberId, String itemName, int itemPrice) {
        Member member = memberRepository.findById(memberId);
        int discountPrice = discountPolicy.discount(member, itemPrice);

        return new Order(memberId, itemName, itemPrice, discountPrice);
    }
}
```

- 상황: 우리는 `DiscountPolicy`를 `FixDiscountPolicy`에서 `RateDiscountPolicy`로 교체하고 싶어요
- 현재 코드에서 `OrderServiceImpl`이 인터페이스인 `DiscountPolicy`뿐만 아니라 `FixDiscountPolicy`까지 의존
- 우리는 `DiscountPolicy`를 편하게 바꾸기 위해 인터페이스를 만들었는데 구현체가 직접 `OrderServiceImpl` 안에 있기 때문에 `OrderServiceImpl` 코드를 수정해야 함
- 이를 해결하고 싶어요 → 관심사를 분리하자!
  - `OrderServiceImpl`에서는 주문 서비스를 구현하는 것에 관심이 있음
  - `DiscountPolicy`를 어떤 것을 쓰든지 `DiscountPolicy`이기만 하면 됨
  - 어떤 `DiscountPolicy`를 사용할지 설정해주는 별도 설정 클래스를 만들면 됨

```java
public class AppConfig {
		// AppConfig가 OrderService에서 사용할 MemberRepository와 DiscountPolicy를 정해줌
		// 생성자를 통해 *주입* (인젝션)
    public OrderService orderService() {
        return new OrderServiceImpl(memberRepository(), discountPolicy());
    }
    public MemberRepository memberRepository() {
        return new MemoryMemberRepository();
    }
    public DiscountPolicy discountPolicy() {
		    return new FixDiscountPolicy();    // 우리는 이제 이 부분만 수정해서 교체 가능!
    }
}
```

```java
public class OrderServiceImpl implements OrderService {

    private final MemberRepository memberRepository;
    private final DiscountPolicy discountPolicy;

    public OrderServiceImpl(MemberRepository memberRepository, DiscountPolicy discountPolicy) {
        this.memberRepository = memberRepository;
        this.discountPolicy = discountPolicy;
    }
		...
}
```

- 이제 `OrderServiceImpl`은 의존관계는 외부에 맡기고 실행에만 집중할 수 있음
  - 클라이언트 입장에서 보면 의존관계를 마치 외부에서 주입해 주는 것 같다고 해서 **DI(Dependency Injection**, **의존관계 주입**)이라고 함

- 서비스를 실행할 때에 이제 `AppConfig`에서 가져옴

```java
MemberService memberService = new MemberServiceImpl();
// 이 코드를 아래로 바꾸어서 실행
AppConfig appconfig = new AppConfig();
MemberService memberService = new appConfig.memberService();
```

- **DIP 만족**

### IoC, DI, 컨테이너

- IoC(Inversion of Control)
  - 제어의 역전
  - 프로그램의 흐름을 직접 제어하지 않고 외부에서 관리하는 것
  - 프레임워크가 IoC의 대표적 예시
- DI(Dependency Infection)
  - 의존관계 주입
  - 의존관계는 “정적인 클래스 의존 관계”와 “실행 시점에 결정되는 동적인 인스턴스 의존 관계”를 분리해서 생각
  - 정적인 의존관계만으로는 어떤 인스턴스가 주입될지 알 수 없음
  - 애플리케이션 실행시점에 외부에서 실제 구현 객체를 생성하고 클라이언트에 전달해서 클라이언트와 서버의 실제 의존관계가 연결되는 것 → 의존관계 주입
  - 의존관계 주입으로 우리는 정적인 클래스 의존 관계는 그대로 두면서도 동적인 객체 인스턴스 의존 관계를 쉽게 변경할 수 있음
- IoC 컨테이너, DI 컨테이너
  - 위에서 만든 AppConfig 같이 객체를 생성, 관리하면서 의존관계를 연결해주는 것을 ‘IoC 컨테이너’ 또는 ‘DI 컨테이너’라고 함 (어샘블러, 오브젝트 팩토리 등으로도 불림)
  - 의존관계 주입에 초점을 맞추어 주로 DI 컨테이너라고 함

### 순수 Java → Spinrg

- `ApplicationContext`를 ‘스프링 컨테이너’라고 함
- 스프링 컨테이너는 `@Configuration`이 붙은 클래스를 설정 정보로 사용
- `@Bean`이 붙은 메서드를 모두 호출해 반환된 객체를 스프링 컨테이너에 등록  
  → 등록된 객체를 ‘스프링 빈’이라고 함
- 스프링 빈은 `@Bean`이 붙은 메서드 명을 스프링 빈의 이름으로 사용
- 스프링 빈은 `applicationContext.getBean()`메서드로 get 가능

```java
@Configuration
public class AppConfig {

    @Bean
    public MemberService memberService() {
        return new MemberServiceImpl(memberRepository());
    }

    @Bean
    public MemberRepository memberRepository() {
        return new MemoryMemberRepository();
    }

    @Bean
    public OrderService orderService() {
        return new OrderServiceImpl(memberRepository(), discountPolicy());
    }

    @Bean
    public DiscountPolicy discountPolicy() {
//        return new FixDiscountPolicy();
        return new RateDiscountPolicy();
    }
}
```

```java
// 스프링 컨테이너에서 memberService 스프링 빈을 가져오는 방법
ApplicationContext applicationContext = new AnnotationConfigApplicationContext(AppConfig.class);
MemberService memberService = applicationContext.getBean("memberService", MemberService.class);

```

- 같은 역할을 하는 것 같은데 스프링 컨테이너를 사용하면 복잡하기만 한 것 같고 왜 사용할까?

## 스프링 컨테이너

- `ApplicationContext`: 스프링 컨테이너. 인터페이스.
  - 정확히는 스프링 컨테이너는 BeanFactory와 ApplicationContext가 있지만 BeanFactory를 직접 사용하는 경우는 거의 없으므로 AppicationContext를 스프링 컨테이너라고 함

### 스프링 컨테이너

- 스프링 컨테이너는 XML을 기반으로도 만들 수 있고, 애노테이션 기반의 자바 설정 클래스로도 만들 수 있음
- 스프링 컨테이너는 설정 정보를 참고해서 의존관계를 주입함

### BeanFactory와 ApplicationContext

- BeanFactory는 스프링 컨테이너의 최상위 인터페이스
  - 스프링 빈을 관리하고 조회하는 역할 담당 (`getBean()` 등)
- ApplicationContext
  - BeanFactory의 기능을 모두 상속받아서 제공 (`[ApplicationContext] → [BeanFactory]`)
  - 빈을 관리 조회하는 기능은 물론 수 많은 부가기능이 필요
  - ApplicationContext가 제공하는 부가기능
    - 다양한 interface를 상속받음
    - `MessageSource`: 한국에서 들어오면 한국어, 영어권에서 들어오면 영어로 출력
    - `EnvironmentCapable`: 로컬/개발/운영을 구분해서 처리
    - `ApplicationEventPublisher`: 이벤트를 발행하고 구독하는 모델을 편리하게 지원
    - `ResourceLoader`: 파일, 클래스패스, 외부 등에서 리소스를 편리하게 조회

### 스프링 빈

- 빈 이름은 기본적으로 메서드 이름을 사용
  - 빈 이름을 직접 부여할 수도 있음 (`@Bean(name="memberService2")`)
- 빈을 조회할 때에는 부모 타입으로 조회하면 자식 타입도 같이 나옴
  - `Object`로 조회하면 모든 스프링 빈이 나옴

- `context.getBeanDefinitionNames()`: 스프링에 등록된 모든 빈 이름 조회
- `context.getBean()`: 빈 이름으로 빈 인스턴스 조회
  - `context.getBean(빈이름 , 타입)` or `context.getBean(타입)`
  - 같은 타입이 둘 이상 있으면 이름을 지정해주어야 함 (`NoUniqueBeanDefinitionException`)
  - 조회 대상 스프링 빈이 없으면 예외 발생 (`NoSuchBeanDefinitionException`)
- `context.getBeansOfType(MemberRepository.class)`: 특정 타입의 빈을 조회 (return: `Map<String, MemberRepository>`)

- 빈은 스프링이 내부에서 사용하는 빈과 사용자가 정의한 빈이 있음
  - `beanDefinition.getRole()`으로 구분 가능
  - `ROLE_APPLICATION`: 일반적으로 사용자가 정의한 빈
  - `ROLE_INFRASTRUCTURE`: 스프링이 내부에서 사용하는 빈

- `BeanDefinition`는 빈 설정 메타정보로 다양한 형태의 설정 정보가 추상화되어있는 것

### `@Configuration` 역할

- 싱글톤을 위한 것!
- 클래스를 보면 매번 `new memberRepository()`처럼 새로운 인스턴스를 불러오는 것처럼 보임
- 하지만 `@Configuration` 어노테이션으로 하나의 인스턴스만 사용하도록 해줌
- 스프링 빈에 등록된 `AppConfig`의 Class를 출력해보면 `AppConfig`가 아니라 `AppConfig@CGLIB`인 것을 확인할 수 있다
  - 바이트 코드를 조작해서 새롭게 작성
  - 해당 이름의 스프링 빈이 없으면 새로 생성
  - 해당 이름의 스프링 빈이 등록되어 있다면 가져와서 사용
  - `AppConfig@CGLIB`은 `AppConfig`의 자식 클래스
- 이 어노테이션을 사용하지 않으면 스프링 빈에 등록할 수는 있지만 싱글톤을 보장하지 않음

## 컴포넌트 스캔

- `@ComponentScan`은 `@Component` 어노테이션이 붙은 클래스를 스캔해서 스프링 빈으로 등록
- 그러면 필드가 없는데 어떻게 의존관계 주입을 하지?
- `@Autowired`를 사용한 의존관계 주입!

```java
// MemberServiceImpl에서 memberRepository의 의존관계를 주입할 때
private final MemberRepository memberRepository;

// 생성자에 @Autowired로 의존관계 주입 가능
@Autowired
public MemberServiceImpl(MemberRepository memberRepository) {
    this.memberRepository = memberRepository;
}
```

- `@Component`가 붙은 클래스명으로 앞글자는 소문자로 변환되어 스프링 빈 등록(memberServiceImpl 처럼)

### 탐색 시작 위치 지정

- `basePackages`
  - 탐색 패키지 지정. 이 패키지를 포함한 하위 패키지 모두 탐색
  - `basePackages = {"hello.core", "hello.service"}` 처럼 지정 가능
- `basePackageClasses`
  - 지정한 클래스의 패키지를 탐색 시작 위치로 지정
  - `basePackageClasses = AutoAppConfig.class`
- 탐색 시작 위치의 Default는 ComponentScan이 붙은 클래스의 패키지 위치부터 시작
- 권장 방법은 설정 정보 클래스의 위치를 프로젝트 최상단에 두고 basePackage 지정 생략
  - 사실 `@ComponentScan` 자체도 필요 없는게 `@SpringBootApplication`에 `@ComponentScan`이 들어있음

### 컴포넌트 스캔 대상

- `@Component` 은 물론이고 다른 어노테이션도 @Component가 붙어있기 때문에 컴포넌트 스캔이 된다
  - `@Component`
  - `@Controller`
  - `@Service`
  - `@Repository`
  - `@Configuration`
- 참고
  - 어노테이션은 상속 기능이 없음 ← 어노테이션에 어노테이션 붙이는 건 스프링의 기능

### 필터

- `includeFilters`, `excludeFilters`
- `excludeFilters`를 가끔 사용
- 최대한 스프링의 기본 설정에 최대한 맞추어 사용을 권장
- 타입 옵션
  - `FilterType.ANNOTATION`
  - `FilterType.ASSIGNABEL_TYPE`
  - `FilterType.ASPECTJ`
  - `FilterType.REGEX`
  - `FilterType.CUSTOM`

### 빈 이름이 중복될 경우?

- 컴포넌트 스캔으로 빈이 자동으로 등록되던 중 중복되는 이름이 있다면 오류가 발생
  - `ConflictingBeanDefinitionException`
- 그럼 만약 수동으로 빈을 등록한 것과 자동으로 등록되는 것의 이름이 중복된다면 수동 등록이 우선권을 가짐
  - 오버라이딩됨
  - 하지만 스프링부트에서는 버그를 막기 위해 오류를 발생시킴

## 의존관계 자동 주입

- 의존관계 주입은 크게 4가지 방법이 있음
  - 생성자 주입
  - 수정자 주입
  - 필드 주입
  - 일반 메서드 주입

### 생성자 주입 *

- 생성자를 통해 의존 관계를 주입
- 위에서의 `@Autowired`를 사용한 의존관계 주입이 여기에 해당
- 생성자 호출 시점에 딱 한 번만 호출되는 것이 보장
  - **불변**, **필수** 의존관계에 사용
- **생성자가 딱 1개만 있다면 `@Autowired`를 생략해도 자동으로 주입됨 (당연히 스프링 빈에만 해당)**
- 생성자 주입을 사용해야 하는 이유
  - 대부분의 의존관계 주입은 **불변**해야함
  - 테스트할 때에 **누락**을 막을 수 있음
  - **`final`** 키워드를 사용할 수 있음
- 대부분의 경우에 생성자 주입을 사용함

### 수정자 주입

- setter로도 주입 가능함

```java
private MemberRepository memberRepository;
private DiscountPolicy discountPolicy;

@Autowired
public void setMemberRepository(MemberRepository memberRepository) {
    this.memberRepository = memberRepository
}

@Autowired
public void setDiscountPolicy(DiscountPolicy discountPolicy) {
    this.discountPolicy = discountPolicy
}
```

- 의존 관계 주입 단계(호출 순서)
  - 생성자 → 수정자 순서

### 필드 주입

- 필드에 바로 주입

```java
@Autowired private MemberRepository memberRepository;
@Autowired private DiscountPolicy discountPolicy;
```

- 간결해서 좋을 것 같지만 외부에서 변경이 어렵기 때문에 테스트가 어려움 → 안쓰는게 좋음

### 메서드 주입

- 일반 메서드를 통해서 주입

```java
private MemberRepository memberRepository;
private DiscountPolicy discountPolicy;

@Autowired
public void init(MemberRepository memberRepository, DiscountPolicy discountPolicy) {
    this.memberRepository = memberRepository;
    this.discountPolicy = discountPolicy;
}
```

- 잘 사용하지 않음

### 옵션 처리

- `required = false` 로 없으면 호출이 안되게 할 수 있고
- `@Nullable` 로 null
- `Optional` 로 Optional.empty로 받을 수도 있음

```java
@Autowired(required = false)
public void setNoBean1(Member noBean1) {

}

@Autowired
public void setNoBean2(@Nullable Member noBean2) {

}

@Autowired
public void setNoBean3(Optional<Member> noBean3) {

}
```

### `@Autowired` 중복 클래스 해결

- 필드 명, 파라미터 명을 바꾼다 ← 타입 매칭을 시도하고 여러 빈이 있을 때 추가로 필드 명 매칭을 하기 때문에
- `@Quilifier`를 사용한다 ← 주입 시에 추가적인 방법을 제공하는 것이지 빈 이름 변경은 아님
- `@Primary`를 사용한다 ← 여러 빈이 매칭될 경우 해당 어노테이션을 가진 게 우선권을 가짐

### lombok?

- 어노테이션으로 getter, setter 등을 만들어 줌
- `@Getter`
- `@Setter`
- `@ToString`
- `@RequiredArgsConstructor` : final 필수값을 생성자로 만들어 줌

## 스프링 빈 이벤트 라이프사이클

- 스프링 컨테이너 생성 → 스프링 빈 생성 → 의존관계 주입 → 초기화 콜백 → 사용 → 소멸전 콜백 → 스프링 종료
  - 생성자 주입은 스프링 빈 생성 단계에서 의존관계 주입이 일어남

### 빈 생명주기 콜백

- 인터페이스 `InitializingBean`, `DisposableBean`
  - `InitializingBean`:
    - `afterPropertiesSet()` 을 구현하여 초기화 메서드를 호출
  - `DisposableBean`:
    - `destroy()`을 구현하여 종료시 콜백
  - 이 두 개는 스프링 전용 인터페이스 → 해당 코드는 스프링 전용 인터페이스에 의존한다는 문제
  - 초기화 소멸 메서드 이름 변경 불가
  - 내가 코드를 고칠 수 없는 외부 라이브러리에 적용 불가
- 빈 등록 초기화, 소멸 메서드
  - `@Bean(initMethod = "init', destroyMethod = "close")` 처럼 빈 설정하면서 초기화 소멸 메서드를 지정할 수 있다
  - 대부분의 라이브러리는 `close`, `shutdown` 라는 이름을 종료 메서드로 사용하는데  
  `@Bean`의 `destroyMethod` 기본값이 `(inferred)`로 `close`, `shutdown` 같은 메서드를 자동으로 추론하여 호출한다
  - 추론 기능을 사용하지 않으려면 `destroyMethod=""`처럼 비워놓기
- 어노테이션 `@PostConstruct`, `@PreDestroy`
  - 각 어노테이션을 init method, close method에 붙인다
  - 자바 표준
  - 평소에는 해당 어노테이션을 사용하되 외부 라이브러리에 필요하다면 `@Bean` 설정을 할 것

## 빈 스코프

- 빈 스코프는 빈이 존재할 수 있는 범위
- 스프링이 지원하는 스코프
  - 싱글톤
  - 프로토타입
  - 웹 관련 스코프
    - request
    - session
    - application

### 싱글톤 빈 요청

1. 싱글톤 스코프의 빈을 스프링 컨테이너에 요청
2. 스프링 컨테이너는 본인이 관리하는 스프링 빈 반환
3. 이후에 같은 요청이 와도 같은 객체 인스턴스 빈을 반환

### 프로토타입 빈 요청

1. 프로토타입 스코프의 빈을 요청한다
2. 스프링 컨테이너는 프로토타입 빈을 생성하고 필요한 의존관계를 주입
3. (이후에 관리하지 않음)
   - 그래서 `@PreDestroy`와 같은 종료메서드가 작동하지 않음

### 웹 스코프

- 웹 스코프는 웹 환경에서만 동작
- 웹 스코프는 프로토타입과 다르게 스프링이 해당 스코프의 종료 시점까지 관리함 (종료 메서드가 호출됨)
- 종류
  - request: HTTP 요청 하나가 들어오고 나갈때까지 유지되는 스코프
  - session: HTTP Session과 동일한 생명주기를 가지는 스코프
  - application: `ServletContext`와 동일한 생명주기를 가지는 스코프
  - websocket: 웹소켓과 동일한 생명주기를 가지는 스코프
- 해당 스코프를 가진 빈은 바로 등록해서 사용할 수 없음 (request가 들어오는 시점에 생성되어야 하기 때문에)
  - Provider를 통해 빈의 생성을 지연해야 함
  - 또는 프록시로 가능

### 프록시

```java
@Scope(value = "request", proxyMode = ScopedProxyMode.TARGET_CLASS)
```

- 이렇게 하면 가짜 프록시 클래스를 만들어두고 미리 주입할 수 있다 (!)
- Provider는 일일이 추가해주기 귀찮으니까..
- 가짜 프록시 빈은 사용자 요청이 오면 그제서야 진짜 빈을 요청하는 위임 로직이 들어있음
- 싱글톤 빈과 비슷하게 보이기 때문에 주의해서 사용하여야 함

### 싱글톤 빈과 프로토타입 빈을 함께 사용할 때 생기는 문제

- clientBean이 싱글톤 빈일때 프로토타입 빈과 의존관계일 때,
- 의존관계 자동 주입으로 주입 시점에 스프링 컨테이너에 프로토타입 빈을 요청
- 이때 clientBean은 프로토타입 빈을 내부 필드에 보관하게 됨 (정확하게는 참조값)
- 의도는 사용할 때마다 프로토타입 빈을 생성하는 거겠지만 **해당 프로토타입 빈은 이미 생성되었기 때문에 없어지고 새로 생성되지 않음(필드가 공유될 수 있음)**
- `Provider`로 해결 가능

### Provider

- 의존관계 주입은 DI, 의존관게를 찾는 것은 DL (Dependency Lookup)
- 스프링은 필요한 의존관계를 찾을 수 있는 DL 기능을 제공한다

```java
@Autowired
private ObjectProvider<PrototypeBean> prototypeBeanProvider;

public int logic() {
    PrototypeBean prototypeBean = prototypeBeanProvider.getObject();
    ...
}
```

- `ObjectProvider`로 스프링 컨테이너에서 새로운 프로토타입 빈을 꺼내 쓸 수 있음 (DL)
  - 기본적으로 `ObjectFactory`를 상속받고 여러 기능이 추가됨
- 하지만 스프링에 의존적임

- 자바 표준에서 제공하는 Provider도 있음

```java
//javax.inject:javax.inject:1 gradle 추가하여야 함
@Autowired
private Provider<PrototypeBean> prototypeBeanProvider;

public int logic() {
    PrototypeBean prototypeBean = prototypeBeanProvider.getObject();
    …
}
```

- build.grade에 추가해주어야 함