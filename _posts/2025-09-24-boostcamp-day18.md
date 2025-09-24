---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 18: CV 이론, CNN의 시각화 과제 궁금한 점"
date: 2025-09-24 18:42:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, cv, cnn, visualization]
description: "Computer Vison에 대해 배우자."
keywords: [numpy, colab, cv, computer vision, convoution, cnn, visualization, cam, grad-cam]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Visualizing CNN 과제 중 Insight

## 1. PyTorch 모델 내부 들여다보기

- 레이어 가중치 확인: 모델의 특정 레이어에 접근하여 가중치(weight) 텐서를 직접 확인할 수 있다.
  - TensorFlow/Keras: `model.get_layer('name').get_weights()`
  - PyTorch: `model.module_name.layer_name.weight` (계층적 접근)
- 중간 레이어 출력(활성화 맵) 확인: `register_forward_hook`을 사용하면 모델의 순전파(forward pass) 중 특정 레이어가 내보내는 결과물(피처 맵)을 실시간으로 "낚아챌" 수 있다.
  - 이는 모델이 각 단계에서 이미지의 어떤 특징에 반응하는지(예: 초기 레이어는 선/모서리, 깊은 레이어는 구체적 형태)를 이해하는 데 결정적이다.
  - 각 레이어의 출력은 채널(필터) 수만큼의 작은 이미지 묶음이며, 이를 격자 형태로 시각화하여 각 필터의 역할을 분석한다.
- Hook에 추가 정보 전달: `functools.partial`을 사용하면 `register_forward_hook`이 기본적으로 제공하지 않는 추가 인자(예: 레이어 이름)를 hook 함수에 미리 "고정"시켜 전달할 수 있다.

---

## 2. 두 종류의 그래디언트(Gradient) 이해하기

- Saliency Map이나 Grad-CAM에서 사용하는 그래디언트는 모델 학습용 그래디언트와 목적과 대상이 다르다.

| 구분 | 모델 학습용 그래디언트 | 모델 해석용 그래디언트 (Saliency) |
| :--- | :--- | :--- |
| 목표 | 손실(Loss) 최소화 | 예측 근거 해석 |
| 미분 대상| 가중치 (Weights) | 입력 이미지 (Input Image) |
| 수식 | `∂Loss / ∂Weight` | `∂Score / ∂Image` |

- 모델 해석용 그래디언트는 "최종 예측 점수를 바꾸기 위해 입력 이미지의 어떤 픽셀을 바꿔야 하는가?"에 대한 답이며, 이 값 자체가 바로 Saliency Map의 핵심 데이터가 된다.
- 일반적으로 Softmax 통과 후의 확률이 아닌, 통과 전의 로짓(logit) 또는 스코어(score)를 기준으로 미분해야 더 선명한 해석 결과를 얻을 수 있다.

---

## 3. 시각화 기법 (Saliency Map, CAM, Grad-CAM)

### Saliency Map
- 실체: 최종 점수를 입력 이미지로 미분한 값. 원본 이미지와 크기가 동일한 2차원 숫자 배열(행렬)이다.
- 의미: 각 숫자는 해당 위치의 픽셀이 예측에 얼마나 중요한지를 나타낸다.
- 시각화: 이 숫자 배열을 히트맵으로 변환하여 원본 이미지 위에 겹쳐서 보여준다.

### CAM (Class Activation Mapping) 
- 개념: Grad-CAM의 전신. "모델이 이미지의 어느 부분을 보고 특정 클래스를 예측했는가?"를 보여주는 최초의 기법 중 하나이다. 
- 과정: 
  1. A (피처 맵): 마지막 Conv Layer의 출력(`(C, H, W)`)을 얻는다. 
  2. w (가중치): 특정 클래스를 예측하는 데 사용된 마지막 FC Layer의 가중치(`(C)`)를 가져온다. 이 가중치가 각 피처 맵 채널의 '중요도' 역할을 한다. 
  3. 가중합: `w`와 `A`를 곱하고 채널 축으로 합산하여 저해상도 히트맵(`(H, W)`)을 생성한다. **그래디언트 계산이 필요 없다.** 
  4. 업샘플링: 저해상도 히트맵을 원본 이미지 크기로 확대하여 시각화한다.

### Grad-CAM
- 과정:
  1. A (피처 맵): 목표 레이어의 출력(`(C, H, W)`)
  2. dA (그래디언트): 최종 점수를 피처 맵 `A`로 미분한 값(`(C, H, W)`)
  3. α (채널 중요도): 그래디언트 `dA`를 채널별로 평균(`(C)`)
  4. 가중합: `α`와 `A`를 곱하고 채널 축으로 합산하여 저해상도 히트맵(`(H, W)`) 생성
  5. 업샘플링: 저해상도 히트맵을 원본 이미지 크기로 확대하여 시각화
- 업샘플링: 작은 해상도의 데이터를 큰 해상도로 키우는 기술.
  - Bilinear Interpolation: 주변 4개 픽셀의 가중 평균으로 부드럽게 값을 채움 (Grad-CAM 시각화에 주로 사용).
  - Transposed Convolution: 업샘플링 방식을 네트워크가 직접 학습 (U-Net, GAN 등에서 사용).