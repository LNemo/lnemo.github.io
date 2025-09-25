---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 19: CV 이론, Segmentation & Computational Imaging"
date: 2025-09-25 18:30:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, cv, cnn, vit, segmentation, detection, computational imaging]
description: "Computer Vison에 대해 배우자."
keywords: [numpy, colab, cv, computer vision, convoution, cnn, visualization, DETR, MaskFormer, camera noise, super resolution, deblurring, flickering]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Segmentation & Computational Imaging

오늘은 CV에서 Segmentation, Detection, Computational Imaging에 대해 정리해보았습니다.

## Semantic Segmentation
- 이미지의 각 픽셀을 특정 카테고리로 분류하는 작업
- 개별 객체(instance)를 구분하지 않고, 오직 픽셀의 의미론적(semantic) 카테고리에만 집중함
- 적용 분야
  - 의료 이미지 (장기 분할 등)
  - 자율 주행 (도로, 사람, 차선 등 영역 구분)
  - 이미지 편집 및 컴퓨터 사진학

### Semantic Segmentation Architectures
- FCN (Fully Convolutional Networks)
  - Semantic Segmentation을 위한 최초의 종단 간(end-to-end) 아키텍처
  - 기존 CNN의 Fully Connected Layer를 1x1 Convolution Layer로 대체하여 공간 정보를 보존하고, 임의의 크기 이미지를 입력받아 동일한 크기의 분할 맵을 출력
  - 다운샘플링으로 작아진 해상도를 업샘플링(Upsampling)을 통해 복원
  - Skip Connection을 도입하여 깊은 레이어의 의미 정보(coarse)와 얕은 레이어의 상세 정보(fine)를 결합함으로써 예측의 정확도를 높임

- U-Net
  - FCN에 기반한 모델로, 주로 의료 이미지 분할에서 높은 성능을 보임
  - 구조
    - Contracting Path (Encoder): 이미지를 압축하며 문맥(context) 정보를 추출. 반복적인 Convolution과 Max Pooling으로 특징맵의 크기는 절반으로 줄고 채널 수는 두 배로 늘어남
    - Expanding Path (Decoder): 압축된 특징맵을 다시 확장하며 정확한 지역화(localization) 수행. 업샘플링(Up-convolution)을 통해 특징맵의 크기를 키우고, Contracting Path의 동일 레벨 특징맵을 가져와 연결(concatenate)하여 공간 정보를 복원함

## Object Detection
- 이미지 내에 있는 객체의 종류(Classification)와 위치(Localization)를 바운딩 박스로 찾아내는 작업
- 출력 형태: (클래스, x_min, y_min, x_max, y_max)
- 적용 분야
  - 자율 주행
  - 광학 문자 인식 (OCR)

### Object Detection Architectures
- Two-stage Detector (R-CNN 계열)
  - '객체가 있을 만한 위치'를 먼저 제안하고, 해당 영역에 대해 분류를 수행하는 2단계 방식
  - R-CNN: Selective Search로 약 2,000개의 후보 영역(Region Proposal)을 추출하고, 각 영역마다 CNN을 통과시켜 분류를 수행해 매우 느림
  - Fast R-CNN: 이미지 전체에 대해 CNN을 한 번만 계산하여 속도를 개선
  - Faster R-CNN: 후보 영역 추출 과정(Region Proposal Network, RPN)까지 학습시켜 종단 간 학습이 가능해졌고 속도와 정확도를 모두 높임

- One-stage Detector (YOLO 계열)
  - 후보 영역 제안과 분류를 하나의 네트워크에서 동시에 처리하는 방식
  - YOLO (You Only Look Once): 이미지를 그리드(grid)로 나누고 각 셀이 바운딩 박스와 클래스 확률을 직접 예측. 구조가 간단하고 매우 빠름
  - One-stage Detector의 한계: 학습 시 배경(negative) 샘플이 객체(positive) 샘플보다 훨씬 많아 발생하는 클래스 불균형(Class Imbalance) 문제로 정확도가 낮았음
  - Focal Loss: 이 문제를 해결하기 위해 등장한 손실 함수로, 쉽게 분류되는 샘플(대부분 배경)의 손실 가중치를 낮추고, 어렵게 분류되는 샘플에 집중하여 학습하도록 함
  - RetinaNet: Focal Loss를 사용하는 One-stage Detector로, Two-stage Detector에 필적하는 정확도를 달성함

## Instance Segmentation
- Semantic Segmentation과 Object Detection을 결합한 형태
- 같은 클래스의 객체라도 개별 인스턴스로 구분하여 각 픽셀 단위로 분할
- Mask R-CNN
  - Faster R-CNN을 확장한 구조
  - 기존의 클래스 예측, 바운딩 박스 예측 브랜치에 병렬로 마스크를 예측하는 브랜치를 추가하여 각 RoI(Region of Interest)에 대한 마스크를 생성함

## Transformer-based Methods

### DETR
- 인코더-디코더 트랜스포머 구조
- 인코더에서 이미지 특징을 강화하고 디코더에서 Object Detection을 수행
- 과정
  - Backbone
    - 기본적인 Convolutional Neural Network를 사용해 공간 정보가 포함된 Feature Map을 추출
    - 여기에 Positional Encoding을 더해 위치 정보를 주입
  - Encoder
    - Feature Map을 `d x HW` 형태의 시퀀스(토큰)로 만들어 Encoder에 입력
    - Self-attention을 통해 토큰 간의 전역적인 관계를 학습하여 특징을 강화
  - Decoder
    - 학습 가능한 N개의 Object Query를 입력받아 '어떤 객체가 어디에 있는지'를 질의하는 형태
    - Auto-regressive(순차적) 방식이 아닌, N개의 예측을 병렬적으로 동시에 처리
  - Prediction Head
    - Decoder의 출력을 FFN(Feed Forward Network)에 통과시켜 클래스와 바운딩 박스를 예측
    - Object Query 개수와 동일한 수의 출력이 나옴
    - 클래스 예측 시 'no object'라는 특별한 레이블을 출력하면 해당 쿼리는 객체가 없다고 판단한 것

### MaskFormer
- Detection을 넘어 Segmentation 분야에 트랜스포머를 본격적으로 적용한 연구
- Semantic/Instance Segmentation 문제를 모두 '마스크 분류(Mask Classification)' 문제로 일반화할 수 있다는 통찰을 제시
- 과정
  - Pixel-level Module
    - Backbone → (Image Features) → Pixel Decoder → (Per-pixel Embeddings) 순으로 픽셀 단위의 임베딩 생성
  - Transformer Module
    - Image Features와 학습 가능한 N개의 쿼리를 Transformer Decoder에 입력하여 N개의 세그먼트 임베딩을 예측
    - 각 쿼리는 '특정 유형의 객체/영역이 이미지에 있는가?'를 질의
  - Segmentation Module
    - Transformer Module의 출력(세그먼트 임베딩)을 MLP에 통과시켜 N개의 클래스 예측과 N개의 마스크 임베딩을 생성
    - 마스크 임베딩은 Pixel-level Module의 Per-pixel Embedding과 내적(inner product)을 통해 최종 마스크 예측을 수행

### SAM (Segment Anything Model)
- TASK: Promptable Segmentation
  - 점, 상자, 텍스트 등 다양한 형태의 프롬프트(prompt)를 입력받아 유효한 마스크를 출력하는 것을 목표로 함
- 과정
  - Image Encoder: MAE 방식으로 사전 학습된 ViT를 사용해 이미지 임베딩을 생성
  - Prompt Encoder
    - Sparse prompts(점, 상자, 텍스트)와 Dense prompts(마스크)를 각각 다른 방식으로 처리
    - Dense prompts(mask): Convolution layer를 통과한 후 이미지 임베딩과 element-wise summation 수행
    - Sparse prompts(points, box, text):
      - 각 점이나 상자 꼭짓점의 위치 정보를 Positional Encoding으로, 유형 정보(전경/배경 등)를 Learned Type Embedding으로 변환하여 더함
      - Type Embedding은 학습을 통해 얻어짐
        - points: {foreground, background}
        - box: {top-left, bottom-right}
  - Mask Decoder
    - Input: [Output Tokens, Prompt Tokens]
    - 프롬프트 토큰과 이미지 임베딩 간의 상호작용을 통해 최종 마스크를 디코딩
    - 내부 과정:
      1. 토큰 간의 Self-attention
      2. 토큰을 쿼리로 이미지 임베딩에 대한 Cross-attention
      3. MLP
      4. 이미지 임베딩을 쿼리로 토큰에 대한 Cross-attention
      5. 위 4개 과정을 두 번 반복
      6. Transposed Convolution을 통해 이미지 임베딩을 업샘플링
      7. 업데이트된 토큰과 업샘플링된 이미지 임베딩을 사용해 최종적으로 마스크와 품질 점수(IoU)를 예측
- 프롬프트의 모호성을 고려하여, 하나의 프롬프트에 대해 3개의 마스크를 동시에 예측
- 학습 시에는 3개의 예측 중 Ground Truth와 비교하여 손실(loss)이 가장 낮은 마스크에 대해서만 역전파(backpropagation)를 수행
- 별도의 Data Engine을 구성하여 SA-1B 데이터셋 구축
  - 고품질의 분할 마스크는 웹에서 대량으로 구하기 어렵기 때문
  - 구성: 1100만 개 이미지, 11억 개 이상의 마스크
  - 3단계 과정
    1. Assisted-manual stage 
       - 공개 데이터셋으로 초기 학습된 SAM을 이용해 주석가가 대화형으로 마스크를 생성하고 수정
       - 6회 반복 학습을 통해 12만 개 이미지에서 430만 개 마스크 수집
    2. Semi-automatic stage
       - 마스크 다양성 확보를 목표로 함
       - 1단계 데이터로 학습한 객체 탐지기(Faster R-CNN)가 생성한 바운딩 박스를 SAM에 프롬프트로 제공해 마스크를 자동 생성하고, 주석가는 누락된 객체 위주로 추가 작업
       - 5회 반복 학습을 통해 총 1,020만 개 마스크 수집
    3. Fully automatic stage
       - 사람 개입 없이 완전 자동으로 마스크를 대량 생성
       - 32x32 점 그리드를 프롬프트로 제공
       - 모델의 IoU 예측, 마스크 안정성, NMS(비최대억제)를 통해 고품질 마스크만 선별
       - 이 단계에서 최종적으로 11억 개 이상의 마스크를 생성

## Computational Imaging

- Camera의 특수기능을 처리하기 위한 신호처리부분을 ISP(Image Signal Processor)라고 한다.
- 여기서는 복원이나 품질 향상을 중심으로 설명
- computational photography 기법도 deep learning 기반으로 설계
- 주로 UNet 사용
- Loss는 L2, L1 Loss를 주로 사용하지만 간단한 Loss function으로 만족스러운 결과를 얻지 못하는 경우에 대해서는 adversarial loss, perceptual loss 등을 사용한다.
- training data와 evaluation data에 대해 얻기 힘듦

### Training data in computational imaging

- 복원이나 품질 향상이 필요한 경우들과 Training Data, Eval Data를 어떻게 얻는지 정리
- 
#### Camera Noise
* 원인 - 낮은 조도에 의해서 발생한 미세한 전기 신호가 입력으로 들어와 증폭, 노이즈가 같이 증폭됨, read noise, quantization noise 등
* 기본적인 노이즈 가정: Gaussian noise 사용 - Gaussian 분포를 따르는 noise가 더해짐

#### Super Resolution
* 원인 - 카메라 센서의 한계로 인한 저해상도 이미지
* 고해상도의 이미지들을 모아서 down-sampling과 adding noise하면서 저해상도로 만들어줌
* RealSR 연구에서는 광학줌을 사용
  * 카메라에서 광학 줌을 한 것을 고해상도 사진(목표)으로 두고 줌을 하지 않고 크롭한 것을 저해상도 사진(input)으로 두고 학습하는 것이라고 생각

#### Deblurring
* 원인 - motion blur, lens blur 여기서는 모션블러만 초점을 맞추어서
* 간단한 합성데이터를 사용 (super resolution처럼 합성이지만 조금이라도 realistic한 데이터를 만들기 위해 노력함) high gram-rate camera(GoPro)를 사용한 블러 이미지 합성
  * 원본 frame rate의 exposure time의 시간 만큼 여러장의 high framerate frames을 평균내어 더함 → 해당 사진에 nonlinear CRF(Camera Response Function)를 적용해 approximated frame을 만듦
  * 하지만 이 방법은 boundary 부분에서 자연스럽지 않음
* 그래서 RealBlur가 등장
  * Dual Camera와 Beam splitter라는 특수 거울 사용
  * Beam splitter는 빛을 반은 통과 반은 반사
  * 카메라 하나는 low shutter speed, 하나는 high shutter speed로 각각 blurry 이미지와 sharp 이미지를 촬영하게 함

#### video motion magnification 비디오 모션 증폭
* 기존에 t에서 t+1로 갈때에 delta를 더해준다면, magnified frame에서는 delta에 alpha를 곱하여 더해줌
* real data pair가 없음
* 그래서 object segmentation dataset과 background image dataset을 적절하게 합성

### Advanced loss functions

- L2 or L1 loss는 perceptually well-aligned하지 않다 → 동일한 MSE라도 reference image와 비슷할 수도 있지만 완전히 달라보일 수도 있다
- 사람이 보는 관점과 비슷하게 loss를 만들기 위해 여러가지 방법이 제시

#### Adversarial loss
* Discriminator를 넣은게 핵심
* 구조
  * Generator는 fake data 생성
  * Discriminator가 fake data와 real data를 구분
  * Generator가 fake data와 real data를 구분하지 못하도록 data를 만들게 학습
  * Discriminator는 잘 구분하도록 학습

#### Perceptual loss
* loss 측정을 위해 pre-trained network가 필요함 (예를들어 VGG)
* pre-trained network는 학습하지 않고 fix해서 사용
* loss network를 통과시켜서 loss를 구함
  * feature reconstruction loss — 원본 이미지와 예측 이미지, 두 이미지의 loss를 구함 (일반적으로 L1이지만 L2도 가능)
  * sytle reconstruction loss — style target과 예측 이미지의 feature map을 구해서 gram matrices로 바꾸어 loss를 측정
    * Gram matrix — feature maps`(HxWxC)`를 `(CxH*W)`와 `(H*WxC)`로 flatten하여 곱해주어 구함

### Extension to video
- Flickering problem
  * 프레임마다 독립적인 프로세싱 → 일관되지 않는 번쩍이는 비디오
  - Recurren network로 해결
    - 현재 처리할 프레임(I_t), 이전 프레임(I_t-1), 독립적으로 처리된 현재 프레임(P_t), 그리고 이전에 생성된 결과 프레임(O_t-1)을 모두 입력으로 사용
    - ConvLSTM과 같은 구조를 통해 시간적 정보를 통합하여 시간적으로 일관성 있는 결과 프레임(O_t)을 생성
    - 생성된 결과(O_t)는 다시 다음 프레임(t+1)을 처리할 때 입력으로 사용됨
  - Loss Functions
    - 전체 손실 함수는 세 가지 요소의 가중합으로 구성됨
    - Perceptual loss
      - 생성된 프레임(O_t)이 독립적으로 처리된 프레임(P_t)의 내용(content)과 시각적으로 유사하도록 보장하는 역할
    - Short-term temporal loss
      - 현재 프레임의 결과(O_t)가 바로 이전 프레임의 결과(O_t-1)를 현재 시점으로 변환(warping)한 것과 일치하도록 하여 단기적인 일관성을 유지
      - 변환(warping)은 두 프레임 간의 움직임을 예측하는 Optical Flow를 사용하며, 가려짐(occlusion) 영역은 손실 계산에서 제외함
    - Long-term temporal loss
      - 현재 프레임의 결과(O_t)가 비디오의 첫 번째 결과 프레임(O_1)을 현재 시점으로 변환(warping)한 것과 일치하도록 하여 장기적인 일관성을 유지
      - 모든 프레임 쌍을 비교하는 대신 첫 프레임하고만 비교하여 계산 효율성을 높임

