---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 17: CV 이론, CNN의 시각화와 데이터 증강"
date: 2025-09-23 18:40:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, cv, cnn, transformer]
description: "Computer Vison에 대해 배우자."
keywords: [numpy, colab, cv, computer vision, convoution, cnn, rnn, lstm, seq2seq, attention, transformer, vit, swin transformer, mae, dino]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Visualizing CNN & Data Augmentation

## CNN 시각화

CNN은 블랙박스로 비유됩니다. 이 블랙박스를 내부를 이해하기 위해서는 CNN을 시각화하는 것이 필요합니다.

- CNN 내부는 어떻게 구성되는가?
- CNN이 높은 성능을 보이는 이유는 무엇인가?
- CNN을 어떻게 개선할 수 있는가?

### Analysis of Model Behaviors

모델 행동 분석은 모델이 데이터를 어떻게 처리하고 학습하는지 분석하는 기법입니다. 모델 행동 분석에는 Embedding Feature Analysis와 Activation Investigation가 있습니다.

#### Embedding Feature Analysis

- Nearest Neighbors
- Dimensionality Reduction

#### Activation Investigation

- Maximally Activating Patches
- Class Visualization

### Model Decision Explanation

모델 결정 설명은 모델이 특정 결정을 내린 이유를 이미지의 특정 부분을 통해 설명하는 기법들입니다.

#### CAM(Class Activation Mapping)

CAM은 모델이 특정 클래스를 예측하는 데에 결정적인 영향을 미친 이미지 영역을 히트맵(heatmap)으로 표시합니다. CAM을 사용하려면 네트워크 마지막의 FC 레이어를 GAP(Global Average Pooling) 레이어로 교체하고 모델을 재학습해야 한다는 제약이 있습니다.

#### Grad-CAM

CAM은 모델 구조를 변경하거나 재학습을 해야한다는 불편함이 있습니다. Grad-CAM은 그런 문제를 해결한 일반화된 기법입니다. 특정 클래스 점수에 대한 마지막 Convolution 레이어의 그래디언트를 가중치로 사용하여 CAM과 유사한 heatmap을 생성합니다. 이미지 분류, 캡셔닝 등 다양한 태스크에 적용 가능합니다.

#### ViT Visualization

ViT에서는 Self-Attention 맵을 시각화하여, 모델이 예측을 위해 이미지의 어떤 패치들에 집중하는지 확인할 수 있습니다. 분류에 사용되는 \<CLS\> 토큰의 어텐션을 보면 모델의 판단 근거를 알 수 있습니다.

#### GAN Dissection

GAN이 학습 과정에서 스스로 해석 가능한 표현을 학습하며, 이를 이용해 이미지에 없던 사물을 추가하는 등 이미지 조작도 가능합니다.

## Data Augmentation

훈련 데이터셋을 실제 데이터 분포의 일부만을 포함합니다. 따라서 이 간극을 메우고 일반화 성능을 높이기 위해서 데이터 증강(Data Augmentation)이 사용됩니다.

### 기본적인 데이터 증강 기법

기본적인 데이터 증강 기법으로는 색상 조절, rotate, flip, crop, Affine transformation(기울이기) 등이 있습니다.

### 최신 데이터 증강 기법

- CutMix
  - 두 개의 훈련 이미지를 잘라 붙여 새로운 이미지를 만듭니다.
  - 레이블도 해당 비율에 맞게 혼합하여 모델이 객체를 더 잘 localize 하도록 돕습니다.
- RandAugment
  - 최적의 증강 기법 조합을 자동으로 탐색하는 방법입니다.
  - 어떤 증강을 적용할지, 얼마나 강하게 적용할지를 무작위로 샘플링하고 평가하여 최상의 정책을 찾습니다.
- Copy-Paste
  - 인스턴스 분할(Instance Segmentation) 작업에서 객체를 복사하여 다른 이미지에 붙여넣는 간단하지만 강력한 증강 기법입니다.

### Synthetic Data

합성 데이터(Synthetic Data)는 훈련 데이터를 얻기 매우 어려운 특정 작업을 위해 데이터를 인공적으로 생성하는 방법입니다. 미세한 움직임이 증폭된 영상을 만드는데 사용됩니다.