---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 20: CV 이론 과제와 보완할 점"
date: 2025-09-26 18:30:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, cv, cnn, vit, segmentation, detection, computational imaging]
description: "4주차 과제 점검과 위클리미션 리뷰"
keywords: [numpy, colab, cv, computer vision, convoution, cnn, visualization, DETR, MaskFormer, camera noise, super resolution, deblurring, flickering, resnet]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Week 4 Review 

오늘은 과제를 진행하면서 보완이 필요하거나 학습이 더 필요한 부분을 리뷰합니다.

## 과제

### 심화-1
- `param.numel()`로 parameter크기 셀 수 있음
- `functools.partial(show_activations_hook, name)`: name 인자를 미리 채워둠
- `register_forward_hook()`: forward 할 때마다 실행할 함수를 등록

## Weekly Mission
- `nn.Sequential(*list)`: list의 레이어들을 순차로 실행하는 레이어
- Conv 레이어는 out_channels만큼의 weight matrix가 존재
- skip-connection은 원래 identity를 더해주는데, 차원이 달라지는 경우 1x1 convolution으로 해결
- 파인튜닝시에 아래와 같이 weight를 학습하지 않도록 고정
  ```python
  for param in model_finetune.parameters():
      param.requires_grad = False
  model_finetune.fc = nn.Linear(512, 10)	# 마지막 fc를 해당 레이어로 대체
  ```
