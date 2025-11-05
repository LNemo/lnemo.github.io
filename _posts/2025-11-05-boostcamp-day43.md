---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 43: Generative AI, Image Generation"
date: 2025-11-05 17:30:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, generative ai, cv, diffusion]
description: "생성형 AI를 이해하자."
keywords: [Generative AI, Image Generation, GAN, Autoencoder, VAE, Diffusion Model, Stable Diffusion, Text-to-Image, Latent Diffusion Models, LDM, StyleGAN, CycleGAN, Pix2Pix, DDPM, DDIM, Classifier-free Guidance, SDXL, FID, Inception Score, CLIP Score]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Image Generation

오늘은 이미지 생성 모델들과 이미지 생성 모델의 평가 방법에 대해 공부하였습니다. 아래에 키워드 중심으로 정리하였습니다.

## GANs

- Generative Adversarial Networks
- Discriminator와 Generator를 적대적으로 학습하는 모델 구조
  - Discriminator(판별자): 진짜 이미지인지 판별
  - Generator(생성자): 잠재 변수 z를 입력으로 받아 학습 데이터의 분포에 가까운 이미지 생성
- 판별자는 데이터의 생성 여부를 잘 판단하도록 학습 → maximization
- 생성자는 판별자가 생성 여부를 판단하지 못하도록 학습 → minimization

### Conditional GAN

- GAN 학습에 조건(condition)을 주입하여 학습
- 주어진 조건에 따라 이미지 생성이 가능하도록 함

### Pix2Pix

- 이미지를 조건으로 이미지를 변환하는 방법
- Conditional GAN 구조를 따름
- 학습을 위해 서로 매칭되는 paired image가 필요

### CycleGAN

- Pix2Pix는 paired images가 필요하지만 많은 paired images를 확보하기 어려움
- CycleGAN은 Unpaired images로 학습하기 위해 cycle consistency loss를 제안한 GAN
- 학습을 위한 목적함수
  - L_GAN: GAN 학습을 위한 adversarial training loss
  - L_cyc: 생성한 이미지를 다시 원본 이미지로 생성했을 때 consistency를 유지하기 위한 loss

### StarGAN

- 단일 생성 모델만으로 여러 도메인 반영할 수 있는 구조 제안
- CycleGAN에서 제안된 cycle consistency loss와 domain classification을 활용
- 학습을 위한 목적함수
  - L_GAN: adversarial training loss
  - L_cls: 도메인을 판한하기 위한 loss
  - L_rec: cycle consistency loss

### ProgressiveGAN

- 고해상도 이미지를 생성하기 위해 저해상도 이미지 생성 구조부터 단계적으로 증강하는 모델 구조 제안
- 적은 비용으로 빠른 수렴 가능
- 과정
  1. 16x16 이미지로 32x32 이미지 생성
  2. 작은 해상도 이미지와 큰 해상도 이미지 결과를 weighted sum하여 사용

### StyleGAN

- ProgressiveGAN 구조에서 style을 주입
- 잠재 공간 Z를 바로 사용하는 것이 아니라 mapping network f를 사용하여 변환된 W를 입력으로 사용
  - Z는 일반적으로 가우시안 분포 → 학습데이터가 선형적으로 구성된다면 데이터 분포를 제대로 반영하지 못할 수 있음
  - ProgressiveGAN 구조에 잠재공간 Z를 변환하여 얻은 W에 affine transform을 적용하여 style y를 계산 (Adaptive Instance Normalization을 통해 반영)

## AE

- Autoencoder
- 인코더와 디코더로 구성되어 입력 이미지를 다시 복원하도록 학습하는 모델 구조
  - Encoder: 입력 이미지를 저차원 잠재 공간으로 매핑하여 잠재 변수 z로 변환
  - Decoder: 잠재 변수를 입력으로 사용하여 원본 이미지를 복원
- 학습을 위한 목적함수
  - reconstruction loss (MSE or MAE)

### VAE

- Variational Autoencoder
- Autoencoder와 동일하게 구성되어 있지만 잠재 공간의 분포를 가정하여 학습
- 인코더에서 잠재 공간의 평균과 표준편차가 나오도록 함

### VQ-VAE

- Vector Quantized-Variational Autoencoder
- 연속적인 잠재 공간이 아니라 이산적인 잠재 공간을 가정하여 학습에 사용
- 이산적인 잠재 공간은 이미지 뿐만아니라 텍스트, 음성과 같은 데이터에 더 적합
- Codebook도 학습

## Diffusion Model

### DDPM

- Denoising Diffusion Probabilistic Models
- 입력 이미지를 forward process를 통해 잠재 공간으로 변환하고 reverse process로 복원하는 구조
- Forward process: 가우시안 노이즈를 추가하며 잠재공간으로 매핑하는 과정
- Reverse process: Forward process에서 추가된 노이즈를 추정하며 제거하는 과정

### DDIM

- Denoising Diffusion Implicit Models
- DDPM은 step수가 많아 이미지 생성에 시간이 많이 소요됨
- DDIM은 stochastic sampling process를 deterministic sampling process로 정의
- 생성에서 모든 step을 거치지 않고 일부만 reverse process를 적용하게 함
- Non-Markovian diffusion process를 적용하여 전체 process의 subset만으로도 좋은 성능을 보임

### CFG

- Classifier Guidance
- CFG는 backward process에서 이전의 노이즈를 추정할 때 학습한 classifier의 기울기를 통해 임의의 클래스 y로 샘플링을 가이드하는데 사용
- DDIM은 CFG를 적용하기 위해 score-based conditioning trick을 적용
  - Score function은 노이즈를 제거하는 과정에서 데이터에 대한 likelihood가 높아지는 방향을 제시
  - CFG는 score function에 class y를 조건부로 주입

- 문제점
  - 기존 diffusion pipeline에 별도의 classifier가 추가되어 복잡해짐
  - 모든 step에 대한 classifier가 필요  

  → Classifier-free Guidance는 Classifier Guidance의 식을 conditional과 unconditional score로 분해하여 재정의

- Classifier-free Guidance는 noise level마다 classifier 학습 없이 class에 대한 guidance를 가중치 w로 조절할 수 있게 됨

### Latent Diffusion Models

- Diffusion 학습 시 image를 사용하는 것이 아니라 encoder를 통해 추출된 저차원의 잠재 변수를 사용
    
  → 고해상도 이미지 학습에 대해 적은 비용
    
- Classifier-free guidance 방식을 통해 생성에 condition을 반영할 수 있음(cross-attention)

## Stable Diffusion

- Stability AI에서 발표한 오픈소스 Text-to-Image Generation 모델
- LDM에서 일부 구조 개선된 모델
- 대량의 이미지, 텍스트 쌍 데이터 LAION-5B로 학습

### Architecture

- Autoencoder
- Image Information Creator
- Text Encoder

#### Autoencoder

- Image를 사용하는 것이 아니라 encoder를 통해 저차원에 매핑된 latent를 사용
  - 변수 감소로 학습 비용이 감소
  - 빠른 학습 및 생성 가능

#### Image Information Creator

- U-Net, Noise Scheduler로 구성

- Noise Scheduler
  - Noise 주입 정도를 결정
  - latent를 받아 noise를 입혀서 noisy latent 생성
- U-Net
  - Noise Prediction을 담당
  - Noise Scheduler에서 생성된 noisy latent와 time embedding을 input으로 받아 가해진 noise를 예측
    - Time embedding은 transformer의 positional encoding과 유사한 형태로 들어감
  - Text Encoder에서 나온 token embeddings과 noisy latent와 cross attention 결합
    - token embeddings는 key, value로, noisy latent는 query로

#### Text Encoder

- Text input을 embedding 형태로 바꿔주기 위해 CLIP Text Encoder 사용
- 77개의 token으로 맞추어 최종 embedding size는 [B, 77, 768]의 형태
- LDM 에서는 BERT encoder를 사용했지만 Stable Diffusion은 OpenAI의 CLIP Text Encoder 사용
  - 후속 연구에서 더 큰 언어모델을 사용하면 생성 이미지 품질이 더 좋아짐이 밝혀짐
  - 이후에 Stable Diffusion 2에서는 더 많은 파라미터의 OpenCLIP을 사용

### Training Process

- Stable Diffusion은 9억개 이상의 파라미터를 가짐
- 대량의 데이터, 많은 GPU가 필요하고 많은 시간이 필요함

1. Input data(image, text)를 각각의 인코더로 latent와 token embedding으로 변환
2. image latent는 random한 timestep만큼 noise scheduler를 이용해 noise를 가함
3. Noisy latent, token mebeddings, time step을 input으로 받아서 가해진 noise를 예측
4. 실제 noise와 예측된 noise의 차이를 통해 모델 학습 진행

### Inference Process

#### Text-to-Image

1. 가우시안 분포를 따르는 noise 상태에서 시작
2. input text의 token embeddings를 input으로 받아 U-Net을 통해 Noise prediction 진행
3. Predicted Noise를 제거 후 위 과정 반복
4. 최종 latent를 autoencoder의 decoder를 통해 image로 변환

#### Inpainting

1. Input image에서 변환된 latent에 noise를 가한 noisy latent에서 시작
2. Input image의 영향도를 높게 가져가기 위해 noise를 적게 가함
3. Input image의 영향도를 낮게 하려면 noise를 강하게
4. 이후 과정은 text-to-image와 동일하게 진행

### 후속 모델들

#### Stable Diffusion 2

- 이미지 해상도를 512x512 → 768x768로 개선
- Text Encoder를 Cliip Text Encoder → OpenCLIP으로 변경 (Text Enoder parameter: 63M → 354M)
- Depth2Img — Text 이외에도 Depth guided Image Generation 가능

#### Stable Diffusion XL

- Stable Diffusion보다 현실적인 이미지 생성
- 높은 색정확도, 정확한 색 대비, 빛, 그림자 조정
- 해상도 1024x1024
- 앞선 모델들과 다르게 two stage model로 구성 (Base + Refiner)
- Text Encoder를 2개 사용 → 더 정교하고 상세한 prompting 가능
- Parameter 수 증가 (865M → 2.6B)
- 정방형 이미지가 아닌 다양한 비율의 이미지 생성 가능
- Autoencoder 성능 향상

#### SDXL Turbo

- SDXL 개선판
- Adversarial Diffusion Distillation 방법론을 적용한 One-Step SDXL
- 원스텝임에도 앞선 모델들보다 높은 수준의 결과 생성

## Evaluation

### Inception Score

- 생성된 이미지의 질(Fidelity)과 다양성(Diversity)을 동시에 측정하는 지표
- Inception v3 모델을 이용 생성된 이미지를 분류
- 생성된 이미지의 likelihood 분포의 KL divergence로 Score 산출
- 높을수록 좋음

#### Fidelity

- 생성된 이미지가 특정 class를 명확히 표현하는가?
- 생성된 이미지가 명확한 class를 갖는다면 해당 이미지의 likelihood 분포(label 분포)는 특정 class에 치우친 분포가 됨

#### Diversity

- 다양한 class가 생성되고 있는가?
- 생성된 이미가 다양한 class를 가진다면 개별 이미지에 대한 likelihood 분포(marginal 분포)의 합은 균일한 분포가 됨

### FID Score

- Frechet Inception Distance
- Pre-trained Image classification model(Inception Network)을 활용해 추출한 벡터간의 거리를 score로 활용
- 거리를 Frechet distance로 측정
  - Frechet distance는 곡선을 이루는 points의 위치와 순서를 고려해 두 곡선 간 유사도를 측정하는 지표
- 낮을수록 좋음

#### Score 방법

- Inception Network를 이용해 구한 실제 이미지의 embedding
- Inception Network를 이용해 구한 생성 이미지의 embedding
- 위 두 embedding의 프레쳇 거리를 이용해 Scoring

### CLIP Score

- 이미지와 Caption과의 상관관계를 평가
- Caption으로부터 CLIP을 이용해 생성된 embedding
- 이미지로부터 CLIP을 이용해 생성된 embedding
- 위 두 embedding의 코사인 유사도를 Score로 사용
- 높을수록 좋음