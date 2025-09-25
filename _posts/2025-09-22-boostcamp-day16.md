---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 16: CV 이론, CNN과 ViT"
date: 2025-09-22 18:40:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, cv, cnn, transformer]
description: "Computer Vison에 대해 배우자."
keywords: [numpy, colab, cv, computer vision, convoution, cnn, rnn, lstm, seq2seq, attention, transformer, vit, swin transformer, mae, dino]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# CNN & ViT

## CV

CV(Computer Vision)은 무엇일까요? 처음에 이 단어를 보았을 때에 전혀 와닿지 않았던 기억이 있습니다. ‘컴퓨터의 미래에 대한 이야기인가?’ 라는 생각도 했었는데, 사실은 컴퓨터의 시각 처리에 관한 것이었습니다. CV는 **기계의 시각에 해당하는 부분을 연구**하는 컴퓨터 과학의 연구 분야입니다. 

| input \| output     | Image                                           | Semantic Attributes |
|---------------------|-------------------------------------------------|---------------------|
| Image               | Image processing /<br>Computational Photography | Computer Vision     |
| Semantic Attributes | Computer Graphics                               |                     |

### CNN

Neural Network는 데이터를 압축해서 weight를 학습하고 새로운 input이 들어왔을 때에 학습된 패턴으로 비교, 대조하여 결과를 도출해냅니다. Neural Network 중 가장 간단한 모델은 바로 CNN(Convolutional Neural Network)입니다. CNN은 말 그대로 Neural Network에 Convolution 연산을 더한 것입니다. 이미지를 Conv 연산해주면서 주변 정보를 압축할 수 있도록 합니다. 해당 지역적인 특징을 뽑아내어 각 Task에 맞도록 모델을 디자인해서 여러 분야에 활용할 수 있습니다.


#### LeNet-5

최초로 Convolution이 Neural Network에 적용된 모델입니다.

Overall architecture
- Conv - Pool - Conv - Pool - FC - FC

#### AlexNet

AlexNet은 LeNet-5보다 모델이 더 깊어지고 커집니다. Convolution을 나누어 계산하면서 병렬계산이 가능하게 하였습니다. 그리고 활성화함수로 ReLU를 도입하면서 Vanishing Gradient 문제를 완화하였습니다. Regularization 중 하나인 Dropout도 적용하였습니다.

Receptive filed 사이즈의 중요성이 대두되었습니다. Receptive filed 사이즈는 한 픽셀이 이전 레이어에서 얼마나 feature를 커버하고 있는지를 나타냅니다. Receptive field는 stride 1인 K x K의 conv layer, P x P인 pooling layer가 있을 때에 ``(P+K-1)X(P+K-1)`` 사이즈를 가집니다.

Overall architecture
- Conv - Pool - LRN - Conv - Pool - LRN - Conv - Conv - Conv - Pool - FC - FC - FC

#### VGGNet

VGGNet은 AlexNet보다 더 깊은 레이어를 가집니다. 16개의 layer, 19개의 layer를 가진 모델이 있습니다. 레이어가 깊어질수록 더 좋은 성능을 보였습니다. Local Response Normalization(LRN)을 사용하지 않았고, 오직 3x3 conv layer와 2x2의 max pooling layer를 사용했습니다.

VGGNet은 다음의 장점을 가집니다:
- 더 깊은 architecture
- 더 간단한 architecture
- 더 좋은 성능
- 더 좋은 일반화

하지만 레이어를 쌓을수록 좋았다면 왜 19개에서 멈췄을까요? 19개에서 더 깊게 레이어를 학습시킬 경우 Vanishing Gradient 문제가 있기 때문입니다.

#### ResNet

ResNet은 Vanishing Gradient 문제를 해결합니다. ResNet은 Conv 레이어에서 weight layer를 통과한 결과값에 통과하기 전의 값인 입력값을 더해서 보냅니다. 이 기법은 Skip Connection이라고 합니다. Skip Connection 덕분에 Vanishing Gradient의 문제가 해결되어 더 깊은 모델을 만들 수 있게 되었습니다.
ResNet은 Pooling layer 대신에 Conv의 stride를 2로 설정하여 수행하였습니다. 


### ViT

ViT는 [이전에 Transformer에서 설명하였듯이](https://lnemo.github.io/posts/boostcamp-day13/#vit) Transformer를 사용한 이미지 모델입니다. Transformer의 인코더 부분만 사용하며 CLS 토큰의 Output을 MLP Head와 연결하여 Class를 분류합니다.

> **ViT는 극단적으로 큰 데이터셋에서만 잘 작동합니다. 왜 그럴까요?**  
> ViT는 CNN의 inductive bias(공간적 근접성 및 위치 불변성)을 가정하지 않기 때문입니다. 그래서 순전히 데이터에서 특성을 학습해야 하고 그렇기 때문에 또 많은 양의 데이터가 필요한 것입니다.  
> 하지만 충분한 학습 데이터가 제공된다면 지역성을 넘는 복잡한 사례를 모델링 할 수 있기 때문에 CNN 기반보다 더 뛰어난 성능을 발휘할 수 있습니다.

#### Swin Transformer

Swin Transformer는 고해상도의 패치로 구성하지만, 그 블록들을 나누어 그 블록 내에서만 어텐션이 계산되도록 만듭니다. 그리고 윗 레이어로 갈수록 계층적인 구조로 구성되어 영상의 전체적인 맥락을 파악하기 좋은 피라미드 구조를 갖게 만들었습니다.

Window를 똑같이 Partitioning을 하게되면 Attention은 해당 window 내에서만 되기 때문에 정보가 지역적으로 국한되는 문제가 있을 수 있습니다. Shifted Window는 window를 살짝 shift함으로써 다음 레이어에서는 다른 window에 포함될 수 있도록 하였습니다.

#### Masked Autoencoders

실제의 상황에서 Transformer의 모델을 적용시키려면 매우 많은 데이터를 필요로 합니다. 이때 데이터를 Self-supervised training하여 Scaling 할 수 있습니다. Masked Autoencoders(MAE)는 Self-supervised training의 한 방법입니다. 

MAE는 input의 일부를 masking하고 인코더에 넣습니다. 마스크된 토큰들을 디코더에 넣게 되면 mask-out된 부분들을 복원하는 학습을 하게 됩니다.

#### DINO

DINO도 Self-supervise training의 한 종류로, student network와 teacher network의 구조로 되어있습니다. Student에는 이미지의 일부를, Teacher에는 전체 이미지를 input으로 넣고 student가 teacher의 예측과 같도록 학습합니다. 이 과정으로 학생 모델은 이미지의 중요한 특징을 스스로 터득하게 됩니다.



