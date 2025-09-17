---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 13: ML LifeCycle, Transformer"
date: 2025-09-17 19:08:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, rnn, lstm, seq2seq, attention, transformer]
description: "머신러닝의 생애주기에 대해 배우자."
keywords: [numpy, colab, ML LifeCycle, regression, linear classifier, neural networks, 데이터, 전처리, rnn, lstm, seq2seq, attention, transformer]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Transformer

## RNN

### RNN

**RNN**(Recurrent Neural Network)은 **순차 데이터**(시계열, 문장 등)를 처리하기 위해 내부 상태를 저장하는 순환적 구조를 가진 딥러닝 모델입니다.

**장점**
- 가변적인 길이의 Input Sequence를 처리할 수 있음
- 입력이 많아져도 모델의 크기는 증가하지 않음
- 이론상 t 시점에서 수행된 계산은 여러 단계 이전의 정보를 사용할 수 있음
- 모든 단계에서 동일한 가중치를 적용함

**단점**
- Recurrent computation이 느림
- 병렬화가 어려움
- **Vanishing gradient**
- **Long-range dependence** 모델링 실패
- 실제로는 여러 단계 이전의 정보에 접근 힘듦

### LSTM

**LSTM**(Long Short-Term Memory)은 RNN의 단점인 vanishing gradient, long-range dependence 문제를 해결하기 위해 등장했습니다. LSTM은 RNN에 Cell state를 추가하여 장기 기억에 도움이 되도록 하였습니다. Cell state는 **Forget gate**에서 얼마나 잊을지, **Input gate**에서 얼마나 기록할지, **Output gate**에서 얼마나 출력할지 정합니다.

### GRU

**GRU**(Gated Recurrent Units)는 LSTM과 유사한 아이디어로 RNN에서 장거리 의존성을 제공합니다. GRU는 장거리 의존성을 위해 Cell state 대신에 기존의 hidden state를 사용하기 때문에 LSTM보다 적은 파라미터를 사용합니다.

## Sequence-to-Sequence

### Many-to-many RNN

기존 RNN을 사용한 seqence-to-sequence 모델은 각 input마다 각각의 output을 가집니다. 여기서 우리는 해당 모델의 한계점을 느낄 수 있습니다. “I love you”라는 문장을 한국어로 “나는 너를 사랑해”라고 번역한다고 합시다. RNN을 사용한 번역은 I, love, you의 번역을 각각 내놓을 것입니다. 그렇게 되면 “나는 사랑해 너를” 이라는 문장이 나올 수 있습니다.

순서의 문제 말고도 입력 길이와 출력 길이가 같다는 것도 분명한 한계점입니다. “사랑해”는 “I love you”이지만 글자수가 다르기 때문에 해당 상황처럼 예측할 수 없는 상황도 생깁니다.

### Encoder-Decoder Structure

Encoder-Decoder Structure는 기존 RNN 모델의 문제점을 해결합니다. 이 Seq2seq 모델은 인코더 RNN과 디코더 RNN으로 구성됩니다. 인코더에서는 RNN을 사용해 각 input마다의 output은 무시하고 hidden state만 통과시킵니다. 가장 마지막의 hidden state를 디코더의 초기 hidden state로 초기화하고 입력으로 \<SOS\> 토큰을 주어 그것을 기준으로 출력을 시작합니다. 한번 예측하기 시작한 디코더는 \<EOS\>가 나오기 전까지 예측을 계속합니다.

Seq2seq은 input의 길이와 output의 길이가 다를 수 있습니다. 또한 인코더에서 모든 input의 정보를 담은 hidden state 덕에 문장을 전체적으로 볼 수 있습니다.

#### Teacher Forcing

Teacher Forcing은 디코더의 올바른 학습을 위한 장치입니다. 디코더가 맨 처음에 학습할 때에는 아무 정보도 없기 때문에 틀린 예측을 할 수 있습니다. 틀린 예측을 하게 되면 해당 값을 사용해서 다음 예측에 사용하기 때문에 다음 예측에 문제가 생기게 됩니다. 따라서 teacher forcing 확률을 설정해주면 해당 확률을 기준으로 현재 예측에 이전 예측값 대신 이전 정답값을 사용하게 할 수 있습니다. 이는 모델의 올바른 학습에 도움이 됩니다.

### Attention

Seq2seq 모델은 RNN의 문제를 어느정도 고친 것처럼 보였지만 여전히 문장히 길어진다면 초기 정보가 잊혀지는 문제가 있습니다. Attention은 이 문제를 해결하기 위한 아이디어입니다. Attention에서는 디코더에서 해당 문장에서 가장 관련성이 높은 토큰에 더 집중하도록 해서 정확성을 더 높입니다. 

#### Attention 함수
- ``Attention(Q, K, V) = Attention 값`` 
Seq2seq에서의 Q는 디코더의 현재 hidden state이고 K, V는 인코더의 hidden states입니다.

#### Dot-Product Attention

내적을 사용하여 Attention Score를 계산합니다.
1. 현재 디코더의 hidden state인 s와 인코더의 hidden state인 h들을 각각 모두 내적합니다. (Attention Score)
2. Attention Score를 softmax 함수를 통과시켜 확률로 만듭니다. (Attention coefficients)
3. 해당 Attention coefficients와 각각의 h를 가중합하여 s와 concat합니다. [a;s]
4. 이후 Seq2seq의 디코더와 동일하게 진행합니다. (해당 과정은 매번 반복)

## Transformer

Transformer는 멀티 헤드 어텐션 메커니즘을 기반으로 하는 딥러닝 아키텍처입니다. 이것을 이해하기 전에 기본 가정을 알 필요가 있습니다. Transformer는 기본적으로 ‘input x가 서로 유기적으로 관련된 여러 요소로 분할될 수 있다’고 하였습니다. 그래서 각 요소는 다른 요소들에 참여하여 본인의 representation을 개선할 수 있습니다. 따라서 Transformer는 주변을 활용하여 본인의 표현을 더 풍부하게 하는 방법이라고 생각할 수 있습니다.

### 출력은?

위의 방법은 자신의 표현을 풍부하게 할 뿐입니다. 이것만으로는 당장 번역은 무슨 분류, 회귀도 할 수 없을 것 같습니다. 또한 입력 데이터의 순서도 전혀 고려되고 있지 않습니다. 아직까지는 아무것도 할 수 없을 것 같습니다.

정말 간단하게로는 해당 출력을 모아서 평균을 내어 특정 분류 모델이나 회귀 모델에 넣는다면 해결될 것 같습니다. 하지만 단순히 평균을 내는 것만으로는 전체의 의미를 반영하는 것은 어려워보입니다.

댜른 방법으로는 입력에 \<CLS\> 토큰을 넣는 것입니다. 이 토큰은 다른 토큰에 어떤 의미도 전달하지 않기 때문에 어떤 토큰에도 치우치지 않습니다. 해당 토큰 위치의 출력에는 전체적인 의미를 가지는 값을 가지도록 합니다. 그 값에 분류 모델과 회귀 모델을 연결할 수도 있습니다.

### Encoder-Decoder

Transformer 모델은 인코더-디코더 아키텍처를 사용했습니다.

![이미지](/assets/img/posts/boostcamp/day13/transformer.png)
_By dvgodoy - https://github.com/dvgodoy/dl-visuals/?tab=readme-ov-file, CC BY 4.0, https://commons.wikimedia.org/w/index.php?curid=151216014_

**인코더**
1. Input Seqence(X)를 인코더에 넣습니다.
2. 순서 정보를 주기 위해 Positional Encoding을 X에 더해줍니다. (이미지엔 없습니다)
3. X로부터 Q, K, V를 계산해서 Attention value를 구합니다.
   - Multi-Headed는 한 토큰에 대해 head의 크기만큼 여러번의 계산을 합니다.
   - 결과를 모두 concat하고 W_o로 Linear transformation합니다. (original input size로 바꿔주기 위해)
   - Multi-head의 의미는 각각 상황에 따라 다른 토큰에 주목할 수도 있기 때문에 성능 향상 측면에서 매우 유리합니다.
4. Residual connection과 Layer Normalization을 진행합니다.
5. 문맥화(contextualized)된 embedding은 Feed-forward Layer를 통과합니다.
   - ``FFN(x) = max(0, xW1 + b1)W2 + b2``
   - 개별적으로 적용됩니다. (하나의 토큰 별로 계산)
   - 이 출력도 여전히 동일한 사이즈의 contextualized된 token embedding

**디코더**
1. 인코더의 출력이 주어지고, output sequence를 auto-regressively하게 생성
2. 순서 정보를 주기 위해 Positional Encoding을 더해줍니다.
3. 해당 입력으로 Q, K, V를 계산해서 Attention value를 구합니다.
   - Masked는 미래의 정보에 대해서는 마스킹 처리를 합니다.
4. 인코더의 K, V, 디코더의 Q를 계산해서 Attention value를 구합니다.
5. FF 레이어를 통과시킵니다.

### BERT
BERT(Bidirectional Encoder Representations from Transformers)는 Transformer에서 인코더만 가져와서 label없이 self-supervised한 모델입니다. BERT의 input은 다음의 구성을 가집니다.
- 두 문장
- Token embedding: pre-trained된 단어 embedding을 가집니다.
  - \<CLS\> — 분류 토큰. 항상 시작에 위치 
  - \<SEP\> — 구분 토큰. 문장의 끝을 표시
- Segment embedding: 각 토큰이 속한 문장을 나타내는 학습된 embedding
- Position embedding: 시퀀스의 각 위치를 위한 학습된 embedding

MLM(Masked Language Modeling)과 NSP(Next Sentence Prediction)으로 훈련시켰는데 BERT는 이 과제에서 약 98%의 정확도를 달성했습니다.

요즘에는 사전 훈련된 BERT가 word embedding을 위한 기본 선택이라고 합니다.

### ViT
ViT(Vision Transformer)는 Transformer를 사용한 이미지 모델입니다. 단어 대신 이미지를 16x16 패치로 분할해 넣으면서 linear embedding sequence가 Transformer에 입력됩니다. 이미지의 순서, 위치는 position embedding으로 지정해줄 수 있습니다. 최종적으로 \<CLS\> 토큰 위에 MLP를 추가하여 이미지를 분류합니다.

ViT는 극단적으로 큰 데이터셋에서만 잘 작동합니다. 왜 그럴까요?  
ViT는 CNN의 inductive bias(공간적 근접성 및 위치 불변성)을 가정하지 않기 때문입니다. 그래서 순전히 데이터에서 특성을 학습해야 하고 그렇기 때문에 또 많은 양의 데이터가 필요한 것입니다.  
하지만 충분한 학습 데이터가 제공된다면 지역성을 넘는 복잡한 사례를 모델링 할 수 있기 때문에 CNN 기반보다 더 뛰어난 성능을 발휘할 수 있습니다.

---
**피어세션을 통해 알아간 것**
- LSTM에서 forget gate = 1, input gate = 0이라면 RNN과 같다는 것이 무엇인지
  - f=1, i=0 이라면 c_t = c_(t-1) 이 되어 결국 c_t를 사용하지 않음
  - 따라서 cell state가 없는 RNN과 똑같은 상태라고 설명한 것
- f=1, i=0이면 셀 상태가 무한히 보장된다는 것은?
  - 이전까지의 기억을 잊지않고 계속 유지한다는 것