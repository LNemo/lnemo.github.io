---
layout: post
title: "[부스트캠프 AI Tech 8기] Day 21: NLP 이론, Tokenization"
date: 2025-09-29 19:00:00+0900
categories: [boostcamp]
tags: [boostcamp, ai, nlp, tokenization, word2vec]
description: "Natural Language Processing에 대해 배우자."
keywords: [colab, tokenization, embedding, byte pair encoding, bpe, rnn, language modeling, seq2seq, self-attention]
image:
  path: /assets/img/posts/boostcamp/boostcamp.jpg
  alt: "네이버 부스트캠프"
comment: true
---

# Tokenization

## Tokenization
Tokenization은 주어진 텍스트를 Token 단위로 분리하는 것입니다. Tokenization은 크게 Word-level Tokenization, Character-level Tokenization, Subword-level Tokenization가 있습니다.

### Word-level Tokenization

Word-level Tokenization은 단어 단위로 토큰을 구분합니다. 일반적으로 띄어쓰기를 기준으로 구분하지만, 한국어에서는 형태소를 기준으로 단어를 구분합니다.

```
['The', 'devil', 'is', 'in', 'the', 'details']
```

사전에 없는 단어가 등장하는 경우 [UNK] 토큰으로 처리해야 한다는 단점이 있습니다.

### Character-level Tokenization

Character-level Tokenization은 토큰을 글자 단위로 구분합니다. 글자 단위로 구분하기 때문에 다른 언어라도 같은 문자를 사용한다면 토큰으로 처리 가능합니다. 또 Out of Vocabulary 문제가 발생하지 않습니다.

```
['T', 'h', 'e', ' ', 'd', 'e', 'v', 'i', 'l', ' ', 'i', 's', 'i', 'n', ' ', 't', 'h', 'e', 'd', 'e', 't', 'a', 'i', 'l', 's']
```

하지만 주어진 텍스트에 대해 토큰이 지나치게 많아지고, 성능이 좋지 않다는 단점이 있습니다.

### Subword-level Tokenization

Subword-level Tokenization은 토큰을 subword 단위로 구분합니다. Subword의 범위는 여러가지 방법에 따라 다양하게 결정됩니다. 

```
['The', ' ', 'de', 'vil', ' ', 'is', ' ', 'in', ' ', 'the', ' ', 'de', 'tail', 's']
```

Character-level Tokenization에 비해 사용되는 토큰의 평균 개수가 적고 OoV 문제도 없으며 성능도 뛰어나다는 장점이 있습니다. 

#### Byte Pair Encoding

BPE(Byte Pair Encoding)은 Subword-level Tokenization의 대표적인 예시입니다. 빈도수가 높은 단어쌍들을 단어 목록에 추가하는 방식입니다.

1. 글자 단위의 단어 목록을 만든다
2. 가장 빈도수가 높은 단어쌍을 토큰으로 추가한다
3. 단어쌍이 추가된 단어 목록에서 또 단어쌍을 만들어 빈도수가 높은 단어쌍을 추가한다 - 해당 과정을 종료지점까지 반복

#### WordPiece

학습 데이터 내의 likelihood를 최대화하는 단어쌍을 단어 목록에 추가합니다. BERT, DistilBERT, ELECTRA에 활용되었습니다.

#### SentencePiece

공백을 Token으로 활용하여 Subword 위치가 띄어쓰기 뒤에 위치하는지 다른 Subword에 이어서 위치하는지 구분합니다. ALBERT와 XLNet, T5에 활용되었습니다.

## Word Embedding

우리는 단어를 그대로 이해할 수 있지만 컴퓨터는 단어를 문자 그대로 이해할 수 없습니다. 따라서 embedding을 통해 컴퓨터가 이해할 수 있도록 토큰을 변경해주어야 합니다.

### One-Hot Encoding

One-Hot Encoding은 단어를 Categorical variable로 인코딩한 벡터로 표현합니다. 단어의 벡터는 각각의 차원이 각각의 단어를 뜻합니다. 따라서 해당 단어를 표현하는 차원은 1, 나머지 다른 모든 차원은 0으로 표현됩니다.(Sparse representation)

```
봄  = [1, 0, 0, 0]
여름 = [0, 1, 0, 0]
가을 = [0, 0, 1, 0]
겨울 = [0, 0, 0, 1]
```

따라서 서로 다른 단어들의 Dot-product similarity는 항상 0이며 단어들 간의 유클리드 거리는 항상 루트2 입니다.

### Distributed Vector Representation

Distributed Vector(또는 Dense Vector)는 단어의 의미를 여러 차원에 걸쳐 0이 아닌 형태로 표현하는 방식입니다. 유클리드 거리, 내적, 코사인 유사도로 단어 사이의 의미론적 유사도도 구할 수 있습니다. 대표적인 방법으로 Word2Vec이 있습니다.

### Word2Vec
Word2Vec에서는 주변 단어의 정보들을 이용해 단어 벡터를 표현합니다. Word2Vec의 핵심 아이디어는 “cat의 의미는 확률분포 `P(w|cat)`에 의해서 결정된다”라는 말에서 볼 수 있듯이 단어의 의미는 주변에 의해서 결정된다는 것입니다. 

Word2Vec에서는 두 가지의 weight matrix가 있습니다. 이웃한 단어의 Embedding이 더 높은 내적 값을 가지도록 학습합니다. 그렇다면 서로 관계있는 단어끼리는 유사한 vector를 가지게 됩니다.

그렇게 학습된 단어 벡터들은 단어들 간의 관계를 나타내기 때문에 `vec[queen] - vec[king] = vec[woman] - vec[man]`와 같은 의미 관계도 실제로 확인할 수 있습니다.