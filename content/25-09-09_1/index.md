---
emoji: ☀️
title: "[NLP] 파이토치 트랜스포머를 활용한 자연어 처리와 컴퓨터비전 심층학습 -토큰화"
date: '2025-09-09 00:00:00'
author: 변우중
tags: NLP 자연어 자연어처리 토큰화
categories: NLP
---

참고 : 윤대희 외. (2023). *파이토치 트랜스포머를 활용한 자연어 처리와 컴퓨터비전 심층학습*. 위키북스.

소스코드: https://github.com/wikibook/pytorchtrf

위의 교재와 소스코드를 참고하였으며, 대부분의 내용은 직접 찾아보며 학습하였습니다.

&nbsp;

자연어 처리에서 토큰화하는 것에 대해 알아보자.

토큰을 나누는 기준은 공백 분할, 정규표현식 사용, 어휘사전 적용, 머신러닝 활용하는 방법이 있다. **어휘 사전**을 구축할 때, 너무 크게 구축하면 차원의 저주에 빠지고, 너무 작게 구축하면 OOV(Out of Vocab) 존재 가능성이 있으므로 그 크기를 적절히 정해야 하며, **출현빈도는 고려되지만 순서관계는 표현하지 못한다는 점**을 기억하자.

&nbsp;

## 토큰화 라이브러리

---

### jamo 라이브러리

- h2j (Hangul to Jamo): 한글 → 자모(자음과 모음 / 한글 글자를 초성,중성,종성 단위로 쪼갬) 변환한다.
  자모 단위로 쪼개었을 때 → 음운단위 학습 가능해져, **희귀단어**나 **신조어 처리**에 유리하다.

- j2hcj (Jamo to Hangul ConJoining): 자모 → 한글 변환한다. (초성+중성+(종성) 다시 합쳐 완전한 한글글자 만듦)

&nbsp;

### KoNLPy 라이브러리

* KoNLPy: 한국어 형태소 분석 라이브러리
* Okt: “Open Korean Text” 형태소 분석기

```python
from konlpy.tag import Okt

okt = Okt()

text = "한글 자연어 처리를 공부해봅시다."

# 형태소 분석
print(okt.morphs(text))
# ['한글', '자연어', '처리', '를', '공부', '해봅시다', '.']

# 형태소 + 품사 태깅
print(okt.pos(text))
# [('한글', 'Noun'), ('자연어', 'Noun'), ('처리', 'Noun'),
#  ('를', 'Josa'), ('공부', 'Noun'), ('해봅시다', 'Verb'), ('.', 'Punctuation')]

# 명사 추출
print(okt.nouns(text))
# ['한글', '자연어', '처리', '공부']

# 구 추출 - 연속된 명사 묶음을 그대로 반환
print (okt.phrases(text))
# ['한글', '한글 자연어', '한글 자연어 처리', '공부', '자연어', '처리']
```

* Kkma (꼬꼬마): 서울대학교 IDS(Intelligent Data Systems) 연구실에서 개발한 한국어 형태소/구문 분석기
  * 꼬꼬마는 `Okt`보다 **세밀한 품사 태깅**이 가능하다. (예: 구체적인 조사 구분)
  * 문장 분리(splitting) 기능이 기본 제공된다.
  * 속도는 Okt보다 느린 편이지만, 정확성은 높다.

```python
from konlpy.tag import Kkma

kkma = Kkma()

text = "한글 자연어 처리를 공부해봅시다."

# 형태소 분석
print(kkma.morphs(text))
# ['한글', '자연어', '처리', '를', '공부', '해보', 'ㅂ시다', '.'] -> ㅂ시다!!!

# 형태소 + 품사 태깅
print(kkma.pos(text))
# [('한글', 'NNG'), ('자연어', 'NNG'), ('처리', 'NNG'), ('를', 'JKO'),
# ('공부', 'NNG'), ('해보', 'VV'), ('ㅂ시다', 'EFA'), ('.', 'SF')]


# 명사 추출
print(kkma.nouns(text))
# ['한글', '자연어', '처리', '공부']

# 문장 추출
print (kkma.sentences(text))
# ['한글 자연어 처리를 공부 해봅시다.']
```

&nbsp;

### NLTK(Natural Language Toolkit) 라이브러리

* NLTK 라이브러리는 토큰화(tokenization), 품사 태깅, 파싱, 텍스트 분류, 코퍼스 제공 등 다양한 기능을 지원한다.
* 영어 기반이라 한국어 토큰화는 잘 안 된다!! → 한국어는 `KoNLPy`의 `Okt`, `Kkma`, `Mecab` 같은 형태소 분석기 이용하는 것이 더 좋음
* tokenize:

```python
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

text = "Hello world. This is NLTK. Let's learn tokenization! 123 test."

print(sent_tokenize(text))
# ['Hello world.', 'This is NLTK.', "Let's learn tokenization!", '123 test.']

print(word_tokenize(text))
# ['Hello', 'world', '.', 'This', 'is', 'NLTK', '.',
#  'Let', "'s", 'learn', 'tokenization', '!', '123', 'test', '.']


tokenizer = RegexpTokenizer(r'\w+')  # 알파벳 대소문자(a-z, A-Z), 숫자(0-9), 아래밑줄(_) 1개이상 포함 정규화
print(tokenizer.tokenize(text))
# ['Hello', 'world', 'This', 'is', 'NLTK', 'Let', 
# 's', 'learn', 'tokenization', '123', 'test']
```

&nbsp;

### spaCy 라이브러리

* 사이썬(Cython) 기반으로 개발된 오픈소스 라이브러리

* NLTK보다 빠른 속도, 높은 정확도를 가짐

* spaCy는 `spacy.load("en_core_web_sm")`로 모델을 불러와서 모델에 문장을 입력값으로 넣으면, 객체 지향적으로 처리된 결과를 속성값으로 접근할 수 있다.

* `doc` : 전체 문서 객체 (`spacy.tokens.doc.Doc`)

  `token` : 문서의 토큰 단위 (`spacy.tokens.token.Token`)

  `span` : 특정 구간 (문장, 구 등) (`spacy.tokens.span.Span`)

  → 각 객체에서 속성들을 접근할 수 있는데, 너무 많으니 생략!!

&nbsp;

&nbsp;

## 하위 단어 토큰화 (Subword Tokenization)

---

하위 단어 토큰화는 단어(word)를 더 작은 단위인 subword(하위 단어)로 나누는 방법이다. 완전한 형태소 분석보다는 **빈도 기반 분리**에 가깝다. **OOV(Out-Of-Vocabulary, 사전에 없는 단어) 문제를 해결**하기 위해 고안된 것!!

희귀 단어 처리

- '돈쭐내다'라는 단어가 사전에 없으면? → 전통적인 토큰화에서는 '돈'+'쭐'+'내다'로 쪼개서 처리함
  - 하위단어 토큰화를 적용한다면, 원래 있는 토큰 '돈'과 하위단어 '쭐', '내', '다'로 쪼개고
  - '돈'+'쭐내다(subword 패턴 조합)'로 신조어 패턴을 학습할 수 있음
- 신조어, 오탈자도 subword로 분리하면 어느 정도 의미 유지 가능하다.

단어 집합 크기 줄이기

- 모든 단어를 사전에 등록하면 수십만~수백만 단어 필요 → 메모리, 연산량 ↑
- subword로 쪼개면 수만 단위의 작은 사전으로 충분해진다.

&nbsp;

### 바이트 페어 인코딩 (BPE)

* 어휘 사전 추가 방법
  * **병합 점수 = 특정 쌍이 코퍼스(여러 문장)에서 함께 등장한 횟수(빈도)**
  * 가장 빈도가 높은 쌍을 합친 새로운 subword를 사전에 추가한다. (ex. ('돈','쭐') 쌍이 가장 많이 등장한 쌍이면 '돈쭐'을 어휘 사전에 추가함)
  * 이 과정을 vocab_size가 될 때까지 반복한다. (ex. subword로 연결된 '돈쭐'과 '내' 쌍이 가장 많이 등장하면 '돈쭐내'를 어휘 사전에 추가함)

* sentencepiece 라이브러리

  * 구글이 만든 언어 독립적인 서브워드 토크나이저
  * BPE(Byte Pair Encoding), Unigram LM 방식을 지원한다.

  * 띄어쓰기 기반이 아니라 **문자 단위 입력**을 사용하기 때문에, 한국어·중국어·일본어 등 띄어쓰기가 애매한 언어에도 잘 동작한다.

  * SentencePieceTrainer 모듈

    * 토크나이저 학습 모듈 불러옴 → **새로운 subword 토큰화 모델을 학습**할 때 사용함

    * 실행 결과
      `spm.model` → 학습된 토크나이저 모델 파일

      `spm.vocab` → 서브워드 사전 (토큰과 점수)

    * 주요 파라미터
      `--input` : 학습할 텍스트 파일

      `--model_prefix` : 출력 모델/사전 파일 이름 prefix

      `--vocab_size` : 어휘 크기 (예: 8000, 32000)

      `--model_type` : `unigram`(기본), `bpe`, `char`, `word`

      `--character_coverage` : 문자 커버율 (예: 한국어는 1.0, 일본어/중국어는 0.9995)

      `--input_sentence_size` : 학습에 사용할 문장 샘플 크기

* sentencepiece 라이브러리의 SentencePieceTrainer 모듈로 BPE 모델 학습 & 적용

```python
from sentencepiece import SentencePieceTrainer

# bpe 모델 학습
# input.txt: 학습할 말뭉치 (한 줄 = 하나의 문장)
# character_coverage=1.0: 학습 코퍼스에 등장하는 모든 문자를 vocab에 포함(한국어/한자/이모지 다룰 것)
# unk_id: 어휘 사전에 없는 OOV를 의미하는 unk 토큰의 id (기본값: 0)
# bos_id: 문장이 시작되는 지점을 의미하는 bos 토큰의 id (기본값: 1)
# eos_id: 문장이 끝나는 지점을 의미하는 eos 토큰의 id (기본값: 2)
SentencePieceTrainer.Train(
    "--input=input.txt\
    --model_prefix=spm\
    --vocab_size=8000\
    --model_type=bpe\
    --character_coverage=1.0"
)
```

```python
from sentencepiece import SentencePieceProcessor

# 1. 학습된 모델 로드
sp = SentencePieceProcessor()
sp.load("spm.model")   # Trainer가 만든 모델 파일

# 2. 문장을 subword 단위로 토큰화
text = "돈쭐내다라는 신조어를 SentencePiece로 실험해봅시다."

# 서브워드 단위로 분리 (문장 -> 서브워드)
pieces = sp.encode_as_pieces(text)
print("Pieces:", pieces)
# Pieces: ['▁돈', '쭐', '내', '다', '라는', '▁신', '조', '어',
# '를', '▁', 'S', 'e', 'n', 't', 'e', 'n', 'c', 'e',
# 'P', 'i', 'e', 'c', 'e', '로', '▁실', '험', '해', '봅', '시다', '.']


# 토큰 id로 변환 (문장 -> 정수 인코딩)
ids = sp.encode_as_ids(text)
print("IDs:", ids)
# IDs: [189, 3427, 1718, 1622, 165, 111, 1706,
# 1647, 1644, 1620, 2099, 1958, 1997, 1928, 1958,
# 1997, 2047, 1958, 2232, 2009, 1958, 2047, 1958, 1635,
# 136, 1875, 1639, 2029, 743, 1623]


# 다시 문장으로 복원 (정수 인코딩 -> 문장)
decoded_ids = sp.decode_ids(ids)
print("Decoded_ids:", decoded_ids)
# Decoded_ids: 돈쭐내다라는 신조어를 SentencePiece로 실험해봅시다.


# 다시 문장으로 복원 (서브워드 -> 문장)
decoded_pieces = sp.decode_pieces(text)
print("Decoded_pieces:", decoded_pieces)
# Decoded_pieces: 돈쭐내다라는 신조어를 SentencePiece로 실험해봅시다.
```

* 학습된 어휘사전 확인

```python
from sentencepiece import SentencePieceProcessor

tokenizer = SentencePieceProcessor()
tokenizer.load("spm.model")

vocab = {idx: tokenizer.id_to_piece(idx) for idx in range(tokenizer.get_piece_size())}
print(list(vocab.items())[:100])  # 100개 확인
# [(0, '<unk>'), (1, '<s>'), (2, '</s>'), (3, '니다'), (4, '▁이'), (5, '▁있'), (6, '습니다'), (7, '▁하'), (8, '▁그'), (9, '▁사'), (10, '▁대'), (11, '으로'), (12, '▁아'), (13, '▁국'), (14, '▁정'), (15, '합니다'), (16, '▁지'), (17, '에서'), (18, '▁수'), (19, '하는'), (20, '▁가'), (21, '하고'), (22, '입니다'), (23, '▁보'), (24, '..'), (25, '▁일'), (26, '▁한'), (27, '▁없'), (28, '▁해'), (29, '▁제'), (30, '▁부'), (31, '▁생'), (32, '▁자'), (33, '▁것'), (34, '▁나'), (35, '▁주'), (36, '▁국민'), (37, '▁안'), (38, '▁다'), (39, '들이'), (40, '▁시'), (41, '▁어'), (42, '▁기'), (43, '▁1'), (44, '▁저'), (45, '다고'), (46, '▁않'), (47, '▁공'), (48, '▁인'), (49, '▁전'), (50, '▁위'), (51, '▁경'), (52, '▁생각'), (53, '▁되'), (54, '▁모'), (55, '▁있는'), (56, '▁2'), (57, '▁청'), (58, '세요'), (59, '▁우'), (60, '▁여'), (61, '▁무'), (62, '지만'), (63, '▁문'), (64, '▁많'), (65, '니까'), (66, '▁사람'), (67, '▁바'), (68, '▁말'), (69, '▁조'), (70, '▁있습니다'), (71, '▁내'), (72, '▁대한'), (73, '에게'), (74, '이라'), (75, '▁고'), (76, '다는'), (77, '▁받'), (78, '▁의'), (79, '▁상'), (80, '들은'), (81, '▁현'), (82, '▁만'), (83, '▁소'), (84, '▁중'), (85, '는데'), (86, '▁더'), (87, '▁불'), (88, '▁합니다'), (89, '하여'), (90, '▁못'), (91, '해서'), (92, '▁우리'), (93, '▁개'), (94, '▁세'), (95, '▁비'), (96, '하게'), (97, '▁미'), (98, '▁법'), (99, '▁도')]
# vocab size : 8000
```

&nbsp;

### 워드피스 (Wordpiece)

* 구글 번역(Google Neural Machine Translation)에서 제안됨

* 어휘 사전 추가 방법

  * **병합 점수 = 글자쌍의 등장횟수 / 각 글자의 등장 횟수의 곱**

  * **초기 상태**: 어휘 사전(vocab)을 문자 단위로 시작 (예: `['ㄷ', 'ㅗ', 'ㄴ', '쭐', '내', '다']`)
  * **병합 과정**: 코퍼스에서 가장 자주 등장하는 **subword 쌍**을 합쳐 새로운 토큰으로 추가
    * `'ㄷ'`,`'ㅗ'`,`'ㄴ'` 의 등장 횟수 대비 `"돈"`이 자주 나오면 `'ㄷ'+'ㅗ'+'ㄴ' → '돈'``
  * **사전 확장**: 원하는 vocab_size (예: 30k)까지 반복
    * 많이 쓰이는 건 그대로 하나의 토큰, 드물게 쓰이는 건 작은 subword 단위로 쪼개짐

* 허깅페이스 Tokenizer 라이브러리

```python
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import Sequence, NFD, Lowercase
from tokenizers.pre_tokenizers import Whitespace

# 1. WordPiece 모델 정의
# - WordPiece 알고리즘을 기반으로 토크나이저를 생성
# - unk_token="[UNK]" : 사전에 없는 토큰(OOV)을 처리할 때 사용되는 특별 토큰
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

# 2. 정규화기(Normalizer) 설정
# - 입력 텍스트를 표준화하는 과정
# - NFD() : 유니코드 정규화 클래스 (예: é -> e + ´)
# - Lowercase() : 모든 문자를 소문자로 변환하는 정규화 클래스
# - 정규화 클래스는 NFKD, NFC, NFKC, strip 등이 더 존재함
# - Sequence([...]) : 여러 정규화기를 순서대로 적용
tokenizer.normalizer = Sequence([NFD(), Lowercase()])

# 3. PreTokenizer 설정 (사전 토큰화 클래스)
# - 본격적인 subword 학습 전에 단어 단위로 먼저 나누는 과정
# - Whitespace() : 공백 단위로 먼저 토큰 분리하는 클래스
# - 사전 토큰화 클래스는 Sequence(), CharDelimiterSplit(), Digits() 등이 더 존재함
tokenizer.pre_tokenizer = Whitespace()

# 4. WordPiece 토크나이저 학습
# - "../datasets/corpus.txt" : 학습에 사용할 말뭉치 파일
# - 파일 안의 텍스트를 기반으로 WordPiece subword 사전을 학습
tokenizer.train(["../datasets/corpus.txt"])

# 5. 학습된 토크나이저 저장
# - "../models/petition_wordpiece.json" : 토크나이저 설정 및 vocab이 포함된 JSON 파일
# - 나중에 이 파일을 불러와서 같은 토크나이저를 재사용 가능
tokenizer.save("../models/petition_wordpiece.json")

```

```python
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder

# 1. 학습된 모델 로드
tokenizer = Tokenizer.from_file("../models/petition_wordpiece.json")
tokenizer.decoder = WordPieceDecoder()

# 2. 문장을 subword 단위로 토큰화
text = "돈쭐내다라는 신조어를 SentencePiece로 실험해봅시다."

# 인코딩
encoded = tokenizer.encode(text)

# 인코딩 타입
print("encoded_type:", type(encoded))
# encoded_type: <class 'tokenizers.Encoding'>

# 문장 토큰화
tokens = encoded.tokens
print("tokens:", tokens)
# tokens: ['돈', '##ᄍ', '##ᅮᆯ', '##내', '##다', '##라는', '신',
# '##조', '##어를', 'se', '##nt', '##en', '##ce', '##p', '##i'
#  '##ec', '##e', '##로', '실험', '##해보', '##ᆸ시다', '.']

# 정수 인코딩
ids = encoded.ids
print("ids:", ids)
# ids: [7917, 4280, 7521, 7731, 7478, 7906,
# 7755, 7649, 11207, 26328, 23624, 10456, 21575,
# 4295, 4263, 10611, 4288, 7495, 12904, 9150, 9008, 13]

# 디코딩 (정수 인코딩 -> 문장 변환)
print("decoded:", tokenizer.decode(ids))
# decoded: 돈쭐내다라는 신조어를 sentencepiece로 실험해봅시다.
```

&nbsp;

토큰화하는 방법은 여러가지 존재한다.

번역/챗봇에서는 subword 방식(WordPiece, SentencePiece), 한국어 문법 분석에서는 형태소 기반(Okt, Kkma, Mecab), 소셜미디어 신조어 처리에서는 subword + 자모 단위 보완하도록 토큰화를 진행할 수 있다. 또한, 어떤 단어/서브워드를 어휘 사전에 포함시킬지에 따라 모델 성능이 크게 달라진다.

결국, **프로젝트 목적에 맞는 토큰화 선택이 중요하고, 어휘사전 구축이 핵심이다.**



```toc
```

