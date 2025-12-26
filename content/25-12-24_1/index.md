---
emoji: ☀️
title: "[LLM] 테디노트의 RAG 비법노트 끄적끄적-7"
date: '2025-12-24 00:00:00'
author: 변우중
tags: LLM 프롬프트 RAG Cache 캐시 캐싱 LangChain 랭체인
categories: LLM
---
참고 : 테디노트의 RAG 비법노트 (https://fastcampus.co.kr/data_online_teddy)

소스코드: https://github.com/teddylee777/langchain-kr

위키독스: https://wikidocs.net/book/14314

&nbsp;

오늘은 캐시 방법에 대해 정리해보고자 합니다.

레츠기륏~!

&nbsp;

&nbsp;

## Cache

---

### 1. InMemory Cache (인메모리 캐시)

LLM 호출에서 인메모리 캐시를 사용한다면,<br>동일한 질문이 들어왔을 때 LLM(OpenAI 등) 서버로 요청을 전달하지 않고, **메모리에 미리 저장해둔 답변을 즉시 꺼내어 응답합니다.** (노트북 커널 재시작하면 캐시가 삭제됩니다.)

그렇기에, **LLM 호출 비용은 들지 않습니다**.

하지만, **질문이 약간 바뀌면(띄어쓰기 하나라도) 다시 호출하게 됩니다.** 그 이유는 `InMemoryCache`는 내부적으로 **Dictionary(Hash Map) 구조를 사용**하기 때문입니다.

```python
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())  # 인메모리 캐시 설정
```

위 코드를 작성한 후에 LLM 호출하게 되면 질문과 답변을 메모리에 저장해두게 됩니다.

&nbsp;

### 2. Semantic Cache (`RedisSemanticCache` 등)

`InMemoryCache`는 질문이 약간 바뀔 때 LLM을 다시 호출해야 하는 단점이 있었습니다.

반면에, **`Semantic Cache`는 질문이 바뀌어도 유사한 질문을 찾아 저장된 캐시를 사용한다는 특징을 가집니다.**

다만, **유사도 계산 시 임베딩 모델을 이용해야 하므로 비용이 들 수 있습니다.**

| **구분**      | **일반 인메모리 캐시 (InMemoryCache)** | **시맨틱 캐시 (RedisSemanticCache 등)**       |
| ------------- | -------------------------------------- | --------------------------------------------- |
| **비교 방식** | 문자열 완전 일치 (Exact Match)         | **벡터 유사도 비교 (Similarity Match)**       |
| **유연성**    | 띄어쓰기, 조사 하나만 틀려도 실패      | "날씨 어때?"와 "날씨 알려줘"를 같게 인식 가능 |
| **비용**      | 없음                                   | 유사도 계산을 위한 임베딩 모델 호출 비용 발생 |

아래 코드와 같이 작성 후, LLM 호출하게 되면 캐시를 저장하고 이후에는 저장된 캐시를 통해 답변을 하게 됩니다.

```python
from langchain.globals import set_llm_cache
from langchain_community.cache import RedisSemanticCache
from langchain_openai import OpenAIEmbeddings

# 1. 시맨틱 캐시는 '유사도'를 측정해야 하므로 임베딩 모델이 반드시 필요합니다.
embeddings = OpenAIEmbeddings()

# 2. 시맨틱 캐시 설정 (Redis 사용 예시)
set_llm_cache(RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=embeddings,
    score_threshold=0.1  # 이 점수가 낮을수록 더 '비슷해야' 캐시를 반환합니다.
))
```

여기서는 **Localhost에 설치된 Redis 데이터베이스**의 메모리에 저장하게 됩니다.

&nbsp;

### 3. SQLite Cache

SQLite Cache는  별도의 데이터베이스 서버 설치 없이, **내 컴퓨터의 하드디스크에 파일(`.db`) 형태로 데이터를 저장**하는 비휘발성 캐싱 방식입니다. (당연히 비휘발성입니다.)

LangChain에서 `InMemoryCache`의 휘발성 문제를 해결하면서도, `Redis`처럼 복잡한 서버 설정이 부담스러울 때 가장 많이 사용하는 **가장 간편한 영구 저장용 캐시**입니다.

 `InMemoryCache`와 동일하게 질문이 **완전히 동일**해야 캐시를 사용합니다

```python
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
import os

# 캐시 디렉토리를 생성합니다.
if not os.path.exists("cache"):
    os.makedirs("cache")

# SQLiteCache를 사용합니다.
set_llm_cache(SQLiteCache(database_path="cache/llm_cache.db"))
```

위의 코드를 작성 후,  LLM 호출하게 되면 로컬에 db 형태로 캐시를 저장하고, 이후 저장된 캐시를 통해 답변을 하게 됩니다.

&nbsp;

&nbsp;

여기서 끄읕.

```toc

```
