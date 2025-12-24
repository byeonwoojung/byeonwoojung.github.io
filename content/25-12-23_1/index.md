---
emoji: ☀️
title: "Dense Retriever와 Sparse Retriever"
date: '2025-12-23 00:00:00'
author: 변우중
tags: LLM 프롬프트 RAG Retriever
categories: LLM RAG
---
오늘은 Dense Retriever와 Sparse Retriever에 대해 정리해보고자 합니다.

레츠기릿~!

&nbsp;

&nbsp;

## Dense Retriever vs Sparse Retriever

---

### 임베딩(Embedding)

Dense Retriever는 텍스트를 고차원 공간의 **밀집 벡터(Dense Vector)**로 변환합니다.

- **Sparse Vector:** 단어 사전 전체 크기의 벡터 중 대부분이 0인 형태 (단어 중복 위주)
- **Dense Vector:** 보통 768차원이나 1024차원의 고정된 크기 안에 모든 숫자가 의미 있는 수치로 채워진 형태

| **구분**      | **Sparse Retriever (BM25 등)**    | **Dense Retriever (DPR 등)**               |
| ------------- | --------------------------------- | ------------------------------------------ |
| **매칭 방식** | 키워드 중심 (Exact Match)         | 문맥 및 의미 중심 (Semantic Match)         |
| **특징**      | "사과"가 포함된 문서를 잘 찾음    | "애플"이나 "과일"이라는 단어도 문맥상 이해 |
| **장점**      | 빠르고, 도메인 지식 없이도 안정적 | 동의어 처리와 추상적인 질문 답변에 강함    |
| **단점**      | 오타나 유의어 처리에 취약함       | 학습 데이터가 많이 필요하고 계산 비용이 큼 |

### 왜 Dense Retriever를 쓸까?

1. **미세한 의미 차이 파악:** "파이썬 설치 방법"과 "Python 인스톨 가이드"가 같은 의미임을 이해합니다.
2. **질문-답변 성능 향상:** RAG(검색 증강 생성) 시스템에서 LLM이 답변하기 가장 좋은 문맥을 가져오는 데 탁월합니다.

### 대표적인 모델: DPR (Dense Passage Retrieval)

Meta(구 Facebook)에서 제안한 모델로, 두 개의 BERT 모델을 각각 질문용과 문서용 인코더로 학습시켜 성능을 극대화한 방식이 가장 유명합니다.

### 한계와 보완 (Hybrid Search)

Dense Retriever는 학습 데이터에 없는 특수한 고유명사나 품번(예: "A-1234") 같은 수치 매칭에는 약할 수 있습니다. 그래서 최근에는 **Sparse와 Dense를 섞은 Hybrid Search** 방식을 실무에서 가장 많이 사용합니다.

> 최근에는 고성능 벡터 데이터베이스(Chroma, Pinecone, FAISS 등)를 사용하여 수백만 개의 Dense Vector 중에서 가장 유사한 것을 0.1초 내외로 검색할 수 있게 되었습니다.

&nbsp;

&nbsp;

### 예제로 확인

Sparse Retriever와 Dense Retriever를 예제를 통해서 직접 비교해보겠습니다.

```python
from langchain_core.documents import Document

# 1. 예시 데이터 (문서 청크)
texts = [
    "아이폰 15 프로는 티타늄 소재를 사용하여 가볍습니다.",
    "갤럭시 S24 울트라는 AI 번역 기능을 제공합니다.",
    "점심 메뉴로 김치찌개와 제육볶음이 인기입니다.",
    "애플의 새로운 스마트폰은 C타입 충전 단자를 지원합니다.", # '아이폰' 단어 없음
]

# LangChain Document 객체로 변환
documents = [Document(page_content=t) for t in texts]
documents
"""출력:
[Document(metadata={}, page_content='아이폰 15 프로는 티타늄 소재를 사용하여 가볍습니다.'),
 Document(metadata={}, page_content='갤럭시 S24 울트라는 AI 번역 기능을 제공합니다.'),
 Document(metadata={}, page_content='점심 메뉴로 김치찌개와 제육볶음이 인기입니다.'),
 Document(metadata={}, page_content='애플의 새로운 스마트폰은 C타입 충전 단자를 지원합니다.')]
"""
```

LangChain Document 객체로 변환한 후에,<br>검색어 "**아이폰 15 프로 충전**"에 대한 **두 Retriever 결과를 비교**해보겠습니다.

&nbsp;

1. **BM25Retriever - Sparse Retriever 예시**

   ```python
   from langchain_community.retrievers import BM25Retriever
   
   # 1. Sparse Retriever (BM25)
   print("--- [A. Sparse Retriever (BM25)] ---")
   # BM25Retriever 객체 생성
   sparse_retriever = BM25Retriever.from_documents(documents)
   sparse_retriever.k = 1  # 상위 1개만 검색
   
   # 검색어: "아이폰 15 프로 충전"
   sparse_result = sparse_retriever.invoke("아이폰 15 프로 충전")
   print(f"검색어: '아이폰 15 프로 충전'")
   print(f"결과: {sparse_result[0].page_content}")
   
   """출력:
   --- [A. Sparse Retriever (BM25)] ---
   검색어: '아이폰 15 프로 충전'
   결과: 아이폰 15 프로는 티타늄 소재를 사용하여 가볍습니다.
   """
   ```

* **검색어 분해(토큰화):** `["아이폰", "15", "프로", "충전"]` (4개의 키워드)

* **첫번째와 네번째 문서가 유력한 후보**인데

      "아이폰 15 프로는 티타늄 소재를 사용하여 가볍습니다.",
      ...
      "애플의 새로운 스마트폰은 C타입 충전 단자를 지원합니다."

* **BM25**는 **1) 토큰이 문서 전체에서 얼마나 희귀한지, 2) 토큰이 각 문서에서 얼마나 자주 등장하는지(빈도 포화 계산을 통해 너무 많이 등장하면 더이상 점수가 오르지 않음), 3) 각 문서 길이는 얼마나 짧은지** 를 계산하여 점수를 매깁니다.
  * 이때 첫번째 문서 `"아이폰 15 프로는 티타늄 소재를 사용하여 가볍습니다."`가 `아이폰`, `15`, `프로`가 여러 개 겹치면서 다른 문서들과 비교했을 때 토큰들의 희귀성들이 큰 차이 없고 문서 길이도 큰 차이 없기 때문에
  * **Sparse Retriever인 BM25Retriever는 첫번째 문서를 가져오게 됩니다.**

&nbsp;

2. **허깅페이스의 jhgan/ko-sroberta-multitask 모델 - Dense Retriever 예시**

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 2. Dense Retriever (Vector Store)
print("--- [B. Dense Retriever (Vector/FAISS)] ---")
# 임베딩 모델 로드 (한국어 성능 좋은 모델)
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# 벡터 저장소(VectorStore) 생성 후 Retriever로 변환
vectorstore = FAISS.from_documents(documents, embedding_model)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 검색어: "아이폰 15 프로 충전"
dense_result = dense_retriever.invoke("아이폰 15 프로 충전")
print(f"검색어: '아이폰 15 프로 충전'")
print(f"결과: {dense_result[0].page_content}")
"""출력:
--- [B. Dense Retriever (Vector/FAISS)] ---
검색어: '아이폰 15 프로 충전'
결과: 애플의 새로운 스마트폰은 C타입 충전 단자를 지원합니다.
"""
```

* **검색어 벡터:** **`아이폰 15 프로 충전`**을 **아이폰 스마트폰의 특정 기종의 충전과 관련한 의미**를 가진 좌표로 변환할 것임
* **네번째 문서가 첫번째 문서보다 특별하게 `충전`이라는 핵심 의도가 가까우므로 네번째 문서를 가져오게 됩니다.**

&nbsp;

&nbsp;

Dense Retriever과 같은 Semantic Retriever은<br>벡터 검색은 "의미"를 너무 중시한 나머지 "정확성"을 놓칠 때가 많습니다.

특히, 고유명사를 무시하는 경우가 존재하여<br>최근에는 하이브리드로 Retriever를 이용하는 경우가 많다고 합니다.

이것은 다음에 다루도록 해보겠습니다.

그럼 이번 글은 여기까지 끄읕.



```toc

```
