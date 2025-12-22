---
emoji: ☀️
title: "[LLM] 테디노트의 RAG 비법노트 끄적끄적-4"
date: '2025-12-14 00:00:00'
author: 변우중
tags: NLP 자연어 자연어처리 LLM 프롬프트 RAG
categories: NLP LLM
---
참고 : 테디노트의 RAG 비법노트 (https://fastcampus.co.kr/data_online_teddy)

소스코드: https://github.com/teddylee777/langchain-kr

&nbsp;

오늘도 갓 테디노트님 끄적 레츠기릿.

&nbsp;

그 전에 잠시!!`<br>`딕셔너리 키워드 인자 전달하는 파이썬 문법 확인하고 가고자 합니다~

```python
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate


examples = [
    {
        "question": "스티브 잡스와 아인슈타인 중 누가 더 오래 살았나요?",
        "answer": """이 질문에 추가 질문이 필요한가요: 예.
추가 질문: 스티브 잡스는 몇 살에 사망했나요?
중간 답변: 스티브 잡스는 56세에 사망했습니다.
추가 질문: 아인슈타인은 몇 살에 사망했나요?
중간 답변: 아인슈타인은 76세에 사망했습니다.
최종 답변은: 아인슈타인
""",
    },
    {
        "question": "네이버의 창립자는 언제 태어났나요?",
        "answer": """이 질문에 추가 질문이 필요한가요: 예.
추가 질문: 네이버의 창립자는 누구인가요?
중간 답변: 네이버는 이해진에 의해 창립되었습니다.
추가 질문: 이해진은 언제 태어났나요?
중간 답변: 이해진은 1967년 6월 22일에 태어났습니다.
최종 답변은: 1967년 6월 22일
""",
    }
]

example_prompt = PromptTemplate.from_template(
    # question, answer 키워드 인자 사용
    "Question:\n{question}\nAnswer:\n{answer}"
)

# **examples[0]로 키워드 인자 전달
print(example_prompt.format(**examples[0]))
"""출력:
Question:
스티브 잡스와 아인슈타인 중 누가 더 오래 살았나요?
Answer:
이 질문에 추가 질문이 필요한가요: 예.
추가 질문: 스티브 잡스는 몇 살에 사망했나요?
중간 답변: 스티브 잡스는 56세에 사망했습니다.
추가 질문: 아인슈타인은 몇 살에 사망했나요?
중간 답변: 아인슈타인은 76세에 사망했습니다.
최종 답변은: 아인슈타인
"""
```

example의 첫번째

```
{
    "question": "스티브 잡스와 아인슈타인 중 누가 더 오래 살았나요?",
    "answer": """이 질문에 추가 질문이 필요한가요: 예.
추가 질문: 스티브 잡스는 몇 살에 사망했나요?
중간 답변: 스티브 잡스는 56세에 사망했습니다.
추가 질문: 아인슈타인은 몇 살에 사망했나요?
중간 답변: 아인슈타인은 76세에 사망했습니다.
최종 답변은: 아인슈타인
""",
}
```

에서 언패킹하는 방법으로 PromptTemplate.from_template에 키워드 인자를 전달하고, 그 키워드를 직접 사용해 프롬프트 템플릿을 만들 수 있습니다.

&nbsp;

그럼 진짜 레츠기륏!

&nbsp;

## FewShotPromptTemplate

---

LLM에게 예시 몇 가지를 `FewShotPromptTemplate`을 활용해 던져주는 방법을 알아봅시다.

사실, 프롬프트에 예시를 직접 포함해도 되지만 `<br>`예시를 선택적으로 삽입하는 모듈과 함께 쓰면 좋기에 활용성이 괜찮습니다.

```python
example_prompt = PromptTemplate.from_template(
    # question, answer 키워드 인자 사용
    "Question:\n{question}\nAnswer:\n{answer}"
)

prompt = FewShotPromptTemplate(
    # 예시
    examples=examples,
    # 예시 프롬프트 템플릿
    example_prompt=example_prompt,
    suffix="Question:\n{question}\nAnswer:",
    input_variables=["question"],
)

question = "Google이 창립된 연도에 Bill Gates의 나이는 몇 살인가요?"
final_prompt = prompt.format(question=question)
print(final_prompt)

"""출력:
Question:
스티브 잡스와 아인슈타인 중 누가 더 오래 살았나요?
Answer:
이 질문에 추가 질문이 필요한가요: 예.
추가 질문: 스티브 잡스는 몇 살에 사망했나요?
중간 답변: 스티브 잡스는 56세에 사망했습니다.
추가 질문: 아인슈타인은 몇 살에 사망했나요?
중간 답변: 아인슈타인은 76세에 사망했습니다.
최종 답변은: 아인슈타인


Question:
네이버의 창립자는 언제 태어났나요?
Answer:
이 질문에 추가 질문이 필요한가요: 예.
추가 질문: 네이버의 창립자는 누구인가요?
중간 답변: 네이버는 이해진에 의해 창립되었습니다.
추가 질문: 이해진은 언제 태어났나요?
중간 답변: 이해진은 1967년 6월 22일에 태어났습니다.
최종 답변은: 1967년 6월 22일


Question:
Google이 창립된 연도에 Bill Gates의 나이는 몇 살인가요?
Answer:
"""
```

`prompt`에는  예시들을 예시 프롬프트 템플릿에 각 채워 나열한 후에 `<br>`마지막에 `suffix` 내용을 채워 주게 됩니다.

`suffix`에서 `question`은 `input_variables` 내용과 연결이 되는 부분이며,`<br>`사용자 입력(질문)을 넣는 부분입니다.

이렇게 몇가지 예시를 주면, LLM 모델이 답변을 생성할 때 해당 예시를 참고하게 됩니다.

OK.

&nbsp;

> ### 💡깨알 Tip
>
> ⭐️ **Example Selector을 이용하여 예시 선택하기** ⭐️
>
> ExampleSelector 중에서 langchain_core.example_selectors의 **MaxMarginalRelevanceExampleSelector**와 **SemanticSimilarityExampleSelector**을 알아봅시다.
>
> ```python
> from langchain_core.example_selectors import (
>     MaxMarginalRelevanceExampleSelector,
>     SemanticSimilarityExampleSelector,
> )
> from langchain_openai import OpenAIEmbeddings
> from langchain_chroma import Chroma
>
> examples = [
>     {
>         "question": "스티브 잡스와 아인슈타인 중 누가 더 오래 살았나요?",
>         "answer": """이 질문에 추가 질문이 필요한가요: 예.
> 추가 질문: 스티브 잡스는 몇 살에 사망했나요?
> 중간 답변: 스티브 잡스는 56세에 사망했습니다.
> 추가 질문: 아인슈타인은 몇 살에 사망했나요?
> 중간 답변: 아인슈타인은 76세에 사망했습니다.
> 최종 답변은: 아인슈타인
> """,
>     },
>     {
>         "question": "네이버의 창립자는 언제 태어났나요?",
>         "answer": """이 질문에 추가 질문이 필요한가요: 예.
> 추가 질문: 네이버의 창립자는 누구인가요?
> 중간 답변: 네이버는 이해진에 의해 창립되었습니다.
> 추가 질문: 이해진은 언제 태어났나요?
> 중간 답변: 이해진은 1967년 6월 22일에 태어났습니다.
> 최종 답변은: 1967년 6월 22일
> """,
>     },
>     {
>         "question": "율곡 이이의 어머니가 태어난 해의 통치하던 왕은 누구인가요?",
>         "answer": """이 질문에 추가 질문이 필요한가요: 예.
> 추가 질문: 율곡 이이의 어머니는 누구인가요?
> 중간 답변: 율곡 이이의 어머니는 신사임당입니다.
> 추가 질문: 신사임당은 언제 태어났나요?
> 중간 답변: 신사임당은 1504년에 태어났습니다.
> 추가 질문: 1504년에 조선을 통치한 왕은 누구인가요?
> 중간 답변: 1504년에 조선을 통치한 왕은 연산군입니다.
> 최종 답변은: 연산군
> """,
>     },
>     {
>         "question": "올드보이와 기생충의 감독이 같은 나라 출신인가요?",
>         "answer": """이 질문에 추가 질문이 필요한가요: 예.
> 추가 질문: 올드보이의 감독은 누구인가요?
> 중간 답변: 올드보이의 감독은 박찬욱입니다.
> 추가 질문: 박찬욱은 어느 나라 출신인가요?
> 중간 답변: 박찬욱은 대한민국 출신입니다.
> 추가 질문: 기생충의 감독은 누구인가요?
> 중간 답변: 기생충의 감독은 봉준호입니다.
> 추가 질문: 봉준호는 어느 나라 출신인가요?
> 중간 답변: 봉준호는 대한민국 출신입니다.
> 최종 답변은: 예
> """,
>     },
> ]
>
> example_selector = SemanticSimilarityExampleSelector.from_examples(
>     # 선택 가능한 예시 목록
>     examples,
>     # 의미적 유사성을 측정하는 데 사용되는 임베딩을 생성하는 임베딩 클래스
>     OpenAIEmbeddings(),
>     # 임베딩을 저장하고 유사성 검색을 수행하는 데 사용되는 VectorStore 클래스
>     Chroma,
>     # 생성할 예시의 수
>     k=1,
> )
>
> question = "Google이 창립된 연도에 Bill Gates의 나이는 몇 살인가요?"
>
> # 입력과 가장 유사한 예시를 선택합니다.
> selected_examples = example_selector.select_examples({"question": question})
>
> print(f"입력에 가장 유사한 예시:\n{question}\n")
> for example in selected_examples:
>     print(f'question:\n{example["question"]}')
>     print(f'answer:\n{example["answer"]}')
>
> """출력:
> 입력에 가장 유사한 예시:
> Google이 창립된 연도에 Bill Gates의 나이는 몇 살인가요?
>
> question:
> 네이버의 창립자는 언제 태어났나요?
> answer:
> 이 질문에 추가 질문이 필요한가요: 예.
> 추가 질문: 네이버의 창립자는 누구인가요?
> 중간 답변: 네이버는 이해진에 의해 창립되었습니다.
> 추가 질문: 이해진은 언제 태어났나요?
> 중간 답변: 이해진은 1967년 6월 22일에 태어났습니다.
> 최종 답변은: 1967년 6월 22일
> """
> ```
>
> **`SemanticSimilarityExampleSelector`의 `from_examples()` 메서드로 Chroma DB에 `OpenAIEmbeddings()` 임베딩 모델로 예시를 선택하는 객체를 생성할 수 있습니다.**
>
> 이후에 그 생성한 **객체의 `select_examples()` 메서드에 딕셔너리 형태의 입력을 넣어주면** `<br>`**예시들 중에서 입력과 유사한 예시 k개를 선택해 줍니다.**
>
> 여기서는 **나이를 물어보는 예시를 정확하게 선택**해주고 있습니다.

&nbsp;

그렇다면,

**example_selector을 실전에서 어떻게 사용하냐?**

1. 예시 프롬프트 `example_prompt` 템플릿 생성하기
2. 임베딩 모델, 벡터 DB 선정해서 `example_selector` 객체 생성하기
3. `FewShotPromptTemplate`에서 `example_selector`, `example_prompt`, `suffix`로 프롬프트 완성하기
4. chain 생성 후 호출

```python
# chain에서 사용해보기

from langchain_core.example_selectors import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_teddynote.messages import stream_response

# llm, examples 설정 필요

example_prompt = PromptTemplate.from_template(
    "Question:\n{question}\nAnswer:\n{answer}"
)

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Chroma,
    k=1,
)

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Question:\n{question}\nAnswer:",
    input_variables=["question"],
)

# 체인 생성
chain = prompt | llm

# 결과 출력
answer = chain.stream(
    {"question": "Google이 창립된 연도에 Bill Gates의 나이는 몇 살인가요?"}
)
stream_response(answer)
"""출력:
이 질문에 추가 질문이 필요한가요: 예.  
추가 질문: Google이 창립된 연도는 언제인가요?  
중간 답변: Google은 1998년에 창립되었습니다.  
추가 질문: Bill Gates는 언제 태어났나요?  
중간 답변: Bill Gates는 1955년 10월 28일에 태어났습니다.  
추가 질문: 1998년에 Bill Gates의 나이는 몇 살인가요?  
중간 답변: 1998년 - 1955년 = 43년. 10월 28일 이전에는 42세, 이후에는 43세입니다.

최종 답변: 1998년에 Bill Gates의 나이는 42세 또는 43세입니다. (생일인 10월 28일 이전에는 42세, 이후에는 43세입니다.)
"""
```

굳.

&nbsp;

아, 참고로 `MaxMarginalRelevanceExampleSelector` 사용법은 `SemanticSimilarityExampleSelector`를 `MaxMarginalRelevanceExampleSelector`로 바꾸기만 하면 됩니다.

&nbsp;

⭐️ **둘의 차이점 비교** ⭐️

| 구분                | SemanticSimilarityExampleSelector      | MaxMarginalRelevanceExampleSelector (MMR)                           |
| ------------------- | -------------------------------------- | ------------------------------------------------------------------- |
| 핵심 목표           | query와**가장 유사한 예시 선택** | query와 유사하면서**예시 간 중복 최소화**                     |
| 기본 개념           | **관련성(relevance)만 고려**     | **관련성 + 다양성(diversity)**                                |
| 선택 기준           | `sim(query, example)`                | `λ·sim(query, example) − (1−λ)·max(sim(example, selected))` |
| 예시 간 중복        | 높음 (비슷한 예시가 몰릴 수 있음)      | 낮음 (서로 다른 케이스 위주)                                        |
| 다양성 고려         | ❌                                     | ✅                                                                  |
| 파라미터            | `k`                                  | `k`, `lambda_mult (λ)`                                         |
| lambda 영향         | 해당 없음                              | λ↑ → 유사도 중심 / λ↓ → 다양성 중심                           |
| 계산 비용           | 낮음                                   | 중간 (반복 선택)                                                    |
| 프롬프트 정보량     | 제한적                                 | 풍부 (케이스 커버리지 ↑)                                           |
| Few-shot 안정성     | 보통                                   | 높음                                                                |
| RAG 적합성          | △                                     | ✅                                                                  |
| Tool Calling 적합성 | ❌                                     | ✅                                                                  |
| Agent / Reasoning   | ❌                                     | ✅                                                                  |
| 추천 사용 상황      | 단순 Q&A, 포맷 학습                    | API 선택, 복합 추론, edge case 포함                                 |

GPT 선생님이 알려주셨습니다.

`MaxMarginalRelevanceExampleSelector`에서 `<br>λ·sim(query, example) − (1−λ)·max(sim(example, selected))` 식을 보았을 때,

##### `sim(query, example)`, `max(sim(example, selected)`를 이용하고 있고, 그들의 계수가 서로 반비례함을 볼 수 있습니다.

즉, **query와 유사성과 선택된 예시와의 유사성을 동시에 고려한다**는 것입니다.`<br>`**"파라미터 λ가 크다" = "질문과 유사한 것들을 고르지만, 반대로 선택되는 것들 사이에는 유사하지 않도록 한다"**라고 할 수 있습니다.

(‼️ langchain의 MaxMarginalRelevanceExampleSelector는 `lambda_mult` 설정이 되지 않습니다. 직접 커스텀해서 만들어야 한다고 합니다.)

&nbsp;

&nbsp;

**하지만, Example Selector 문제점이 있다고 합니다.**

상황 정의부터 합시다.

1. 예시에 `instruction`, `input`, `answer`와 같이 여러 변수가 있음
2. LLM에게 예시처럼 `instruction`, `input`의 값을 던져주고, `answer` 값을 받아와야 하는데
3. 그러지 않고 `instruction`만 던져주었을 때 올바른 답변을 해주지 못함

`instruction`, `input`을 함께 고려하여 유사도 계산을 해야하는데 `<br>`그러지 못하여 오류가 난다고 합니다.

&nbsp;

그런데,

**갓 테디노트님께서 만드신 CustomExampleSelector() 모듈에서 `search_key` 설정을 통해 직접 유사도를 계산하고자 하는 변수를 설정할 수 있도록 해주었습니다.**

```python
from langchain_teddynote.prompts import CustomExampleSelector

# 커스텀 예제 선택기 생성
# examples에서 유사도 계산하는 변수를 instruction로 지정
# search_key의 기본 값은 instruction.
custom_selector = CustomExampleSelector(examples=examples, embedding_model=OpenAIEmbeddings(), search_key="instruction")

# 커스텀 예제 선택기를 사용했을 때 결과
custom_selector.select_examples({"instruction": "다음 문장을 회의록 작성해 주세요"})
"""출력:
[{'instruction': '당신은 회의록 작성 전문가 입니다. 주어진 정보를 바탕으로 회의록을 작성해 주세요',
  'input': '2023년 12월 25일, XYZ 회사의 마케팅 전략 회의가 오후 3시에 시작되었다. 회의에는 마케팅 팀장인 김수진, 디지털 마케팅 담당자인 박지민, 소셜 미디어 관리자인 이준호가 참석했다. 회의의 주요 목적은 2024년 상반기 마케팅 전략을 수립하고, 새로운 소셜 미디어 캠페인에 대한 아이디어를 논의하는 것이었다. 팀장인 김수진은 최근 시장 동향에 대한 간략한 개요를 제공했으며, 이어서 각 팀원이 자신의 분야에서의 전략적 아이디어를 발표했다.',
  'answer': '\n회의록: XYZ 회사 마케팅 전략 회의\n일시: 2023년 12월 25일\n장소: XYZ 회사 회의실\n참석자: 김수진 (마케팅 팀장), 박지민 (디지털 마케팅 담당자), 이준호 (소셜 미디어 관리자)\n\n1. 개회\n   - 회의는 김수진 팀장의 개회사로 시작됨.\n   - 회의의 목적은 2024년 상반기 마케팅 전략 수립 및 새로운 소셜 미디어 캠페인 아이디어 논의.\n\n2. 시장 동향 개요 (김수진)\n   - 김수진 팀장은 최근 시장 동향에 대한 분석을 제시.\n   - 소비자 행동 변화와 경쟁사 전략에 대한 통찰 공유.\n\n3. 디지털 마케팅 전략 (박지민)\n   - 박지민은 디지털 마케팅 전략에 대해 발표.\n   - 온라인 광고와 SEO 최적화 방안에 중점을 둠.\n\n4. 소셜 미디어 캠페인 (이준호)\n   - 이준호는 새로운 소셜 미디어 캠페인에 대한 아이디어를 제안.\n   - 인플루언서 마케팅과 콘텐츠 전략에 대한 계획을 설명함.\n\n5. 종합 논의\n   - 팀원들 간의 아이디어 공유 및 토론.\n   - 각 전략에 대한 예산 및 자원 배분에 대해 논의.\n\n6. 마무리\n   - 다음 회의 날짜 및 시간 확정.\n   - 회의록 정리 및 배포는 박지민 담당.\n'}]
"""
```

원래는 다른 예시(교정 전문가 관련)가 선택됐었는데 `<br>`지금은 정확하게 선택됨을 알 수 있습니다.

&nbsp;

&nbsp;

## LangChain Hub

---

LangChain Hub에서 프롬프트를 당겨올 수도 있습니다.

예시 프롬프트: https://smith.langchain.com/hub/rlm/rag-prompt

```python
from langchain import hub

# 가장 최신 버전의 프롬프트를 가져옵니다.
prompt = hub.pull("rlm/rag-prompt")
prompt
# 출력: input_variables=['context', 'question'] input_types={} partial_variables={} metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'} messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"), additional_kwargs={})]
```

이때, 프롬프트는 수정될 수 있으니 프롬프트 버전 해시를 지정해두는 것이 좋습니다.`<br>`버전 해시는 해당 프롬프트에서 commit 부분 들어가면 나와 있습니다.

```python
# 특정 버전의 프롬프트를 가져오기 위해 버전 해시 지정
prompt = hub.pull("rlm/rag-prompt:50442af1")
prompt
```

프롬프트를 허브에 업로드를 할 수도 있습니다.

```python
from langchain import hub

# 프롬프트를 허브에 업로드합니다.
hub.push("teddynote/simple-summary-korean", prompt)
```

자신의 "ID/레포지토리"를 입력하면 됩니다.

&nbsp;
&nbsp;

여기서 끄읕.

&nbsp;

```toc

```
