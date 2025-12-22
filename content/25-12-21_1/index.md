---
emoji: ☀️
title: "[LLM] 테디노트의 RAG 비법노트 끄적끄적-5"
date: '2025-12-21 00:00:00'
author: 변우중
tags: NLP 자연어 자연어처리 LLM 프롬프트 RAG
categories: NLP LLM
---
참고 : 테디노트의 RAG 비법노트 (https://fastcampus.co.kr/data_online_teddy)

소스코드: https://github.com/teddylee777/langchain-kr

&nbsp;

오늘은 출력 파서에 대해 알아보고자 합니다.

며칠 글이 올라오지 않았던 것은<br>streamlit 활용한 UI에서 사용자가 프롬프트 템플릿을 선택하고 업로드한 PDF를 기반해 질문하면, PDF를 파싱해서 RAG를 구현하는 미니 프로젝트를 했었습니다.

RAG 개발 관련한 게시글은 좀 더 깊이 있게 공부한 후에 올리고자 합니다.

고럼 레츠고~!

&nbsp;

&nbsp;

LLM은 수렴보다 발산을 더 잘하기 떄문에 응답의 포맷을 정해주는 것이 중요합니다.<br>**그래서 LLM의 답변을 원하는 구조대로 강제하기 위해 출력 파서를 이용하는 것이 좋습니다.**

저는 Pydantic 파싱 방법을 많이 이용했던 것 같습니다.

&nbsp;

## PydanticOutputParser: 가장 유용한 내맘대로 파서

---

⭐️ OutputParser에서 가장 중요한 메서드 ⭐️

1. `get_format_instructions()`: 출력 정보의 형식을 정의하는 지침 제공

2. `parse()`: 모델의 출력을 특정 스키마에 맞는지 검증, 스키마 구조로 변환

이 메서드들은 파서 객체를 생성하고, `get_format_instructions()` 메서드로 어떻게 출력결과를 파싱하는지 출력하여 확인 가능합니다.

````python
# BaseModel: 부모로부터 상속받음
# description: 필드에 대한 설명(자세히 작성 필요)
class EmailSummary(BaseModel):
    person: str = Field(description="메일을 보낸 사람")
    email: str = Field(description="메일을 보낸 사람의 이메일 주소")
    subject: str = Field(description="메일 제목")
    summary: str = Field(description="메일 본문을 요약한 텍스트")
    date: str = Field(description="메일 본문에 언급된 미팅 날짜와 시간")


# PydanticOutputParser 생성
parser = PydanticOutputParser(pydantic_object=EmailSummary)

parser.get_format_instructions()
"""출력:
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"properties": {"person": {"description": "메일을 보낸 사람", "title": "Person", "type": "string"}, "email": {"description": "메일을 보낸 사람의 이메일 주소", "title": "Email", "type": "string"}, "subject": {"description": "메일 제목", "title": "Subject", "type": "string"}, "summary": {"description": "메일 본문을 요약한 텍스트", "title": "Summary", "type": "string"}, "date": {"description": "메일 본문에 언급된 미팅 날짜와 시간", "title": "Date", "type": "string"}}, "required": ["person", "email", "subject", "summary", "date"]}
```
"""
````

이렇게 설정이 됩니다.

이후, **`parser.get_format_instructions()`를 프롬프트에 출력 포맷으로 함께 주면 됩니다**.

그리고, LLM 응답(문자열 형식)을 **parser의 `parse()` 메서드를 이용하여 파싱을 하면 앞서 정의했던 클래스대로 파싱을 하게 됩니다**. (여기서는 앞서 정의한 Pydantic 객체 형식이 됨)

```python
parser.parse(output)
"""출력:
EmailSummary(person='김철수', email='chulsoo.kim@bikecorporation.me', subject='"ZENESIS" 자전거 유통 협력 및 미팅 일정 제안', summary='바이크코퍼레이션 김철수 상무가 ZENESIS 자전거의 상세 브로슈어(기술 사양, 배터리 성능, 디자인) 요청과 유통 및 마케팅 협력 논의를 위해 미팅을 제안함.', date='1월 15일 화요일 오전 10시')
"""
```

(참고로, `print()`로 출력하면 안의 내용만 출력하게 됩니다.)

&nbsp;

또한, **chain에서 LLM 뒤에 parser를 연결해주면<br>`invoke()` 메서드의 출력 결과를 자동으로 파싱하여 Pydantic 객체의 응답을 받을 수 있습니다.**

```python
chain = prompt | llm | parser
chain.invoke(
    {
        "email_conversation": email_conversation,
        "question": "이메일 내용중 주요 내용을 추출해 주세요.",
    }
)
"""출력:
EmailSummary(person='김철수', email='chulsoo.kim@bikecorporation.me', subject='"ZENESIS" 자전거 유통 협력 및 미팅 일정 제안', summary='바이크코퍼레이션 김철수 상무가 ZENESIS 자전거에 대한 상세 브로슈어 요청(기술 사양, 배터리 성능, 디자인)과 유통 및 마케팅 협력 논의를 위해 미팅을 제안함.', date='1월 15일 화요일 오전 10시')
"""
```

&nbsp;

그런데,

> ### 💡깨알 Tip
>
> 제가 경험한 바로는 **LLM이 애초에 응답해주는 결과가 불안정하면 파서가 제대로 작동되지 않는 경우**가 있습니다.
>
> **즉, 프롬프트에 출력 포맷을 정해주었다고 해서 항상 출력 포맷을 맞추어 답변해주지 않는 경우가 있습니다.**
>
> 앞의 예시에서 이러한 출력 결과를 받을 때가 있었습니다.
>
> ```
> AIMessage(content='
> json\n{\n  "person": "김철수",\n  "email": "chulsoo.kim@bikecorporation.me",\n  "subject": "\\"ZENESIS\\" 자전거 유통 협력 및 미팅 일정 제안",\n  "summary": "바이크코퍼레이션 김철수 상무가 ZENESIS 자전거에 대한 상세 브로슈어(기술 사양, 배터리 성능, 디자인)를 요청하고, 유통 전략과 마케팅 계획 수립을 위해 협력 가능성을 논의하고자 1월 15일 오전 10시에 미팅을 제안함.",\n  "date": "1월 15일 오전 10시"\n}\n
> ', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 151, 'prompt_tokens': 601, 'total_tokens': 752, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4.1-mini-2025-04-14', 'system_fingerprint': 'fp_376a7ccef1', 'id': 'chatcmpl-CpTzmI5b3Zm40LzHPuNQnkCzOWMqz', 'finish_reason': 'stop', 'logprobs': None}, id='run-3520fb7f-43a8-44ce-8c80-5c30f61d8ea4-0', usage_metadata={'input_tokens': 601, 'output_tokens': 151, 'total_tokens': 752, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})
> ```
>
> 여기서 **출력 형식이 `AIMessage` 객체라 함은 파서가 실제로 실행되지 않아 LLM 결과 그대로인 상태**인데<br>content 값이 ````json`로 시작한 것으로 볼 때 이 경우는 **LLM이 응답으로 코드 블록을 내준 것**으로 보입니다.
>
> => 이렇게 **파싱이 실패하는 경우, 저는 LLM에게 응답받는 것을 재시도하는 로직**을 함께 넣었었습니다.

&nbsp;

**Chain에 parser을 연결하는 것은 LLM의 응답결과를 받은 '후처리' 방식이기 때문에 안정적인 파싱이 불가**할 수 있습니다.

&nbsp;

하지만,

⭐️ **LLM의 `.with_structured_output()` 메서드를 이용하면 애초에 LLM에게 구조화된 응답받을 수 있도록 할 수 있습니다.** ⭐️

```python
llm_with_structered = ChatOpenAI(
    temperature=0, model_name="gpt-4.1-mini"
).with_structured_output(EmailSummary)

answer = llm_with_structered.invoke(email_conversation)
answer
"""출력:
EmailSummary(person='김철수', email='chulsoo.kim@bikecorporation.me', subject='"ZENESIS" 자전거 유통 협력 및 미팅 일정 제안', summary='바이크코퍼레이션의 김철수 상무가 이은채 대리에게 ZENESIS 자전거에 대한 상세 브로슈어 요청과 함께, 기술 사양, 배터리 성능, 디자인 정보가 필요하다고 전달했습니다. 또한, 1월 15일 화요일 오전 10시에 미팅을 제안하며 협력 가능성을 논의하고자 합니다.', date='2024-01-08')
"""
```

이처럼

LLM을 `with_structured_output()` 메서드를 붙여 생성하고 invoke() 메서드를 호출하면

**LLM 응답 형식 `AIMessage`가 아닌,<br>여기서 Pydantic 객체인 EmailSummary 객체로 응답이 나오는 것을 볼 수 있습니다.**

&nbsp;

⚠️ **알아두어야 할 점**

1. LLM마다 `with_structured_output()` 메서드를 지원하지 않을 수 있으니 찾아보고 사용해야 합니다.
2. 스트리밍 출력은 불가능합니다. `invoke()` 호출만 가능합니다.

&nbsp;

&nbsp;

출력 파서는 Pydantic 파서 외에 다양한 형식의 파서가 존재합니다.

이번 글에서는 그 중에 가장 유용한 Pydantic 파서만 다뤄보았는데 다른 것들은 필요시 다루고자 합니다.

이번 글은 여기서 마무리하겠슴다 🫡

```toc

```
