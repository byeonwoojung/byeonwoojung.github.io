---
emoji: ☀️
title: "[LLM] 테디노트의 RAG 비법노트 끄적끄적-3"
date: '2025-12-13 00:00:00'
author: 변우중
tags: NLP 자연어 자연어처리 토큰화 LLM
categories: NLP LLM
---
참고 : 테디노트의 RAG 비법노트

소스코드: https://github.com/teddylee777/langchain-kr

&nbsp;

오늘도 작성하는데 Pydantic 파싱 따로 공부한 것 좀 끄적이고 가겠습니다.

레츠기릿!

&nbsp;

## PydanticOutputParser

---

BaseModel/Field, partial_variables, 결과 포맷 지시사항, parser 등을 알아야 합니다.

예시로 바로 봅시다.

```python
# LangChain + PydanticOutputParser “전체 예시 코드”
# - BaseModel / Field로 출력 스키마 정의
# - parser.get_format_instructions()를 partial_variables로 프롬프트에 주입
# - prompt | llm | parser 로 "구조화된 결과"를 강제 파싱
# - RunnablePassthrough로 단일 입력(str)도 체인 내부에서 dict로 매핑

from typing import Optional
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough


# 1) 출력 스키마 정의 (LLM 출력이 반드시 이 형태로 파싱되어야 함)
class CountryInfo(BaseModel):
    country: str = Field(description="국가 이름")
    capital: str = Field(description="수도")
    population_million: Optional[int] = Field(
        default=None,
        description="인구(백만 단위). 모르면 null"
    )


# 2) Pydantic 파서 생성
parser = PydanticOutputParser(pydantic_object=CountryInfo)

# 3) 프롬프트 템플릿 (format_instructions는 '사용자 정의 변수명'임)
prompt = PromptTemplate(
    template=(
        "너는 정확히 지정된 형식으로만 답하는 도우미야.\n"
        "{format_instructions}\n"
        "국가: {country}\n"
        "주의: 추가 설명 금지. JSON만 출력."
    ),
    input_variables=["country"],
    partial_variables={
        # parser가 요구하는 출력 형식 지침을 프롬프트에 '고정 문자열'로 주입
        "format_instructions": parser.get_format_instructions()
    },
)

# 4) LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 5) 체인 구성 (Prompt → LLM → Pydantic Parser)
chain = prompt | llm | parser

# (A) dict로 실행
result_a = chain.invoke({"country": "대한민국"})
print("A) dict 입력 결과:", result_a)
print("A) 타입:", type(result_a))

# (B) 단일 값(str) 입력을 받고 싶으면 RunnablePassthrough로 내부에서 dict로 매핑
chain_str_input = (
    {"country": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

result_b = chain_str_input.invoke("일본")
print("\nB) str 입력 결과:", result_b)
print("B) 타입:", type(result_b))

```

### 1) `BaseModel` / `Field`는 무엇인가?

- **`BaseModel`**: “출력 데이터는 반드시 이 구조여야 한다”를 정의하는 **스키마(계약서)**
  - 필드 누락, 타입 불일치, 구조 깨진다? → **즉시 에러** 발생하도록 합니다.
- **`Field`**: 각 필드의 의미/제약(설명, 기본값 등)을 붙이는 **메타데이터**
  - LLM에게 “이 필드는 이런 의미”라고 알려줘서 출력 품질을 올려줄 수 있습니다.

### 2) `partial_variables`를 넣으면 항상 형식이 맞나?

- **아닙니다. 프롬프트에 지침 문자열을 미리 채워 넣어주는 기능이라, LLM이 지침을 따를 확률만 높여줄 뿐입니다.**
- ⭐️ **진짜 강제는 parser가 합니다.** ⭐️
  - LLM 출력이 스키마와 다르면 `PydanticOutputParser`가 파싱 실패로 **에러 발생**

### 3) `format_instructions`는 원래 있는 변수인가?

- **아닙니다. 예약어가 아닙니다.** `{format_instructions}`는 내가 프롬프트 템플릿에 만든 변수명일 뿐 착각 NO.
- `partial_variables={"format_instructions": ...}`로 그 변수 자리에 들어갈 값을 채운 것이라, 변수명은 `schema_guide`, `output_rules` 등으로 바꿔도 됩니다. (프롬프트의 `{...}`와만 일치하면 OK)

실제로 프롬프트는 아래와 유사하게 만들어집니다.

```
너는 정확히 지정된 형식으로만 답하는 도우미야.
The output should be formatted as a JSON instance that conforms to the JSON schema below.

{
  "title": "CountryInfo",
  "type": "object",
  "properties": {
    "country": {
      "type": "string",
      "description": "국가 이름"
    },
    "capital": {
      "type": "string",
      "description": "수도"
    },
    "population_million": {
      "type": "integer",
      "description": "인구(백만 단위). 모르면 null"
    }
  },
  "required": ["country", "capital"]
}

국가: 대한민국
주의: 추가 설명 금지. JSON만 출력.
```

**이렇게 출력 형식을 정해준 후에, chain_str_input에서 마지막 parser가 형식이 맞는지 확인합니다.**
**형식이 맞지 않으면 이때 에러가 발생하는 것입니다.**

이해 완.

&nbsp;

&nbsp;











```toc

```
