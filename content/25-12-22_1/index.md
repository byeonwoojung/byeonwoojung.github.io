---
emoji: ☀️
title: "[LLM] 테디노트의 RAG 비법노트 끄적끄적-6"
date: '2025-12-22 00:00:00'
author: 변우중
tags: NLP 자연어 자연어처리 LLM 프롬프트 RAG
categories: NLP LLM
---
참고 : 테디노트의 RAG 비법노트 (https://fastcampus.co.kr/data_online_teddy)

소스코드: https://github.com/teddylee777/langchain-kr

&nbsp;

지난 번에 OutputParser는 필요 시 다루고자 했는데<br>제 성격상 바로 또 남겨놔야해서.. 오늘은 나머지 OutputParser 끄적이고 가겠습니다..

레츠고오!!

&nbsp;



## 이외의 OutputParser

---

요약해서 적어두고자 합니다.

1. `CommaSeparatedListOutputParser`를 chain에 달아 스트리밍 출력할 때는 각 응답 1개씩을 리스트로 변환하는 경우가 있다. (모든 응답을 묶어서 리스트로 반환하지 않음)

&nbsp;

2. `StructuredOutputParser`는 `ResponseSchema` 클래스를 이용해 정의하고 `from_response_schemas()` 메서드를 이용해 파서 초기화를 해준다. 로컬과 덜 강력한 모델에 유용한 파서라고 한다.

&nbsp;

3. `JsonOutputParser`는 Pydantic 파서에서 했던 방식과 같이 데이터 구조를 클래스를 이용해 정의해주고 `pydantic_object` 파라미터 값에 해당 클래스명 값을 전달해주어, 파서로 정의하여 사용한다.<br>이때, chain을 구성하여 `invoke()` 메서드를 호출하면 답변을 **딕셔너리 형태로 파싱되어 값을 받을 수 있다.**

&nbsp;

4. **`PandasDataFrameOutputParser`는 LLM이 Pandas 데이터프레임에서 질문에 맞는 조회하는 쿼리문(예: `column:Age`, `mean:Age[0..4]`)을 응답으로 내놓은 것을 바탕으로 그 문자열을 파싱하여 실제로 데이터프레임에서 조회하여 딕셔너리 형태로 결과를 출력한다.**

   * 즉, **`parser=PandasDataFrameOutputParser(dataframe=df)`로 정의했을 때**
   * **`parser.get_instructions()`로 데이터프레임의 컬럼 구조 등 정보를 LLM에게 주고 LLM이 정보를 조회하는 명령 문자열을 답변으로 내놓으면**<br>(LLM에게는 실제 데이터가 아닌 컬럼 구조에 대한 정보만 주는 것임)
   * **parser가 그 명령 문자열을 파싱하여 dataframe에 전달한 '실제 df의 데이터에서 읽고' 답변을 내놓는다**.

   * ⭐️ **장점: 모든 데이터를 LLM에게 전달하지 않고, 데이터 구조를 조회하는 방법만 받아와 파서로 데이터를 조회할 수 있다.** ⭐️ 

&nbsp;

5. `DateTimeOutputParser`는 `format()` 메서드를 이용해 파싱하고자 하는 datetime 형식을 정해주어 그 형식의 답변을 받는다. 최종 응답을 `strftime()` 메서드를 이용해 datetime 형식을 문자열 형식으로 바꾸어 이용할 수 있다.
   * `datetime.datetime(1998, 9, 4, 0, 0)` -> `.strftime("%Y-%m-%d")`을 이용해 `'1998-09-04'` 이렇게 바꾸어 활용 가능합니다.

&nbsp;

6. `EnumOutputParser`는 enum 라이브러리의 Enum 모듈을 상속모델로 하여 클래스 정의를 한 후, 해당 파서의 enum 파라미터에 해당 클래스를 전달함으로써 파서 객체를 생성할 수 있다. **chain에서 최종 응답 형식은 Enum 멤버 객체이다.**

   ```python
   from enum import Enum
   from langchain_core.prompts import PromptTemplate
   from langchain_openai import ChatOpenAI
   
   class Colors(Enum):
       RED = "빨간색"
       GREEN = "초록색"
       BLUE = "파란색"
   
   parser = EnumOutputParser(enum=Colors)
   parser.get_format_instructions()
   """⭐️ 출력 ⭐️:
   'Select one of the following options: 빨간색, 초록색, 파란색'
   """
   ```

   * **`get_format_instructions()` 메서드로 출력해보면 `Colors` 클래스 안에 정의된 값들(`RED`, `GREEN`, `BLUE`)의 Value(`빨간색`, `초록색`, `파란색`)를 읽어온다. (즉, Enum 멤버가 아닌 Enum 값을 가져옴)**

   ```python
   prompt = PromptTemplate.from_template(
       """다음의 물체는 어떤 색깔인가요?
   
   Object: {object}
   
   Instructions: {instructions}"""
   ).partial(instructions=parser.get_format_instructions())
   
   chain = prompt | ChatOpenAI() | parser
   response = chain.invoke({"object": "하늘"})
   print(response)
   """⭐️ 출력 ⭐️:
   Colors.BLUE
   """
   ```

   * ⭐️ 하지만, **chain에서 `parser`를 통해 응답받은 결과는 Enum 멤버 객체이다.**<br>**즉, LLM의 텍스트 응답을 Python Enum 타입으로 변환해주어 Color.BLUE 멤버 객체로 바꾸어 출력을 해준다.** ⭐️

   * ⭐️ **이렇게 멤버 객체로 바꾸어주기 때문에<br>`response.value`로 접근해서 우리가 원하는 값으로 매핑하여 볼 수 있습니다.** ⭐️

&nbsp;

추가로,

이전 글(https://byeonwoojung.github.io/25-12-13_1/)에 적었던 내용인데,

* **`OutputFixingParser`를 이용해서 다시 고쳐달라는 요청을 보낼 수 있고, `RetryOutputParser`을 이용해서 처음 보냈던 프롬프트와 잘못된 답변을 모두 함께 주어 재시도를 수행할 수도 있습니다.**

  ```python
  from langchain.output_parsers import OutputFixingParser
  
  # 기존 parser를 감싸서 정의
  fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())
  
  # 이제 체인에서 에러가 나면 스스로 수정 요청을 보냅니다.
  chain = prompt | model | fixing_parser
  ```

* **또는, `with_fallbacks`을 이용해서 플랜 B 체인을 준비해둘 수도 있습니다.**

  ```python
  chain = (prompt | model | parser).with_fallbacks([backup_chain])
  ```

이들을 chain에 함께 두면 좋을 듯 합니다.

&nbsp;

&nbsp;

여기까지 끊고 갑니다~

```toc

```
