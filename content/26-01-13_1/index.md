---
emoji: ☀️
title: "[LLM] 테디노트의 RAG 비법노트 끄적끄적-13"
date: '2026-01-13 00:00:00'
author: 변우중
tags: LLM 프롬프트 RAG LangChain 랭체인 Retriver 리트리버 Reranker 리랭커 Vectorstore 벡터스토어
categories: LLM RAG
---

참고 : 테디노트의 RAG 비법노트 (https://fastcampus.co.kr/data_online_teddy)<br>소스코드: https://github.com/teddylee777/langchain-kr<br>위키독스: https://wikidocs.net/book/14314

&nbsp;

오늘은 정말 뜬금없이 갖가지 내용들을 끄적이고자 합니다..<br>**@chain 데코레이터**, **Configurable**, **Route** 등에 대해 정리하고자 합니다.

레츠기릿



## @chain 데코레이터

---

`from langchain_core.runnables import chain`으로 가져와서<br>`@chain` 데코레이터를 이용해 **함수를 `chain`으로 변환**하도록 합시다.

즉, `chain`은 `Runnable` 객체이기 때문에 LCEL 인터페이스에 따라 `invoke()` 메서드 등을 활용할 수 있습니다.



chain.get_graph().print_ascii()로 체인의 그래프 출력 가능<br>(`!pip install -qU grandalf` 그래프 그리는 라이브러리 설치해야 함)

`chain.get_prompts()`로 체인의 프롬프트 확인 가능





## Helper 함수와 Wrapper 함수

---

```python
"""
Helper 함수: Wrapper 함수를 통해서만 사용된다는 의미로 _를 붙임
- 실제 로직이 실행되는 함수
"""
def _multiple_length_function(text1, text2):  # 두 텍스트의 길이를 곱하는 함수
    return len(text1) * len(text2)

"""
Wrapper 함수: 2개 인자를 받는 함수로 연결하는 함수
- Helper 함수에서 실제 로직이 실행되기 위해, 형식을 맞춰주는 함수
- RunnableLambda에서 인자는 1개로 받아야 하므로, 딕셔너리 활용
- 이 함수에서만 사용한다는 의미와 dict 기본 명령어 겹치지 않기 위해 _dict라는 이름을 사용
"""
def multiple_length_function(
    _dict,
):  # 딕셔너리에서 "text1"과 "text2"의 길이를 곱하는 함수
    return _multiple_length_function(_dict["text1"], _dict["text2"])
```



## Configurable

---

`chain`에 `config`로 전달하는 딕셔너리(`RunnableConfig`)에서 주로 쓰는 약속된 키가 몇 가지 존재합니다. `chain.invoke(입력값, config)`로 config 설정 가능합니다. 사용하는 방법은 `Configurable`에 챕터에서 자세히 보고자 합니다.

1. `callbacks` (가장 중요 ⭐)

   - **역할**: 실행 과정을 지켜보는 감시자(핸들러)를 등록합니다.


   - **용도**: Log 출력, LangSmith 추적, 스트리밍 처리 등.


   - **예시**: 

     ```
     {"callbacks": [ConsoleCallbackHandler()]}
     ```

2. `tags`

   - **역할**: 이 실행에 태그(꼬리표)를 붙입니다.


   - **용도**: 나중에 로그나 LangSmith에서 "이 태그 달린 것만 보여줘" 하고 필터링할 때 씁니다.


   - **예시**: 

     ```
     {"tags": ["my-tag", "experiment-1"]}
     ```


3. `metadata`

   - **역할**: 추가적인 정보를 딕셔너리로 저장합니다.


   - **용도**: 사용자 ID, 세션 ID 처럼 실행 로직에는 영향이 없지만 기록해두고 싶은 정보들.


   - **예시**: 

     ```
     {"metadata": {"user_id": "123", "session_id": "abc"}}
     ```

4. `run_name`

   - **역할**: 이 실행(Run)의 이름을 강제로 지정합니다.


   - **용도**: LangSmith 트레이스 화면에서 "RunnableLambda" 대신 "내 수정 함수" 처럼 예쁜 이름으로 보고 싶을 때.


   - **예시**: 

     ```
     {"run_name": "MyCustomParsingJob"}
     ```

5. `recursion_limit`

   - **역할**: 체인이 너무 깊게 뺑뺑이 도는 것을 막습니다. (기본값 25)


   - **용도**: 무한 루프 방지.


   - **예시**: 

     ```
     {"recursion_limit": 10}
     ```


&nbsp;

이외에도 `configurable_field()` 메서드를 이용해 LLM 모델을 생성한 후<br>**`config`에서 `configurable`키를 설정해 LLM 모델의 필드(파라미터)를 동적으로 바꿀 수 있습니다.**

&nbsp;

&nbsp;

앞서 얘기한 **configurable**에 대해 더 구체적이고 명확하게 정리하고자 합니다.

### 동적 변경을 허용할 '속성/Runnable 대안' 정의하는 방법

1. `.configurable_fields()`: 동적 변경 가능한 속성(필드)을 정의하는 메서드

   [동적 변경할 ChatOpenAI의 model_name 속성 정의]

   ```python
   from langchain.prompts import PromptTemplate
   from langchain_core.runnables import ConfigurableField
   from langchain_openai import ChatOpenAI
   
   # model_name: 동적 설정 가능 (default:gpt-4o)
   model = ChatOpenAI(temperature=0, model_name="gpt-4o").configurable_fields(
   	# ChatOpenAI의 필드 model_name을 동적으로 변경할 것임을 설정
       model_name=ConfigurableField(
           id="gpt_version",  # model_name의 id 설정
           name="Version of GPT",  # model_name의 이름 설정
           # model_name의 설명 설정
           description="Official model name of GPTs. ex) gpt-4o, gpt-4o-mini",
       )
   )
   ```

   * `ChatOpenAI`에 `configurable_fields()` 메서드를 이용해 속성 `model_name`을 동적 변경 가능한 속성으로 정의함
   * 속성 `model_name`에 `ConfigurableField`을 이용해 정의함
   * 속성 변경 시, `model_name`의 `id` 값인 `gpt_version`에 변경할 값을 주면 됨

   &nbsp;

   [HubRunnable을 이용한 랭체인 허브 프롬프트 동적 변경 정의]

   ```python
   from langchain.runnables.hub import HubRunnable
   
   prompt = HubRunnable("teddynote/rag-prompt-korean").configurable_fields(
       # 소유자 저장소 커밋을 설정하는 ConfigurableField
       owner_repo_commit=ConfigurableField(  # ⭐️ owner_repo_commit는 "teddynote/rag-prompt-korean" 부분을 말하는 것 ⭐️
           # 필드의 ID
           id="hub_commit",
           # 필드의 이름
           name="Hub Commit",
           # 필드에 대한 설명
           description="Korean RAG prompt by teddynote",
       )
   )
   prompt
   ```

   * `HubRunnable`에 `configurable_fields()` 메서드를 이용해 속성 `owner_repo_commit`을 동적 변경 가능한 속성으로 정의함
     * **정확히 말하자면, LangChain Hub에서 프롬프트 불러오는 repo 속성을 변경하는 것입니다!!!**
   * 속성 `owner_repo_commit`에 `ConfigurableField`을 이용해 정의함
   * 속성 변경 시, `owner_repo_commit`의 `id` 값인 `hub_commit`에 변경할 값을 주면 됨



2. `.configurable_alternatives()`: Runnable 객체 대안을 정의하는 메서드

   [LLM의 객체 대안 정의]

   ```python
   from langchain.prompts import PromptTemplate
   from langchain_anthropic import ChatAnthropic
   from langchain_core.runnables import ConfigurableField
   from langchain_openai import ChatOpenAI
   
   # ⭐️ configurable_alternatives: Runnable 객체 자체를 바꿈 ⭐️
   
   llm = ChatAnthropic(
       temperature=0, model="claude-3-5-sonnet-20240620"
   ).configurable_alternatives(
       # 이 필드에 id를 부여합니다.
       # 최종 실행 가능한 객체를 구성할 때, 이 id를 사용하여 이 필드를 구성할 수 있습니다.
       ConfigurableField(id="llm"),
       # 기본 키를 설정합니다.
       # ⭐️ configurable에서 llm에 이 키를 지정하면 위에서 초기화된 기본 LLM(ChatAnthropic(temperature=0, model="claude-3-5-sonnet-20240620"))이 사용됩니다. ⭐️
       default_key="anthropic",
       # 'openai'라는 이름의 새 옵션을 추가하며, 이는 `ChatOpenAI()`와 동일합니다.
       openai=ChatOpenAI(model="gpt-4o-mini"),
       # 'gpt4'라는 이름의 새 옵션을 추가하며, 이는 `ChatOpenAI(model="gpt-4")`와 동일합니다.
       gpt4o=ChatOpenAI(model="gpt-4o"),
       # 여기에 더 많은 구성 옵션을 추가할 수 있습니다.
   )
   prompt = PromptTemplate.from_template("{topic} 에 대해 간단히 설명해주세요.")
   chain = prompt | llm
   ```

   * `ChatAnthropic`에 `configurable_alternatives()` 메서드를 이용해 LLM 모델 자체를 변경 가능하도록 정의함<br>(속성 변경이 아니므로 메서드에 필드명 없이 바로 `ConfigurableField` 넣음)

   * LLM 모델 변경 시

     * `id` 값인 `llm`에 변경할 값을 주면 됨

     * `id`에 줄 수 있는 변경 가능한 값

       * `anthropic`: `default_key`로 설정한 원래 모델 설정<br>(`ChatAnthropic(temperature=0, model="claude-3-5-sonnet-20240620")`)을 이용할 때, `id`에 줄 값
       *  `openai`: `ChatOpenAI(model="gpt-4o-mini")`로 설정
       * `gpt4o`: `ChatOpenAI(model="gpt-4o")`로 설정

     * 모델 변경 방법 예시

       ```python
       chain.with_config(configurable={"llm": "openai"}).invoke({"topic": "뉴진스"})
       ```

   [프롬프트 객체 대안 정의]

   ```python
   # 언어 모델을 초기화하고 temperature를 0으로 설정합니다.
   llm = ChatOpenAI(temperature=0)
   
   prompt = PromptTemplate.from_template(
       "{country} 의 수도는 어디야?"  # 기본 프롬프트 템플릿
   ).configurable_alternatives(
       # 이 필드에 id를 부여합니다. (⭐️ 변경할 값을 prompt 변수명에 넣으면 됨 ⭐️)
       ConfigurableField(id="prompt"),
       # 기본 키를 설정합니다. -> ⭐️ configurable에서 prompt에 capital로 넣으면 원래 프롬프트 "{country} 의 수도는 어디야?"로 설정됨 ⭐️
       default_key="capital",
       # 'area'이라는 새로운 옵션을 추가합니다.
       area=PromptTemplate.from_template("{country} 의 면적은 얼마야?"),
       # 'population'이라는 새로운 옵션을 추가합니다.
       population=PromptTemplate.from_template("{country} 의 인구는 얼마야?"),
       # 'eng'이라는 새로운 옵션을 추가합니다.
       eng=PromptTemplate.from_template("{input} 을 영어로 번역해주세요."),
       # 여기에 더 많은 구성 옵션을 추가할 수 있습니다.
   )
   
   # 프롬프트와 언어 모델을 연결하여 체인을 생성합니다.
   chain = prompt | llm
   ```

   LLM 객체 대안 설정과 비슷합니다 !!



3. `ConfigurableField()`: 동적으로 변경할 필드 정의

   * `id` : 동적으로 변경할 필드명을 대신할 변수명(필수로 지정)<br>ex ChatOpenAI의 model_name 필드명을 gpt_version으로 대신함 -> 나중에 gpt_version에 속성값을 지정함

   * `name`: id에 대한 이름
     * 동적으로 변경하는 필드명이 무엇인지 이름을 정함

   * `description`: 해당 필드에 대한 설명

&nbsp;

### 속성을 변경하는 방법

1. `.invoke(config={"configurable": {id: 바꾸고자 하는 값}})`: chain에서 `invoke()` 호출 시 config 설정으로 속성을 동적으로 변경합니다.

   ```python
   # gpt_version(model_name의 ConfigurableField에서 id)을 gpt-3.5-turbo로 동적 설정
   model.invoke(
       "대한민국의 수도는 어디야?",
       config={"configurable": {"gpt_version": "gpt-3.5-turbo"}},
   )
   ```

2. `.with_config(configurable={id: 바꾸고자하는 값})`: 속성 또는 객체를 동적 변경을 위한 Runnable의 메서드<br>(`configurable` 값에 바꿀 값을 전달합니다.)

   ```python
   chain.with_config(configurable={"llm": "openai"}).invoke({"topic": "뉴진스"})
   ```

차이점: `invoke`에서 config 설정은 **"이번 한 번만 실행할 때 적용"**하는 느낌이고, `.with_config()`는 "설정이 적용된 새로운 체인 객체를 아예 만들어내는" 느낌입니다. (그래서 변수에 저장해서 재사용 가능)

&nbsp;

&nbsp;

## Route

---

라우팅은 이전 단계의 출력이 경로를 정하는 것입니다.<br>**간단한 질문 분류**나 **주제 분류** 등에 유용한 방식이죠.

유저의 질문이 들어왔을 때 라우팅을 하는 방법은<br>`RunnableLambda`, `RunnableBranch` 2가지가 존재합니다.

&nbsp;

### `RunnableLambda`에서 Routing

`route`를 정해주는 `route_chain` 생성

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

route_prompt = PromptTemplate.from_template(
    """주어진 사용자 질문을 `수학`, `과학`, 또는 `기타` 중 하나로 분류하세요. 한 단어 이상으로 응답하지 마세요.

<question>
{question}
</question>

Classification:"""
)

# 체인 생성
route_chain = (
    route_prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)
```

&nbsp;

`route`별 수행해야 할 `chain` 생성

```python
math_chain = (
    PromptTemplate.from_template(
        """You are an expert in math. \
Always answer questions starting with "깨봉선생님께서 말씀하시기를..". \
Respond to the following question:

Question: {question}
Answer:"""
    )
    | ChatOpenAI(model="gpt-4o-mini")
)

science_chain = (
    PromptTemplate.from_template(
        """You are an expert in science. \
Always answer questions starting with "아이작 뉴턴 선생님께서 말씀하시기를..". \
Respond to the following question:

Question: {question}
Answer:"""
    )
    | ChatOpenAI(model="gpt-4o-mini")
)

general_chain = (
    PromptTemplate.from_template(
        """Respond to the following question concisely:

Question: {question}
Answer:"""
    )
    | ChatOpenAI(model="gpt-4o-mini")
)
```

&nbsp;

정해진 `route`에 따라 `chain`으로 안내하는 교차로 역할의 함수 정의

```python
# RunnableLambda로 실행되는 함수이므로, 인자는 1개를 받아와야 함
# info = {"topic": route_chain 결과, "question": 사용자 입력}
def route(info):
    # 주제에 "수학"이 포함되어 있는 경우
    if "수학" in info["topic"].lower():
        # datascience_chain을 반환
        return math_chain
    # 주제에 "과학"이 포함되어 있는 경우
    elif "과학" in info["topic"].lower():
        # art_chain을 반환
        return science_chain
    # 그 외의 경우
    else:
        # general_chain을 반환
        return general_chain
```

&nbsp;

이들을 모두 엮어주는 메인 chain

```python
from operator import itemgetter
from langchain_core.runnables import RunnableLambda

full_chain = (
    {"topic": route_chain, "question": itemgetter("question")}
    | RunnableLambda(
        # 경로를 지정하는 함수를 인자로 전달합니다.
        route
    )
    | StrOutputParser()
)
```

`{"topic": route_chain, "question": itemgetter("question")}`부터 이해가 안 갈 수 있지만

1가지 기억합시다.

> **"chain에는 딕셔너리 형태로 계속 무언가가 전달된다."**

즉,

**전달되는 것과 출력되는 것들에 집중하고<br>항상 입력을 딕셔너리로 받아온 후, itemgetter 등으로 뽑아서 사용된다는 것을 잊지 말자.**

&nbsp;

사용자 입력은

1. `chain`에 전달돼서 주제를 가져와 `topic`에 저장하고
1. `question`에 그대로 저장됩니다.

`topic`과 `question`은 `RunnableLambda`에 의해<br>`route` 함수에서 `info` 딕셔너리로 그대로 받아와 `topic`에 따라 루트가 정해집니다.

결국, 해당 주제별 체인에 `topic`과 `question`이 전달되면서 변수에 맞게 프롬프트가 채워지는 거죠.

&nbsp;

`full_chain.get_graph().print_ascii()`를 하면

```python
+-------------------------------+                      
                    | Parallel<topic,question>Input |                      
                    +-------------------------------+                      
                             ***           ***                             
                           **                 **                           
                         **                     **                         
              +----------------+                  **                       
              | PromptTemplate |                   *                       
              +----------------+                   *                       
                       *                           *                       
                       *                           *                       
                       *                           *                       
                +------------+                     *                       
                | ChatOpenAI |                     *                       
                +------------+                     *                       
                       *                           *                       
                       *                           *                       
                       *                           *                       
              +-----------------+             +--------+                   
              | StrOutputParser |             | Lambda |                   
              +-----------------+             +--------+                   
                             ***           ***                             
                                **       **                                
                                  **   **                                  
                    +--------------------------------+                     
                    | Parallel<topic,question>Output |                     
                    +--------------------------------+                     
                                     *                                     
                                     *                                     
                                     *                                     
                              +-------------+                              
                              | route_input |                              
                          ****+-------------+****                          
                     *****           *           *****                     
                 ****                *                ****                 
              ***                    *                    ***              
+----------------+          +----------------+          +----------------+ 
| PromptTemplate |          | PromptTemplate |          | PromptTemplate | 
+----------------+          +----------------+          +----------------+ 
         *                           *                           *         
         *                           *                           *         
         *                           *                           *         
  +------------+              +------------+              +------------+   
  | ChatOpenAI |*             | ChatOpenAI |              | ChatOpenAI |   
  +------------+ ****         +------------+          ****+------------+   
                     *****           *           *****                     
                          ****       *       ****                          
                              ***    *    ***                              
                             +--------------+                              
                             | route_output |                              
                             +--------------+                              
                                     *                                     
                                     *                                     
                                     *                                     
                            +-----------------+                            
                            | StrOutputParser |                            
                            +-----------------+                            
                                     *                                     
                                     *                                     
                                     *                                     
                        +-----------------------+                          
                        | StrOutputParserOutput |                          
                        +-----------------------+
```

아름다운 그림이 나와요.

&nbsp;

&nbsp;

### `RunnableBranch`에서 Routing

`RunnableBranch`는 `if-elif-else`의 `chain` 버전이라고 생각하시면 됩니다.

우선, "`route`를 정해주는 `route_chain` 생성", "`route`별 수행해야 할 `chain` 생성"을 한 후

**교차로 역할인 `route` 함수에서 `if-elif-else`문을 <br>`RunnableBranch`에서 3개의 `Runnable` 형식으로 나열해주면 됩니다**.

```python
from operator import itemgetter
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    # 주제에 "수학"이 포함되어 있는지 확인하고, 포함되어 있다면 math_chain을 실행합니다.
    (lambda x: "수학" in x["topic"].lower(), math_chain),
    # 주제에 "과학"이 포함되어 있는지 확인하고, 포함되어 있다면 science_chain을 실행합니다.
    (lambda x: "과학" in x["topic"].lower(), science_chain),
    # 위의 조건에 해당하지 않는 경우 general_chain을 실행합니다.
    general_chain,
)
# 주제와 질문을 입력받아 branch를 실행하는 전체 체인을 정의합니다.
full_chain = (
    {"topic": route_chain, "question": itemgetter("question")} | branch | StrOutputParser()
)
```

`RunnableBranch`안에서 `Runnable`인 `lamda` 함수로 if, elif, else 순으로 나열하면 됩니다.

`lamda` 함수는 `(lamda 입력: 조건, 실행할 것)` 식으로 작성하며<br>앞에서부터 해당하는 조건에서 실행됩니다.

&nbsp;

&nbsp;

여기서 끊고 넘어가도록 하겠습니다. 👍🏻

```toc

```
