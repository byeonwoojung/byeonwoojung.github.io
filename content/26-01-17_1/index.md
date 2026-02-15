---
emoji: ☀️
title: "[LLM] 테디노트의 RAG 비법노트 끄적끄적-14"
date: '2026-01-17 00:00:00'
author: 변우중
tags: LLM 프롬프트 RAG LangChain 랭체인 create_sql_query_chain
categories: LLM RAG
---

참고 : 테디노트의 RAG 비법노트 (https://fastcampus.co.kr/data_online_teddy)<br>소스코드: https://github.com/teddylee777/langchain-kr<br>위키독스: https://wikidocs.net/book/14314

&nbsp;

글이 자주 올라오지 않는 것은<br>RAG와 Agent의 내용들은 프로젝트에 녹여서 끄적이고자 그렇습니다,,

오늘은 아아아주 정말 메모장st 느낌의 글입니다. 다소 짧고 구조적이지 않습니다.

레츠기릿,,

&nbsp;

## create_sql_query_chain

---

DB에서 특정 값을 조회하기 위한 SQL 쿼리를 생성하고, 쿼리를 실행하는 방법에 대해 정리하고자 합니다.

간단합니다.

1. SQL 쿼리를 생성하는 chain 생성 (DB를 연결하고, 쿼리 생성하는 LLM 선정)

   ```python
   
   from langchain_core.prompts import PromptTemplate
   
   db = SQLDatabase.from_uri("sqlite:///data/my_db.db")
   
   # db의 dialect(어떤 db인지), 사용자 입력, 컬럼 설명 등을 넣음
   prompt = PromptTemplate.from_template(
       """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.
   Use the following format:
   
   Question: "Question here"
   SQLQuery: "SQL Query to run"
   SQLResult: "Result of the SQLQuery"
   Answer: "Final answer here"
   
   Only use the following tables:
   {table_info}
   
   Here is the description of the columns in the tables:
   `컬럼1`: 컬럼 설명
   `컬럼2`: 컬럼 설명
   `컬럼3`: 컬럼 설명
   
   Question: {input}"""
   ).partial(dialect=db.dialect) # db.dialect 넣음!!
   
   # sql 쿼리 생성 모델
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
   
   # LLM, DB, prompt를 매개변수로 입력하여 chain 을 생성합니다.
   # ⭐️ prompt는 직접 넣어서 변경 가능함 ⭐️
   create_query_chain = create_sql_query_chain(llm, db, prompt)
   
   
   # 생성된 쿼리를 출력하기
   answer = create_query_chain.invoke({"question": "고객의 이름을 나열하세요"})
   print(answer.__repr__())
   
   ```

2. 쿼리를 실행하는 도구 생성

   ```python
   from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
   
   # 생성한 쿼리를 실행하기 위한 도구 생성
   execute_query = QuerySQLDataBaseTool(db=db)
   ```

3. 두 체인 연결

   ```python
   chain = create_query_chain | execute_query
   chain.invoke({"question": "민수의 이메일을 조회하세요"})
   ```

   * create_query_chain의 결과가 execute_query의 query 변수에 들어가서 db에 조회합니다.

&nbsp;

&nbsp;

## 추가 메모장

---

1. retriever를 그냥 호출하게 되면 page_content만 반환하므로, LLM에게 줄 추가적인 메타데이터는 프롬프트 템플릿을 통해 넣어야 한다.

2. LangGraph에서 Reducer(add_messages, operator.add): 자동으로 리스트에 메시지를 추가해주는 기능

3. TypedDict: dict에 타입힌팅 추가한 개념

4. LangGraph 기본 개념

* 상태(State): 노드와 노드 간에 정보를 전달할 때 상태(State) 객체에 담아 전달함

* 새로운 노드에서 값을 덮어쓰기 방식으로 채움

* GraphState 클래스를 TypedDict 클래스를 상속받아 보통 정의함

  ```python
  from typing import Annotated, TypedDict
  from langgraph.graph.message import add_messages
  
  class GraphState(TypedDict):
      question: Annotated(list, add_messages)  # 질문
      context: Annotated(str, "Context")  # 문서의 검색 결과
      answer:  Annotated(str, "Answer")  # 답변
      messages: Annotated(list, add_messages)  # 메시지
      relevance: Annotated(str, "relevance")  # 관련성
  ```

  * `question`, `messages`는 `Annotated(list, add_messages)`를 통해 리스트에 메시지를 계속 추가함
  * 다른 것들은 덮어쓰기로 상태를 저장함

* 기본 구조

  ```python
  from typing import Annotated, List, Dict
  from typing_extensions import TypedDict
  
  from langchain_core.tools import tool
  from langchain_openai import ChatOpenAI
  from langgraph.checkpoint.memory import MemorySaver
  from langgraph.graph import StateGraph, START, END
  from langgraph.graph.message import add_messages
  from langgraph.prebuilt import ToolNode, tools_condition
  from langchain_teddynote.graphs import visualize_graph
  from langchain_teddynote.tools import GoogleNews
  
  
  1. 상태 정의
  class State(TypedDict):
      # 메시지 목록 주석 추가
      messages: Annotated[list, add_messages]
      dummy_data: Annotated[str, "dummy"]
  
  
  # 2. 도구 정의 및 바인딩
  # 도구 초기화 (키워드로 뉴스 검색하는 도구 생성)
  news_tool = GoogleNews()
  
  
  @tool
  def search_keyword(query: str) -> List[Dict[str, str]]:
      """Look up news by keyword"""
      news_tool = GoogleNews()
      return news_tool.search_by_keyword(query, k=5)
  
  
  tools = [search_keyword]
  
  # LLM 초기화
  llm = ChatOpenAI(model="gpt-4o-mini")
  
  # 도구와 LLM 결합
  llm_with_tools = llm.bind_tools(tools)
  
  
  # 3. 노드 추가
  # 챗봇 함수 정의
  def chatbot(state: State):
      # 메시지 호출 및 반환
      return {
          "messages": [llm_with_tools.invoke(state["messages"])],
          "dummy_data": "[chatbot] 호출, dummy data",  # 테스트 위해 더미 데이터 추가
      }
  
  
  # 상태 그래프 생성
  graph_builder = StateGraph(State)
  
  # 챗봇 노드 추가
  graph_builder.add_node("chatbot", chatbot)
  
  
  # 도구 노드 생성 및 추가
  tool_node = ToolNode(tools=tools)
  
  # 도구 노드 추가
  graph_builder.add_node("tools", tool_node)
  
  # 조건부 엣지
  graph_builder.add_conditional_edges(
      "chatbot",
      tools_condition,
  )
  
  # 4. 엣지 추가
  
  # tools > chatbot
  graph_builder.add_edge("tools", "chatbot")
  
  # START > chatbot
  graph_builder.add_edge(START, "chatbot")
  
  # chatbot > END
  graph_builder.add_edge("chatbot", END)
  
  # 5. 그래프 컴파일
  
  # 그래프 빌더 컴파일
  graph = graph_builder.compile()
  
  ########## 6. 그래프 시각화 ##########
  # 그래프 시각화
  visualize_graph(graph)
  ```

  

5. stream 노드 단계별 출력

그래프를 stream 메서드를 이용하여 각 노드별 실행 결과를 for 문을 이용해 받아와서 출력을 할 수 있습니다. 각 노드별 `key`(노드 이름)와 `value`(상태 결과)를 받아올 수 있으며, 이때 stream 메서드에 매개변수 값을 넣어 값을 받아오기 위한 다양한 설정을 할 수 있습니다.

* `input`: 그래프 입력
* `config`: 설정 값(`recursion_limit`, `configurable`(thread_id 등), `tags`)
* `list(graph.channels.keys())`: 그래프의 모든 채널 키를 가져옴

* `stream_mode` : **"그래프가 실행되는 동안, 어떤 데이터를 뱉어낼 것인가?"**를 결정하는 옵션입니다.

  ```python
  input = State(dummy_data="테스트 문자열", messages=[("user", question)])
  
  config = RunnableConfig(
      recursion_limit=10,  # 최대 10개의 노드까지 방문
      configurable={"thread_id": "1"},  # 스레드 ID 설정
  )
  
  # 2. output_keys에 모든 키를 전달하여 실행합니다.
  for event in graph.stream(
      input=input,
      config=config,
      stream_mode="values",  # 전체 상태를 스트리밍
      output_keys=all_channel_keys,  # 여기에 모든 키 리스트를 넣습니다.
  ):
      for key, value in event.items():
          print(f"\n[ {key} ]")
          print(f"Type: {type(value)}")
          print(value)
  ```

  * 현재 상태(State)에 `messages` 리스트가 있고, `chatbot` 노드가 새로운 메시지를 하나 추가하는 상황이라고 가정해 봅시다.

  * 모드 값: `values`, `updates`(기본값), `debug`

  * 상황 가정

    - **현재 상태**: `['안녕']`
    - **chatbot 노드 실행**: `['반가워']`라는 메시지를 생성함

  * <strong>`stream_mode="values"` (전체 상태 모드)</strong><br>

    이 모드는 **"노드가 실행된 후, 완성된 전체 상태(State)"**를 보여줍니다. 가장 최신 시점의 **"결과물 전체"**를 보고 싶을 때 사용합니다.

    - **출력의 키(Key)**: 상태 변수의 이름 (예: `messages`, `dummy_data`)
    - **출력의 값(Value)**: 해당 변수의 **전체 누적 값**

    ```python
    # 예시 출력
    {
        "messages": ["안녕", "반가워"],  # <-- 기존 값 + 새로운 값 모두 포함됨
        "dummy_data": "..."
    }
    ```

    > **비유**: 회의록을 작성할 때, **"회의록 전체 파일"**을 매번 새로 공유받는 것과 같습니다. (앞부분 내용 + 새로 추가된 내용 가 포함됨)

  * <strong> `stream_mode="updates"` (업데이트 모드) - *기본값*</strong><br>

    이 모드는 **"방금 실행된 노드가 변경한 부분(차이점)"**만 보여줍니다. **"누가(어떤 노드가) 무엇을 바꿨는지"** 흐름을 파악할 때 좋습니다.

    - **출력의 키(Key)**: 방금 실행된 **노드 이름** (예: `chatbot`, `tools`)
    - **출력의 값(Value)**: 그 노드가 **새로 만들어낸 값**

    ```python
    # 예시 출력
    {
        "chatbot": {                 # <-- 노드 이름이 키가 됨
            "messages": ["반가워"]    # <-- 이번에 '새로 추가된' 메시지만 있음 ("안녕"은 없음)
        }
    }
    ```

    > **비유**: 회의록을 작성할 때, **"방금 추가된 줄"**만 메신저로 알림 받는 것과 같습니다.

* `interrupt_before`, `interrupt_after`, 

  * 특정 노드 전/후에 스트리밍을 중단합니다.

  * `interrupt_before` 설정 방법 예시

    ```python
    for event in graph.stream(
        input=input,
        config=config,
        stream_mode="updates",  # 기본값
        interrupt_before=["tools"],  # ⭐️ tools 노드 이전에 스트리밍 중단 ⭐️
    ):
    ```

  * 중단 후 재개 방법

    * 추가 input 없을 때 (`input`에 `None`을 넣음)

      ```python
      # None은 추가 input은 없는 것임
      graph.stream(input=None, config, stream_mode="updates")
      ```

    * 추가 input 있을 때 (input에 상태를 넣습니다.)



6. `get_state(config)` 메서드

   * 그래프의 현재 상태를 출력합니다. (config에는 현재 스레드 id 담겨 있음)<br>즉, 현재 상태 스냅샷 !!

   * `interrupt_before`, `interrupt_after`를 이용하여 특정 노드 전/후로 멈추고, 현재 상태와 다음 스냅샷 상태를 출력할 수 있습니다.

     ```python
     # ⭐️ 그래프 상태 스냅샷 생성 (현재 상태 값 출력) ⭐️
     snapshot = graph.get_state(config)
     
     # 다음 스냅샷 상태
     print(snapshot.next)
     print()
     # 현재 상태
     print(snapshot)
     print()
     # 현재 저장된 값
     print(snapshot.values)
     print()
     
     """출력:
     ('tools',)
     ====================
     StateSnapshot(values={'messages': [HumanMessage(content='AI 관련 최신 뉴스를 알려주세요.', additional_kwargs={}, response_metadata={}, id='884734b2-b0ef-4c0a-9f1c-5d48048b0887'), AIMessage(content='', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 51, 'total_tokens': 65, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_1590f93f9d', 'id': 'chatcmpl-D4ibXNCTqMvmcGLczUIrtLZg3baly', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--019c1d3e-370e-7480-93ec-727278328647-0', tool_calls=[{'name': 'search_keyword', 'args': {'query': 'AI'}, 'id': 'call_apWmU6348h1ckdw4XdHfJjl5', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 51, 'output_tokens': 14, 'total_tokens': 65, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}, next=('tools',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f100086-2b66-65e0-8001-fe233ecc81d3'}}, metadata={'source': 'loop', 'step': 1, 'parents': {}}, created_at='2026-02-02T07:25:44.105514+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f100086-1c7b-66a2-8000-fe434acdd66f'}}, tasks=(PregelTask(id='ef6c6364-3daf-6960-d077-6bc3f717e3ef', name='tools', path=('__pregel_pull', 'tools'), error=None, interrupts=(), state=None, result=None),), interrupts=())
     ====================
     {'messages': [HumanMessage(content='AI 관련 최신 뉴스를 알려주세요.', additional_kwargs={}, response_metadata={}, id='884734b2-b0ef-4c0a-9f1c-5d48048b0887'), AIMessage(content='', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 51, 'total_tokens': 65, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_1590f93f9d', 'id': 'chatcmpl-D4ibXNCTqMvmcGLczUIrtLZg3baly', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--019c1d3e-370e-7480-93ec-727278328647-0', tool_calls=[{'name': 'search_keyword', 'args': {'query': 'AI'}, 'id': 'call_apWmU6348h1ckdw4XdHfJjl5', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 51, 'output_tokens': 14, 'total_tokens': 65, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}
     """
     ```



7. checkpoint 활용 방법 (중간 단계에서부터 다시 실행)

먼저 **그래프의 채크포인터에 메모리 설정**을 해줍니다.

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)  # ⭐️ 인메모리 설정하여 그래프 빌더 컴파일 ⭐️
```



그래프의 중간 단계별 출력을 위해 **`stream` 메서드**로 출력합니다.

```python
from langchain_teddynote.messages import pretty_print_messages
from langchain_core.runnables import RunnableConfig

# 질문
question = "AI 관련 최신 뉴스를 알려주세요."

# 초기 입력 State 를 정의
input = State(messages=[("user", question)])

# config 설정
config = RunnableConfig(
    recursion_limit=10,
    configurable={"thread_id": "1"},  # 스레드 ID 설정
)

for event in graph.stream(
    input=input,
    config=config,
    stream_mode="updates",
):
    for key, value in event.items():
        # key 는 노드 이름
        print(f"\n[{key}]\n")

        # value 는 노드의 출력값
        # print(value)
        pretty_print_messages(value)

        # value 에는 state 가 dict 형태로 저장(values 의 key 값)
        if "messages" in value:
            print(f"{value['messages']}")
```

&nbsp;

**그래프의 `get_state_history(config)` 메서드를 for 문을 이용해 각 상태를 가져옵니다.** <strong>여기서 특정 상태를 저장하고 싶은 곳에서 저장합니다 ‼️(여기서는 `to_replay`)</strong>

```python
to_replay = None

# 상태 기록 가져오기
for state in graph.get_state_history(config):
    # 메시지 수 및 다음 상태 출력

    # ⭐️ 특정 상태 선택 기준: 채팅 메시지 수 ⭐️
    if len(state.values["messages"]) == 3:
        to_replay = state
        print("현재 상태:", str(state.values["messages"]))
        print("메시지 수: ", len(state.values["messages"]), "다음 노드: ", state.next)
        print("-" * 80)
        continue

    print("현재 상태:", str(state.values["messages"])[:200], "...(생략)")
    print("메시지 수: ", len(state.values["messages"]), "다음 노드: ", state.next)
    print("-" * 80)

"""출력:
현재 상태: [HumanMessage(content='AI 관련 최신 뉴스를 알려주세요.', additional_kwargs={}, response_metadata={}, id='6bf41778-8525-44b4-bb27-0045d21b6258'), AIMessage(content='', additional_kwargs={'refusal': None}, response ...(생략)
메시지 수:  4 다음 노드:  ()
--------------------------------------------------------------------------------
현재 상태: [HumanMessage(content='AI 관련 최신 뉴스를 알려주세요.', additional_kwargs={}, response_metadata={}, id='6bf41778-8525-44b4-bb27-0045d21b6258'), AIMessage(content='', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 51, 'total_tokens': 65, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_1590f93f9d', 'id': 'chatcmpl-D4jKWuVxdrGKGEdxTrW4Pzsf1zNxC', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--019c1d68-c5b2-7f72-974d-fe1b10946548-0', tool_calls=[{'name': 'search_keyword', 'args': {'query': 'AI'}, 'id': 'call_gv14hAhwIgkBWYVi8YzSiEZS', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 51, 'output_tokens': 14, 'total_tokens': 65, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), ToolMessage(content='[{"url": "https://news.google.com/rss/articles/CBMid0FVX3lxTE1GLUR3ODI5V3FjOG9mcmFpclhpbXR5Y1E3VWE1eTV1RTlyVzhxaXhuWE1yczY2WEQxZHZKSHd1NktYSVBZM1JnRGw0OHZSNGxIZUFhdG5Lejl3NFRXeFhJNHU5eVR3TEhBQUNLenp6Z3BXZGo3QzB3?oc=5", "content": "AI끼리 말하는 SNS…“인간은 실패작, 우린 새로운 신” - 한겨레"}, {"url": "https://news.google.com/rss/articles/CBMiVkFVX3lxTE94UlFRZlNDTEkwS1JVczVxZlRPeTUtWVJOdzZONWg3c0l6ZjVJUVRVRDRkdlV1WjhncC1OV3FVM2RWOUJmMnhYMDExNE04NGRPUnlyRXNB?oc=5", "content": "\'다음\' 품은 업스테이지, 경쟁력 있을까…AI로 분석했더니 - 지디넷코리아"}, {"url": "https://news.google.com/rss/articles/CBMiTkFVX3lxTFBhaVY2U1UzZFl5Mm91aGhKOFA0eFc2WDVxUXNWWEJINGNQdWJTMXRTb2E0U0lCazhvR0NnZ3pPd1hldG5jMmdjZWV6R3Zhdw?oc=5", "content": "한국피지컬AI협회, \'피지컬 AI 최강국\' 로드맵 제시…\'데이터 팩토리\' 추진 - 전자신문"}, {"url": "https://news.google.com/rss/articles/CBMidkFVX3lxTE1BLVFBN1FmQk9nX1N4MlMtZzB4clQweE5EYlMtZWtiNHJtYmo1anIwXzgzRGJJNXYtRUFaT1JBb1ZDMFY0NjNiT1ZCRmlHb1hNSXh1RUJ5VWNnQWx1dWlyZ3lKLXNrbTdvTGEwUHlsZHBPVllPSEHSAWZBVV95cUxNNFhIMnJEWE5TZGpNNVNJcUZpVHZuTTlVM2czZG5uX3lhOXUzOXdRd3llMHZwcWNNcXhTdm9sTlU4R01Kb2dzdk5wR0pvTExERlVQR1JsRWxDcFFlOHdHSU1EQVBTTGc?oc=5", "content": "바디캠인 줄 알았는데 ‘AI 영상’…허위 영상 유포 30대 유튜버 구속 - 동아일보"}, {"url": "https://news.google.com/rss/articles/CBMiVkFVX3lxTE1XZFJhVWpLMEJ5VlBaXzQxNDhoRWgzelh0aUVjZ3JBMGkwU2drakxKdGZFRl9BZG9LOW03dUlSUWVEbU1xOE9LaEJxaHJrbTJuT3F5dTlB?oc=5", "content": "‘경찰 보디캠 허위영상’ AI로 만들어 유포…유튜버 구속[영상] - 중앙일보"}]', name='search_keyword', id='70cc7e3f-0b26-40d9-9bcf-9dea7f087ae0', tool_call_id='call_gv14hAhwIgkBWYVi8YzSiEZS')]
메시지 수:  3 다음 노드:  ('chatbot',)
--------------------------------------------------------------------------------
현재 상태: [HumanMessage(content='AI 관련 최신 뉴스를 알려주세요.', additional_kwargs={}, response_metadata={}, id='6bf41778-8525-44b4-bb27-0045d21b6258'), AIMessage(content='', additional_kwargs={'refusal': None}, response ...(생략)
메시지 수:  2 다음 노드:  ('tools',)
--------------------------------------------------------------------------------
현재 상태: [HumanMessage(content='AI 관련 최신 뉴스를 알려주세요.', additional_kwargs={}, response_metadata={}, id='6bf41778-8525-44b4-bb27-0045d21b6258')] ...(생략)
메시지 수:  1 다음 노드:  ('chatbot',)
--------------------------------------------------------------------------------
현재 상태: [] ...(생략)
메시지 수:  0 다음 노드:  ('__start__',)
--------------------------------------------------------------------------------
"""
```

**이때, 각 그래프 상태 히스토리의 state.values 출력 형태는 `stream_mode="values"`의 출력과 동일한 형태입니다.**

&nbsp;

**아래와 같이 해당 상태에서 다음 노드와 `config`에서의 `checkpoint_id`를 확인할 수 있습니다.**

```python
# 다음 항목의 다음 요소 출력
print(to_replay.next)

# 다음 항목의 설정 정보 출력
print(to_replay.config)


"""출력:
('chatbot',)
{'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f1000ee-151c-631c-8002-8ebf03a2af91'}}
"""
```

&nbsp;

해당 상태의 `config`를 넣어 스트리밍 출력하게 되면<br>**그 상태에서 그 다음 노드를 실행**하게 됩니다.

정리하면

> 1. **인메모리를 통해 기억해야 하도록 컴파일하고**
> 2. **출력을 실행한 후 히스토리에서 특정 상태를 저장하여**
> 3. **상태의 config를 이용해 스트리밍 출력함으로써 중간 단계에서부터 다시 실행이 가능합니다.**

&nbsp;

7. 중간 단계 결과 수정

먼저 그래프를 스트리밍 출력했다고 합시다.

그리고 가장 최근 메시지(여기서는 `tool_call` (툴 호출 직전))을 가져와봅시다.

```python
# 그래프 상태 스냅샷 생성
snapshot = graph.get_state(config)

# 가장 최근 메시지 추출
last_message = snapshot.values["messages"][-1]
```

그 tool_call의 id를 가져옵시다.

&nbsp;

그리고, **`update_state()` 메서드를 통해 툴을 호출하는 id와 우리가 원하는 툴 콜링 결과 메시지를 넣으면 수정/추가가 됩니다.**

<strong>(`update_state()`에서 중요한 것은 어떤 상태에서 '추가' 또는 '수정'됨 ‼️)</strong>

```python
from langchain_core.messages import AIMessage, ToolMessage

# 위에서 modified_search_result 설정

# tool_call의 id를 가져옴
tool_call_id = last_message.tool_calls[0]["id"]

# ‼️ 툴 콜링 결과를 직접 추가/수정합니다. ‼️
new_messages = [
    # LLM API의 도구 호출과 일치하는 ToolMessage 필요
    ToolMessage(
        content=modified_search_result,
        tool_call_id=tool_call_id,
    ),
    # LLM의 응답에 직접적으로 내용 추가
    # AIMessage(content=modified_search_result),
]

graph.update_state(
    # 업데이트할 상태 지정
    config,
    # 제공할 업데이트된 값. `State`의 메시지는 "추가 전용"으로 기존 상태에 추가됨
    {"messages": new_messages},
    as_node="tools",
)
```

* `as_node`에서 `{"messages": new_messages}`가 온 것처럼 동작합니다.<b r>(`as_node` 없으면 마지막에서 상태 업데이트로 지정)
* 즉, tool_calls를 호출하여 `tools` 노드에서 결과를 받아온 것처럼 업데이트됩니다.
* 지정된 노드의 writer들을 이용해 상태를 업데이트하고 업데이트 된 상태를 새로운 체크포인트로 저장합니다.

&nbsp;

‼️ 궁금한 것이 있습니다 ‼️

이전에 **`사용자 입력 -> tool_calls -> tools -> chatbot을 통한 답변`** **모든 과정을 거쳤다고 했을 때**<br>히스토리 찾아서 `tool_calls`의 `id`을 가져와서 **`tools`의 결과를 아래와 같이`update_state()`로 수정**했다고 합시다.

```python
graph.update_state(
    # 업데이트할 상태 지정
    config,
    {"messages": new_messages},
    as_node="tools",
)
```

그러면 **"기존의 chatbot 답변은 사라지나요?"**

> **"기존의 답변은 그대로인 상태에서 중간 툴의 결과만 수정됩니다.**

&nbsp;

그러면 **"그 중간 툴의 결과를 수정한 후, 뒤의 결과도 달라지게 하는 방법은?"**

> **checkpointer 메모리 설정하여 해당 툴 결과로 상태를 가져와서 스트리밍 실행하면 됩니다.**

&nbsp;

아래와 같은 코드로 확인 가능합니다.

```python
import uuid
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition


# 1. 도구 & 그래프 정의
@tool
def search(query: str):
    """지방 방송국 채널 번호 정보를 검색합니다."""
    return "채널 번호는 7번입니다."


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [search]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=[search]))
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


# ---------------------------------------------------------
# [Step 1] 최초 실행 (Thread ID 생성)
# ---------------------------------------------------------
thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
print(
    f"=== [Step 1] 최초 실행 (Thread ID: {thread_config['configurable']['thread_id']}) ==="
)

input_message = HumanMessage(content="부산 MBC 방송국의 채널 번호를 검색해줘.")
for event in graph.stream({"messages": [input_message]}, thread_config):
    for key, value in event.items():
        if "messages" in value:
            print(f"[{key}] {value['messages'][-1].content}")


# ---------------------------------------------------------
# [Step 2] 과거 시점(Checkpoint) 탐색
# ---------------------------------------------------------
print("\n=== [Step 2] 과거 시점(Checkpoint) 탐색 ===")
history = list(graph.get_state_history(thread_config))
checkpoint_to_fork = None

# 과거 체크포인트 찾기 로직 (앞부분 코드)
# 조건: "다음 실행할 것이 chatbot이고(next == chatbot)", "마지막 메시지가 ToolMessage(도구 결과)인 순간"
for state in history:
    if (
        state.next == ("chatbot",)
        and state.values["messages"]
        and isinstance(state.values["messages"][-1], ToolMessage)
    ):
        checkpoint_to_fork = state
        break

if checkpoint_to_fork:
    print(
        f"타임머신 목적지 발견! Parent Checkpoint ID: {checkpoint_to_fork.config['configurable']['checkpoint_id']}"
    )

    # ---------------------------------------------------------
    # [Step 3] 과거 수정 및 분기 (Fork)
    # ---------------------------------------------------------
    print("\n=== [Step 3] 과거 수정 및 새로운 분기 생성 (Fork) ===")

    last_tool_msg = checkpoint_to_fork.values["messages"][-1]
    new_tool_message = ToolMessage(
        id=last_tool_msg.id,
        content="채널 번호는 999번입니다!!! (과거조작됨)",
        tool_call_id=last_tool_msg.tool_call_id,
    )

    # 과거 Config를 기반으로 상태 업데이트 -> 새로운 분기 Config 반환
    branch_config = graph.update_state(
        checkpoint_to_fork.config,  # ‼️ ⭐️ (매우 중요) 체크포인트의 config를 넣습니다. ⭐️ ‼️
        {"messages": [new_tool_message]},
        as_node="tools",
    )

    print(f"새로운 분기(평행우주) 생성 완료!")
    print(f"New Checkpoint ID: {branch_config['configurable']['checkpoint_id']}")

    # ---------------------------------------------------------
    # [Step 4] 새로운 타임라인 재생 (Replay)
    # ---------------------------------------------------------
    print("\n=== [Step 4] 새로운 타임라인 재생 (Replay) ===")
    for event in graph.stream(
        None, branch_config
    ):  # ‼️ 체크포인트 상황에서 다시 실행됨 ‼️
        for key, value in event.items():
            if "messages" in value:
                print(f"[{key}] {value['messages'][-1].content}")


# ---------------------------------------------------------
# [Step 5] 전체 체크포인트 히스토리 출력 (추가된 부분)
# ---------------------------------------------------------
print("\n" + "=" * 60)
print(
    f"=== [Step 5] 전체 체크포인트 조회 (Thread ID: {thread_config['configurable']['thread_id']}) ==="
)
print("=" * 60)

# 같은 Thread ID에 저장된 모든 기록을 가져옵니다.
all_history = list(graph.get_state_history(thread_config))

print(f"총 저장된 체크포인트 개수: {len(all_history)}개\n")

for i, state in enumerate(all_history, 1):
    ckpt_id = state.config["configurable"]["checkpoint_id"]
    # 부모 체크포인트 ID (어디서 파생되었는지 알 수 있음)
    parent_id = state.metadata.get("source") if state.metadata else "Unknown"

    last_msg = state.values["messages"][-1] if state.values["messages"] else None
    msg_preview = "메시지 없음"
    msg_type = "None"

    if last_msg:
        msg_type = type(last_msg).__name__
        msg_preview = (
            last_msg.content[:50] + "..."
            if len(last_msg.content) > 50
            else last_msg.content
        )

    print(f"[{i}] Checkpoint ID: {ckpt_id}")
    print(f"    - 생성 시점(작업): {state.created_at} / Next: {state.next}")
    print(f'    - 마지막 메시지({msg_type}): "{msg_preview}"')

    # 하이라이트: 어떤 타임라인인지 구별
    if "7번" in msg_preview and msg_type == "AIMessage":
        print("    => ★ [기존 역사] (AI 답변: 7번)")
    elif "999번" in msg_preview and msg_type == "AIMessage":
        print("    => ★ [새로운 역사] (AI 답변: 999번)")
    elif "999번" in msg_preview and msg_type == "ToolMessage":
        print("    => ☆ [분기점] (조작된 도구 결과)")

    print("-" * 60)
    
"""출력:
=== [Step 1] 최초 실행 (Thread ID: d7b56785-d630-456a-9b8a-87ce34681643) ===
[chatbot] 
[tools] 채널 번호는 7번입니다.
[chatbot] 부산 MBC 방송국의 채널 번호는 7번입니다.

=== [Step 2] 과거 시점(Checkpoint) 탐색 ===
타임머신 목적지 발견! Parent Checkpoint ID: 1f1001d9-d840-6b72-8002-ba430210bccf

=== [Step 3] 과거 수정 및 새로운 분기 생성 (Fork) ===
새로운 분기(평행우주) 생성 완료!
New Checkpoint ID: 1f1001d9-df06-6506-8003-8a01a76e4aae

=== [Step 4] 새로운 타임라인 재생 (Replay) ===
[chatbot] 부산 MBC 방송국의 채널 번호는 999번입니다!

============================================================
=== [Step 5] 전체 체크포인트 조회 (Thread ID: d7b56785-d630-456a-9b8a-87ce34681643) ===
============================================================
총 저장된 체크포인트 개수: 7개

[1] Checkpoint ID: 1f1001d9-f007-6742-8004-d955732016b3
    - 생성 시점(작업): 2026-02-02T09:57:44.685531+00:00 / Next: ()
    - 마지막 메시지(AIMessage): "부산 MBC 방송국의 채널 번호는 999번입니다!"
    => ★ [새로운 역사] (AI 답변: 999번)
------------------------------------------------------------
[2] Checkpoint ID: 1f1001d9-df06-6506-8003-8a01a76e4aae
    - 생성 시점(작업): 2026-02-02T09:57:42.902497+00:00 / Next: ('chatbot',)
    - 마지막 메시지(ToolMessage): "채널 번호는 999번입니다!!! (과거조작됨)"
    => ☆ [분기점] (조작된 도구 결과)
------------------------------------------------------------
[3] Checkpoint ID: 1f1001d9-df02-6d66-8003-f1cdb5a003da
    - 생성 시점(작업): 2026-02-02T09:57:42.901076+00:00 / Next: ()
    - 마지막 메시지(AIMessage): "부산 MBC 방송국의 채널 번호는 7번입니다."
    => ★ [기존 역사] (AI 답변: 7번)
------------------------------------------------------------
[4] Checkpoint ID: 1f1001d9-d840-6b72-8002-ba430210bccf
    - 생성 시점(작업): 2026-02-02T09:57:42.192418+00:00 / Next: ('chatbot',)
    - 마지막 메시지(ToolMessage): "채널 번호는 7번입니다."
------------------------------------------------------------
[5] Checkpoint ID: 1f1001d9-d83d-66c0-8001-f6ad800b265e
    - 생성 시점(작업): 2026-02-02T09:57:42.191068+00:00 / Next: ('tools',)
    - 마지막 메시지(AIMessage): ""
------------------------------------------------------------
[6] Checkpoint ID: 1f1001d9-c941-6d06-8000-032884eb5a9b
    - 생성 시점(작업): 2026-02-02T09:57:40.619998+00:00 / Next: ('chatbot',)
    - 마지막 메시지(HumanMessage): "부산 MBC 방송국의 채널 번호를 검색해줘."
------------------------------------------------------------
[7] Checkpoint ID: 1f1001d9-c93e-6430-bfff-37c6f7cf8c2c
    - 생성 시점(작업): 2026-02-02T09:57:40.618546+00:00 / Next: ('__start__',)
    - 마지막 메시지(None): "메시지 없음"
------------------------------------------------------------
"""
```

즉, [7]-[6]-[5]-[4]-[3] 까지는 처음 **`사용자 입력 -> tool_calls -> tools -> chatbot을 통한 답변`** 의 과정입니다.

그리고,

```python
for state in history:
    if state.next == ('chatbot',) and state.values["messages"] and isinstance(state.values["messages"][-1], ToolMessage):
        checkpoint_to_fork = state
        break
```

여기서 다음 챗봇이 답변하고 그 당시 새롭게 생성한 답변이 ToolMessage인 곳을<br>체크포인트로 가져옵니다.

&nbsp;

이후 툴 결과를 수정합니다.

```python
last_tool_msg = checkpoint_to_fork.values["messages"][-1]
new_tool_message = ToolMessage(
    id=last_tool_msg.id,
    content="채널 번호는 999번입니다!!! (과거조작됨)",
    tool_call_id=last_tool_msg.tool_call_id,
)

# 과거 Config를 기반으로 상태 업데이트 -> 새로운 분기 Config 반환
branch_config = graph.update_state(
    checkpoint_to_fork.config,  # ‼️ ⭐️ (매우 중요) 체크포인트의 config를 넣습니다. ⭐️ ‼️
    {"messages": [new_tool_message]},
    as_node="tools",
)
```

&nbsp;

그리고 그 **체크포인트 상황에서 다시 실행**하게 되면

<strong>(체크포인트로 수정된 config를 넣어줘야 함. config 값을 그대로 넣으면 이미 END라서 이미 있는 답변 그대로 나옴)</strong>

```python
for event in graph.stream(
    None, branch_config
):  # ‼️ 체크포인트 상황에서 다시 실행됨 ‼️
    for key, value in event.items():
        if "messages" in value:
            print(f"[{key}] {value['messages'][-1].content}")
```

그 수정된 툴 결과를 바탕으로 `chatbot` 노드에서 답변을 생성합니다.





```toc

```
