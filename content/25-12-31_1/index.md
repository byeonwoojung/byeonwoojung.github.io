---
emoji: ☀️
title: "[LLM] 테디노트의 RAG 비법노트 끄적끄적-10"
date: '2025-12-31 00:00:00'
author: 변우중
tags: LLM 프롬프트 RAG LangChain 랭체인
categories: LLM
---
참고 : 테디노트의 RAG 비법노트 (https://fastcampus.co.kr/data_online_teddy)<br>소스코드: https://github.com/teddylee777/langchain-kr<br>위키독스: https://wikidocs.net/book/14314

&nbsp;

이번에는 '**LCEL에서 메모리 사용하는 방법**'을 간단히 정리하고,<br>'**SQLite DB를 이용한 메모리 저장하는 방법**'을 정리해보고자 합니다.

상당히 복잡해보여서 연습이 중요할 것 같습니다.

(**주의‼️ 마음의 준비를 하고, 심호흡 한번 크게 하고 들어가야 합니다.**)

레츠기릿~!

&nbsp;

## LCEL에서 메모리 사용하는 방법

---

1. **프롬프트 템플릿에 `MessagesPlaceholder(variable_name="chat_history")` 박아두기**!
2. **`memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")` 메모리 생성하면서<br>`memory_key`의 값과 `MessagesPlaceholder`에 들어가는 `variable_name` 값을 동일하게 하기**
   * **메모리에서 지정한 `memory_key`는 프롬프트 템플릿에서 이전 대화 자리에 박아두는 변수명과 같아야 함**

3. **`RunnablePassthrough.assign()`에서 `RunnableLambda(memory.load_memory_variables) | itemgetter(memory.memory_key)`을 `chat_history`에 지정한 `runnable`을 만들어 chain에서 이용함**

   ```python
   from langchain_core.runnables import RunnableLambda, RunnablePassthrough
   
   runnable = RunnablePassthrough.assign(  chat_history=RunnableLambda(memory.load_memory_variables)
       | itemgetter("chat_history")  # RunnableLambda(memory.load_memory_variables)의 반환값에서 chat_history만 추출
       # | itemgetter(memory.memory_key) # 위와 똑같음
   )
   
   runnable.invoke({"input": "hi"})
   # {'input': 'hi'}과 chat_history 값이 합쳐져 아래의 프롬프트 템플릿의 변수값에 들어갈 것임 ({'input': 'hi', 'chat_history': [대화들]} 꼴)
   ```

   * **`RunnableLambda(memory.load_memory_variables)`는 `memory.load_memory_variables({})`와 같은 것임**
     * 즉, `chat_history` 키의 값에 **대화들 리스트**를 담은 딕셔너리일 것임
   * 그리고, ⭐️ **`memory.load_memory_variables({})`의 결과값(대화들 담긴 딕셔너리)에서 `memory.memory_key`값('chat_history' 키의 값, 즉 대화들)만 추출해서 `chat_history` 변수에 전달함** ⭐️
     * `memory.load_memory_variables({})`<br>: {'chat_history': [대화들]} 꼴
     * `memory.memory_key`값('chat_history' 키의 값)만 추출<br>: [대화들] 꼴
     * `chat_history` 변수에 전달한 후의 최종 `RunnablePassthrough.assign()` 값<br>: {'chat_history': [대화들]}
     * `runnable.invoke({"input": "hi"})` 값<br>: `{'input': 'hi'}`과 `chat_history` 값이 합쳐져 `{'input': 'hi', 'chat_history': [대화들]}` 꼴이 됨

   &nbsp;

   ⭐️ **결국, runnable를 invoke 호출했을 때<br>`{'input': 'hi', 'chat_history': [대화들]}`이 되므로 각각 사용자 입력값과 이전 대화로 프롬프트 템플릿에서 이용할 수 있습니다!!**⭐️

   (좀 복잡하지만 그 과정을 명확히 알고 있어야 본격적으로 chain에 연결할 때 헷갈리지 않을 것 같습니다..)

&nbsp;

방금 만들었던 **이전 대화와 새로운 질문을 반환하는 `runnable`**을 가장 맨 앞에 붙여<br>그 결과를 프롬프트에 전달하는 'runnable-프롬프트-LLM-파서' 이렇게 chain을 구성하면 될 듯 합니다.

&nbsp;

&nbsp;

## SQLite DB를 이용한 메모리 저장 방법

---

이전까지는 인메모리에 캐시를 저장하는 방법을 정리해봤는데<br>사실, 대화 세션을 일회성으로 이용하는 것이면 데이터베이스에 저장하지 않아도 되긴 합니다.

하지만, 이전 대화를 나중에 다시 사용해야 한다면 데이터베이스에 유저별 테이블 형식으로 저장해두고, 그 기반으로 답변을 할 수 있도록 하는 것이 좋습니다.

&nbsp;

흐름부터 정리하겠습니다.

```
invoke() 호출
    ↓
config에서 user_id, session_id 추출 (history_factory_config 참조)
    ↓
get_chat_history(user_id, session_id) 호출 → ChatMessageHistory 반환
    ↓
chat_history를 프롬프트에 주입 (history_messages_key 사용)
    ↓
question을 프롬프트에 주입 (input_messages_key 사용)
    ↓
LLM 실행 → 응답 반환
    ↓
새 대화(질문+응답)를 ChatMessageHistory에 자동 저장
```

처음에 질문을 했을 때 config에서 어떠한 데이터베이스에서 어떠한 테이블에서 어떠한 컬럼값을 참고할지를 정합니다.

이후, 해당 config를 참고하여 이전 대화를 가져온 후 프롬프트에 넣습니다.

그리고, 처음 입력한 질문을 프롬프트에 넣으면서 LLM이 실행하고 응답을 받습니다.

마지막에 그 새로운 대화를 해당 데이터베이스의 테이블에 데이터로 저장합니다.

&nbsp;

코드로 자세하게 봅시다.

1. **`ConfigurableFieldSpec`**을 이용해 **DB 조회할 파라미터(컬럼명 등)를 설정**해두어야 합니다.

   ```python
   from langchain_core.runnables.utils import ConfigurableFieldSpec
   
   config_fields = [
       ConfigurableFieldSpec(
           id="user_id",  # ⭐️ get_chat_history 함수에서 사용할 파라미터 ⭐️
           annotation=str,
           name="User ID",
           description="Unique identifier for a user.",
           default="",
           is_shared=True,
       ),
       ConfigurableFieldSpec(
           id="conversation_id", # ⭐️ get_chat_history 함수에서 사용할 파라미터 ⭐️
           annotation=str,
           name="Conversation ID",
           description="Unique identifier for a conversation.",
           default="",
           is_shared=True,
       ),
   ]
   ```

   

2. **`get_chat_history`** 함수에서 **DB에서 사용할 파라미터명을 인자로 받아**, **`SQLChatMessageHistory`**을 이용해 **DB 메시지 히스토리를 관리**하도록 정의합니다.

   ```python
   def get_chat_history(user_id, conversation_id):
       return SQLChatMessageHistory(
           table_name=user_id,          # (유저 ID별) 테이블
           session_id=conversation_id,  # 대화 ID 컬럼
           connection="sqlite:///sqlite.db",  # 데이터베이스 파일명
       )
   ```

   * `table_name`: (보통 유저 ID별) 테이블명 설정
   * `session_id`: 테이블에서 session_id 값에 넣는 대화 세션 ID 값 설정
   * `connection`: 데이터베이스 파일명 설정

   

3. `RunnableWithMessageHistory()` 객체 생성하면서 **대화 내용을 기록해주는 함수를 기존 체인에 연결**해줍니다.

   ```python
   from langchain_core.runnables.history import RunnableWithMessageHistory
   
   chain_with_history = RunnableWithMessageHistory(
       chain,
       get_chat_history,  # 대화 기록을 가져오는 함수를 설정합니다.
       input_messages_key="question",  # 입력 메시지의 키를 "question"으로 설정 (⭐️ chain에서 prompt 템플릿의 입력 변수와 같아야 함 ⭐️)
       history_messages_key="chat_history",  # 대화 기록 메시지의 키를 "chat_history"로 설정 (⭐️ chain에서 prompt 템플릿-메시지플레이스홀더의 입력 변수와 같아야 함 ⭐️)
       history_factory_config=config_fields,  # 대화 기록 조회시 참고할 파라미터를 설정합니다. (⭐️ get_chat_history에서 사용하는 ConfigurableFieldSpece들을 전달하기 위한 것 ⭐️)
   )
   ```

   * `chain`: 기존 체인(프롬프트-LLM-파서 연결)
   * `get_chat_history`: DB에서 데이터베이스 파일명-대화 세션-(유저별) 테이블을 찾아가서 메시지 히스토리 관리함
   * `input_messages_key`: 기존 체인에서 프롬프트 템플릿에서 사용하는 입력변수 값을 넣음
   * `history_messages_key`: 기존 체인에서 프롬프트 템플릿에서 `MessagesPlaceholder`에서 사용하는 변수 값을 넣음
   * `history_factory_config`: 대화 기록 조회할 때 참고할 파라미터 정보들 설정

   

4. config에 configurable 키 하위에 get_chat_history에서  **DB 메시지 히스토리를 관리**할 때 사용하는 **파라미터(대화 세션 ID, 테이블명)들을 설정**합니다.

   ```python
   # config 설정
   """
   configurable 키 하위에 user_id, conversation_id 값을 설정
   (get_chat_history에서 DB 메시지 히스토리 관리할 때 사용하는 값들)
   """
   config = {"configurable": {"user_id": "user1", "conversation_id": "conversation1"}}
   ```

   * **table_name 테이블명=user_id=user1**
   * **session_id 컬럼 값=conversation_id=conversation1**

   

5. 앞에서 기존 체인과 이전 대화 기록 조회하는 것을 연결한 **`chain_with_history`에서 `invoke()` 메서드에 입력변수와 config을 전달하여 호출**합니다.

   ```python
   chain_with_history.invoke({"question": "내 이름이 뭐라고?"}, config)
   ```

&nbsp;

굉장히 복잡하고 어려워보이지만<br>갓 테디노트님 강의 들으니 이해하기 쉽더라고요,, 좋슴다....

&nbsp;

&nbsp;

여기서 끄읕.<br>다음은 문서 로드하는 방법인데, 간단히 정리하고 다른 것을 쓰고자 합니다! See ya.

```toc

```
