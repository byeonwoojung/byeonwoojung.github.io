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

1. retriever를 그냥 호출하게 되면 page_content만 반환하므로 프롬프트 템플릿에 넣어야 한다.











```toc

```
