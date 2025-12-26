---
emoji: ☀️
title: "[LLM] 테디노트의 RAG 비법노트 끄적끄적-8"
date: '2025-12-25 00:00:00'
author: 변우중
tags: LLM 프롬프트 RAG Ollama 허깅페이스 LangChain 랭체인
categories: LLM
---
참고 : 테디노트의 RAG 비법노트 (https://fastcampus.co.kr/data_online_teddy)

소스코드: https://github.com/teddylee777/langchain-kr

위키독스: https://wikidocs.net/book/14314

&nbsp;

오늘은 맥 사용자(저요,,,)의 허깅페이스 모델을 로컬로 가져와`<br>`mps를 이용해 GPU 가속화하는 방법을 먼저 보고 가겠습니다~

레츠기리잇~!

&nbsp;

## Mac(MPS) 환경에서 HuggingFacePipeline 사용 시 주의사항

---

### `from_model_id` vs `pipeline` 직접 주입

LangChain을 사용하여 로컬 LLM을 구동할 때, 특히 Apple Silicon(M1/M2/M3) 환경에서 GPU 가속(MPS)을 설정하는 과정에서 `ValueError`가 발생하는 경우가 많습니다.

**이는 LangChain이 내부적으로 Hugging Face 모델을 로드하는 방식(`from_model_id`)과 실제 `transformers` 라이브러리의 `pipeline`이 장치(Device)를 처리하는 방식의 차이 때문입니다.**

&nbsp;

1. `transformers.pipeline`

   - **소속 라이브러리:** `transformers` (Hugging Face)
   - **정체:** 모델 추론(Inference)을 위한 **End-to-End 실행 엔진**입니다.
   - **역할:** 모델 다운로드, 토크나이저 로드, 입력 텍스트 전처리(Pre-processing), 모델 추론(Model Inference), 결과 후처리(Post-processing)의 전 과정을 수행합니다.
   - **Device 처리 특징:**

     - ⭐️ **`device` 인자로 문자열(`"cpu"`, `"cuda"`, `"mps"`)과 정수(GPU ID)를 모두 지원합니다.** ⭐️
2. `HuggingFacePipeline.from_model_id`

   - **소속 라이브러리:** `langchain_huggingface`
   - **정체:** LangChain에서 `HuggingFacePipeline` 객체를 쉽게 생성하기 위해 제공하는 **팩토리 메서드(Factory Method)이자 래퍼(Wrapper)**입니다.
   - **역할:** 내부적으로 `transformers.pipeline`을 호출하여 파이프라인을 생성하고, 이를 LangChain 객체로 감쌉니다.
   - **Device 처리 특징:**

     - 사용자의 편의를 위해 `device` 파라미터를 주로 정수형(Integer)으로 입력받도록 설계되었습니다. (예: `-1`은 CPU, ⭐️ **`0`은 첫 번째 CUDA GPU** ⭐️)
     - 이 과정에서 **유효성 검사 로직(Validation Logic)**이 포함되는데, 이 로직이 NVIDIA GPU(CUDA)를 기준으로 작성되어 있어 Mac(MPS) 환경과 호환성 충돌을 일으킬 수 있습니다.

&nbsp;

결국,

**맥북 사용자는 mps 사용하기 위해서는<br>`HuggingFacePipeline.from_model_id`가 아닌, `transformers.pipeline`을 이용하여야 합니다.**

```python
# ✅ mps는 허깅페이스 라이브러리인 transformers의 pipeline을 사용하여야 한다.

import torch
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

# 1. Transformers의 pipeline을 직접 생성합니다.
# 여기서 device="mps"를 문자열로 명시하면 에러 없이 강제로 MPS를 잡습니다.
pipe = pipeline(
    task="text-generation",
    model="microsoft/Phi-3-mini-4k-instruct",
    device="mps",  # <--- 핵심: 여기서 "mps"를 직접 지정
    model_kwargs={"torch_dtype": torch.float16}, # Mac에서 메모리 절약 및 속도 향상
    max_new_tokens=256,
    top_k=50,
    temperature=0.1,
    do_sample=True, # 경고 메시지 해결
)

# 2. 생성된 pipe를 LangChain 객체에 주입합니다.
llm = HuggingFacePipeline(pipeline=pipe)

# 3. 실행
print(llm.invoke("Hugging Face is"))
```

맥북 사용자(mps 이용)는 위 코드를 이용해야 하며,

```python
# CUDA 용
from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 256,
        "top_k": 50,
        "temperature": 0.1,
        "do_sample": True,
    },
    device=0   # from_model_id은 cuda를 0으로 설정되어 있어서 mps는 사용 불가능
)
llm.invoke("Hugging Face is")
```

위 코드는 맥북 사용자(mps 이용)는 사용할 수 없습니다.

OK.

&nbsp;

&nbsp;

## Ollama

---

Ollama는 복잡한 설정 없이 CLI(명령줄) 한 줄로 모델을 다운로드하고 실행할 수 있게 해줍니다. Hugging Face `transformers` 라이브러리를 사용하여 직접 모델을 로드해올 수 있지만 Ollama를 이용하면 편리하게 모델을 사용할 수 있습니다.

물론 AWS, GCP에서 EC2를 빌려와서 그 안에서 Ollama를 설치해서 사용 가능합니다.

&nbsp;

### Ollama를 사용하는 방법

**gguf 파일(허깅페이스에서 gguf를 제공하는 LLM)을 models 폴더에 넣어줍니다**.<br>

**models 폴더에 Modelfile 파일을 작성해주는데**

```
FROM (허깅페이스에서 가져온 모델명).gguf

TEMPLATE """(모델이 학습한 템플릿 내용)
"""

SYSTEM """(시스템 프롬프트)"""

PARAMETER stop <s>
PARAMETER stop </s>
```

이런 식으로 작성해줍니다. 그러면 허깅페이스에서 가져온 모델을 읽고 모델이 학습했던 템플릿과 시스템 프롬프트를 특정 명령어를 통해서 알아서 읽게 됩니다.

(\<s>, \</s>과 같은 것들을 스페셜 토큰이라 하며 모델이 어디까지 무엇인지 알아들을 수 있는 키워드입니다. `PARAMETER stop </s>`을 작성하지 않으면 혼잣말을 계속하거나 횡설수설하는 현상이 일어날 때가 있습니다.)

&nbsp;

### Ollama 관련 터미널 사용법 (가상환경과 무관)

1. `ollama` : Ollama가 작동하는지 알 수 있음
2. `ollama list`(또는 `ollama ls`): 내 로컬 Ollama 시스템 전용 폴더에 저장된 모델 목록을 보여줌
3. `ollama ps`: 현재 실행 중인(메모리에 올라간) 모델 목록을 보여줌
4. `ollama rm [모델명]`: 모델을 삭제하는 명령어임
5. `ollama pull [모델명:모델크기b]`: Ollama 라이브러리에 있는 모델을 내 로컬 `ollama list`에 등록
6. `ollama create [내가부르고싶은모델명] -f Modelfile`<br>: 내 로컬에 있는 모델을 내 로컬  `ollama list`에 등록<br>(`Modelfile` 파일에 있는 모델을 읽어 [내가부르고 싶은 모델명]으로 등록함)
7. `ollama run [모델명:모델크기b]`: 내 로컬  `ollama list`에 있는 모델을 메모리에 올림

&nbsp;

### langchain에서 ollama 사용법

`langchain_ollama`의 `ChatOllama` 모듈을 로컬 Ollama 시스템 전용 폴더에 저장된 모델명을 가져와 이용하면 됩니다.

먼저, 터미널에서 `ollama ls`로 모델명을 파악한 후

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="모델명", temperature=0.1, format='json')
```

이렇게 **`model` 파라미터에 해당 모델명을 넣어 생성**하면 해당 모델을 손쉽게 사용할 수 있습니다. (Chain 연결도 쌉가능.)

&nbsp;

그리고,

**`temperature`(답변의 창의성 정도)와 `format`(출력 형식), `num_predict`(최대 출력 토큰 수),  `top_k`(단어 선택의 다양성 정도),  `top_p`(누적 확률이 P가 될 때까지의 단어들만 후보군 넣음),  `repeat_penalty`(반복 방지),  `seed`(결과 재현성 확보),  `stop`(모델이 생성을 멈춰야 하는 문자열 지정), `keep_alive`(모델 메모리 유지 시간) 등을 설정**할 수도 있습니다.

&nbsp;

만약,

GPU를 잘 활용하고 있는지 보고 싶다면<br>코드가 스트리밍 중일 때 터미널에서 `ollama ps`를 입력해서 '100% GPU'인지 확인 가능합니다.

참고로, Ollama는 자동적으로 환경을 감지해서 GPU 가속을 수행합니다.

&nbsp;

&nbsp;

오늘은 여기까지 작성하는데,<br>Ollama 외에도 GPT4ALL 모델을 이용하여 LLM 모델을 로드하여 사용할 수 있습니다.(프로그램과 라이브러리 모두 존재함) 취향껏 써보면 될 듯 합니다.

끄읕.

```toc

```
