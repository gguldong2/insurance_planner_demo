# 스트리밍 방식 전환 — 변경 사항 정리

## 개요

LLM 응답을 전체 생성 완료 후 한 번에 반환하던 방식(blocking)에서,
토큰이 생성되는 즉시 클라이언트로 흘려보내는 방식(SSE 스트리밍)으로 전환했다.

핵심 목표: **TTFT(Time To First Token) 단축** — 사용자가 첫 글자를 더 빠르게 볼 수 있도록.

---

## 변경 파일 목록

| 파일 | 변경 종류 |
|---|---|
| `backend/main.py` | `/chat/stream` SSE 엔드포인트 추가 |
| `frontend/src/App.jsx` | axios 블로킹 → fetch ReadableStream 스트리밍으로 교체 |
| `engine5/engine.py` | `query_engine_stream`, `query_engine_stream_with_metadata` 함수 추가 |

---

## 1. `backend/main.py`

### 변경 내용

기존 `/chat` 엔드포인트는 그대로 유지하고, 스트리밍 전용 엔드포인트 `/chat/stream`을 추가했다.

| | 기존 `/chat` | 신규 `/chat/stream` |
|---|---|---|
| LangGraph 실행 | `ainvoke()` | `astream_events()` |
| 응답 형식 | JSON 단일 반환 | SSE(`text/event-stream`) |
| 응답 타이밍 | 생성 완료 후 전송 | 토큰 생성 즉시 전송 |

### SSE 이벤트 타입

클라이언트는 두 가지 이벤트를 수신한다.

```
data: {"type": "chunk", "text": "토큰"}        ← LLM 생성 중 실시간 전송
data: {"type": "done", "answer": "전체답변", ...} ← 그래프 완료 후 메타데이터 포함
data: {"type": "error", "message": "..."}       ← 오류 발생 시
```

### 핵심 구현 방식

```python
from fastapi.responses import StreamingResponse
import json

@app.post("/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    async def event_generator():
        final_result = None
        async for event in app_graph.astream_events(initial_state, config, version="v2"):
            # generator 노드에서 LLM 토큰 생성 시마다 즉시 전송
            if (
                event["event"] == "on_chat_model_stream"
                and event.get("metadata", {}).get("langgraph_node") == "generator"
            ):
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    yield f"data: {json.dumps({'type': 'chunk', 'text': chunk.content})}\n\n"

            # 그래프 전체 완료 시 메타데이터 수집
            elif event["event"] == "on_chain_end" and not event.get("metadata", {}).get("langgraph_node"):
                output = event["data"].get("output", {})
                if isinstance(output, dict) and "final_answer" in output:
                    final_result = output

        if final_result:
            yield f"data: {json.dumps({'type': 'done', 'answer': final_result['final_answer'], ...})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```

### LangGraph 이벤트 구조

`astream_events(version="v2")`는 그래프 실행 중 다양한 이벤트를 yield한다.

| 이벤트 타입 | 발생 시점 | 사용 여부 |
|---|---|---|
| `on_chain_start` | 노드 실행 시작 | 미사용 |
| `on_chat_model_stream` | LLM 토큰 1개 생성 시마다 | **사용** (generator 노드만 필터링) |
| `on_chain_end` | 노드 또는 그래프 완료 | **사용** (langgraph_node 없는 것 = 그래프 전체 완료) |

---

## 2. `frontend/src/App.jsx`

### 변경 내용

| | 기존 | 변경 후 |
|---|---|---|
| HTTP 방식 | `axios.post('/chat')` | `fetch('/chat/stream')` + `ReadableStream` |
| 응답 처리 | 완료 후 `response.data.answer` 표시 | `chunk` 이벤트마다 메시지에 append |
| 상태 관리 | `isLoading` 하나 | `isLoading`(분석 중) + `isStreaming`(토큰 수신 중) 분리 |
| 전송 버튼 비활성화 | `isLoading` 중 | `isLoading \|\| isStreaming` 중 |

### axios 대신 fetch를 쓰는 이유

axios는 응답이 완전히 끝날 때까지 기다렸다가 전달하기 때문에 스트리밍 불가.
`fetch`의 `response.body.getReader()`는 바이트 단위로 실시간 수신이 가능하다.

### 핵심 구현 방식

```javascript
const response = await fetch(`${API_BASE}/chat/stream`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: userMessage.text }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = '';

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  buffer += decoder.decode(value, { stream: true });
  const lines = buffer.split('\n');
  buffer = lines.pop(); // 잘린 줄은 다음 청크와 합쳐서 처리

  for (const line of lines) {
    if (!line.startsWith('data: ')) continue;
    const event = JSON.parse(line.slice(6).trim());

    if (event.type === 'chunk') {
      // 토큰을 메시지 끝에 실시간 append
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1].text += event.text;
        return next;
      });
    } else if (event.type === 'done') {
      // 최종 답변으로 교체 + 메타데이터 저장
      setDebugData(event);
      setIsStreaming(false);
    }
  }
}
```

---

## 3. `engine5/engine.py`

### 변경 내용

기존 `query_engine` (blocking)은 그대로 유지하고, 스트리밍용 함수 2개를 추가했다.

### 추가된 함수

#### `query_engine_stream` — 토큰만 필요할 때

```python
from api.engine5.engine import query_engine_stream

async for chunk in query_engine_stream("질문"):
    print(chunk, end="", flush=True)
```

#### `query_engine_stream_with_metadata` — 메타데이터도 필요할 때

```python
from api.engine5.engine import query_engine_stream_with_metadata

meta, stream = await query_engine_stream_with_metadata("질문")
async for chunk in stream:
    print(chunk, end="", flush=True)

# 스트림 소진 후 meta 확인
# meta = {
#   "status": "success",
#   "metadata": {
#     "intent": ...,
#     "tasks": [...],
#     "task_statuses": [...],
#     "plan_candidates": [...]
#   }
# }
```

### engine4 인터페이스와의 비교

engine4의 `query_lightrag_streaming` / `query_lightrag_streaming_with_metadata`와 동일한 호출 패턴을 따른다.

| | engine4 (LightRAG) | engine5 (LangGraph) |
|---|---|---|
| 단순 스트리밍 함수 | `query_lightrag_streaming` | `query_engine_stream` |
| 메타데이터 포함 함수 | `query_lightrag_streaming_with_metadata` | `query_engine_stream_with_metadata` |
| 반환 타입 | `tuple[dict, AsyncIterator[str]]` | `tuple[dict, AsyncIterator[str]]` |
| 메타데이터 확정 시점 | LLM 생성 **전** (RAG 검색 완료 후) | LLM 생성 **후** (그래프 완료 후) |

> **주의**: engine5의 `meta_ref`는 스트림을 완전히 소진한 후에 유효한 값이 채워진다.

---

## 환경 설정 (`frontend/.env.local`)

프론트엔드가 백엔드를 바라보는 URL은 `.env.local`에서 관리한다.
이 파일은 git에 커밋되지 않으므로 각 환경에 맞게 직접 생성해야 한다.

```
# frontend/.env.local
VITE_API_BASE=https://<백엔드-터널-또는-도메인-주소>
```

기본값 (`frontend/.env`):

```
VITE_API_BASE=http://localhost:8080
```
