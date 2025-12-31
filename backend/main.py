import os
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks  #BackgroundTasks 온라인 평가를 위해 추가
###### 온라인 평가를 위해 추가 ######
import random
from pathlib import Path
import json
##################################
from pydantic import BaseModel, Field
from typing import List, Optional
from backend.graph import app_graph
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware  # <-- 이 부분 추가(Frontend(5173 포트)가 Backend(8080 포트)에 요청을 보내려면 CORS 설정이 필수)



load_dotenv()

app = FastAPI(title="AgensGraph Agent API")

# --- CORS Middleware 추가 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시에는 ["http://localhost:5173"] 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic V2 Model
class ChatRequest(BaseModel):
    query: str = Field(..., description="사용자 질문")
    graph_schema: str = Field(default="Node: Person(name)... Edge: KNOWS...", description="Graph 스키마")
    sql_schema: str = Field(default="CREATE TABLE logs...", description="SQL 스키마")

class ChatResponse(BaseModel):
    answer: str
    logs: List[str]

# # 문서 업로드 요청 모델
# class DocumentRequest(BaseModel):
#     texts: List[str] = Field(..., description="저장할 텍스트 리스트")
#     metadatas: Optional[List[dict]] = Field(None, description="각 텍스트의 메타데이터 (옵션)")

# @app.post("/documents", summary="지식 문서 업로드 (Vector DB)")
# async def upload_documents(req: DocumentRequest):
#     """
#     Qdrant Vector DB에 문서를 임베딩하여 저장합니다.
#     """
#     try:
#         count = add_texts_to_vector_db(req.texts, req.metadatas)
#         return {"status": "success", "added_count": count}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))




################################  온라인 평가  #################################
# =========================
# 4/5 온라인 평가(운영 모니터링) 설정
# =========================
# - ENABLE_ONLINE_CAPTURE: /chat에서 평가용 payload를 "큐(파일)"에 적재할지 여부
# - ONLINE_EVAL_SAMPLE_RATE: 정상 케이스에서 랜덤 샘플링 비율 (예: 0.05 = 5%)
# - ONLINE_EVAL_CTXCHARS_MIN: ctx_chars가 이 값보다 작으면(컨텍스트 급감) 100% 수집
# - ONLINE_EVAL_QUEUE_PATH: 큐 파일 경로
#
# ⚠️ 중요:
# - ENABLE_ONLINE_CAPTURE=false면 "아예 안 쌓임" (큐 파일에 적재 자체를 안 함)
# - ENABLE_ONLINE_CAPTURE=true면 "수집만" 함 (RAGAS 평가는 worker가 돌 때만 발생)
ENABLE_ONLINE_CAPTURE = os.getenv("ENABLE_ONLINE_CAPTURE", "false").lower() == "true"
ONLINE_EVAL_SAMPLE_RATE = float(os.getenv("ONLINE_EVAL_SAMPLE_RATE", "0.05"))
ONLINE_EVAL_CTXCHARS_MIN = int(os.getenv("ONLINE_EVAL_CTXCHARS_MIN", "1000"))
ONLINE_EVAL_QUEUE_PATH = Path(os.getenv("ONLINE_EVAL_QUEUE_PATH", "eval/online_queue.jsonl"))

# (선택) 큐 파일 무한 증가 방지용 상한 (MB)
ONLINE_QUEUE_MAX_MB = int(os.getenv("ONLINE_QUEUE_MAX_MB", "100"))  # 기본 100MB
ONLINE_QUEUE_MAX_BYTES = ONLINE_QUEUE_MAX_MB * 1024 * 1024


def _calc_contexts_and_stats(result: dict) -> tuple[list[str], int, int]:
    """
    result(state)에서 contexts(list[str]), ctx_chars(총 글자수), k(문서개수)를 계산
    - contexts는 retrieved_docs[].text 기준으로 통일 (3/5와 동일)
    """
    docs = result.get("retrieved_docs") or []
    contexts = [d.get("text") for d in docs if d.get("text")]
    ctx_chars = sum(len(c) for c in contexts)
    k = len(docs)
    return contexts, ctx_chars, k


def _queue_is_too_large(path: Path) -> bool:
    """
    큐 파일이 너무 커지면(디스크 무한 증가) 수집을 자동 차단할 때 사용
    """
    try:
        if not path.exists():
            return False
        return path.stat().st_size >= ONLINE_QUEUE_MAX_BYTES
    except Exception:
        # stat 실패 시 보수적으로 "크다"로 처리하지 않고 통과
        return False


def should_capture_for_online_eval(result: dict) -> bool:
    """
    ✅ Rule + Random 샘플링
    - "전 요청 trace(LangSmith)"는 이미 1/5로 전수 저장되고,
    - 여기서는 "RAGAS 평가용 payload를 큐에 쌓을지"만 결정한다.
    """
    # 0) 전체 스위치: 수집 자체를 끄면 무조건 False
    if not ENABLE_ONLINE_CAPTURE:
        return False
    # 1) 큐 파일이 너무 커졌으면 수집 중단 (운영 안전장치)
    if _queue_is_too_large(ONLINE_EVAL_QUEUE_PATH):
        return False
    # 2) 문제 가능성 높은 케이스는 100% 수집
    contexts, ctx_chars, k = _calc_contexts_and_stats(result)
    # error가 있으면 무조건 수집
    if result.get("error"):
        return True
    # retriever 실패(컨텍스트 없음)도 무조건 수집
    if len(contexts) == 0:
        return True
    # 컨텍스트 급감(운영에서 매우 중요한 이상신호) → 무조건 수집
    if ctx_chars < ONLINE_EVAL_CTXCHARS_MIN:
        return True
    # 3) 나머지 정상 케이스는 랜덤 샘플링 (예: 5%)
    return random.random() < ONLINE_EVAL_SAMPLE_RATE


def enqueue_eval_payload(payload: dict) -> None:
    """
    ✅ (가벼운 시작) 파일 기반 큐에 append
    - 이 함수는 BackgroundTasks로 호출되어 /chat 응답 지연을 최소화한다.
    - 파일은 디스크에 남기 때문에 서버 재시작해도 보존됨(삭제하지 않는 한).
    """
    ONLINE_EVAL_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ONLINE_EVAL_QUEUE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


###############################################################################


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, background_tasks: BackgroundTasks):
    try:
        # Pydantic V2: req.dict() -> req.model_dump()
        # 필요한 필드만 추출하여 초기 상태 구성
        inputs = {
            "question": req.query,
            "graph_schema": req.graph_schema,
            "sql_schema": req.sql_schema,
            "mode": "vector",
            "generated_query": "",
            "query_result": "",
            "context": [],
            "evaluation": {},
            "final_answer": "",
            "retry_count": 0,
            "trace_log": [],
            "retrieved_docs": [],   # ✅ 추가 (KeyError 방지 + 평가 재료 자리)
            "node_models": {},      # ✅ 추가 (노드별 모델 기록 자리)
            "error": None,          # ✅ (선택) AgentState에 있으면 초기화 권장
        }
        
            # ✅ LangSmith에서 보기 편하게 “실행 이름/메타”를 주는 패턴
        config = {
            "run_name": "chat_request",  # LangSmith에서 run 이름으로 노출
            "tags": ["env:dev", "entry:chat"],
            "metadata": {
                "has_gt": False,
                "app": "agens-graphagent",
                # (선택) 온라인 평가 수집 on/off 상태도 남기면 운영 디버깅에 좋음
                "online_capture": ENABLE_ONLINE_CAPTURE,
            }

        }

        result = await app_graph.ainvoke(inputs, config=config)  #비동기(LangGraph의 비동기 실행 메서드)

        # =========================
        # ✅ 4/5: 온라인 평가용 payload "수집" (RAGAS 실행은 여기서 안 함)
        # =========================
        if should_capture_for_online_eval(result):
            contexts, ctx_chars, k = _calc_contexts_and_stats(result)

            payload = {
                "question": inputs["question"],
                "answer": result.get("final_answer"),
                "contexts": contexts,                 # ✅ RAGAS 입력용
                "mode": result.get("mode"),
                "node_models": result.get("node_models") or {},
                "meta": {
                    "k": k,
                    "ctx_chars": ctx_chars,           # ✅ 컨텍스트 급감 탐지 핵심값
                    "error": result.get("error"),
                },
                # (선택) 추후 trace와 조인하고 싶으면 request_id 같은 것을 넣는 게 좋음
                # "request_id": ...
            }

        # =========================
        # ✅ 응답 로그는 요약만 (마지막 10줄)
        # =========================
        logs = result.get("trace_log", []) or []
        return ChatResponse(
            answer=result.get("final_answer", "No answer"),
            # logs=result.get("trace_log", [])
            logs=logs[-10:]   # ✅ 마지막 10줄만
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 직접 실행 시 디버깅용
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8080, reload=True)