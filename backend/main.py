# backend/main.py
import os
import uvicorn
import uuid
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

# 우리가 만든 LangGraph 앱 임포트
from backend.graph import app_graph
from backend.logging_utils import get_logger, setup_logging
from dotenv import load_dotenv

load_dotenv()
setup_logging()
logger = get_logger(__name__)

app = FastAPI(title="AgensGraph Agent API (Neuro-Symbolic)")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    query: str = Field(..., description="사용자 질문")
    # 기존의 schema 필드들은 더 이상 필요 없으므로 제거하거나 Optional로 둠
    
class ChatResponse(BaseModel):
    answer: str
    logs: List[str]
    intent: Optional[str] = None
    request_id: Optional[str] = None

# --- (선택사항) 온라인 평가 로직은 유지하되, 새 State에 맞게 조정 필요 ---
# 여기서는 핵심 채팅 로직 위주로 작성합니다.

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    request_id = uuid.uuid4().hex[:12]
    started_at = time.perf_counter()
    try:
        initial_state = {
            "question": req.query,
            "intent": "",
            "keywords": [],
            "concept_id": None,
            "context": [],
            "final_answer": "",
            "trace_log": [],
            "node_models": {},
            "request_id": request_id,
        }
        logger.info("chat request started", extra={
            "request_id": request_id,
            "query_len": len(req.query),
        })

        config = {
            "run_name": "neuro_symbolic_chat",
            "tags": ["env:dev", "version:2.0"],
            "metadata": {"user_id": "demo_user", "request_id": request_id}
        }

        result = await app_graph.ainvoke(initial_state, config=config)

        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.info("chat request finished", extra={
            "request_id": request_id,
            "intent": result.get("intent"),
            "duration_ms": duration_ms,
            "log_count": len(result.get("trace_log", []) or []),
        })

        return ChatResponse(
            answer=result.get("final_answer", "죄송합니다. 답변을 생성하지 못했습니다."),
            logs=result.get("trace_log", []) or [],
            intent=result.get("intent"),
            request_id=request_id,
        )

    except Exception as e:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.exception("chat request failed", extra={
            "request_id": request_id,
            "duration_ms": duration_ms,
        })
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8080, reload=True)