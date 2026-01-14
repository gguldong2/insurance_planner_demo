# backend/main.py
import os
import uvicorn
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# 우리가 만든 LangGraph 앱 임포트
from backend.graph import app_graph
from dotenv import load_dotenv

load_dotenv()

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
    intent: Optional[str] = None # 디버깅용: 분류된 의도 확인

# --- (선택사항) 온라인 평가 로직은 유지하되, 새 State에 맞게 조정 필요 ---
# 여기서는 핵심 채팅 로직 위주로 작성합니다.

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        # 1. 초기 상태 구성 (AgentState 스키마 준수)
        # graph.py의 AgentState 정의: question, intent, keywords, concept_id, context, final_answer, trace_log, node_models
        initial_state = {
            "question": req.query,
            "intent": "",           # Router가 채움
            "keywords": [],         # Router가 채움
            "concept_id": None,     # Retriever가 채움
            "context": [],          # Retriever가 채움
            "final_answer": "",     # Generator가 채움
            "trace_log": [],        # 각 노드가 채움
            "node_models": {}
        }
        
        # 2. LangGraph 실행 (Config 설정)
        config = {
            "run_name": "neuro_symbolic_chat",
            "tags": ["env:dev", "version:2.0"],
            "metadata": {"user_id": "demo_user"}
        }

        # ainvoke로 비동기 실행
        result = await app_graph.ainvoke(initial_state, config=config)

        # 3. 결과 반환
        return ChatResponse(
            answer=result.get("final_answer", "죄송합니다. 답변을 생성하지 못했습니다."),
            logs=result.get("trace_log", []) or [],
            intent=result.get("intent") # 프론트엔드에서 어떤 의도로 분류됐는지 확인 가능
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8080, reload=True)