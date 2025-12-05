import os
import uvicorn
from fastapi import FastAPI, HTTPException
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
    allow_origins=["http://localhost:5173"],  # 실제 배포 시에는 ["http://localhost:5173"] 권장
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

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        # Pydantic V2: req.dict() -> req.model_dump()
        # 필요한 필드만 추출하여 초기 상태 구성
        inputs = {
            "question": req.query,
            "graph_schema": req.graph_schema,
            "sql_schema": req.sql_schema,
            "mode": "",
            "retry_count": 0,
            "trace_log": []
        }
        
        result = await app_graph.ainvoke(inputs)
        
        return ChatResponse(
            answer=result.get("final_answer", "No answer"),
            logs=result.get("trace_log", [])
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 직접 실행 시 디버깅용
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8080, reload=True)