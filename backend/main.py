# backend/main.py
import logging
import time
import traceback
from typing import Any, Dict, List, Optional
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.graph import app_graph

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(title="AgensGraph Agent API (Planner-based)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = Field(..., description="사용자 질문")


class ChatResponse(BaseModel):
    answer: str
    logs: List[str]
    tasks: List[str] = []
    primary_task: Optional[str] = None
    request_id: str
    task_statuses: List[Dict[str, Any]] = []


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    request_id = uuid4().hex[:12]
    started = time.perf_counter()
    logger.info("[/chat][%s] query=%s", request_id, req.query)
    try:
        initial_state: Dict[str, Any] = {
            "question": req.query,
            "request_id": request_id,
            "tasks": [],
            "concept_keywords": [],
            "product_keywords": [],
            "analysis_notes": [],
            "resolved_concepts": [],
            "task_plan": [],
            "task_results": [],
            "response_sections": [],
            "final_answer": "",
            "trace_log": [],
            "node_models": {},
            "execution_metrics": {},
        }

        config = {
            "run_name": "planner_chat",
            "tags": ["env:dev", "architecture:planner", "version:3.0"],
            "metadata": {"user_id": "demo_user"},
        }

        result = await app_graph.ainvoke(initial_state, config=config)
        tasks = result.get("tasks", []) or []

        task_results = result.get("task_results", []) or []
        task_statuses = [
            {
                "task_type": x.get("task_type"),
                "status": x.get("status"),
                "evidence_count": x.get("evidence_count", len(x.get("evidence", []))),
                "duration_ms": x.get("duration_ms"),
            }
            for x in task_results
        ]
        total_ms = int((time.perf_counter() - started) * 1000)
        logger.info("[/chat][%s] done tasks=%s total_ms=%s", request_id, tasks, total_ms)
        return ChatResponse(
            answer=result.get("final_answer", "답변을 생성하지 못했습니다."),
            logs=result.get("trace_log", []) or [],
            tasks=tasks,
            primary_task=tasks[0] if tasks else None,
            request_id=request_id,
            task_statuses=task_statuses,
        )

    except Exception as exc:
        logger.exception("[/chat][%s] failed: %s", request_id, exc)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8080, reload=True)
