"""FastAPI entrypoint for the insurance QA backend."""

import logging
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.graph import app_graph
from backend.logging_utils import setup_logging

load_dotenv()
setup_logging()
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
    """Incoming chat request from the frontend."""

    query: str = Field(..., description="사용자 질문")


class ChatResponse(BaseModel):
    """Normalized API response returned to the frontend."""

    answer: str
    logs: List[str]
    tasks: List[str] = []
    primary_task: Optional[str] = None
    request_id: str
    task_statuses: List[Dict[str, Any]] = []

    intent: Optional[str] = None
    task_candidates: List[str] = []
    required_tasks: List[str] = []
    concept_keywords: List[str] = []
    product_keywords: List[str] = []
    user_filters: Dict[str, Any] = {}
    resolved_concepts: List[Dict[str, Any]] = []
    task_plan: List[Dict[str, Any]] = []
    task_results: List[Dict[str, Any]] = []
    response_sections: List[Dict[str, Any]] = []
    guarded_sections: List[Dict[str, Any]] = []
    plan_candidates: List[Dict[str, Any]] = []
    allowed_entities: Dict[str, Any] = {}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Run the LangGraph workflow for a single user question."""
    request_id = uuid4().hex[:12]
    started = time.perf_counter()
    logger.info("chat request started", extra={"request_id": request_id, "query": req.query})
    try:
        initial_state: Dict[str, Any] = {
            "question": req.query,
            "request_id": request_id,
            "intent": "",
            "tasks": [],
            "task_candidates": [],
            "required_tasks": [],
            "concept_keywords": [],
            "product_keywords": [],
            "analysis_notes": [],
            "user_filters": {},
            "resolved_concepts": [],
            "task_plan": [],
            "task_results": [],
            "response_sections": [],
            "guarded_sections": [],
            "plan_candidates": [],
            "allowed_entities": {},
            "final_answer": "",
            "trace_log": [],
            "node_models": {},
            "execution_metrics": {},
        }

        config = {
            "run_name": "planner_chat",
            "tags": ["env:dev", "architecture:planner", "version:4.0"],
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
        logger.info(
            "chat request finished",
            extra={
                "request_id": request_id,
                "intent": result.get("intent"),
                "tasks": tasks,
                "total_ms": total_ms,
                "task_count": len(tasks),
                "plan_candidate_count": len(result.get("plan_candidates", []) or []),
            },
        )
        return ChatResponse(
            answer=result.get("final_answer", "답변을 생성하지 못했습니다."),
            logs=result.get("trace_log", []) or [],
            tasks=tasks,
            primary_task=tasks[0] if tasks else None,
            request_id=request_id,
            task_statuses=task_statuses,
            intent=result.get("intent"),
            task_candidates=result.get("task_candidates", []) or [],
            required_tasks=result.get("required_tasks", []) or [],
            concept_keywords=result.get("concept_keywords", []) or [],
            product_keywords=result.get("product_keywords", []) or [],
            user_filters=result.get("user_filters", {}) or {},
            resolved_concepts=result.get("resolved_concepts", []) or [],
            task_plan=result.get("task_plan", []) or [],
            task_results=task_results,
            response_sections=result.get("response_sections", []) or [],
            guarded_sections=result.get("guarded_sections", []) or [],
            plan_candidates=result.get("plan_candidates", []) or [],
            allowed_entities=result.get("allowed_entities", {}) or {},
        )
    except Exception as exc:
        logger.exception("chat request failed", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8080, reload=True)
