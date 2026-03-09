# scripts/offline_run.py
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ✅ 루트 실행 기준으로 backend import가 깨지면 아래 한 줄로 보완
# (scripts/에서 실행해도 루트를 import path에 넣어줌)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._utils import now_tag, read_jsonl, append_jsonl, env_int, env_bool

# 프로젝트의 app_graph
from backend.graph import app_graph  # ✅ 이 경로가 너 프로젝트 기준

QUESTIONS_PATH = Path("eval/questions.jsonl")
RUNS_DIR = Path("eval/runs")

DEFAULT_K = env_int("OFFLINE_RETRIEVAL_K", 3)
CONCURRENCY = env_int("OFFLINE_CONCURRENCY", 3)  # 비동기 동시 작업 갯수
OFFLINE_TIMEOUT_SEC = env_int("OFFLINE_TIMEOUT_SEC", 120)  # 요청 타임아웃(초)
PROGRESS_EVERY = env_int("OFFLINE_PROGRESS_EVERY", 10)     # N개마다 진행 로그



def make_inputs(question: str, graph_schema: str = "", sql_schema: str = "") -> Dict[str, Any]:
    # 너 /chat과 동일한 초기 state를 유지 (KeyError 방지)
    return {
        "question": question,
        "graph_schema": graph_schema,
        "sql_schema": sql_schema,
        "mode": "vector",
        "generated_query": "",
        "query_result": "",
        "context": [],
        "evaluation": {},
        "final_answer": "",
        "retry_count": 0,
        "trace_log": [],
        "retrieved_docs": [],
        "node_models": {},
        "error": None,
        # (선택) retriever가 k를 state에서 보도록 만들어뒀다면
        # "retrieval_k": DEFAULT_K,
    }

def extract_contexts_from_state(state: Dict[str, Any]) -> List[str]:
    docs = state.get("retrieved_docs") or []
    contexts = []
    for d in docs:
        t = d.get("text")
        if t:
            contexts.append(t)
    return contexts

def calc_ctx_chars(contexts: List[str]) -> int:
    return sum(len(c) for c in contexts if c)

def extract_top_score(state: Dict[str, Any]) -> Optional[float]:
    docs = state.get("retrieved_docs") or []
    for d in docs:
        s = d.get("score")
        if isinstance(s, (int, float)):
            return float(s)
    return None

async def run_one(q: Dict[str, Any], run_name: str, tags: List[str]) -> Dict[str, Any]:
    inputs = make_inputs(q["question"])

    config = {
        "run_name": run_name,
        "tags": tags,
        "metadata": {
            "qid": q.get("id"),
            "type": q.get("type"),
            "has_gt": False,
            "offline": True,
        },
    }

    try:
        result = await asyncio.wait_for(
            app_graph.ainvoke(inputs, config=config),
            timeout=OFFLINE_TIMEOUT_SEC,
        )
        contexts = extract_contexts_from_state(result)
        ctx_chars = calc_ctx_chars(contexts)
        k = len(result.get("retrieved_docs") or [])
        mode = result.get("mode")
        top_score = extract_top_score(result)

        return {
            "id": q.get("id"),
            "type": q.get("type"),
            "question": q.get("question"),
            "mode": mode,
            "final_answer": result.get("final_answer"),
            "contexts": contexts,  # ✅ RAGAS 입력용
            "retrieved_docs": result.get("retrieved_docs") or [],
            "node_models": result.get("node_models") or {},
            "meta": {
                "k": k,
                "ctx_chars": ctx_chars,      # ✅ 컨텍스트 급감 탐지 핵심
                "top_score": top_score,
                "error": result.get("error"),
            },
            # 운영에서 너무 길면 여기 저장을 끄거나 마지막 30줄만 저장하는 방식 추천
            "trace_log": (result.get("trace_log") or [])[-30:],
        }
    except Exception as e:
        return {
            "id": q.get("id"),
            "type": q.get("type"),
            "question": q.get("question"),
            "mode": "vector",
            "final_answer": "",
            "contexts": [],
            "retrieved_docs": [],
            "node_models": {},
            "meta": {
                "k": 0,
                "ctx_chars": 0,
                "top_score": None,
                "error": f"{type(e).__name__}: {e}",
            },
            "trace_log": [],
        }

async def main():
    run_tag = os.getenv("RUN_TAG") or now_tag()  # ex) RUN_TAG=run_A_20251230_...
    out_path = RUNS_DIR / f"{run_tag}_outputs.jsonl"
    rows = read_jsonl(QUESTIONS_PATH)

    # --- resume: 이미 outputs에 기록된 id는 스킵 ---
    done_ids = set()
    if out_path.exists():
        prev = read_jsonl(out_path)
        for r in prev:
            if r.get("id"):
                done_ids.add(r["id"])

    if done_ids:
        rows = [r for r in rows if r.get("id") not in done_ids]
        print(f"↩️ Resume: skip already done={len(done_ids)}, remaining={len(rows)}")

    run_name = os.getenv("LANGSMITH_RUN_NAME", "offline_eval")
    base_tags = ["offline:true", f"run:{run_tag}"]

    sem = asyncio.Semaphore(CONCURRENCY)

    total = len(rows)
    done = 0
    lock = asyncio.Lock()
    PROGRESS_EVERY = env_int("OFFLINE_PROGRESS_EVERY", 10)  # 없으면 상단/다른 위치에서 선언
    
    async def runner(q):
        nonlocal done 
        async with sem:
            tags = base_tags + [f"type:{q.get('type','unknown')}"]
            rec = await run_one(q, run_name=run_name, tags=tags)
            append_jsonl(out_path, rec)

            async with lock:
                done += 1
                if done % PROGRESS_EVERY == 0 or done == total:
                    print(f"[offline_run] {done}/{total} saved -> {out_path}", flush=True)

    await asyncio.gather(*[runner(q) for q in rows])

    print(f"✅ Saved: {out_path} (n={len(rows)})")

if __name__ == "__main__":
    asyncio.run(main())
