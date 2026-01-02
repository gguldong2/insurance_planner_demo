# scripts/langsmith_eval_min.py
import asyncio
from backend.graph import app_graph
from langsmith import aevaluate

async def target(inputs: dict) -> dict:
    # inputs: {"question": "..."}
    state = {
        "question": inputs["question"],
        "graph_schema": "",
        "sql_schema": "",
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
    }
    out = await app_graph.ainvoke(state, config={"tags": ["langsmith:eval"]})
    return {
        "answer": out.get("final_answer", ""),
        "contexts": [d.get("text") for d in (out.get("retrieved_docs") or []) if d.get("text")],
        "raw": out,
    }

def format_strict_evaluator(inputs: dict, outputs: dict) -> dict:
    ans = outputs.get("answer", "") or ""
    ok = all(k in ans for k in ["1) 결론:", "2) 근거:", "3) 예외/주의:"]) and ("[출처:" in ans) and ("제" in ans and "조" in ans)
    return {"key": "format_strict", "score": 1.0 if ok else 0.0}

async def main():
    dataset_name = "seoultech-controlled"  # LangSmith에 만든 dataset 이름
    await aevaluate(
        target,
        data=dataset_name,
        evaluators=[format_strict_evaluator],
    )

if __name__ == "__main__":
    asyncio.run(main())
