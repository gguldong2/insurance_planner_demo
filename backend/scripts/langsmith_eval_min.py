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

    # ✅ contexts는 "retrieved_docs.text"에서 구성
    contexts = []
    for d in (out_state.get("retrieved_docs") or []):
        txt = d.get("text")
        if txt:
            contexts.append(txt)


    # return {
    #     "answer": out.get("final_answer", ""),
    #     "contexts": [d.get("text") for d in (out.get("retrieved_docs") or []) if d.get("text")],
    #     "raw": out,
    # }
    ##################### 수정 ##################
    # 수정 포인트: offline_ragas_eval.py가 쓰는 키가 answer=final_answer, contexts=contexts 형태였으니, LangSmith evaluator도 그걸 그대로 먹게 만드는 게 안정적
    return {
        "final_answer": out_state.get("final_answer", ""),
        "contexts": contexts,
        "mode": out_state.get("mode"),
        "trace_log": out_state.get("trace_log", []),
    }


def ragas_evaluator(example: dict, outputs: dict) -> dict:
    """
    LangSmith code evaluator:
    - returns {"results":[{"key":..., "score":...}, ...]}
    """
    question = (example.get("inputs") or {}).get("question") or (example.get("inputs") or {}).get("query") or ""
    answer = outputs.get("final_answer", "") or ""
    contexts = outputs.get("contexts") or []

    # contexts 없으면 RAGAS 계산 불가 → 스킵
    if not contexts:
        return {
            "results": [
                {"key": "ragas_skipped", "value": "empty_contexts"}
            ]
        }

    from ragas import evaluate
    from ragas.metrics import context_precision, faithfulness, answer_relevancy

    dataset = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }

    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, faithfulness, answer_relevancy],
    )

    df = result.to_pandas()
    row = df.iloc[0].to_dict()

    # LangSmith에 여러 점수 컬럼으로 노출
    results = []
    for k, v in row.items():
        try:
            results.append({"key": f"ragas_{k}", "score": float(v)})
        except Exception:
            pass

    return {"results": results}


def format_strict_evaluator(inputs: dict, outputs: dict) -> dict:
    ans = outputs.get("answer", "") or ""
    ok = all(k in ans for k in ["1) 결론:", "2) 근거:", "3) 예외/주의:"]) and ("[출처:" in ans) and ("제" in ans and "조" in ans)
    return {"key": "format_strict", "score": 1.0 if ok else 0.0}

async def main():
    dataset_name = "seoultech-controlled"  # LangSmith에 만든 dataset 이름
    await aevaluate(
        target,
        data=dataset_name,
        evaluators= [format_strict_evaluator, 
                    ragas_evaluator],   # ✅ 추가(RAGAS 평가)
        experiment_prefix="controlled_ragas",   # 이름 취향
    )

if __name__ == "__main__":
    asyncio.run(main())
