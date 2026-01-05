# scripts/offline_ragas_eval.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
from datasets import Dataset  # ✅ 추가

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._utils import read_jsonl, append_jsonl, env_bool

RUNS_DIR = Path("eval/runs")
METRICS_DIR = Path("eval/metrics")

def main():
    run_tag = os.getenv("RUN_TAG")
    if not run_tag:
        raise ValueError("RUN_TAG is required. e.g., RUN_TAG=run_A")

    in_path = RUNS_DIR / f"{run_tag}_outputs.jsonl"
    out_path = METRICS_DIR / f"{run_tag}_metrics.jsonl"

    rows = read_jsonl(in_path)

    # --- build ragas dataset (skip empty contexts) ---
    eval_rows = []
    skipped = []
    for r in rows:
        contexts = r.get("contexts") or []
        if len(contexts) == 0:
            skipped.append(r)
            continue
        eval_rows.append(r)

    # ✅ 스킵만 기록(운영 KPI로 중요)
    for r in skipped:
        append_jsonl(out_path, {
            "id": r.get("id"),
            "type": r.get("type"),
            "question": r.get("question"),
            "meta": r.get("meta") or {},
            "scores": {},
            "skipped": True,
            "skip_reason": "empty_contexts",
        })

    if len(eval_rows) == 0:
        print(f"⚠️ No rows to evaluate (all empty contexts). Saved skips: {out_path}")
        return

    # --- RAGAS ---
    # ⚠️ ragas 버전별 API 차이가 있을 수 있음
    from ragas import evaluate

    # ✅ [변경] ragas 버전에 따라 metrics import 경로가 바뀔 수 있어서 fallback 처리
    try:
        # (구버전/일부 버전)
        from ragas.metrics import context_precision, faithfulness, answer_relevancy
    except Exception:
        # (신버전 계열에서 collections로 이동한 경우)
        from ragas.metrics.collections import context_precision, faithfulness, answer_relevancy

    # ✅ [추가] .env에서 "평가용 LLM"을 명시적으로 고정하기 위한 설정 읽기
    eval_model = os.getenv("RAGAS_EVAL_MODEL", "gpt-4o-mini")
    eval_temp_raw = os.getenv("RAGAS_EVAL_TEMPERATURE", "0")
    try:
        eval_temp = float(eval_temp_raw)
    except Exception:
        eval_temp = 0.0

    # ✅ [추가] RAGAS 권장 패턴: llm_factory로 LLM 객체 만들고 evaluate(llm=...)로 주입
    # - OPENAI_API_KEY는 일반적으로 환경변수로 세팅되어 있어야 함
    # - base_url이 필요하면 OPENAI_BASE_URL(or custom) 환경변수로도 통제 가능
    from openai import OpenAI
    from ragas.llms import llm_factory

    # (선택) OpenAI 클라이언트에 base_url/timeout 등 주고 싶으면 여기서 지정
    # 예: base_url = os.getenv("OPENAI_BASE_URL") or None
    client = OpenAI()

    # provider는 환경에 따라 생략 가능하지만, 명시하면 더 안전한 편
    llm = llm_factory(
        eval_model,
        provider="openai",
        client=client,
        temperature=eval_temp,
    )

    print(f"🧪 RAGAS eval model fixed: model={eval_model}, temperature={eval_temp}")

    dataset_dict  = {
        "question": [r["question"] for r in eval_rows],
        "answer": [r.get("final_answer", "") for r in eval_rows],
        "contexts": [r.get("contexts") or [] for r in eval_rows],
    }
    dataset = Dataset.from_dict(dataset_dict)  # ✅ dict -> Dataset

    # ✅ [변경] evaluate()에 llm=llm 주입 → 평가 모델이 확실히 고정됨
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy], # context_precision는 일단 배제(GT 없으므로)
        llm=llm,
    )
    df = result.to_pandas()

    for i, r in enumerate(eval_rows):
        scores = {}
        for col in df.columns:
            # metric 컬럼만 추출 (비정상 값 방지)
            try:
                scores[col] = float(df.loc[i, col])
            except Exception:
                pass

        append_jsonl(out_path, {
            "id": r.get("id"),
            "type": r.get("type"),
            "question": r.get("question"),
            "meta": r.get("meta") or {},
            "scores": scores,
            "skipped": False,
            "skip_reason": None,
            # ✅ (선택) 나중에 “이 결과가 어떤 평가모델로 나온건지” 추적용으로 남기고 싶으면:
            # "eval_model": eval_model,
            # "eval_temperature": eval_temp,
        })

    print(f"✅ Saved metrics: {out_path}")
    print(f"   evaluated={len(eval_rows)}, skipped_empty_contexts={len(skipped)}")

if __name__ == "__main__":
    main()
