# scripts/offline_ragas_eval.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

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
    from ragas.metrics import context_precision, faithfulness, answer_relevancy

    dataset = {
        "question": [r["question"] for r in eval_rows],
        "answer": [r.get("final_answer", "") for r in eval_rows],
        "contexts": [r.get("contexts") or [] for r in eval_rows],
    }

    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, faithfulness, answer_relevancy],
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
        })

    print(f"✅ Saved metrics: {out_path}")
    print(f"   evaluated={len(eval_rows)}, skipped_empty_contexts={len(skipped)}")

if __name__ == "__main__":
    main()
