# # scripts/offline_ragas_eval.py
# from __future__ import annotations

# import os
# import sys
# from pathlib import Path
# from typing import Any, Dict, List
# from dotenv import load_dotenv
# from datasets import Dataset  # ✅ 추가

# load_dotenv()
# ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(ROOT))

# from scripts._utils import read_jsonl, append_jsonl, env_bool

# RUNS_DIR = Path("eval/runs")
# METRICS_DIR = Path("eval/metrics")

# def main():
#     run_tag = os.getenv("RUN_TAG")
#     if not run_tag:
#         raise ValueError("RUN_TAG is required. e.g., RUN_TAG=run_A")

#     in_path = RUNS_DIR / f"{run_tag}_outputs.jsonl"
#     out_path = METRICS_DIR / f"{run_tag}_metrics.jsonl"

#     rows = read_jsonl(in_path)

#     # --- build ragas dataset (skip empty contexts) ---
#     eval_rows = []
#     skipped = []
#     for r in rows:
#         contexts = r.get("contexts") or []
#         if len(contexts) == 0:
#             skipped.append(r)
#             continue
#         eval_rows.append(r)

#     # ✅ 스킵만 기록(운영 KPI로 중요)
#     for r in skipped:
#         append_jsonl(out_path, {
#             "id": r.get("id"),
#             "type": r.get("type"),
#             "question": r.get("question"),
#             "meta": r.get("meta") or {},
#             "scores": {},
#             "skipped": True,
#             "skip_reason": "empty_contexts",
#         })

#     if len(eval_rows) == 0:
#         print(f"⚠️ No rows to evaluate (all empty contexts). Saved skips: {out_path}")
#         return

#     # --- RAGAS ---
#     from ragas import evaluate

#     # ✅ metrics import 경로 fallback
#     try:
#         from ragas.metrics import faithfulness, answer_relevancy
#     except Exception:
#         from ragas.metrics.collections import faithfulness, answer_relevancy

#     # ✅ 평가용 LLM 설정(.env)
#     eval_model = os.getenv("RAGAS_EVAL_MODEL", "gpt-4o-mini")
#     eval_temp_raw = os.getenv("RAGAS_EVAL_TEMPERATURE", "0")
#     try:
#         eval_temp = float(eval_temp_raw)
#     except Exception:
#         eval_temp = 0.0

#     # ✅ 임베딩 모델 설정(.env) — 없으면 기본값 사용
#     # 권장: text-embedding-3-small (비용/속도 균형)
#     emb_model = os.getenv("RAGAS_EMBED_MODEL", "text-embedding-3-small")

#     # ✅ RAGAS 권장: llm_factory / embedding_factory 로 “RAGAS가 기대하는 인터페이스”로 생성해 주입
#     from openai import OpenAI

#     from ragas.llms import llm_factory

#     # embedding_factory 경로는 버전 차이가 있어서 fallback로 안전하게 처리
#     # try:
#     #     from ragas.embeddings import embedding_factory
#     # except Exception:
#     #     from ragas.embeddings.base import embedding_factory

#     client = OpenAI()

#     llm = llm_factory(
#         eval_model,
#         provider="openai",
#         client=client,
#         temperature=eval_temp,
#     )

#     # # ✅ 핵심: embeddings를 명시적으로 만들어 evaluate에 주입
#     # embeddings = embedding_factory(
#     #     "openai",
#     #     model=emb_model,
#     #     client=client,
#     # )
#     # ✅ RAGAS 전용 OpenAIEmbeddings 사용 (embed_query 제공 기대)
#     try:
#         from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
#     except Exception:
#         try:
#             from ragas.embeddings.openai import OpenAIEmbeddings as RagasOpenAIEmbeddings
#         except Exception as e:
#             raise ImportError(
#                 "RAGAS OpenAIEmbeddings import 경로를 찾지 못했습니다. "
#                 "ragas 버전에 맞게 경로 조정이 필요합니다."
#             ) from e

#     embeddings = RagasOpenAIEmbeddings(
#         model=emb_model,
#         client=client,
#     )



#     print(f"🧪 RAGAS eval model fixed: model={eval_model}, temperature={eval_temp}")
#     print(f"🧩 RAGAS embed model fixed: model={emb_model}")

#     dataset_dict = {
#         "question": [r["question"] for r in eval_rows],
#         "answer": [r.get("final_answer", "") for r in eval_rows],
#         "contexts": [r.get("contexts") or [] for r in eval_rows],
#     }
#     dataset = Dataset.from_dict(dataset_dict)

#     # ✅ embeddings=embeddings 주입 (answer_relevancy가 임베딩을 쓰므로 필수)
#     result = evaluate(
#         dataset=dataset,
#         metrics=[faithfulness, answer_relevancy],
#         llm=llm,
#         embeddings=embeddings,
#     )
#     df = result.to_pandas()

#     for i, r in enumerate(eval_rows):
#         scores = {}
#         for col in df.columns:
#             try:
#                 scores[col] = float(df.loc[i, col])
#             except Exception:
#                 pass

#         append_jsonl(out_path, {
#             "id": r.get("id"),
#             "type": r.get("type"),
#             "question": r.get("question"),
#             "meta": r.get("meta") or {},
#             "scores": scores,
#             "skipped": False,
#             "skip_reason": None,
#             # "eval_model": eval_model,
#             # "eval_temperature": eval_temp,
#             # "embed_model": emb_model,
#         })

#     print(f"✅ Saved metrics: {out_path}")
#     print(f"   evaluated={len(eval_rows)}, skipped_empty_contexts={len(skipped)}")

# if __name__ == "__main__":
#     main()

##############################################################


# scripts/offline_ragas_eval.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
from datasets import Dataset

# ✅ LangChain 관련 라이브러리 명시적 import
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._utils import read_jsonl, append_jsonl

RUNS_DIR = Path("eval/runs")
METRICS_DIR = Path("eval/metrics")

def main():
    run_tag = os.getenv("RUN_TAG")
    if not run_tag:
        raise ValueError("RUN_TAG is required. e.g., RUN_TAG=run_A")

    in_path = RUNS_DIR / f"{run_tag}_outputs.jsonl"
    out_path = METRICS_DIR / f"{run_tag}_metrics.jsonl"

    rows = read_jsonl(in_path)

    # --- Build Ragas Dataset ---
    eval_rows = []
    skipped = []
    for r in rows:
        contexts = r.get("contexts") or []
        if len(contexts) == 0:
            skipped.append(r)
            continue
        eval_rows.append(r)

    # 스킵된 항목 기록
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

    # --- Ragas Metrics Import ---
    # 버전 호환성을 위해 try-except로 처리
    try:
        from ragas.metrics import faithfulness, answer_relevancy
    except ImportError:
        from ragas.metrics.collections import faithfulness, answer_relevancy

    # --- LLM & Embeddings Setup (핵심 수정) ---
    eval_model_name = os.getenv("RAGAS_EVAL_MODEL", "gpt-4o-mini")
    eval_temp = float(os.getenv("RAGAS_EVAL_TEMPERATURE", "0"))

    print(f"🧪 RAGAS eval setup: LLM={eval_model_name}, Embeddings=text-embedding-3-small")

    # 1. 평가용 LLM (LangChain Wrapper 사용 권장)
    lc_llm = ChatOpenAI(model=eval_model_name, temperature=eval_temp)
    evaluator_llm = LangchainLLMWrapper(lc_llm)

    # 2. 평가용 Embeddings (명시적 생성하여 에러 방지)
    # answer_relevancy 등에서 이 객체의 .embed_query()를 호출합니다.
    evaluator_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # --- Dataset 생성 ---
    dataset_dict = {
        "question": [r["question"] for r in eval_rows],
        "answer": [r.get("final_answer", "") for r in eval_rows],
        "contexts": [r.get("contexts") or [] for r in eval_rows],
        # GT가 있다면 여기에 "ground_truth" 추가
    }
    dataset = Dataset.from_dict(dataset_dict)

    # --- Evaluate ---
    from ragas import evaluate

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=evaluator_llm,            # ✅ 명시적 주입
        embeddings=evaluator_embeddings, # ✅ 명시적 주입 (AttributeError 해결)
    )
    df = result.to_pandas()

    # --- Save Results ---
    for i, r in enumerate(eval_rows):
        scores = {}
        for col in df.columns:
            if col in ["faithfulness", "answer_relevancy", "context_precision"]: # 필요한 컬럼만
                try:
                    scores[col] = float(df.loc[i, col])
                except Exception:
                    pass
        
        # Ragas가 생성한 기타 컬럼도 안전하게 포함하고 싶다면:
        # scores = {k: float(v) for k, v in df.iloc[i].to_dict().items() if isinstance(v, (int, float))}

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