import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import context_precision, faithfulness, answer_relevancy

# --- OpenAI evaluator (LLM-as-a-judge) ---
from langchain_openai import ChatOpenAI
import openai

# ragas 버전에 따라 wrapper/embeddings 경로가 달라질 수 있어 안전하게 처리
try:
    from ragas.llms import LangchainLLMWrapper
except Exception:
    LangchainLLMWrapper = None

try:
    # 일부 버전에서 제공
    from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
except Exception:
    RagasOpenAIEmbeddings = None


###############################################################################
# Paths & env
###############################################################################

# online_eval_worker.py 위치: backend/scripts/online_eval_worker.py
# backend 디렉토리: parents[1]
BACKEND_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BACKEND_DIR / ".env"

# backend/.env 로드
load_dotenv(ENV_PATH)

# 메인 서버와 경로를 맞추기 위해 동일 env 사용
QUEUE = Path(os.getenv("ONLINE_EVAL_QUEUE_PATH", str(BACKEND_DIR / "eval/online_queue.jsonl")))
OUT = Path(os.getenv("ONLINE_EVAL_OUT_PATH", str(BACKEND_DIR / "eval/online_metrics.jsonl")))

# evaluator 모델(심판 LLM)
RAGAS_EVAL_MODEL = os.getenv("RAGAS_EVAL_MODEL", "gpt-4o-mini")
RAGAS_EVAL_TEMPERATURE = float(os.getenv("RAGAS_EVAL_TEMPERATURE", "0"))

# 평가할 metric (reference-free)
METRICS = [context_precision, faithfulness, answer_relevancy]


###############################################################################
# Helpers
###############################################################################

def read_and_clear_queue(queue_path: Path) -> List[Dict[str, Any]]:
    """
    큐 파일을 읽고 삭제(비우기).
    - 서버 재시작/worker 재시작에도 남아있을 수 있으니, worker가 처리 후 비우는 방식 유지
    """
    if not queue_path.exists():
        return []

    lines = queue_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(l) for l in lines if l.strip()]

    # 큐 비우기
    queue_path.unlink()
    return rows


def append_out(out_path: Path, row: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_ragas_evaluators():
    """
    OpenAI evaluator LLM/Embeddings를 생성해서 metrics에 주입.
    - LangChain ChatOpenAI 사용
    - RAGAS metric 객체에 llm/embeddings 속성이 있으면 세팅
    """
    # OpenAI client (embeddings용; OPENAI_API_KEY는 .env에서 로드)
    openai_client = openai.OpenAI()

    # evaluator LLM
    lc_llm = ChatOpenAI(
        model=RAGAS_EVAL_MODEL,
        temperature=RAGAS_EVAL_TEMPERATURE,
    )

    if LangchainLLMWrapper is not None:
        evaluator_llm = LangchainLLMWrapper(lc_llm)
    else:
        # 버전에 따라 wrapper 없이도 동작하는 경우가 있음
        evaluator_llm = lc_llm

    # evaluator embeddings (metric이 요구할 때만)
    evaluator_embeddings = None
    if RagasOpenAIEmbeddings is not None:
        try:
            evaluator_embeddings = RagasOpenAIEmbeddings(client=openai_client)
        except TypeError:
            # 어떤 버전은 client 인자를 받지 않을 수 있어 fallback
            evaluator_embeddings = RagasOpenAIEmbeddings()

    # metrics에 주입
    for m in METRICS:
        if hasattr(m, "llm"):
            m.llm = evaluator_llm
        if evaluator_embeddings is not None and hasattr(m, "embeddings"):
            m.embeddings = evaluator_embeddings

    return evaluator_llm, evaluator_embeddings


###############################################################################
# Main
###############################################################################

def main():
    rows = read_and_clear_queue(QUEUE)
    if not rows:
        return

    # contexts 없는 건 스킵 (RAGAS는 contexts 기반 metric이므로)
    eval_rows = [r for r in rows if r.get("contexts") and r.get("answer") and r.get("question")]
    if not eval_rows:
        return

    # evaluator 세팅 (OpenAI)
    build_ragas_evaluators()

    # RAGAS 입력 dataset
    dataset = {
        "question": [r["question"] for r in eval_rows],
        "answer": [r["answer"] for r in eval_rows],
        "contexts": [r["contexts"] for r in eval_rows],
    }

    result = evaluate(
        dataset=dataset,
        metrics=METRICS,
    )
    df = result.to_pandas()

    for i, r in enumerate(eval_rows):
        scores = {col: float(df.loc[i, col]) for col in df.columns}
        append_out(
            OUT,
            {
                **r,
                "scores": scores,
                "evaluated_at": time.time(),
                "eval_meta": {
                    "ragas_eval_model": RAGAS_EVAL_MODEL,
                    "ragas_eval_temperature": RAGAS_EVAL_TEMPERATURE,
                    "metrics": [m.name if hasattr(m, "name") else str(m) for m in METRICS],
                },
            },
        )


if __name__ == "__main__":
    main()



## 주기적으로 실행 (cron / tmux / systemd)
# python scripts/online_eval_worker.py