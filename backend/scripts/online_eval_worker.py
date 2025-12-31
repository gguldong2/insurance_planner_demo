import json
import time
from pathlib import Path

from ragas import evaluate
from ragas.metrics import context_precision, faithfulness, answer_relevancy

QUEUE = Path("eval/online_queue.jsonl")
OUT = Path("eval/online_metrics.jsonl")

def read_and_clear_queue():
    if not QUEUE.exists():
        return []
    rows = [json.loads(l) for l in QUEUE.read_text(encoding="utf-8").splitlines() if l.strip()]
    QUEUE.unlink()  # 큐 비우기
    return rows

def append_out(row):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    rows = read_and_clear_queue()
    if not rows:
        return

    # contexts 없는 건 스킵
    eval_rows = [r for r in rows if r.get("contexts")]

    if not eval_rows:
        return

    dataset = {
        "question": [r["question"] for r in eval_rows],
        "answer": [r["answer"] for r in eval_rows],
        "contexts": [r["contexts"] for r in eval_rows],
    }

    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, faithfulness, answer_relevancy],
    )
    df = result.to_pandas()

    for i, r in enumerate(eval_rows):
        scores = {col: float(df.loc[i, col]) for col in df.columns}
        append_out({
            **r,
            "scores": scores,
            "evaluated_at": time.time(),
        })

if __name__ == "__main__":
    main()




## 주기적으로 실행 (cron / tmux / systemd)
# python scripts/online_eval_worker.py