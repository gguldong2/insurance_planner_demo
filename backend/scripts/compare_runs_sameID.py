# scripts/compare_runs.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._utils import read_jsonl

METRICS_DIR = Path("eval/metrics")
RUNS_DIR = Path("eval/runs")

def load_metrics(run_tag: str) -> pd.DataFrame:
    p = METRICS_DIR / f"{run_tag}_metrics.jsonl"
    rows = read_jsonl(p)
    # scores는 dict → 컬럼으로 펼치기
    flat = []
    for r in rows:
        base = {
            "id": r.get("id"),
            "type": r.get("type"),
            "question": r.get("question"),
            "skipped": bool(r.get("skipped")),
            "skip_reason": r.get("skip_reason"),
        }
        meta = r.get("meta") or {}
        base["k"] = meta.get("k")
        base["ctx_chars"] = meta.get("ctx_chars")
        base["top_score"] = meta.get("top_score")
        base["error"] = meta.get("error")

        scores = r.get("scores") or {}
        for k, v in scores.items():
            base[k] = v
        flat.append(base)
    df = pd.DataFrame(flat).set_index("id")
    return df

def main():
    run_a = os.getenv("RUN_A")
    run_b = os.getenv("RUN_B")
    if not run_a or not run_b:
        raise ValueError("Set RUN_A and RUN_B. e.g., RUN_A=run_A RUN_B=run_B")

    df_a = load_metrics(run_a)
    df_b = load_metrics(run_b)

    # 공통 ID만 비교
    common = df_a.index.intersection(df_b.index)
    df_a = df_a.loc[common].copy()
    df_b = df_b.loc[common].copy()

    # metric 컬럼 탐지: 숫자형 + score 컬럼
    ignore_cols = {"type", "question", "skipped", "skip_reason", "error"}
    candidates = [c for c in df_a.columns if c in df_b.columns and c not in ignore_cols]
    metric_cols = []
    for c in candidates:
        # ctx_chars/k/top_score도 숫자 비교 대상에 포함
        if pd.api.types.is_numeric_dtype(df_a[c]) or c in ("ctx_chars", "k", "top_score"):
            metric_cols.append(c)

    # delta 계산
    delta = (df_b[metric_cols] - df_a[metric_cols])
    delta.columns = [f"delta__{c}" for c in metric_cols]

    df = (
        df_a.add_suffix("__A")
        .join(df_b.add_suffix("__B"))
        .join(delta)
    )

    # --- 운영형 보강 1: 스킵(빈 컨텍스트) 비율 비교 ---
    skip_a = df_a["skipped"].mean()
    skip_b = df_b["skipped"].mean()
    print("\n=== Skip rate (empty_contexts) ===")
    print(f"{run_a}: {skip_a:.3f} | {run_b}: {skip_b:.3f} | delta: {skip_b - skip_a:+.3f}")

    # --- 운영형 보강 2: metric 임계치 미달 비율 ---
    # 임계치는 프로젝트에 맞게 조정해
    thresholds = {
        "faithfulness": 0.6,
        "answer_relevancy": 0.5,
        "context_precision": 0.5,
    }
    print("\n=== Below-threshold rate ===")
    for m, th in thresholds.items():
        # ragas 컬럼명이 버전에 따라 다를 수 있어 partial match
        col_a = next((c for c in df_a.columns if c == m), None)
        if col_a is None:
            # 부분 매칭
            col_a = next((c for c in df_a.columns if m in c), None)
        col_b = next((c for c in df_b.columns if c == m), None)
        if col_b is None:
            col_b = next((c for c in df_b.columns if m in c), None)
        if not col_a or not col_b:
            continue

        ra = (df_a[col_a] < th).mean()
        rb = (df_b[col_b] < th).mean()
        print(f"{m} < {th}: {run_a}={ra:.3f}, {run_b}={rb:.3f}, delta={rb-ra:+.3f}")

    # --- 핵심: 평균 delta ---
    delta_cols = [c for c in df.columns if c.startswith("delta__")]
    mean_delta = df[delta_cols].mean().sort_values()
    print("\n=== Mean Delta (B - A) ===")
    print(mean_delta.to_string())

    # --- ctx_chars 급감 탐지(운영 포인트) ---
    # 1) ctx_chars delta 분포
    if "delta__ctx_chars" in df.columns:
        d = df["delta__ctx_chars"].fillna(0)
        bins = {
            ">= +2000": (d >= 2000).sum(),
            "+500 ~ +2000": (d.between(500, 2000)).sum(),
            "-500 ~ +500": (d.between(-500, 500)).sum(),
            "-2000 ~ -500": (d.between(-2000, -500)).sum(),
            "<= -2000": (d <= -2000).sum(),
        }
        print("\n=== Delta Distribution (ctx_chars) ===")
        print(bins)

        # 2) 급감 Worst Top N
        worst_ctx = df.sort_values("delta__ctx_chars").head(15)[
            ["type__A", "question__A", "ctx_chars__A", "ctx_chars__B", "delta__ctx_chars", "k__A", "k__B"]
        ]
        print("\n=== Worst 15 (ctx_chars delta) ===")
        print(worst_ctx.to_string())

    # --- metric worst top: faithfulness delta 기준 (있을 때) ---
    faith_delta_col = next((c for c in df.columns if c.startswith("delta__") and "faithfulness" in c), None)
    if faith_delta_col:
        worst_faith = df.sort_values(faith_delta_col).head(10)[
            ["type__A", "question__A", faith_delta_col]
        ]
        print("\n=== Worst 10 (faithfulness delta) ===")
        print(worst_faith.to_string())

    # --- type별 평균 delta ---
    type_col = "type__A"
    if type_col in df.columns:
        by_type = df.groupby(type_col)[delta_cols].mean().sort_values(delta_cols[0], ascending=True)
        print("\n=== Mean Delta by Type ===")
        print(by_type.to_string())

if __name__ == "__main__":
    main()
