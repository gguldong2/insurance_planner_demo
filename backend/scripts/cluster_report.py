# scripts/cluster_report.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows = []
    if not p.exists():
        raise FileNotFoundError(str(p))
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(p: Path, rows: List[Dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def get_score(rec: Dict[str, Any], key: str) -> float | None:
    v = (rec.get("scores") or {}).get(key)
    if isinstance(v, (int, float)):
        return float(v)
    return None

def pass_rule(rec: Dict[str, Any], faith_th: float, rel_th: float) -> bool:
    faith = get_score(rec, "faithfulness")
    rel = get_score(rec, "answer_relevancy")
    if faith is None or rel is None:
        return False
    return (faith >= faith_th) and (rel >= rel_th)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controlled_tag", required=True)
    ap.add_argument("--natural_tag", required=True)
    ap.add_argument("--map", required=True, help="mapping jsonl path")
    ap.add_argument("--faith_th", type=float, default=0.7)
    ap.add_argument("--rel_th", type=float, default=0.7)
    args = ap.parse_args()

    metrics_dir = Path("eval/metrics")

    ctrl_path = metrics_dir / f"{args.controlled_tag}_metrics.jsonl"
    nat_path = metrics_dir / f"{args.natural_tag}_metrics.jsonl"
    map_path = Path(args.map)

    ctrl_rows = read_jsonl(ctrl_path)
    nat_rows = read_jsonl(nat_path)
    maps = read_jsonl(map_path)

    ctrl_by_id = {r["id"]: r for r in ctrl_rows if r.get("id")}
    nat_by_id = {r["id"]: r for r in nat_rows if r.get("id")}

    cluster_rows = []
    tpass_nfail_rows = []

    for m in maps:
        tid = m["controlled_id"]
        nids = m["natural_ids"]

        ctrl = ctrl_by_id.get(tid)
        if not ctrl:
            continue

        ctrl_pass = pass_rule(ctrl, args.faith_th, args.rel_th)

        nat_recs = [nat_by_id.get(nid) for nid in nids]
        nat_recs = [r for r in nat_recs if r]

        nat_pass_flags = [pass_rule(r, args.faith_th, args.rel_th) for r in nat_recs]
        nat_pass_count = sum(1 for x in nat_pass_flags if x)
        nat_pass_rate = nat_pass_count / max(1, len(nids))

        # 집계(자연형)
        def agg(metric: str) -> Tuple[float | None, float | None]:
            vals = [get_score(r, metric) for r in nat_recs]
            vals = [v for v in vals if v is not None]
            if not vals:
                return None, None
            return sum(vals) / len(vals), min(vals)

        nat_faith_mean, nat_faith_min = agg("faithfulness")
        nat_rel_mean, nat_rel_min = agg("answer_relevancy")

        bucket = "other"
        if ctrl_pass and nat_pass_count == 0:
            bucket = "T_pass_N_all_fail"
        elif ctrl_pass and nat_pass_count < len(nids):
            bucket = "T_pass_N_partial_fail"
        elif (not ctrl_pass) and nat_pass_count > 0:
            bucket = "T_fail_N_some_pass"
        elif (not ctrl_pass) and nat_pass_count == 0:
            bucket = "T_fail_N_all_fail"

        row = {
            "controlled_id": tid,
            "natural_ids": nids,
            "controlled_pass": ctrl_pass,
            "natural_pass_count": nat_pass_count,
            "natural_pass_rate": nat_pass_rate,
            "scores": {
                "controlled": {
                    "faithfulness": get_score(ctrl, "faithfulness"),
                    "answer_relevancy": get_score(ctrl, "answer_relevancy"),
                },
                "natural_mean": {
                    "faithfulness": nat_faith_mean,
                    "answer_relevancy": nat_rel_mean,
                },
                "natural_min": {
                    "faithfulness": nat_faith_min,
                    "answer_relevancy": nat_rel_min,
                },
            },
            "bucket": bucket,
        }
        cluster_rows.append(row)

        # “통제형 성공 & 자연형 실패” 케이스만 뽑기
        if bucket in ("T_pass_N_all_fail", "T_pass_N_partial_fail"):
            tpass_nfail_rows.append(row)

    out_cluster = metrics_dir / f"{args.natural_tag}_cluster_report.jsonl"
    out_fail = metrics_dir / f"{args.natural_tag}_Tpass_Nfail.jsonl"
    write_jsonl(out_cluster, cluster_rows)
    write_jsonl(out_fail, tpass_nfail_rows)

    print(f"✅ cluster report: {out_cluster} (n={len(cluster_rows)})")
    print(f"✅ Tpass_Nfail   : {out_fail} (n={len(tpass_nfail_rows)})")
    print(f"   thresholds: faithfulness>={args.faith_th}, answer_relevancy>={args.rel_th}")

if __name__ == "__main__":
    main()
