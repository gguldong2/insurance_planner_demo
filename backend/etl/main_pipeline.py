"""Insurance ETL pipeline entrypoint.

Backward compatibility:
- Legacy single-file layout under backend/data/01~06.json still works.
- Expanded layout under backend/data/common and backend/data/products/* is also supported.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.etl.common import DBContext
from backend.etl.loaders.benefit_loader import BenefitLoader
from backend.etl.loaders.clause_loader import ClauseLoader
from backend.etl.loaders.concept_loader import ConceptLoader
from backend.etl.loaders.product_loader import ProductLoader
from backend.etl.loaders.rider_loader import RiderLoader
from backend.etl.loaders.term_loader import TermLoader
from backend.logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

LOADER_REGISTRY = {
    "product": (ProductLoader, "01_products.json"),
    "rider": (RiderLoader, "02_riders.json"),
    "concept": (ConceptLoader, "03_concepts.json"),
    "benefit": (BenefitLoader, "04_benefits.json"),
    "clause": (ClauseLoader, "05_clauses.json"),
    "term": (TermLoader, "06_terms.json"),
}


def _legacy_steps(base_dir: Path) -> List[Tuple[str, Path]]:
    return [(name, base_dir / filename) for name, (_, filename) in LOADER_REGISTRY.items()]


def _expanded_steps(base_dir: Path) -> List[Tuple[str, Path]]:
    steps: List[Tuple[str, Path]] = []
    common_dir = base_dir / "common"
    products_dir = base_dir / "products"

    if common_dir.exists():
        for name in ["concept", "term"]:
            _, filename = LOADER_REGISTRY[name]
            path = common_dir / filename
            if path.exists():
                steps.append((name, path))

    if products_dir.exists():
        for product_dir in sorted([p for p in products_dir.iterdir() if p.is_dir()]):
            for name in ["product", "rider", "benefit", "clause"]:
                _, filename = LOADER_REGISTRY[name]
                path = product_dir / filename
                if path.exists():
                    steps.append((name, path))

    return steps


def discover_steps(data_root: str = "backend/data") -> List[Tuple[str, str]]:
    """Discover ETL input files from either the legacy or expanded layout."""
    base_dir = Path(data_root)
    if not base_dir.exists():
        return []

    expanded = _expanded_steps(base_dir)
    if expanded:
        return [(name, str(path)) for name, path in expanded]

    legacy = _legacy_steps(base_dir)
    return [(name, str(path)) for name, path in legacy if path.exists()]


# 명령어 인자(--target, --loader)를 받아 로더를 통제하도록 수정
def main():
    parser = argparse.ArgumentParser(description="Insurance ETL Pipeline")
    parser.add_argument("--target", type=str, choices=["all", "graph", "vector"], default="all", help="적재할 타겟 DB (all, graph, vector)")
    parser.add_argument("--loader", type=str, default="all", help="실행할 특정 로더 이름 (product, rider, concept, benefit, clause, term)")
    parser.add_argument("--data-root", type=str, default="backend/data", help="데이터 루트 디렉토리")

    args = parser.parse_args()

    pipeline_started = time.perf_counter()
    logger.info("etl pipeline started", extra={"target": args.target, "loader": args.loader, "data_root": args.data_root})

    ctx = DBContext()

    try:
        steps = discover_steps(args.data_root)
        if not steps:
            logger.warning("etl pipeline found no input files", extra={"data_root": args.data_root})

        for name, file_path in steps:
            if args.loader != "all" and args.loader != name:
                continue

            loader_cls, _ = LOADER_REGISTRY[name]
            loader = loader_cls(ctx)
            started = time.perf_counter()
            logger.info("etl loader started", extra={"loader": name, "file_path": file_path, "target": args.target})
            loader.run(file_path, target_mode=args.target)
            logger.info(
                "etl loader finished",
                extra={
                    "loader": name,
                    "file_path": file_path,
                    "target": args.target,
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                },
            )

        logger.info("etl pipeline finished", extra={"duration_ms": int((time.perf_counter() - pipeline_started) * 1000)})
        print("\n=== 🎉 Pipeline Finished ===")

    except Exception as e:
        print(f"\n❌ Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
