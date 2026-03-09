import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.etl.common import DBContext
from backend.etl.loaders.product_loader import ProductLoader
from backend.etl.loaders.rider_loader import RiderLoader
from backend.etl.loaders.concept_loader import ConceptLoader
from backend.etl.loaders.benefit_loader import BenefitLoader
from backend.etl.loaders.clause_loader import ClauseLoader
from backend.etl.loaders.term_loader import TermLoader


# 명령어 인자(--target, --loader)를 받아 로더를 통제하도록 수정
def main():
    # CLI 인자 파싱
    parser = argparse.ArgumentParser(description="Insurance ETL Pipeline")
    parser.add_argument("--target", type=str, choices=["all", "graph", "vector"], default="all", 
                        help="적재할 타겟 DB (all, graph, vector)")
    parser.add_argument("--loader", type=str, default="all", 
                        help="실행할 특정 로더 이름 (product, rider, concept, benefit, clause, term)")
    
    args = parser.parse_args()

    print(f"=== 🏁 Start ETL Pipeline [Target: {args.target.upper()} | Loader: {args.loader.upper()}] ===")
    
    ctx = DBContext()
    
    try:
        # (LoaderInstance, FilePath, LoaderName)
        steps = [
            (ProductLoader(ctx), "backend/data/01_products.json", "product"),
            (RiderLoader(ctx),   "backend/data/02_riders.json", "rider"),
            (ConceptLoader(ctx), "backend/data/03_concepts.json", "concept"),
            (BenefitLoader(ctx), "backend/data/04_benefits.json", "benefit"),
            (ClauseLoader(ctx),  "backend/data/05_clauses.json", "clause"),
            (TermLoader(ctx),    "backend/data/06_terms.json", "term"),
        ]

        for loader, file_path, name in steps:
            # 1. 특정 로더 필터링
            if args.loader != "all" and args.loader != name:
                continue

            # 2. 파일 존재 확인 및 실행
            if os.path.exists(file_path):
                # target_mode 전달
                loader.run(file_path, target_mode=args.target)
            else:
                print(f"⚠️ Skip {name}: File not found ({file_path})")

        print("\n=== 🎉 Pipeline Finished ===")

    except Exception as e:
        print(f"\n❌ Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ctx.close()

if __name__ == "__main__":
    main()