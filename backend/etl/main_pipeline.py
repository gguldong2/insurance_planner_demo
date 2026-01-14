import sys
import os

# 상위 경로 추가 (backend 모듈 인식)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.etl.common import DBContext
from backend.etl.loaders.product_loader import ProductLoader
from backend.etl.loaders.rider_loader import RiderLoader
from backend.etl.loaders.concept_loader import ConceptLoader
from backend.etl.loaders.benefit_loader import BenefitLoader
from backend.etl.loaders.clause_loader import ClauseLoader
from backend.etl.loaders.term_loader import TermLoader

def main():
    print("=== 🏁 Start Modular ETL Pipeline ===")
    
    # 1. DB Context 초기화 (연결)
    ctx = DBContext()
    
    try:
        # 2. 로더 정의 (순서 중요! 부모 노드 -> 자식 노드)
        steps = [
            (ProductLoader(ctx), "backend/data/01_products.json"),
            (RiderLoader(ctx),   "backend/data/02_riders.json"),
            (ConceptLoader(ctx), "backend/data/03_concepts.json"),
            (BenefitLoader(ctx), "backend/data/04_benefits.json"),
            (ClauseLoader(ctx),  "backend/data/05_clauses.json"),
            (TermLoader(ctx),    "backend/data/06_terms.json"),
        ]

        # 3. 파이프라인 실행
        for loader, file_path in steps:
            if os.path.exists(file_path):
                loader.run(file_path)
            else:
                print(f"⚠️ Warning: File not found {file_path}")

        print("\n=== 🎉 All ETL Jobs Finished Successfully ===")

    except Exception as e:
        print(f"\n❌ Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ctx.close()

if __name__ == "__main__":
    main()