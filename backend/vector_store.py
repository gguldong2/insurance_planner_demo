# import os

# #OpenAI
# # from langchain_openai import OpenAIEmbeddings
# # from langchain_qdrant import QdrantVectorStore
# # from qdrant_client import QdrantClient
# # from qdrant_client.http.models import Distance, VectorParams
# # from dotenv import load_dotenv

# #HuggingFace
# from langchain_huggingface import HuggingFaceEmbeddings 
# from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import Distance, VectorParams
# from dotenv import load_dotenv


# import torch # torch 임포트 필요
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"🚀 Embedding Device: {device}")


# load_dotenv()

# QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
# COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "seoultech_regulations")


# # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# # [변경] 한국어 성능이 좋은 로컬 모델로 설정
# embeddings = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-m3",
#     model_kwargs={'device': device}, # 동적 할당
#     encode_kwargs={"normalize_embeddings": True},
# )

# def get_vector_store():
#     """QdrantVectorStore 객체 반환"""
#     return QdrantVectorStore.from_existing_collection(
#         embedding=embeddings,
#         collection_name=COLLECTION_NAME,
#         url=QDRANT_URL
#     )


# # [변경] 비동기 검색 함수
# async def search_documents(query: str, k: int = 3):
#     """
#     질문과 유사한 문서를 비동기적으로 검색합니다.
#     LangChain의 asimilarity_search를 사용합니다.
#     """
#     try:
#         store = get_vector_store()
#         # [핵심] await + asimilarity_search 사용
#         docs = await store.asimilarity_search(query, k=k)
#         return docs
#     except Exception as e:
#         print(f"⚠️ Vector Search Error: {e}")
#         return []

########################## ↑↑기본 방식(langchain)↑↑#############################


import os
import torch
from typing import List, Dict
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

# [라이브러리 로드]
try:
    # 임베딩 모델 (BGE-M3)
    from FlagEmbedding import BGEM3FlagModel
    # 리랭커 모델 (BGE-Reranker)
    from FlagEmbedding import FlagReranker
except ImportError:
    print("❌ FlagEmbedding 설치 필요: pip install FlagEmbedding")
    exit()

load_dotenv()

# ==========================================
# [설정]
# ==========================================
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "seoultech_regulations")

# 모델 설정
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3" # 리랭커 모델 (고성능)

# 검색 설정
DENSE_LIMIT = 50
SPARSE_LIMIT = 50
DENSE_THRESHOLD = 0.70  # Dense 1차 필터링 기준
RRF_K = 60              # RRF 상수

# 1. GPU 자동 감지
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device setup: {device}")

# 2. 모델 로드 (전역 로드)
print("⏳ Loading Embedding Model...")
embedding_model = BGEM3FlagModel(EMBEDDING_MODEL_NAME, use_fp16=True, device=device)

print("⏳ Loading Reranker Model...")
reranker_model = FlagReranker(RERANKER_MODEL_NAME, use_fp16=True, device=device)

# 3. Qdrant 연결
client = QdrantClient(url=QDRANT_URL)

# ---------------------------------------------------------
# [Helper] 결과 반환용 객체
# ---------------------------------------------------------
class SimpleDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"<Document id={self.metadata.get('id', 'unknown')} score={self.metadata.get('_score', 0):.4f}>"

# ---------------------------------------------------------
# [핵심] 수동 RRF + 리랭킹 + 로깅 함수
# ---------------------------------------------------------
async def search_documents(query: str, k: int = 3):
    """
    1. Dense(50) / Sparse(50) 검색
    2. Dense Threshold(0.7) 적용 및 로깅
    3. RRF Fusion (Python 구현)
    4. Re-ranking (Top N)
    5. Final Selection (Top K)
    """
    print(f"\n🔍 [Search Start] Query: {query}")
    
    try:
        # 1. 쿼리 임베딩 (Dense + Sparse)
        output = embedding_model.encode([query], return_dense=True, return_sparse=True, return_colbert_vecs=False)
        query_dense = output['dense_vecs'][0]
        query_sparse = output['lexical_weights'][0]
        
        # Sparse 변환
        sparse_indices = [int(key) for key in query_sparse.keys()]
        sparse_values = [float(val) for val in query_sparse.values()]

        # -----------------------------------------------------
        # 2. 개별 검색 (Dense & Sparse)
        # -----------------------------------------------------
        # 2-1. Dense Search (Threshold 없이 일단 50개 가져와서 파이썬에서 자름 -> 로깅 위해)
        raw_dense_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=models.NamedVector(name="default", vector=query_dense),
            limit=DENSE_LIMIT,
            with_payload=True
        )

        # 2-2. Dense Thresholding & Logging
        filtered_dense = []
        cut_count = 0
        for res in raw_dense_results:
            if res.score >= DENSE_THRESHOLD:
                filtered_dense.append(res)
            else:
                cut_count += 1
        
        # 2-3. Sparse Search
        raw_sparse_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=models.NamedVector(
                name="sparse", 
                vector=models.SparseVector(indices=sparse_indices, values=sparse_values)
            ),
            limit=SPARSE_LIMIT,
            with_payload=True
        )

        # -----------------------------------------------------
        # 3. RRF 계산 및 출처 파악 (Overlaps)
        # -----------------------------------------------------
        # 문서 ID를 키로 하여 정보 통합
        # doc_info = { "id": { "doc": point, "dense_rank": ..., "sparse_rank": ..., "rrf_score": ... } }
        all_candidates = {}

        # Dense 처리
        for rank, res in enumerate(filtered_dense, 1):
            doc_id = res.id
            if doc_id not in all_candidates:
                all_candidates[doc_id] = {"point": res, "dense_rank": rank, "sparse_rank": None, "tags": []}
            all_candidates[doc_id]["dense_rank"] = rank
            all_candidates[doc_id]["tags"].append("dense")

        # Sparse 처리
        for rank, res in enumerate(raw_sparse_results, 1):
            doc_id = res.id
            if doc_id not in all_candidates:
                all_candidates[doc_id] = {"point": res, "dense_rank": None, "sparse_rank": rank, "tags": []}
            all_candidates[doc_id]["sparse_rank"] = rank
            all_candidates[doc_id]["tags"].append("sparse")

        # RRF Score 계산: score = 1/(k + rank_d) + 1/(k + rank_s)
        for doc_id, info in all_candidates.items():
            score = 0
            if info["dense_rank"]:
                score += 1 / (RRF_K + info["dense_rank"])
            if info["sparse_rank"]:
                score += 1 / (RRF_K + info["sparse_rank"])
            info["rrf_score"] = score

        # -----------------------------------------------------
        # 4. [Logging 1] 검색/필터링 통계 출력
        # -----------------------------------------------------
        # 통계 집계
        dense_only_cnt = 0
        sparse_only_cnt = 0
        both_cnt = 0
        
        for info in all_candidates.values():
            has_dense = "dense" in info["tags"]
            has_sparse = "sparse" in info["tags"]
            
            if has_dense and has_sparse:
                both_cnt += 1
                info["origin_type"] = "both"
            elif has_dense:
                dense_only_cnt += 1
                info["origin_type"] = "dense_only"
            else:
                sparse_only_cnt += 1
                info["origin_type"] = "sparse_only"

        total_candidates = len(all_candidates)
        
        print("-" * 60)
        print(f"📊 [Retrieval Stats]")
        print(f"   • Dense Raw: {len(raw_dense_results)} (Cut by threshold {DENSE_THRESHOLD}: {cut_count})")
        print(f"   • Dense Valid: {len(filtered_dense)}")
        print(f"   • Sparse Valid: {len(raw_sparse_results)}")
        print(f"   • Overlap Stats:")
        print(f"     - Dense Only: {dense_only_cnt}/{len(filtered_dense)}")
        print(f"     - Sparse Only: {sparse_only_cnt}/{len(raw_sparse_results)}")
        print(f"     - Both: {both_cnt}")
        print(f"     - Total Unique Candidates: {total_candidates}")
        print("-" * 60)

        if total_candidates == 0:
            print("⚠️ No candidates found.")
            return []

        # -----------------------------------------------------
        # 5. Re-ranking (리랭킹)
        # -----------------------------------------------------
        # RRF 점수순 정렬 (리랭킹 대상 선정용, 여기선 전체 다 넣음)
        sorted_candidates = sorted(all_candidates.values(), key=lambda x: x['rrf_score'], reverse=True)
        
        # 리랭킹 모델 입력 준비 [[query, doc_text], ...]
        rerank_pairs = []
        for cand in sorted_candidates:
            # content가 payload에 있다고 가정
            content = cand["point"].payload.get("content", "")
            rerank_pairs.append([query, content])

        # 리랭커 채점 (Batch 처리)
        if rerank_pairs:
            rerank_scores = reranker_model.compute_score(rerank_pairs)
            # 단일 결과일 때 float로 오므로 리스트 변환 처리
            if not isinstance(rerank_scores, list):
                rerank_scores = [rerank_scores]
        else:
            rerank_scores = []

        # 점수 매핑
        for i, cand in enumerate(sorted_candidates):
            cand["rerank_score"] = rerank_scores[i]

        # -----------------------------------------------------
        # 6. 최종 정렬 및 Top K 선정
        # -----------------------------------------------------
        # 리랭킹 점수 높은 순 정렬
        final_ranking = sorted(sorted_candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        # Top K 자르기
        top_k_docs = final_ranking[:k]

        # -----------------------------------------------------
        # 7. [Logging 2] 최종 결과 상세 로그
        # -----------------------------------------------------
        print(f"🏆 [Final Selection] Top {len(top_k_docs)} (Target K={k})")
        
        final_documents = []
        for i, item in enumerate(top_k_docs, 1):
            point = item["point"]
            origin = item["origin_type"]
            score = item["rerank_score"]
            doc_id = point.id
            title = point.payload.get("doc_title", "No Title")
            
            # 로그 출력
            print(f"   {i}. [{origin.upper()}] ID: {doc_id} | Score: {score:.4f} | Title: {title}")
            
            # 반환용 문서 객체 생성
            payload = point.payload
            # 메타데이터에 점수와 출처 주입
            payload["_score"] = score
            payload["_origin"] = origin
            payload["id"] = doc_id
            
            doc = SimpleDocument(
                page_content=payload.get("content", ""),
                metadata=payload
            )
            final_documents.append(doc)
            
        print("-" * 60)
        return final_documents

    except Exception as e:
        print(f"⚠️ Vector Search Error: {e}")
        import traceback
        traceback.print_exc()
        return []