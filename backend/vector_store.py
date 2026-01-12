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
    from FlagEmbedding import BGEM3FlagModel
    from FlagEmbedding import FlagReranker
except ImportError:
    print("❌ FlagEmbedding 설치 필요: pip install FlagEmbedding")
    exit()

load_dotenv()

# ==========================================
# [설정]
# ==========================================
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "cancer_insurance")

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

DENSE_LIMIT = 50
SPARSE_LIMIT = 50
DENSE_THRESHOLD = 0.70
RRF_K = 60

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device setup: {device}")

print("⏳ Loading Embedding Model...")
embedding_model = BGEM3FlagModel(EMBEDDING_MODEL_NAME, use_fp16=True, device=device)

print("⏳ Loading Reranker Model...")
reranker_model = FlagReranker(RERANKER_MODEL_NAME, use_fp16=True, device=device)

client = QdrantClient(url=QDRANT_URL)

class SimpleDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"<Document id={self.metadata.get('id', 'unknown')} score={self.metadata.get('_score', 0):.4f}>"

async def search_documents(query: str, k: int = 3):
    print(f"\n🔍 [Search Start] Query: {query}")
    
    try:
        # 1. 쿼리 임베딩
        output = embedding_model.encode([query], return_dense=True, return_sparse=True, return_colbert_vecs=False)
        query_dense = output['dense_vecs'][0]
        query_sparse = output['lexical_weights'][0]
        
        sparse_indices = [int(key) for key in query_sparse.keys()]
        sparse_values = [float(val) for val in query_sparse.values()]

        # -----------------------------------------------------
        # 2. 검색 (query_points 사용 - 안전한 방식)
        # -----------------------------------------------------
        
        # 2-1. Dense Search
        # [변경점] search() -> query_points() 사용으로 에러 방지
        raw_dense_points = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_dense,
            using="default",
            limit=DENSE_LIMIT,
            with_payload=True
        ).points

        # 2-2. Dense Thresholding
        filtered_dense = []
        cut_count = 0
        for res in raw_dense_points:
            if res.score >= DENSE_THRESHOLD:
                filtered_dense.append(res)
            else:
                cut_count += 1
        
        # 2-3. Sparse Search
        raw_sparse_points = client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.SparseVector(indices=sparse_indices, values=sparse_values),
            using="sparse",
            limit=SPARSE_LIMIT,
            with_payload=True
        ).points

        # -----------------------------------------------------
        # 3. RRF Fusion
        # -----------------------------------------------------
        all_candidates = {}

        # Dense Rank
        for rank, res in enumerate(filtered_dense, 1):
            doc_id = res.id
            if doc_id not in all_candidates:
                all_candidates[doc_id] = {"point": res, "dense_rank": rank, "sparse_rank": None, "tags": []}
            all_candidates[doc_id]["dense_rank"] = rank
            all_candidates[doc_id]["tags"].append("dense")

        # Sparse Rank
        for rank, res in enumerate(raw_sparse_points, 1):
            doc_id = res.id
            if doc_id not in all_candidates:
                all_candidates[doc_id] = {"point": res, "dense_rank": None, "sparse_rank": rank, "tags": []}
            all_candidates[doc_id]["sparse_rank"] = rank
            all_candidates[doc_id]["tags"].append("sparse")

        # Score 계산
        for doc_id, info in all_candidates.items():
            score = 0
            if info["dense_rank"]:
                score += 1 / (RRF_K + info["dense_rank"])
            if info["sparse_rank"]:
                score += 1 / (RRF_K + info["sparse_rank"])
            info["rrf_score"] = score

        # -----------------------------------------------------
        # 4. 통계 로깅
        # -----------------------------------------------------
        dense_only_cnt = 0
        sparse_only_cnt = 0
        both_cnt = 0
        
        for info in all_candidates.values():
            tags = info["tags"]
            if "dense" in tags and "sparse" in tags:
                both_cnt += 1
                info["origin_type"] = "both"
            elif "dense" in tags:
                dense_only_cnt += 1
                info["origin_type"] = "dense_only"
            else:
                sparse_only_cnt += 1
                info["origin_type"] = "sparse_only"

        total_candidates = len(all_candidates)
        
        print("-" * 60)
        print(f"📊 [Retrieval Stats]")
        print(f"   • Dense Raw: {len(raw_dense_points)} (Cut by 0.7: {cut_count})")
        print(f"   • Dense Valid: {len(filtered_dense)}")
        print(f"   • Sparse Valid: {len(raw_sparse_points)}")
        print(f"   • Overlap: Both({both_cnt}), Dense_only({dense_only_cnt}), Sparse_only({sparse_only_cnt})")
        print("-" * 60)

        if total_candidates == 0:
            print("⚠️ No candidates found.")
            return []

        # -----------------------------------------------------
        # 5. Re-ranking & Selection
        # -----------------------------------------------------
        sorted_candidates = sorted(all_candidates.values(), key=lambda x: x['rrf_score'], reverse=True)
        
        rerank_pairs = []
        for cand in sorted_candidates:
            content = cand["point"].payload.get("content", "")
            rerank_pairs.append([query, content])

        if rerank_pairs:
            rerank_scores = reranker_model.compute_score(rerank_pairs)
            if not isinstance(rerank_scores, list):
                rerank_scores = [rerank_scores]
        else:
            rerank_scores = []

        for i, cand in enumerate(sorted_candidates):
            cand["rerank_score"] = rerank_scores[i]

        # 최종 정렬
        final_ranking = sorted(sorted_candidates, key=lambda x: x['rerank_score'], reverse=True)
        top_k_docs = final_ranking[:k]

        print(f"🏆 [Final Selection] Top {len(top_k_docs)}")
        
        final_documents = []
        for i, item in enumerate(top_k_docs, 1):
            point = item["point"]
            origin = item["origin_type"]
            score = item["rerank_score"]
            doc_id = point.id
            title = point.payload.get("doc_title", "No Title")
            
            print(f"   {i}. [{origin.upper()}] ID: {doc_id} | Score: {score:.4f} | Title: {title}")
            
            payload = point.payload
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