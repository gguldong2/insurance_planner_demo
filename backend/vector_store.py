import os

#OpenAI
# from langchain_openai import OpenAIEmbeddings
# from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import Distance, VectorParams
# from dotenv import load_dotenv

#HuggingFace
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv


import torch # torch 임포트 필요
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Embedding Device: {device}")


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "my_knowledge_base")


# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# [변경] 한국어 성능이 좋은 로컬 모델로 설정
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': device}, # 동적 할당
    encode_kwargs={"normalize_embeddings": True},
)

def get_vector_store():
    """QdrantVectorStore 객체 반환"""
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL
    )


# [변경] 비동기 검색 함수
async def search_documents(query: str, k: int = 3):
    """
    질문과 유사한 문서를 비동기적으로 검색합니다.
    LangChain의 asimilarity_search를 사용합니다.
    """
    try:
        store = get_vector_store()
        # [핵심] await + asimilarity_search 사용
        docs = await store.asimilarity_search(query, k=k)
        return docs
    except Exception as e:
        print(f"⚠️ Vector Search Error: {e}")
        return []