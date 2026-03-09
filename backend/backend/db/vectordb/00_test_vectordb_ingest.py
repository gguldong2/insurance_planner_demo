from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# 1. 임베딩 모델 준비 (한국어 성능 좋은 모델)
# 처음 실행 시 모델을 다운로드하느라 시간이 좀 걸릴 수 있습니다.
embeddings = HuggingFaceEmbeddings(
    model_name="nlpai-lab/KURE-v1",
    encode_kwargs={"normalize_embeddings": True},
)

# 2. Qdrant 클라이언트 연결
url = "http://localhost:6333"
client = QdrantClient(url=url)
collection_name = "my_knowledge_base"

# 3. 기존 컬렉션이 있다면 초기화(선택사항) 후 생성
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

# 4. 넣을 데이터 준비 (테스트용)
docs = [
    Document(page_content="AgensGraph는 그래프 데이터베이스와 관계형 데이터베이스를 통합한 멀티모델 DB입니다.", metadata={"source": "manual"}),
    Document(page_content="Qdrant는 벡터 유사도 검색을 지원하는 고성능 벡터 데이터베이스입니다.", metadata={"source": "manual"}),
    Document(page_content="LangGraph는 LLM을 활용한 상태 기반의 워크플로우를 만드는 라이브러리입니다.", metadata={"source": "manual"}),
    Document(page_content="본사 사무실은 서울시 강남구 테헤란로에 위치하고 있습니다.", metadata={"source": "company_info"}),
]

# 5. 데이터 삽입 (Vector Store 생성)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

vector_store.add_documents(docs)

print("✅ 데이터 적재 완료! Qdrant 대시보드에서 확인해보세요.")