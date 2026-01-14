import os
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from FlagEmbedding import BGEM3FlagModel
# (Turn 1에서 만든 GraphLoader가 있다고 가정)
from backend.db.graph_connector import GraphLoader 
from dotenv import load_dotenv

load_dotenv()

class DBContext:
    """DB 연결 및 모델 로딩을 관리하는 Context Manager"""
    def __init__(self):
        print("🔌 [ETL] Connecting to DBs & Loading Model...")
        
        # 1. Qdrant 연결
        self.qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        
        # 2. Graph 연결 (AgensGraph)
        self.graph = GraphLoader()
        
        # 3. Embedding Model (GPU 확인)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Embedding Device: {device}")
        self.embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=device)
        
        # 4. 컬렉션 사전 생성 (Turn 1 로직 통합)
        self._init_collections()
        
    def _init_collections(self):
        cols = ["insurance_knowledge", "concepts", "glossary"]
        size = 1024 # BGE-M3
        for col in cols:
            if not self.qdrant.collection_exists(col):
                self.qdrant.create_collection(
                    collection_name=col,
                    vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE)
                )

    def close(self):
        self.graph.close()
        print("🔌 [ETL] DB Connection Closed.")
