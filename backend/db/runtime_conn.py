# backend/db/runtime_conn.py
import os
import psycopg2
import asyncio
from qdrant_client import QdrantClient, AsyncQdrantClient
from dotenv import load_dotenv

load_dotenv()

class RuntimeDB:
    """
    API 서비스용 DB 연결 관리자 (Singleton)
    AgensGraph(PostgreSQL) & Qdrant 연결
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RuntimeDB, cls).__new__(cls)
            cls._instance._init_connections()
        return cls._instance

    def _init_connections(self):
        print("🔌 [Runtime] Connecting to Databases...")
        
        # 1. Qdrant (Async Client 권장)
        self.qdrant = AsyncQdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        
        # 2. AgensGraph (Psycopg2 is Sync -> wrapped in asyncio later)
        self.pg_conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            dbname=os.getenv("POSTGRES_DB", "postgres"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres")
        )
        self.pg_conn.autocommit = True
        self.graph_path = os.getenv("AGENS_GRAPH_NAME", "insurance_graph")
        
        # AGE 초기화
        with self.pg_conn.cursor() as cur:
            cur.execute("LOAD 'age';")
            cur.execute("SET search_path = ag_catalog, '$user', public;")
        print("✅ [Runtime] DB Connected.")

    async def execute_cypher(self, query: str):
        """
        AgensGraph Cypher 쿼리 비동기 실행 래퍼
        """
        def _run():
            with self.pg_conn.cursor() as cur:
                # AGE Cypher 쿼리 래핑
                wrapped = f"SELECT * FROM cypher('{self.graph_path}', $${query}$$) as (v agtype);"
                try:
                    cur.execute(wrapped)
                    # agtype(JSON) 결과를 Python Dict로 변환
                    return [row[0] for row in cur.fetchall()]
                except Exception as e:
                    print(f"❌ Cypher Error: {e}\nQuery: {query}")
                    return []
        
        # 블로킹 방지를 위해 스레드풀에서 실행
        return await asyncio.to_thread(_run)

    async def search_vector(self, collection: str, vector: list, filter: dict = None, limit: int = 3):
        """Qdrant 비동기 검색"""
        return await self.qdrant.search(
            collection_name=collection,
            query_vector=vector,
            query_filter=filter,
            limit=limit
        )
