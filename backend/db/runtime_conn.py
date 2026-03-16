import os
import json
import psycopg2
import asyncio
from qdrant_client import AsyncQdrantClient
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

        self.qdrant = AsyncQdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

        host = os.getenv("DB_HOST") or os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("DB_PORT") or os.getenv("POSTGRES_PORT", "5432")
        dbname = os.getenv("DB_NAME") or os.getenv("POSTGRES_DB", "postgres")
        user = os.getenv("DB_USER") or os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("DB_PASSWORD")
        if password is None:
            password = os.getenv("POSTGRES_PASSWORD")

        self.graph_path = os.getenv("GRAPH_PATH") or os.getenv("AGENS_GRAPH_NAME", "insurance_graph")

        conn_kwargs = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
        }
        if password:
            conn_kwargs["password"] = password

        self.pg_conn = psycopg2.connect(**conn_kwargs)
        self.pg_conn.autocommit = True

        with self.pg_conn.cursor() as cur:
            try:
                cur.execute(f"CREATE GRAPH {self.graph_path};")
            except Exception:
                pass
            cur.execute(f"SET graph_path = {self.graph_path};")

        print(f"✅ [Runtime] DB Connected. graph_path={self.graph_path}")

    async def execute_cypher(self, query: str):
        def _run():
            with self.pg_conn.cursor() as cur:
                try:
                    cur.execute(f"SET graph_path = {self.graph_path};")
                    cur.execute(query)
                    if cur.description is None:
                        return []
                    rows = cur.fetchall()
                    return self._normalize_rows(rows)
                except Exception as e:
                    print(f"❌ Cypher Error: {e}\nQuery:\n{query}")
                    return []

        return await asyncio.to_thread(_run)

    @staticmethod
    def _normalize_rows(rows):
        out = []
        for row in rows:
            if len(row) == 1:
                v = row[0]
                if isinstance(v, (dict, list, int, float)) or v is None:
                    out.append(v)
                    continue
                if isinstance(v, str):
                    s = v.strip()
                    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                        try:
                            out.append(json.loads(s))
                            continue
                        except Exception:
                            pass
                    out.append(v)
                    continue
                out.append(v)
            else:
                out.append(list(row))
        return out

    async def search_vector(self, collection: str, vector: list, filter: dict = None, limit: int = 3):
        return await self.qdrant.search(
            collection_name=collection,
            query_vector=vector,
            query_filter=filter,
            limit=limit,
        )
