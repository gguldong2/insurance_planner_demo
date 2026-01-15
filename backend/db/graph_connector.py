# backend/db/graph_connector.py
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

class GraphLoader:
    def __init__(self):
        # .env에서 DB 정보 로드
        self.conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            dbname=os.getenv("POSTGRES_DB", "postgres"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres")
        )
        self.conn.autocommit = True
        self.graph_path = os.getenv("AGENS_GRAPH_NAME", "insurance_graph")
        self._init_age()

    def _init_age(self):
        """Apache AGE 확장 로드 및 경로 설정"""
        with self.conn.cursor() as cur:
            try:
                cur.execute("LOAD 'age';")
                cur.execute(f"SET search_path = ag_catalog, '$user', public;")
                # 그래프가 없으면 생성 (에러 무시)
                # cur.execute(f"SELECT create_graph('{self.graph_path}');") 
                print(f"✅ GraphLoader Connected: {self.graph_path}")
            except Exception as e:
                print(f"⚠️ AGE Init Warning: {e}")

    def execute_cypher(self, query: str, params: tuple = None):
        """
        Cypher 쿼리를 실행하는 래퍼 함수.
        Apache AGE는 cypher() 함수로 감싸야 함.
        """
        # 파라미터 처리 로직이 복잡하므로 ETL 단계에서는 f-string 주입 방식을 신중하게 사용
        # (실무에서는 파라미터 바인딩을 권장하지만, ETL 스크립트 편의상 포맷팅 사용)
        wrapped_query = f"""
        SELECT * FROM cypher('{self.graph_path}', $$
            {query}
        $$) as (v agtype);
        """
        with self.conn.cursor() as cur:
            try:
                cur.execute(wrapped_query)
            except Exception as e:
                print(f"❌ Query Failed: {query}\nError: {e}")
                raise e

    def close(self):
        self.conn.close()