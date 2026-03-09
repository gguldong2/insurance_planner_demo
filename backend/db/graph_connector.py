# # backend/db/graph_connector.py
# import psycopg2
# import os
# from dotenv import load_dotenv

# load_dotenv()

# class GraphLoader:
#     def __init__(self):
#         # .env에서 DB 정보 로드
#         self.conn = psycopg2.connect(
#             host=os.getenv("POSTGRES_HOST", "localhost"),
#             port=os.getenv("POSTGRES_PORT", "5432"),
#             dbname=os.getenv("POSTGRES_DB", "postgres"),
#             user=os.getenv("POSTGRES_USER", "postgres"),
#             password=os.getenv("POSTGRES_PASSWORD", "postgres")
#         )
#         self.conn.autocommit = True
#         self.graph_path = os.getenv("AGENS_GRAPH_NAME", "insurance_graph")
#         self._init_age()

#     def _init_age(self):
#         """Apache AGE 확장 로드 및 경로 설정"""
#         with self.conn.cursor() as cur:
#             try:
#                 cur.execute("LOAD 'age';")
#                 cur.execute(f"SET search_path = ag_catalog, '$user', public;")
#                 # 그래프가 없으면 생성 (에러 무시)
#                 # cur.execute(f"SELECT create_graph('{self.graph_path}');") 
#                 print(f"✅ GraphLoader Connected: {self.graph_path}")
#             except Exception as e:
#                 print(f"⚠️ AGE Init Warning: {e}")

#     def execute_cypher(self, query: str, params: tuple = None):
#         """
#         Cypher 쿼리를 실행하는 래퍼 함수.
#         Apache AGE는 cypher() 함수로 감싸야 함.
#         """
#         # 파라미터 처리 로직이 복잡하므로 ETL 단계에서는 f-string 주입 방식을 신중하게 사용
#         # (실무에서는 파라미터 바인딩을 권장하지만, ETL 스크립트 편의상 포맷팅 사용)
#         wrapped_query = f"""
#         SELECT * FROM cypher('{self.graph_path}', $$
#             {query}
#         $$) as (v agtype);
#         """
#         with self.conn.cursor() as cur:
#             try:
#                 cur.execute(wrapped_query)
#             except Exception as e:
#                 print(f"❌ Query Failed: {query}\nError: {e}")
#                 raise e

#     def close(self):
#         self.conn.close()


# backend/db/graph_connector.py
import os
import json
import psycopg2
from dotenv import load_dotenv

load_dotenv()


class GraphLoader:
    """
    ETL 적재용 GraphDB 커넥터 (AgensGraph 전용)
    - CREATE GRAPH / SET graph_path 사용
    - Cypher를 SQL wrapper 없이 "직접 실행"
    """

    def __init__(self):
        host = os.getenv("DB_HOST") or os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("DB_PORT") or os.getenv("POSTGRES_PORT", "5435")
        dbname = os.getenv("DB_NAME") or os.getenv("POSTGRES_DB", "agens")
        user = os.getenv("DB_USER") or os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("DB_PASSWORD") or os.getenv("POSTGRES_PASSWORD", "agens")

        self.graph_path = os.getenv("GRAPH_PATH") or os.getenv("AGENS_GRAPH_NAME", "insurance_graph")

        self.conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
        )
        self.conn.autocommit = True

        self._init_agensgraph()

    def _init_agensgraph(self):
        """AgensGraph 그래프 공간 준비 + graph_path 설정"""
        with self.conn.cursor() as cur:
            # 1) 그래프 생성 (이미 있으면 에러 나도 무시)
            try:
                cur.execute(f"CREATE GRAPH {self.graph_path};")
            except Exception:
                # 이미 존재하는 경우 등은 무시
                pass

            # 2) 현재 세션 graph_path 설정
            cur.execute(f"SET graph_path = {self.graph_path};")

        print(f"✅ GraphLoader Connected (AgensGraph): graph_path={self.graph_path}")

    def execute_cypher(self, query: str, params: tuple = None):
        """
        Cypher(OPEN CYPHER) 직접 실행.
        - params 바인딩도 가능하게 열어둠 (필요할 때만 사용)
        """
        with self.conn.cursor() as cur:
            try:
                # 세션이 끊겼다 다시 붙는 케이스 대비: 안전하게 graph_path 재설정
                cur.execute(f"SET graph_path = {self.graph_path};")

                if params is None:
                    cur.execute(query)
                else:
                    cur.execute(query, params)

                # RETURN 없는 쿼리면 fetch 불가 -> 그냥 종료
                if cur.description is None:
                    return []

                rows = cur.fetchall()
                return self._normalize_rows(rows)

            except Exception as e:
                print(f"❌ Query Failed:\n{query}\nError: {e}")
                raise

    @staticmethod
    def _normalize_rows(rows):
        """
        retrievers.py에서 RETURN { ... } 형태를 많이 쓰므로,
        - 1컬럼 결과면 그 컬럼을 리스트로 풀어주고
        - 문자열(JSON)처럼 보이면 json.loads 시도
        """
        out = []
        for row in rows:
            if len(row) == 1:
                v = row[0]
                if isinstance(v, (dict, list, int, float)) or v is None:
                    out.append(v)
                    continue
                if isinstance(v, str):
                    s = v.strip()
                    # JSON처럼 보이면 파싱 시도
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

    def close(self):
        self.conn.close()