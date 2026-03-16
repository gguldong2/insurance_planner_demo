import os
import json
import psycopg2
from dotenv import load_dotenv

load_dotenv()


class GraphLoader:
    """
    ETL 적재용 GraphDB 커넥터 (AgensGraph 전용)
    - CREATE GRAPH / SET graph_path 사용
    - Cypher를 SQL wrapper 없이 직접 실행
    """

    def __init__(self):
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

        self.conn = psycopg2.connect(**conn_kwargs)
        self.conn.autocommit = True

        self._init_agensgraph()

    def _init_agensgraph(self):
        with self.conn.cursor() as cur:
            try:
                cur.execute(f"CREATE GRAPH {self.graph_path};")
            except Exception:
                pass
            cur.execute(f"SET graph_path = {self.graph_path};")

        print(f"✅ GraphLoader Connected (AgensGraph): graph_path={self.graph_path}")

    @staticmethod
    def _escape_agtype_value(value):
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)[1:-1]
        if isinstance(value, list):
            return [GraphLoader._escape_agtype_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(GraphLoader._escape_agtype_value(v) for v in value)
        if isinstance(value, dict):
            return {k: GraphLoader._escape_agtype_value(v) for k, v in value.items()}
        return value

    def execute_cypher(self, query: str, params=None):
        with self.conn.cursor() as cur:
            try:
                cur.execute(f"SET graph_path = {self.graph_path};")
                if params is None:
                    cur.execute(query)
                else:
                    cur.execute(query, self._escape_agtype_value(params))
                if cur.description is None:
                    return []
                rows = cur.fetchall()
                return self._normalize_rows(rows)
            except Exception as e:
                print(f"❌ Query Failed:\n{query}\nError: {e}")
                raise

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

    def close(self):
        self.conn.close()
