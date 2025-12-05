#DB 연결 및 실행 로직. (AgensGraph 호환성을 위해 psycopg2 사용)

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "password"),
    "dbname": os.getenv("DB_NAME", "postgres")
}
GRAPH_PATH = os.getenv("GRAPH_PATH", "mygraph")

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def execute_query(cypher_query: str):
    """AgensGraph Cypher 쿼리 실행"""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # AgensGraph 경로 설정
        cur.execute(f"SET graph_path = {GRAPH_PATH};")
        cur.execute(cypher_query)
        
        try:
            rows = cur.fetchall()
        except psycopg2.ProgrammingError:
            rows = [] # 결과가 없는 쿼리
            
        conn.commit()
        return rows
    except Exception as e:
        if conn: conn.rollback()
        raise e
    finally:
        if conn: conn.close()

def execute_sql_query(sql_query: str):
    """일반 PostgreSQL SQL 쿼리 실행"""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql_query)
        rows = cur.fetchall()
        conn.commit()
        return rows
    except Exception as e:
        if conn: conn.rollback()
        raise e
    finally:
        if conn: conn.close()