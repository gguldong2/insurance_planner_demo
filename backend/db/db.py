import psycopg2
from psycopg2.extras import RealDictCursor
import os
import asyncio  # 비동기 처리
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
    """동기식 DB 연결 객체 생성 (기존 유지)"""
    return psycopg2.connect(**DB_CONFIG)

def _execute_query_sync(cypher_query: str):
    """
    [내부 함수] 기존의 동기식 Cypher 실행 로직
    이 함수는 직접 호출하지 않고, 아래의 execute_query(async)에서 호출됩니다.
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(f"SET graph_path = {GRAPH_PATH};")
        cur.execute(cypher_query)
        try:
            rows = cur.fetchall()
        except psycopg2.ProgrammingError:
            rows = []
        conn.commit()
        return rows
    except Exception as e:
        if conn: conn.rollback()
        raise e
    finally:
        if conn: conn.close()

def _execute_sql_query_sync(sql_query: str):
    """[내부 함수] 기존의 동기식 SQL 실행 로직"""
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

# -----------------------------------------------------------
# [변경] 외부에서 사용할 Async Wrapper 함수들
# -----------------------------------------------------------

async def execute_query(cypher_query: str):
    """
    Cypher 쿼리를 비동기적으로 실행합니다.
    (내부적으로는 별도 스레드에서 동기 함수를 실행하여 Main Loop 차단 방지)
    """
    return await asyncio.to_thread(_execute_query_sync, cypher_query)

async def execute_sql_query(sql_query: str):
    """
    SQL 쿼리를 비동기적으로 실행합니다.
    """
    return await asyncio.to_thread(_execute_sql_query_sync, sql_query)