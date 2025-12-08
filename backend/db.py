# DB 연결 및 실행 로직 (AgensGraph 호환성을 위해 psycopg2 사용)

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드 (DB 접속 정보 등)
load_dotenv()

# 데이터베이스 접속 설정 (환경 변수가 없으면 기본값 사용)
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "password"),
    "dbname": os.getenv("DB_NAME", "postgres")
}

# AgensGraph에서 사용할 그래프 이름 (RDB의 스키마 개념과 유사)
GRAPH_PATH = os.getenv("GRAPH_PATH", "mygraph")

def get_db_connection():
    """데이터베이스 연결 객체(Connection)를 생성하여 반환"""
    # **DB_CONFIG: 딕셔너리 언패킹을 통해 host, port, user 등을 인자로 전달
    return psycopg2.connect(**DB_CONFIG)

def execute_query(cypher_query: str):
    """
    AgensGraph용 Cypher 쿼리 실행 함수
    - graph_path를 설정한 후 Cypher 쿼리를 수행합니다.
    """
    conn = None
    try:
        conn = get_db_connection()
        # RealDictCursor: 결과를 튜플이 아닌 딕셔너리 형태({'name': 'Alice'})로 반환받기 위해 사용
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # [중요] AgensGraph 경로 설정
        # 이 설정을 먼저 해야 해당 그래프 데이터(Node, Edge)에 접근 가능합니다.
        cur.execute(f"SET graph_path = {GRAPH_PATH};")
        
        # Cypher 쿼리 실행
        cur.execute(cypher_query)
        
        # 결과 가져오기 (Fetch)
        try:
            rows = cur.fetchall()
        except psycopg2.ProgrammingError:
            # CREATE, MERGE, DELETE 등 결과를 반환하지 않는 쿼리를 실행하고 
            # fetchall()을 호출하면 에러가 날 수 있음 -> 빈 리스트로 예외 처리
            rows = [] 
            
        # 트랜잭션 확정 (Commit)
        conn.commit()
        return rows

    except Exception as e:
        # 에러 발생 시 롤백하여 데이터 무결성 보호
        if conn: conn.rollback()
        raise e # 에러를 상위 호출자(Agent)에게 전파
    finally:
        # 연결 종료 (리소스 누수 방지)
        if conn: conn.close()

def execute_sql_query(sql_query: str):
    """
    일반 PostgreSQL SQL 쿼리 실행 함수
    - 통계, 로그, 일반 테이블 데이터 조회 시 사용합니다.
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # 표준 SQL 실행
        cur.execute(sql_query)
        
        # 결과 반환
        rows = cur.fetchall()
        
        conn.commit()
        return rows
        
    except Exception as e:
        if conn: conn.rollback()
        raise e
    finally:
        if conn: conn.close()