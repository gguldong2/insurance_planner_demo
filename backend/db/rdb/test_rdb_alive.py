# test_db.py
from backend.db import execute_query
import sys

# 테스트할 쿼리: 앨리스 찾기
query = "MATCH (n:Person) WHERE n.name = 'Alice' RETURN n"

try:
    print("🔌 DB 연결 및 그래프 조회 시도 중...")
    results = execute_query(query)
    
    print("\n✅ 성공! 조회된 데이터:")
    for row in results:
        print(row)
        
    if not results:
        print("⚠️ 연결은 성공했으나 데이터가 없습니다. (그래프 생성은 됐으나x 노드가 없을 수 있음)")

except Exception as e:
    print(f"\n❌ 실패! 에러 로그:\n{e}")
    print("\n[진단 가이드]")
    print("1. 'FATAL: database does not exist' -> .env의 DB_NAME을 확인하세요.")
    print("2. 'schema mygraph does not exist' -> DB는 맞는데 CREATE GRAPH를 안 했거나 다른 DB에 했습니다.")
    print("3. 'connection refused' -> DB 서버가 안 켜져 있거나 포트가 틀렸습니다.")