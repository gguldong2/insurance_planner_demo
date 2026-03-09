import asyncio
from backend.vector_store import search_documents

# 비동기 함수 실행을 위한 래퍼
async def main():
    query = "AgensGraph가 뭐야?"
    print(f"🔍 질문: {query}")
    
    # 검색 실행
    docs = await search_documents(query, k=2)
    
    print(f"\n✅ 검색 결과 ({len(docs)}건):")
    for i, doc in enumerate(docs):
        print(f"[{i+1}] {doc.page_content}")

if __name__ == "__main__":
    asyncio.run(main())