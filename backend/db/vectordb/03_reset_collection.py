from qdrant_client import QdrantClient

# ==========================================
# [설정]
# ==========================================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "seoultech_regulations"

def main():
    print(f"🔌 Qdrant 연결 중... ({QDRANT_HOST}:{QDRANT_PORT})")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if client.collection_exists(COLLECTION_NAME):
        user_input = input(f"⚠️ 경고: 컬렉션 '{COLLECTION_NAME}'을 정말로 삭제하시겠습니까? (y/n): ")
        if user_input.lower() == 'y':
            print(f"🗑️ 컬렉션 '{COLLECTION_NAME}' 삭제 중...")
            client.delete_collection(COLLECTION_NAME)
            print("✅ 삭제 완료.")
        else:
            print("❌ 삭제가 취소되었습니다.")
    else:
        print(f"ℹ️ 컬렉션 '{COLLECTION_NAME}'이 존재하지 않습니다.")

if __name__ == "__main__":
    main()