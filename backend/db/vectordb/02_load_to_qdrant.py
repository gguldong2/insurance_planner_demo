import pickle
import os
import time
from qdrant_client import QdrantClient, models

# ==========================================
# [설정]
# ==========================================
INPUT_FILE = "processed_data.pkl"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "seoultech_regulations"
VECTOR_SIZE = 1024  # BGE-M3

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 파일을 찾을 수 없습니다: {INPUT_FILE}")
        print("01_process_and_embed.py를 먼저 실행해주세요.")
        return

    # 1. 클라이언트 연결
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        # 연결 테스트
        client.get_collections()
        print(f"✅ Qdrant 연결 성공 ({QDRANT_HOST}:{QDRANT_PORT})")
    except Exception as e:
        print(f"❌ Qdrant 연결 실패: {e}")
        return

    # 2. 데이터 로드
    print(f"📂 데이터 파일 로딩 중... ({INPUT_FILE})")
    with open(INPUT_FILE, 'rb') as f:
        data_points = pickle.load(f)
    
    total_data_count = len(data_points)
    print(f"📊 총 {total_data_count}개의 데이터를 메모리에 로드했습니다.")

    # 3. 컬렉션 생성 (없을 경우에만)
    if not client.collection_exists(COLLECTION_NAME):
        print(f"🆕 컬렉션 '{COLLECTION_NAME}' 생성 중...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "default": models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams()
            }
        )
    else:
        print(f"ℹ️ 컬렉션 '{COLLECTION_NAME}'이 이미 존재합니다. (Upsert 모드)")

    # 4. PointStruct 변환
    print("🔄 데이터 변환 중...")
    points = []
    for item in data_points:
        vector_data = item['vector']
        points.append(models.PointStruct(
            id=item['id'],
            vector={
                "default": vector_data['dense'].tolist(),
                "sparse": models.SparseVector(
                    indices=vector_data['sparse_indices'],
                    values=vector_data['sparse_values']
                )
            },
            payload=item['payload']
        ))

    # 5. 업로드 (배치 처리 및 로깅 강화)
    batch_size = 100
    total_points = len(points)
    success_count = 0
    fail_count = 0
    
    print(f"🚀 업로드 시작 (총 {total_points}건, 배치크기: {batch_size})...")
    start_time = time.time()

    for i in range(0, total_points, batch_size):
        batch = points[i : i + batch_size]
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )
            success_count += len(batch)
        except Exception as e:
            fail_count += len(batch)
            print(f"\n❌ 배치 업로드 실패 (Index: {i}~{i+len(batch)}): {e}")
            # 실패한 ID들을 출력하고 싶다면 아래 주석 해제
            # failed_ids = [p.id for p in batch]
            # print(f"   -> Failed IDs: {failed_ids}")
        
        # 진행률 출력
        current = min(i + batch_size, total_points)
        percent = (current / total_points) * 100
        # \r을 사용하여 같은 줄에 업데이트
        print(f"\r[{percent:.1f}%] 진행 중... (성공: {success_count} / 실패: {fail_count} / 전체: {total_points})", end="")

    end_time = time.time()
    duration = end_time - start_time
    
    print("\n\n=== 🎉 적재 완료 ===")
    print(f"✅ 총 성공: {success_count}건")
    print(f"❌ 총 실패: {fail_count}건")
    print(f"⏱️ 소요 시간: {duration:.2f}초")

if __name__ == "__main__":
    main()