import pickle
import os
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
        print(f"파일을 찾을 수 없습니다: {INPUT_FILE}")
        print("01_process_and_embed.py를 먼저 실행해주세요.")
        return

    # 1. 클라이언트 연결
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # 2. 데이터 로드
    print(f"데이터 파일 로딩 중... ({INPUT_FILE})")
    with open(INPUT_FILE, 'rb') as f:
        data_points = pickle.load(f)
    
    print(f"총 {len(data_points)}개의 데이터를 적재할 준비가 되었습니다.")

    # 3. 컬렉션 생성 (없을 경우에만)
    if not client.collection_exists(COLLECTION_NAME):
        print(f"컬렉션 '{COLLECTION_NAME}' 생성 중...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            # Dense Vector 설정
            vectors_config={
                "default": models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            },
            # Sparse Vector 설정 (BGE-M3용)
            sparse_vectors_config={
                "sparse": models.SparseVectorParams()
            }
        )
    else:
        print(f"컬렉션 '{COLLECTION_NAME}'이 이미 존재합니다. (기존 데이터에 Upsert 수행)")

    # 4. 포인트 변환 및 적재
    # Qdrant에 보낼 PointStruct 형태로 변환
    points = []
    for item in data_points:
        vector_data = item['vector']
        
        points.append(models.PointStruct(
            id=item['id'],
            vector={
                "default": vector_data['dense'].tolist(), # Dense
                "sparse": models.SparseVector(            # Sparse
                    indices=vector_data['sparse_indices'],
                    values=vector_data['sparse_values']
                )
            },
            payload=item['payload']
        ))

    # 5. 업로드 (배치 처리 권장하지만, 10MB 수준이므로 한 번에 전송)
    # 데이터가 많으면 batch_size를 조절해서 나눠서 보내야 함 (여기선 100개씩)
    batch_size = 100
    total_batches = (len(points) + batch_size - 1) // batch_size
    
    print(f"업로드 시작 (총 {len(points)}건)...")
    
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
        print(f"진행률: {min(i + batch_size, len(points))} / {len(points)} 완료")

    print("\n=== 모든 데이터 적재 완료 ===")

if __name__ == "__main__":
    main()