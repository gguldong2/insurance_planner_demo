import os
import re
import uuid
import hashlib
import json
import pickle
from datetime import datetime
from typing import List, Dict, Any

# ==========================================
# [설정] 환경에 맞게 수정
# ==========================================
DATA_DIR = "./md_data"              # .md 파일 위치
OUTPUT_FILE = "processed_data.pkl" # 적재용 데이터 (Pickle)
DEBUG_FILE = "debug_chunks.json"   # 확인용 데이터 (JSON)
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

try:
    from FlagEmbedding import BGEM3FlagModel
    import torch
except ImportError:
    print("라이브러리 설치 필요: pip install FlagEmbedding torch")
    exit()

# ==========================================
# [Class] 마크다운 파서
# ==========================================
class RegulationParser:
    def __init__(self):
        # 정규표현식 패턴
        self.pat_chapter = re.compile(r'^##\s+(제\d+장.*)')       # ## 제1장
        self.pat_section = re.compile(r'^\*\*(제\d+절.*)\*\*')    # **제1절** (Bold)
        self.pat_article = re.compile(r'^###\s+(제\d+조.*)')      # ### 제1조
        self.pat_appendix = re.compile(r'^###\s+(\[별표.*)')      # ### [별표
        self.pat_addenda_header = re.compile(r'^\*\*(부칙.*)\*\*') # **부칙...**
        # 부칙 내 조항 (### 없이 텍스트로 시작하는 경우 대응)
        self.pat_addenda_article = re.compile(r'^(제\d+조\(.*?\))') 
        self.pat_date = re.compile(r'(\d{4}\.\s?\d{1,2}\.\s?\d{1,2})')

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        filename = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        chunks = []
        header_lines = []
        line_idx = 0
        
        # 1. 헤더(메타데이터) 추출: 첫 '##' 나오기 전까지
        while line_idx < len(lines):
            line = lines[line_idx].strip()
            if line.startswith("## "): break
            if line: header_lines.append(line)
            line_idx += 1
            
        header_full_text = "\n".join(header_lines)
        
        # 1-1. 메타데이터 파싱
        doc_title = header_lines[1] if len(header_lines) > 1 else filename.replace(".md", "")
        dept = "교무과" if "교무" in header_full_text else "미분류" # (필요시 정교화)
        
        last_revision_date = None
        dates = self.pat_date.findall(header_full_text)
        if dates:
            last_revision_date = dates[-1] # 가장 마지막 날짜

        # 2. 본문 파싱 상태 변수
        state = {
            "chapter": None,
            "section": None,
            "is_addenda": False
        }
        
        current_chunk = {"title": None, "lines": []}
        
        def save_chunk():
            if current_chunk["title"] and current_chunk["lines"]:
                # 텍스트 조립 (마크다운 원본 유지)
                content_raw = "\n".join(current_chunk["lines"]).strip()
                
                # Payload 생성 (User 요구사항: 원본 그대로)
                payload = {
                    "doc_title": doc_title,
                    "dept": dept,
                    "last_revision_date": last_revision_date,
                    "header_full_text": header_full_text,
                    
                    # 계층 정보 (없으면 None)
                    "chapter": state["chapter"],
                    "section": state["section"],
                    "article": current_chunk["title"],
                    
                    # 마크다운 원본 내용 (적재용)
                    "content": f"{current_chunk['title']}\n{content_raw}"
                }
                
                # 임베딩용 텍스트 생성 (경로 포함 + 마크다운 일부 유지)
                # 예: [파일명] > [장] > [절] > [조] : [내용]
                path_parts = [doc_title]
                if state["chapter"]: path_parts.append(state["chapter"])
                if state["section"]: path_parts.append(state["section"])
                
                path_str = " > ".join(path_parts)
                # 임베딩 텍스트
                vector_text = f"{path_str} > {current_chunk['title']} :\n{content_raw}"
                
                chunks.append({
                    "vector_text": vector_text,
                    "payload": payload
                })
                
            current_chunk["title"] = None
            current_chunk["lines"] = []

        # 라인 순회
        while line_idx < len(lines):
            line = lines[line_idx].rstrip()
            stripped = line.strip()
            
            # (1) 장 (Chapter)
            if self.pat_chapter.match(stripped):
                save_chunk()
                state["chapter"] = self.pat_chapter.match(stripped).group(1)
                state["section"] = None
                state["is_addenda"] = False
                
            # (2) 부칙 (Addenda)
            elif self.pat_addenda_header.match(stripped):
                save_chunk()
                state["chapter"] = "부칙"
                state["section"] = self.pat_addenda_header.match(stripped).group(1)
                state["is_addenda"] = True
                
            # (3) 절 (Section) - 부칙 아닐 때
            elif not state["is_addenda"] and self.pat_section.match(stripped):
                save_chunk()
                state["section"] = self.pat_section.match(stripped).group(1)
                
            # (4) 조 (Article) - ### 제N조
            elif self.pat_article.match(stripped):
                save_chunk()
                current_chunk["title"] = self.pat_article.match(stripped).group(1)
                
            # (5) 별표 (Appendix)
            elif self.pat_appendix.match(stripped):
                save_chunk()
                current_chunk["title"] = self.pat_appendix.match(stripped).group(1)
                state["chapter"] = None # 별표는 독립적이라 가정 (User 요청 반영)
                state["section"] = None
                state["is_addenda"] = False

            # (6) 부칙 내 조항 - 제N조...
            elif state["is_addenda"] and self.pat_addenda_article.match(stripped):
                save_chunk()
                current_chunk["title"] = self.pat_addenda_article.match(stripped).group(1)

            # (7) 내용 누적
            else:
                if current_chunk["title"]:
                    current_chunk["lines"].append(line)
            
            line_idx += 1
            
        save_chunk() # 마지막 처리
        return {"filename": filename, "chunks": chunks}

# ==========================================
# [ID 생성기]
# ==========================================
def generate_id(filename, article_title):
    unique_str = f"{filename}_{article_title}"
    hash_obj = hashlib.md5(unique_str.encode('utf-8'))
    return str(uuid.UUID(hex=hash_obj.hexdigest()))

# ==========================================
# [메인 실행]
# ==========================================
def main():
    print("=== [Step 1] 파싱 및 임베딩 시작 ===")
    
    # 1. 모델 로드
    print(f"모델 로딩 중: {EMBEDDING_MODEL_NAME}")
    model = BGEM3FlagModel(EMBEDDING_MODEL_NAME, use_fp16=True)
    parser = RegulationParser()
    
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.md')]
    all_data_points = []
    anomaly_files = [] # 구조가 이상한 파일
    
    debug_list = [] # JSON 저장용

    for fname in all_files:
        path = os.path.join(DATA_DIR, fname)
        result = parser.parse_file(path)
        chunks = result["chunks"]
        
        # [Anomaly Check] 파싱된 청크가 하나도 없으면 이상한 파일
        if not chunks:
            anomaly_files.append(fname)
            continue
            
        print(f"처리 중: {fname} ({len(chunks)}개 청크)")
        
        # 2. 임베딩 (Dense + Sparse)
        texts = [c["vector_text"] for c in chunks]
        output = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        
        dense_vecs = output['dense_vecs']
        sparse_vecs = output['lexical_weights']

        for i, chunk in enumerate(chunks):
            # ID 생성
            pid = generate_id(fname, chunk['payload']['article'])
            
            # Sparse 데이터 변환 (TokenID: Weight -> Indices, Values)
            sp_indices = [int(k) for k in sparse_vecs[i].keys()]
            sp_values = [float(v) for v in sparse_vecs[i].values()]
            
            # 저장할 데이터 구조
            data_point = {
                "id": pid,
                "vector": {
                    "dense": dense_vecs[i],
                    "sparse_indices": sp_indices,
                    "sparse_values": sp_values
                },
                "payload": chunk["payload"]
            }
            all_data_points.append(data_point)
            
            # 디버그용 (벡터 제외)
            debug_item = chunk["payload"].copy()
            debug_item["vector_text_preview"] = chunk["vector_text"][:50] + "..."
            debug_list.append(debug_item)

    # 3. 결과 저장
    # (1) 적재용 Pickle 저장
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(all_data_points, f)
    
    # (2) 확인용 JSON 저장
    with open(DEBUG_FILE, 'w', encoding='utf-8') as f:
        json.dump(debug_list, f, ensure_ascii=False, indent=2)

    print("\n=== [완료 보고] ===")
    print(f"총 처리된 청크: {len(all_data_points)}개")
    print(f"적재용 데이터 저장됨: {OUTPUT_FILE}")
    print(f"검수용 데이터 저장됨: {DEBUG_FILE}")
    
    if anomaly_files:
        print("\n[주의] 다음 파일들은 구조가 예상과 달라 청크가 생성되지 않았습니다:")
        for f in anomaly_files:
            print(f" - {f}")
    else:
        print("\n모든 파일이 정상적으로 파싱되었습니다.")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"'{DATA_DIR}' 폴더가 없습니다. 생성했습니다.")
    else:
        main()