import os
import re
import uuid
import hashlib
import json
import pickle
from datetime import datetime
from typing import List, Dict, Any

# ==========================================
# [설정]
# ==========================================
DATA_DIR = "./data/md_data"
OUTPUT_FILE = "processed_data.pkl"
DEBUG_FILE = "debug_chunks.json"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# 파일명 분리 패턴 ('〉' 전각문자 또는 '>' 반각문자)
SPLIT_PATTERN = re.compile(r'\s*[〉>]\s*')

try:
    from FlagEmbedding import BGEM3FlagModel
    import torch
except ImportError:
    print("라이브러리 설치 필요: uv add FlagEmbedding torch")
    exit()

# ==========================================
# [Class] 마크다운 파서
# ==========================================
class RegulationParser:
    def __init__(self):
        # 유연한 정규식 (들여쓰기 허용, 띄어쓰기 유연함)
        self.pat_chapter = re.compile(r'^\s*##\s+(제\s*\d+\s*장.*)')
        self.pat_section = re.compile(r'^\s*\*\*(제\s*\d+\s*절.*)\*\*')
        self.pat_article = re.compile(r'^\s*###\s+(제\s*\d+\s*조.*)')
        self.pat_appendix = re.compile(r'^\s*###\s+(\[별표.*)')
        self.pat_addenda_header = re.compile(r'^\s*\*\*(부칙.*)\*\*')
        self.pat_addenda_article = re.compile(r'^\s*(제\s*\d+\s*조\(.*?\))') 

    def extract_meta_from_filename(self, filename: str):
        """파일명 파싱 로직"""
        name_only = filename.replace(".md", "")
        doc_title = name_only
        dept = "미분류"
        
        parts = SPLIT_PATTERN.split(name_only)
        if len(parts) >= 2:
            doc_title = parts[1].strip()
            left_part = parts[0].strip()
            left_tokens = left_part.split(' ', 1)
            if len(left_tokens) == 2 and left_tokens[0].startswith("제"):
                dept = left_tokens[1]
            else:
                dept = left_part
        return doc_title, dept

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        filename = os.path.basename(file_path)
        doc_title, dept = self.extract_meta_from_filename(filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # ---------------------------------------------------------
        # [수정된 핵심 로직] 헤더(메타데이터) 영역 읽기
        # ---------------------------------------------------------
        header_lines = []
        line_idx = 0
        has_title_header = False

        while line_idx < len(lines):
            line = lines[line_idx].rstrip()
            stripped = line.strip()
            
            # [수정] 본문 시작 감지 조건 강화
            # ##(장) 뿐만 아니라, ###(조), **(절/부칙) 중 하나라도 나오면 본문 시작으로 간주
            is_body_start = (
                self.pat_chapter.match(stripped) or 
                self.pat_article.match(stripped) or 
                self.pat_addenda_header.match(stripped) or
                self.pat_section.match(stripped) or
                self.pat_appendix.match(stripped)
            )

            if is_body_start:
                break # 본문 시작! 루프 탈출
            
            if stripped: 
                header_lines.append(line)
                if stripped.startswith("# "): 
                    has_title_header = True
            line_idx += 1
        
        # 제목(#)이 없으면 2번째 줄에 강제 삽입
        if not has_title_header and len(header_lines) >= 2:
            header_lines[1] = "# " + header_lines[1].strip()
        
        header_full_text = "\n".join(header_lines)

        # 날짜 추출
        last_revision_date = None
        date_match = re.findall(r'(\d{4}\.\s?\d{1,2}\.\s?\d{1,2})', filename)
        if not date_match:
            date_match = re.findall(r'(\d{4}\.\s?\d{1,2}\.\s?\d{1,2})', header_full_text)
        if date_match:
            last_revision_date = date_match[-1]

        # 상태 변수
        state = {
            "chapter": None, 
            "section": None,
            "is_addenda": False
        }
        
        current_chunk = {"title": None, "lines": []}
        chunks = []
        
        def save_chunk():
            if current_chunk["title"] and current_chunk["lines"]:
                content_raw = "\n".join(current_chunk["lines"]).strip()
                
                # Payload Content 복원
                payload_content = ""
                if current_chunk['title'].startswith("제") or current_chunk['title'].startswith("[별표"):
                     payload_content = f"### {current_chunk['title']}\n{content_raw}"
                else:
                     payload_content = f"### {current_chunk['title']}\n{content_raw}"

                payload = {
                    "doc_title": doc_title,
                    "dept": dept,
                    "last_revision_date": last_revision_date,
                    "header_full_text": header_full_text,
                    "chapter": state["chapter"],
                    "section": state["section"],
                    "article": current_chunk["title"],
                    "content": payload_content
                }
                
                # Vector Text
                path_parts = [doc_title]
                if state["chapter"]: path_parts.append(state["chapter"])
                if state["section"]: path_parts.append(state["section"])
                path_str = " > ".join(path_parts)
                
                vector_text = f"{path_str} > {current_chunk['title']} :\n{content_raw}"
                if len(vector_text) > 4000:
                    vector_text = vector_text[:4000]

                chunks.append({
                    "vector_text": vector_text,
                    "payload": payload
                })
                
            current_chunk["title"] = None
            current_chunk["lines"] = []

        # 본문 파싱 루프
        while line_idx < len(lines):
            line = lines[line_idx].rstrip()
            stripped = line.strip()
            
            if self.pat_chapter.match(stripped):
                save_chunk()
                state["chapter"] = self.pat_chapter.match(stripped).group(1)
                state["section"] = None
                state["is_addenda"] = False
                
            elif self.pat_addenda_header.match(stripped):
                save_chunk()
                state["chapter"] = "부칙"
                state["section"] = self.pat_addenda_header.match(stripped).group(1)
                state["is_addenda"] = True
                
            elif not state["is_addenda"] and self.pat_section.match(stripped):
                save_chunk()
                state["section"] = self.pat_section.match(stripped).group(1)
                
            elif self.pat_article.match(stripped):
                save_chunk()
                current_chunk["title"] = self.pat_article.match(stripped).group(1)
                
            elif self.pat_appendix.match(stripped):
                save_chunk()
                current_chunk["title"] = self.pat_appendix.match(stripped).group(1)
                state["section"] = None 
                state["is_addenda"] = False

            elif state["is_addenda"] and self.pat_addenda_article.match(stripped):
                save_chunk()
                current_chunk["title"] = self.pat_addenda_article.match(stripped).group(1)

            else:
                if current_chunk["title"]:
                    current_chunk["lines"].append(line)
            
            line_idx += 1
            
        save_chunk()
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
    
    print(f"모델 로딩 중: {EMBEDDING_MODEL_NAME}")
    try:
        model = BGEM3FlagModel(EMBEDDING_MODEL_NAME, use_fp16=True)
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return

    parser = RegulationParser()
    
    all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.md')])
    total_files = len(all_files)
    
    all_data_points = []
    anomaly_files = [] 
    debug_list = [] 

    for idx, fname in enumerate(all_files, 1):
        path = os.path.join(DATA_DIR, fname)
        result = parser.parse_file(path)
        chunks = result["chunks"]
        
        if not chunks:
            anomaly_files.append(fname)
            print(f"[{idx}/{total_files}] [Warning] 청크 생성 실패: {fname}")
            continue
            
        print(f"[{idx}/{total_files}] 처리 중: {fname} ({len(chunks)}개 청크) -> ", end="", flush=True)
        
        try:
            texts = [c["vector_text"] for c in chunks]
            output = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
            
            dense_vecs = output['dense_vecs']
            sparse_vecs = output['lexical_weights']

            for i, chunk in enumerate(chunks):
                pid = generate_id(fname, chunk['payload']['article'])
                
                sp_indices = [int(k) for k in sparse_vecs[i].keys()]
                sp_values = [float(v) for v in sparse_vecs[i].values()]
                
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
                
                debug_item = chunk["payload"].copy()
                debug_item["_vector_text_preview"] = chunk["vector_text"][:200]
                debug_list.append(debug_item)
            
            print("완료!")

        except Exception as e:
            print(f"\n[Error] {fname} 처리 중 에러: {e}")
            continue

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(all_data_points, f)
    
    with open(DEBUG_FILE, 'w', encoding='utf-8') as f:
        json.dump(debug_list, f, ensure_ascii=False, indent=2)

    print("\n=== [완료 보고] ===")
    print(f"총 처리된 파일: {total_files}개")
    print(f"총 생성된 데이터 포인트: {len(all_data_points)}건")
    
    if anomaly_files:
        print("\n[주의] 다음 파일들은 파싱되지 않았습니다:")
        for f in anomaly_files:
            print(f" - {f}")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"'{DATA_DIR}' 폴더가 없습니다.")
    else:
        main()