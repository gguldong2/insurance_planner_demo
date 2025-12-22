# import os
# import re
# import uuid
# import hashlib
# import json
# import pickle
# from datetime import datetime
# from typing import List, Dict, Any

# # ==========================================
# # [설정]
# # ==========================================
# DATA_DIR = "./data/md_data"
# OUTPUT_FILE = "processed_data.pkl"
# DEBUG_FILE = "debug_chunks.json"
# EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# # 파일명 분리 패턴 ('〉' 전각문자 또는 '>' 반각문자)
# SPLIT_PATTERN = re.compile(r'\s*[〉>]\s*')

# try:
#     from FlagEmbedding import BGEM3FlagModel
#     import torch
# except ImportError:
#     print("라이브러리 설치 필요: uv add FlagEmbedding torch")
#     exit()

# # ==========================================
# # [Class] 마크다운 파서
# # ==========================================
# class RegulationParser:
#     def __init__(self):
#         # 유연한 정규식 (들여쓰기 허용, 띄어쓰기 유연함)
#         self.pat_chapter = re.compile(r'^\s*##\s+(제\s*\d+\s*장.*)')
#         self.pat_section = re.compile(r'^\s*\*\*(제\s*\d+\s*절.*)\*\*')
#         self.pat_article = re.compile(r'^\s*###\s+(제\s*\d+\s*조.*)')
#         self.pat_appendix = re.compile(r'^\s*###\s+(\[별표.*)')
#         self.pat_addenda_header = re.compile(r'^\s*\*\*(부칙.*)\*\*')
#         self.pat_addenda_article = re.compile(r'^\s*(제\s*\d+\s*조\(.*?\))') 

#     def extract_meta_from_filename(self, filename: str):
#         """파일명 파싱 로직"""
#         name_only = filename.replace(".md", "")
#         doc_title = name_only
#         dept = "미분류"
        
#         parts = SPLIT_PATTERN.split(name_only)
#         if len(parts) >= 2:
#             doc_title = parts[1].strip()
#             left_part = parts[0].strip()
#             left_tokens = left_part.split(' ', 1)
#             if len(left_tokens) == 2 and left_tokens[0].startswith("제"):
#                 dept = left_tokens[1]
#             else:
#                 dept = left_part
#         return doc_title, dept

#     def parse_file(self, file_path: str) -> Dict[str, Any]:
#         filename = os.path.basename(file_path)
#         doc_title, dept = self.extract_meta_from_filename(filename)

#         with open(file_path, 'r', encoding='utf-8') as f:
#             lines = f.readlines()

#         # ---------------------------------------------------------
#         # [수정된 핵심 로직] 헤더(메타데이터) 영역 읽기
#         # ---------------------------------------------------------
#         header_lines = []
#         line_idx = 0
#         has_title_header = False

#         while line_idx < len(lines):
#             line = lines[line_idx].rstrip()
#             stripped = line.strip()
            
#             # [수정] 본문 시작 감지 조건 강화
#             # ##(장) 뿐만 아니라, ###(조), **(절/부칙) 중 하나라도 나오면 본문 시작으로 간주
#             is_body_start = (
#                 self.pat_chapter.match(stripped) or 
#                 self.pat_article.match(stripped) or 
#                 self.pat_addenda_header.match(stripped) or
#                 self.pat_section.match(stripped) or
#                 self.pat_appendix.match(stripped)
#             )

#             if is_body_start:
#                 break # 본문 시작! 루프 탈출
            
#             if stripped: 
#                 header_lines.append(line)
#                 if stripped.startswith("# "): 
#                     has_title_header = True
#             line_idx += 1
        
#         # 제목(#)이 없으면 2번째 줄에 강제 삽입
#         if not has_title_header and len(header_lines) >= 2:
#             header_lines[1] = "# " + header_lines[1].strip()
        
#         header_full_text = "\n".join(header_lines)

#         # 날짜 추출
#         last_revision_date = None
#         date_match = re.findall(r'(\d{4}\.\s?\d{1,2}\.\s?\d{1,2})', filename)
#         if not date_match:
#             date_match = re.findall(r'(\d{4}\.\s?\d{1,2}\.\s?\d{1,2})', header_full_text)
#         if date_match:
#             last_revision_date = date_match[-1]

#         # 상태 변수
#         state = {
#             "chapter": None, 
#             "section": None,
#             "is_addenda": False
#         }
        
#         current_chunk = {"title": None, "lines": []}
#         chunks = []
        
#         def save_chunk():
#             if current_chunk["title"] and current_chunk["lines"]:
#                 content_raw = "\n".join(current_chunk["lines"]).strip()
                
#                 # Payload Content 복원
#                 payload_content = ""
#                 if current_chunk['title'].startswith("제") or current_chunk['title'].startswith("[별표"):
#                      payload_content = f"### {current_chunk['title']}\n{content_raw}"
#                 else:
#                      payload_content = f"### {current_chunk['title']}\n{content_raw}"

#                 payload = {
#                     "doc_title": doc_title,
#                     "dept": dept,
#                     "last_revision_date": last_revision_date,
#                     "header_full_text": header_full_text,
#                     "chapter": state["chapter"],
#                     "section": state["section"],
#                     "article": current_chunk["title"],
#                     "content": payload_content
#                 }
                
#                 # Vector Text
#                 path_parts = [doc_title]
#                 if state["chapter"]: path_parts.append(state["chapter"])
#                 if state["section"]: path_parts.append(state["section"])
#                 path_str = " > ".join(path_parts)
                
#                 vector_text = f"{path_str} > {current_chunk['title']} :\n{content_raw}"
#                 if len(vector_text) > 4000:
#                     vector_text = vector_text[:4000]

#                 chunks.append({
#                     "vector_text": vector_text,
#                     "payload": payload
#                 })
                
#             current_chunk["title"] = None
#             current_chunk["lines"] = []

#         # 본문 파싱 루프
#         while line_idx < len(lines):
#             line = lines[line_idx].rstrip()
#             stripped = line.strip()
            
#             if self.pat_chapter.match(stripped):
#                 save_chunk()
#                 state["chapter"] = self.pat_chapter.match(stripped).group(1)
#                 state["section"] = None
#                 state["is_addenda"] = False
                
#             elif self.pat_addenda_header.match(stripped):
#                 save_chunk()
#                 state["chapter"] = "부칙"
#                 state["section"] = self.pat_addenda_header.match(stripped).group(1)
#                 state["is_addenda"] = True
                
#             elif not state["is_addenda"] and self.pat_section.match(stripped):
#                 save_chunk()
#                 state["section"] = self.pat_section.match(stripped).group(1)
                
#             elif self.pat_article.match(stripped):
#                 save_chunk()
#                 current_chunk["title"] = self.pat_article.match(stripped).group(1)
                
#             elif self.pat_appendix.match(stripped):
#                 save_chunk()
#                 current_chunk["title"] = self.pat_appendix.match(stripped).group(1)
#                 state["section"] = None 
#                 state["is_addenda"] = False

#             elif state["is_addenda"] and self.pat_addenda_article.match(stripped):
#                 save_chunk()
#                 current_chunk["title"] = self.pat_addenda_article.match(stripped).group(1)

#             else:
#                 if current_chunk["title"]:
#                     current_chunk["lines"].append(line)
            
#             line_idx += 1
            
#         save_chunk()
#         return {"filename": filename, "chunks": chunks}

# # ==========================================
# # [ID 생성기]
# # ==========================================
# def generate_id(filename, article_title):
#     unique_str = f"{filename}_{article_title}"
#     hash_obj = hashlib.md5(unique_str.encode('utf-8'))
#     return str(uuid.UUID(hex=hash_obj.hexdigest()))

# # ==========================================
# # [메인 실행]
# # ==========================================
# def main():
#     print("=== [Step 1] 파싱 및 임베딩 시작 ===")
    
#     print(f"모델 로딩 중: {EMBEDDING_MODEL_NAME}")
#     try:
#         model = BGEM3FlagModel(EMBEDDING_MODEL_NAME, use_fp16=True)
#     except Exception as e:
#         print(f"모델 로드 실패: {e}")
#         return

#     parser = RegulationParser()
    
#     all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.md')])
#     total_files = len(all_files)
    
#     all_data_points = []
#     anomaly_files = [] 
#     debug_list = [] 

#     for idx, fname in enumerate(all_files, 1):
#         path = os.path.join(DATA_DIR, fname)
#         result = parser.parse_file(path)
#         chunks = result["chunks"]
        
#         if not chunks:
#             anomaly_files.append(fname)
#             print(f"[{idx}/{total_files}] [Warning] 청크 생성 실패: {fname}")
#             continue
            
#         print(f"[{idx}/{total_files}] 처리 중: {fname} ({len(chunks)}개 청크) -> ", end="", flush=True)
        
#         try:
#             texts = [c["vector_text"] for c in chunks]
#             output = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
            
#             dense_vecs = output['dense_vecs']
#             sparse_vecs = output['lexical_weights']

#             for i, chunk in enumerate(chunks):
#                 pid = generate_id(fname, chunk['payload']['article'])
                
#                 sp_indices = [int(k) for k in sparse_vecs[i].keys()]
#                 sp_values = [float(v) for v in sparse_vecs[i].values()]
                
#                 data_point = {
#                     "id": pid,
#                     "vector": {
#                         "dense": dense_vecs[i],
#                         "sparse_indices": sp_indices,
#                         "sparse_values": sp_values
#                     },
#                     "payload": chunk["payload"]
#                 }
#                 all_data_points.append(data_point)
                
#                 debug_item = chunk["payload"].copy()
#                 debug_item["_vector_text_preview"] = chunk["vector_text"][:200]
#                 debug_list.append(debug_item)
            
#             print("완료!")

#         except Exception as e:
#             print(f"\n[Error] {fname} 처리 중 에러: {e}")
#             continue

#     with open(OUTPUT_FILE, 'wb') as f:
#         pickle.dump(all_data_points, f)
    
#     with open(DEBUG_FILE, 'w', encoding='utf-8') as f:
#         json.dump(debug_list, f, ensure_ascii=False, indent=2)

#     print("\n=== [완료 보고] ===")
#     print(f"총 처리된 파일: {total_files}개")
#     print(f"총 생성된 데이터 포인트: {len(all_data_points)}건")
    
#     if anomaly_files:
#         print("\n[주의] 다음 파일들은 파싱되지 않았습니다:")
#         for f in anomaly_files:
#             print(f" - {f}")

# if __name__ == "__main__":
#     if not os.path.exists(DATA_DIR):
#         os.makedirs(DATA_DIR)
#         print(f"'{DATA_DIR}' 폴더가 없습니다.")
#     else:
#         main()

###############################################################################################

# import os
# import re
# import uuid
# import hashlib
# import json
# import pickle
# from datetime import datetime
# from typing import List, Dict, Any, Optional

# # ==========================================
# # [설정]
# # ==========================================
# DATA_DIR = "./data/test/"
# OUTPUT_FILE = "processed_data.pkl"
# DEBUG_FILE = "debug_chunks.json"
# EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# # 파일명 분리 패턴 ('〉' 전각문자 또는 '>' 반각문자)
# SPLIT_PATTERN = re.compile(r'\s*[〉>]\s*')

# try:
#     from FlagEmbedding import BGEM3FlagModel
#     import torch
# except ImportError:
#     print("라이브러리 설치 필요: uv add FlagEmbedding torch")
#     exit()

# # ==========================================
# # [Class] 마크다운 파서
# # ==========================================
# class RegulationParser:
#     def __init__(self):
#         # 1. 일반 장/절/조 패턴
#         self.pat_chapter = re.compile(r'^\s*##\s+(제\s*\d+\s*장.*)')
#         self.pat_section = re.compile(r'^\s*\*\*(제\s*\d+\s*절.*)\*\*')
#         self.pat_article = re.compile(r'^\s*###\s+(제\s*\d+\s*조.*)')
        
#         # 2. [수정] 부칙 패턴 (다양한 형식 대응: **부칙**, 부칙, 부 칙, 부칙(날짜) 등)
#         # ^(시작), \s*(공백), (**)?(볼드옵션), 부\s*칙(글자), .*(나머지)
#         self.pat_addenda_header = re.compile(r'^\s*(\*\*)?부\s*칙.*')
        
#         # 3. [수정] 부칙 내 조항 (제1조, 제 1 조 등) - 헤딩기호(#) 없이 시작하는 경우 대응
#         self.pat_addenda_article = re.compile(r'^\s*(제\s*\d+\s*조.*)')

#         # 4. [수정] 별표 패턴 (### [별표, ## [별표, 그냥 [별표 등 대응)
#         self.pat_appendix = re.compile(r'^\s*(#+)?\s*(\[별표.*)')

#     def extract_meta_from_filename(self, filename: str):
#         """파일명 파싱 로직"""
#         name_only = filename.replace(".md", "")
#         doc_title = name_only
#         dept = "미분류"
#         parts = SPLIT_PATTERN.split(name_only)
#         if len(parts) >= 2:
#             doc_title = parts[1].strip()
#             left_part = parts[0].strip()
#             left_tokens = left_part.split(' ', 1)
#             if len(left_tokens) == 2 and left_tokens[0].startswith("제"):
#                 dept = left_tokens[1]
#             else:
#                 dept = left_part
#         return doc_title, dept

#     def parse_file(self, file_path: str) -> Dict[str, Any]:
#         filename = os.path.basename(file_path)
#         doc_title, dept = self.extract_meta_from_filename(filename)

#         with open(file_path, 'r', encoding='utf-8') as f:
#             lines = f.readlines()

#         # 헤더 읽기 및 보정
#         header_lines = []
#         line_idx = 0
#         has_title_header = False

#         while line_idx < len(lines):
#             line = lines[line_idx].rstrip()
#             stripped = line.strip()
#             # 본문 시작 감지 (장, 절, 조, 부칙, 별표 중 하나라도 나오면)
#             is_body_start = (
#                 self.pat_chapter.match(stripped) or 
#                 self.pat_article.match(stripped) or 
#                 self.pat_addenda_header.match(stripped) or
#                 self.pat_section.match(stripped) or
#                 self.pat_appendix.match(stripped)
#             )
#             if is_body_start: break
            
#             if stripped: 
#                 header_lines.append(line)
#                 if stripped.startswith("# "): has_title_header = True
#             line_idx += 1
        
#         if not has_title_header and len(header_lines) >= 2:
#             header_lines[1] = "# " + header_lines[1].strip()
        
#         header_full_text = "\n".join(header_lines)

#         # 날짜 추출
#         last_revision_date = None
#         date_match = re.findall(r'(\d{4}\.\s?\d{1,2}\.\s?\d{1,2})', filename)
#         if not date_match:
#             date_match = re.findall(r'(\d{4}\.\s?\d{1,2}\.\s?\d{1,2})', header_full_text)
#         if date_match: last_revision_date = date_match[-1]

#         # 상태 변수
#         state = {
#             "chapter": None, 
#             "section": None,
#             "mode": "normal" # normal, addenda, appendix
#         }
        
#         # 청크 버퍼: title이 None이면 '조'가 없는 일반 텍스트 덩어리로 간주
#         current_chunk = {"title": None, "lines": []}
#         chunks = []
        
#         def save_chunk():
#             # 저장할 내용이 없으면 스킵
#             if not current_chunk["lines"]: return

#             content_raw = "\n".join(current_chunk["lines"]).strip()
#             if not content_raw: return

#             # [핵심 로직] 부칙/별표에 따른 필드 강제 할당
#             final_chapter = state["chapter"]
#             final_section = state["section"]
#             final_article = current_chunk["title"]
            
#             # 1. 부칙 모드일 때
#             if state["mode"] == "addenda":
#                 final_chapter = "부칙"
#                 final_section = "부칙"
#                 # title이 "__General__" (임시 마커)이거나 None이면 article은 null
#                 if final_article == "__General__" or not final_article:
#                     final_article = None 
            
#             # 2. 별표 모드일 때
#             elif state["mode"] == "appendix":
#                 final_chapter = "별표"
#                 final_section = "별표"
#                 # 별표는 title이 곧 article ([별표 1]...)
            
#             # Content 포맷팅 (User 요청 반영)
#             # 1. 일반 조항: ### 제1조...
#             # 2. 별표: ### [별표... (또는 원본 유지)
#             # 3. 부칙(조 있음): 제1조... (### 없이)
#             # 4. 부칙(조 없음): 내용만
            
#             payload_content = ""
#             if final_article:
#                 # 별표나 일반 조항은 ### 붙임 (LLM 인지용)
#                 if state["mode"] == "appendix":
#                      # 별표는 원본 헤더가 이미 lines에 포함되어 있을 수 있으므로 중복 방지 체크 필요하지만,
#                      # 여기서는 title을 헤더로 쓰고 lines는 내용으로 씀.
#                      payload_content = f"### {final_article}\n{content_raw}"
#                 elif state["mode"] == "addenda":
#                      # 부칙의 조항은 보통 ###을 안 붙임 (User 예시: "제1조(시행일) 이 규정은...")
#                      payload_content = f"{final_article} {content_raw}" 
#                      # 만약 content_raw가 이미 "이 규정은..." 처럼 내용만 있다면 위가 맞고,
#                      # content_raw에 제목이 포함되어 있다면 중복될 수 있음. 
#                      # 코드 구조상 current_chunk['title']은 정규식 그룹 1이고, lines는 그 다음 줄부터임.
#                      # 따라서 헤더를 붙여주는 게 맞음.
#                      payload_content = f"{final_article}\n{content_raw}"
#                 else:
#                      # 일반 조항
#                      payload_content = f"### {final_article}\n{content_raw}"
#             else:
#                 # Article이 없는 경우 (부칙의 단순 텍스트)
#                 payload_content = content_raw

#             payload = {
#                 "doc_title": doc_title,
#                 "dept": dept,
#                 "last_revision_date": last_revision_date,
#                 "header_full_text": header_full_text,
#                 "chapter": final_chapter,
#                 "section": final_section,
#                 "article": final_article, # 부칙 텍스트일 경우 None(null) 들어감
#                 "content": payload_content
#             }
            
#             # Vector Text 생성
#             path_parts = [doc_title]
#             if final_chapter: path_parts.append(final_chapter)
#             if final_section and final_section != final_chapter: path_parts.append(final_section)
            
#             path_str = " > ".join(path_parts)
            
#             # 제목이 있으면 붙이고, 없으면 내용만
#             if final_article:
#                 vector_text = f"{path_str} > {final_article} :\n{content_raw}"
#             else:
#                 vector_text = f"{path_str} :\n{content_raw}"
            
#             if len(vector_text) > 4000:
#                 vector_text = vector_text[:4000]

#             chunks.append({
#                 "vector_text": vector_text,
#                 "payload": payload
#             })
            
#             # 저장 후 버퍼 초기화
#             current_chunk["title"] = None
#             current_chunk["lines"] = []


#         # 본문 파싱 루프
#         while line_idx < len(lines):
#             line = lines[line_idx].rstrip()
#             stripped = line.strip()
            
#             # 1. 별표 감지 (최우선)
#             if self.pat_appendix.match(stripped):
#                 save_chunk()
#                 state["mode"] = "appendix"
#                 state["chapter"] = "별표" # 임시
#                 state["section"] = "별표"
#                 # 별표 제목 전체 추출 (예: [별표 1] 규정명)
#                 match = self.pat_appendix.match(stripped)
#                 current_chunk["title"] = match.group(2) # group 1은 #, group 2가 [별표...
#                 # 별표는 내용에 표가 오므로 lines 수집 시작

#             # 2. 부칙 헤더 감지
#             elif self.pat_addenda_header.match(stripped):
#                 save_chunk()
#                 state["mode"] = "addenda"
#                 state["chapter"] = "부칙"
#                 state["section"] = "부칙"
#                 # 부칙 헤더 자체는 청크로 만들지 않고, 그 아래 내용부터 수집
#                 # title을 None으로 두어 "조"가 없는 텍스트가 나오면 "__General__"로 처리되게 함

#             # 3. 부칙 내 조항 감지 (부칙 모드일 때만)
#             elif state["mode"] == "addenda" and self.pat_addenda_article.match(stripped):
#                 save_chunk()
#                 # 제1조... 추출
#                 current_chunk["title"] = self.pat_addenda_article.match(stripped).group(1)

#             # 4. 일반 Chapter (##) - 일반 모드 또는 부칙 모드 종료(드물지만)
#             elif self.pat_chapter.match(stripped):
#                 save_chunk()
#                 state["mode"] = "normal"
#                 state["chapter"] = self.pat_chapter.match(stripped).group(1)
#                 state["section"] = None

#             # 5. 일반 Section (**제N절**)
#             elif state["mode"] == "normal" and self.pat_section.match(stripped):
#                 save_chunk()
#                 state["section"] = self.pat_section.match(stripped).group(1)

#             # 6. 일반 Article (### 제N조)
#             elif state["mode"] == "normal" and self.pat_article.match(stripped):
#                 save_chunk()
#                 current_chunk["title"] = self.pat_article.match(stripped).group(1)

#             # 7. 내용 누적
#             else:
#                 # 부칙 모드인데 아직 title(제N조)이 없는 경우 -> 일반 텍스트 부칙
#                 if state["mode"] == "addenda" and current_chunk["title"] is None:
#                     current_chunk["title"] = "__General__" # 마커 설정
#                     current_chunk["lines"].append(line)
                
#                 # 별표 모드이거나, 제목이 세팅된 상태면 내용 추가
#                 elif current_chunk["title"] or state["mode"] == "appendix":
#                     current_chunk["lines"].append(line)
            
#             line_idx += 1
            
#         save_chunk() # 마지막 청크 저장
#         return {"filename": filename, "chunks": chunks}

# # ==========================================
# # [ID 생성기]
# # ==========================================
# def generate_id(filename, article_title):
#     # article_title이 None일 경우(부칙 텍스트) 처리
#     title_part = article_title if article_title else "general_text"
#     unique_str = f"{filename}_{title_part}"
#     hash_obj = hashlib.md5(unique_str.encode('utf-8'))
#     return str(uuid.UUID(hex=hash_obj.hexdigest()))

# # ==========================================
# # [메인 실행]
# # ==========================================
# def main():
#     print("=== [Step 1] 파싱 및 임베딩 시작 ===")
    
#     print(f"모델 로딩 중: {EMBEDDING_MODEL_NAME}")
#     try:
#         model = BGEM3FlagModel(EMBEDDING_MODEL_NAME, use_fp16=True)
#     except Exception as e:
#         print(f"모델 로드 실패: {e}")
#         return

#     parser = RegulationParser()
    
#     all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.md')])
#     total_files = len(all_files)
    
#     all_data_points = []
#     anomaly_files = [] 
#     debug_list = [] 

#     for idx, fname in enumerate(all_files, 1):
#         path = os.path.join(DATA_DIR, fname)
#         result = parser.parse_file(path)
#         chunks = result["chunks"]
        
#         if not chunks:
#             anomaly_files.append(fname)
#             print(f"[{idx}/{total_files}] [Warning] 청크 생성 실패: {fname}")
#             continue
            
#         print(f"[{idx}/{total_files}] 처리 중: {fname} ({len(chunks)}개 청크) -> ", end="", flush=True)
        
#         try:
#             texts = [c["vector_text"] for c in chunks]
#             output = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
            
#             dense_vecs = output['dense_vecs']
#             sparse_vecs = output['lexical_weights']

#             for i, chunk in enumerate(chunks):
#                 pid = generate_id(fname, chunk['payload']['article'])
                
#                 sp_indices = [int(k) for k in sparse_vecs[i].keys()]
#                 sp_values = [float(v) for v in sparse_vecs[i].values()]
                
#                 data_point = {
#                     "id": pid,
#                     "vector": {
#                         "dense": dense_vecs[i],
#                         "sparse_indices": sp_indices,
#                         "sparse_values": sp_values
#                     },
#                     "payload": chunk["payload"]
#                 }
#                 all_data_points.append(data_point)
                
#                 debug_item = chunk["payload"].copy()
#                 debug_item["_vector_text_preview"] = chunk["vector_text"][:200]
#                 debug_list.append(debug_item)
            
#             print("완료!")

#         except Exception as e:
#             print(f"\n[Error] {fname} 처리 중 에러: {e}")
#             continue

#     with open(OUTPUT_FILE, 'wb') as f:
#         pickle.dump(all_data_points, f)
    
#     with open(DEBUG_FILE, 'w', encoding='utf-8') as f:
#         json.dump(debug_list, f, ensure_ascii=False, indent=2)

#     print("\n=== [완료 보고] ===")
#     print(f"총 처리된 파일: {total_files}개")
#     print(f"총 생성된 데이터 포인트: {len(all_data_points)}건")
#     print(f"검수용 파일 확인하세요: {DEBUG_FILE}")
    
#     if anomaly_files:
#         print("\n[주의] 다음 파일들은 파싱되지 않았습니다:")
#         for f in anomaly_files:
#             print(f" - {f}")

# if __name__ == "__main__":
#     if not os.path.exists(DATA_DIR):
#         os.makedirs(DATA_DIR)
#         print(f"'{DATA_DIR}' 폴더가 없습니다.")
#     else:
#         main()


###############################################################################################


import os
import re
import uuid
import hashlib
import json
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional

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
        # 1. 일반 장/절/조 패턴
        self.pat_chapter = re.compile(r'^\s*##\s+(제\s*\d+\s*장.*)')
        self.pat_section = re.compile(r'^\s*\*\*(제\s*\d+\s*절.*)\*\*')
        self.pat_article = re.compile(r'^\s*###\s+(제\s*\d+\s*조.*)')
        
        # 2. 부칙 패턴 (부 칙, **부칙**, 부칙(날짜) 등)
        self.pat_addenda_header = re.compile(r'^\s*(\*\*)?부\s*칙.*')
        
        # 3. 부칙 내 조항 (제1조, 제 1 조 등) - # 유무 관계없이 잡음
        self.pat_addenda_article = re.compile(r'^\s*(#+)?\s*(제\s*\d+\s*조.*)')

        # 4. 별표 패턴
        self.pat_appendix = re.compile(r'^\s*(#+)?\s*(\[별표.*)')

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

        # 헤더 읽기 및 보정
        header_lines = []
        line_idx = 0
        has_title_header = False

        while line_idx < len(lines):
            line = lines[line_idx].rstrip()
            stripped = line.strip()
            # 본문 시작 감지
            is_body_start = (
                self.pat_chapter.match(stripped) or 
                self.pat_article.match(stripped) or 
                self.pat_addenda_header.match(stripped) or
                self.pat_section.match(stripped) or
                self.pat_appendix.match(stripped)
            )
            if is_body_start: break
            
            if stripped: 
                header_lines.append(line)
                if stripped.startswith("# "): has_title_header = True
            line_idx += 1
        
        if not has_title_header and len(header_lines) >= 2:
            header_lines[1] = "# " + header_lines[1].strip()
        
        header_full_text = "\n".join(header_lines)

        last_revision_date = None
        date_match = re.findall(r'(\d{4}\.\s?\d{1,2}\.\s?\d{1,2})', filename)
        if not date_match:
            date_match = re.findall(r'(\d{4}\.\s?\d{1,2}\.\s?\d{1,2})', header_full_text)
        if date_match: last_revision_date = date_match[-1]

        state = {
            "chapter": None, 
            "section": None,
            "mode": "normal" 
        }
        
        current_chunk = {"title": None, "lines": []}
        chunks = []
        
        # -------------------------------------------------------
        # [핵심 수정 함수] save_chunk
        # -------------------------------------------------------
        def save_chunk():
            if not current_chunk["lines"]: return
            content_raw = "\n".join(current_chunk["lines"]).strip()
            if not content_raw: return

            final_chapter = state["chapter"]
            final_section = state["section"]
            final_article = current_chunk["title"]
            
            # 모드별 챕터/섹션 보정
            if state["mode"] == "addenda":
                final_chapter = "부칙"
                if final_article == "__General__" or not final_article:
                    final_article = None 
            elif state["mode"] == "appendix":
                final_chapter = "별표"
                final_section = "별표"

            # ---------------------------------------------------------
            # 1. 경로(Path) 문자열 먼저 생성 (이전보다 위로 이동)
            # ---------------------------------------------------------
            path_parts = [doc_title]
            if final_chapter: 
                path_parts.append(final_chapter)
            if final_section and final_section != final_chapter: 
                path_parts.append(final_section)
            
            path_str = " > ".join(path_parts)

            # ---------------------------------------------------------
            # 2. LLM용 Content 구성 (경로 헤더 추가)
            #    예: [출처: 학칙 > 제1장 총칙] \n ### 제1조...
            # ---------------------------------------------------------
            payload_content = ""
            header_str = f"[출처: {path_str}]" # ★ 여기에 경로를 박아넣음

            if final_article:
                if state["mode"] == "appendix":
                     payload_content = f"{header_str}\n### {final_article}\n{content_raw}"
                elif state["mode"] == "addenda":
                     if content_raw.startswith(final_article):
                         payload_content = f"{header_str}\n{content_raw}"
                     else:
                         payload_content = f"{header_str}\n{final_article}\n{content_raw}"
                else:
                     payload_content = f"{header_str}\n### {final_article}\n{content_raw}"
            else:
                payload_content = f"{header_str}\n{content_raw}"

            payload = {
                "doc_title": doc_title,
                "dept": dept,
                "last_revision_date": last_revision_date,
                "header_full_text": header_full_text,
                "chapter": final_chapter,
                "section": final_section,
                "article": final_article, 
                "content": payload_content # ★ 경로가 포함된 내용 저장
            }
            
            # ---------------------------------------------------------
            # 3. Vector용 Text 구성 (검색용)
            # ---------------------------------------------------------
            if final_article:
                vector_text = f"{path_str} > {final_article} :\n{content_raw}"
            else:
                vector_text = f"{path_str} :\n{content_raw}"
            
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
            
            if self.pat_appendix.match(stripped):
                save_chunk()
                state["mode"] = "appendix"
                state["chapter"] = "별표"
                state["section"] = "별표"
                match = self.pat_appendix.match(stripped)
                current_chunk["title"] = match.group(2)

            elif self.pat_addenda_header.match(stripped):
                save_chunk()
                state["mode"] = "addenda"
                state["chapter"] = "부칙"
                raw_header = stripped.replace("**", "").strip()
                state["section"] = raw_header
                current_chunk["title"] = None

            elif state["mode"] == "addenda" and self.pat_addenda_article.match(stripped):
                save_chunk()
                current_chunk["title"] = self.pat_addenda_article.match(stripped).group(2)
                if not stripped.startswith("###"): 
                    current_chunk["lines"].append(line)

            elif self.pat_chapter.match(stripped):
                save_chunk()
                state["mode"] = "normal"
                state["chapter"] = self.pat_chapter.match(stripped).group(1)
                state["section"] = None

            elif state["mode"] == "normal" and self.pat_section.match(stripped):
                save_chunk()
                state["section"] = self.pat_section.match(stripped).group(1)

            elif state["mode"] == "normal" and self.pat_article.match(stripped):
                save_chunk()
                current_chunk["title"] = self.pat_article.match(stripped).group(1)

            else:
                if state["mode"] == "addenda" and current_chunk["title"] is None:
                    current_chunk["title"] = "__General__"
                    current_chunk["lines"].append(line)
                elif current_chunk["title"] or state["mode"] == "appendix":
                    current_chunk["lines"].append(line)
            
            line_idx += 1
            
        save_chunk()
        return {"filename": filename, "chunks": chunks}

# ==========================================
# [ID 생성기] - 메타데이터 기반 해시 생성
# ==========================================
def generate_id(filename, chapter, section, article, content):
    """
    ID 생성 시 충돌 방지를 위해 메타데이터와 본문 일부를 조합하여 해시를 생성합니다.
    """
    safe_filename = str(filename) if filename else ""
    safe_chapter = str(chapter) if chapter else ""
    safe_section = str(section) if section else ""
    safe_article = str(article) if article else "general"
    safe_content_preview = content[:150] if content else "" 

    unique_str = f"{safe_filename}_{safe_chapter}_{safe_section}_{safe_article}_{safe_content_preview}"
    
    hash_obj = hashlib.md5(unique_str.encode('utf-8'))
    return str(uuid.UUID(hex=hash_obj.hexdigest()))

# ==========================================
# [메인 실행]
# ==========================================
def main():
    print("=== [Step 1] 파싱 및 임베딩 시작 ===")
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
                payload = chunk['payload']

                # ID 생성 (경로가 포함된 payload['content'] 사용)
                pid = generate_id(
                    filename=fname,
                    chapter=payload.get('chapter'),
                    section=payload.get('section'),
                    article=payload.get('article'),
                    content=payload.get('content')
                )
                
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
        print("\n[주의] 파싱 실패 파일:", anomaly_files)

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"'{DATA_DIR}' 폴더가 없습니다.")
    else:
        main()