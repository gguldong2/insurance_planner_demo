from qdrant_client import models
from .base_loader import BaseLoader

class ClauseLoader(BaseLoader):
    def run(self, file_path, target_mode="all"):
        data = self.load_json(file_path)
        print(f"⚖️ [Clause] Processing {len(data)} items... (Target: {target_mode})")
        
        points = []
        debug_list = []

        for c in data:
            # ------------------------------------------------------------------
            # [핵심 로직] JSON에서 Tag와 Relation Type 꺼내기 (없으면 기본값)
            # ------------------------------------------------------------------
            # 1. 태그(Tag): EXCLUSION, CONDITION, GENERAL 등
            clause_tag = c.get('tag', 'GENERAL').upper()
            
            # 2. 관계(Relation): RESTRICTS, EVIDENCED_BY, HAS_CLAUSE 등
            rel_type = c.get('relation_type', 'HAS_CLAUSE')

            # ------------------------------------------------------------------
            # 1. GraphDB 적재 (Cypher 쿼리 방식)
            # ------------------------------------------------------------------
            if target_mode in ["all", "graph"]:
                source_id = c['source_node']
                
                # Source ID가 특약(Rider)인지 혜택(Benefit)인지 식별
                source_label = "Rider" if "RIDER" in source_id else "Benefit"
                source_key = "rider_id" if source_label == "Rider" else "benefit_id"

                # ★ 수정 포인트: 
                # 1) SET cl.tag = '{clause_tag}' 추가 (속성 저장)
                # 2) MERGE (s)-[:{rel_type}]->(cl) 로 변경 (동적 연결)
                cypher = f"""
                MATCH (s:{source_label} {{{source_key}: '{source_id}'}})
                MERGE (cl:Clause {{clause_id: '{c['clause_id']}'}})
                SET cl.article_num = '{c['article_num']}',
                    cl.title = '{c['title']}',
                    cl.content = '{c['content']}',
                    cl.type = 'clause',
                    cl.tag = '{clause_tag}'
                MERGE (s)-[:{rel_type}]->(cl)
                """
                
                # DB에 쿼리 실행
                self.ctx.graph.execute_cypher(cypher)

            # ------------------------------------------------------------------
            # 2. VectorDB 준비 (Payload 구성)
            # ------------------------------------------------------------------
            if target_mode in ["all", "vector"]:
                # 검색용 텍스트 뭉치 만들기
                text_chunk = f"{c['article_num']} {c['title']}\n{c['content']}"
                vec = self.embed_text(text_chunk)
                
                payload = {
                    "type": "clause",       # (고정값) 데이터 유형 식별자
                    "tag": clause_tag,      # ★ [추가] 필터링용 태그
                    "node_id": c['clause_id'],
                    "rider_id": c.get('rider_id') if "RIDER" in c.get('source_node', '') else None,
                    "article_no": c['article_num'],
                    "related_concepts": c.get('related_concepts', []),
                    "text": text_chunk
                }
                
                points.append(models.PointStruct(
                    id=models.generate_uuid5(c['clause_id']),
                    vector=vec,
                    payload=payload
                ))
                
                # 디버그용 (옵션)
                debug_item = payload.copy()
                debug_item["_vector_text"] = text_chunk
                debug_list.append(debug_item)

        # ------------------------------------------------------------------
        # 3. VectorDB 업로드 실행
        # ------------------------------------------------------------------
        if points and target_mode in ["all", "vector"]:
            # common.py 등에 정의된 컬렉션 이름 사용
            self.ctx.qdrant.upsert(
                collection_name="insurance_knowledge",
                points=points
            )
            print(f"✅ [Clause] VectorDB Uploaded {len(points)} items.")