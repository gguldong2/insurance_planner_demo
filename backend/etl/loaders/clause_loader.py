from qdrant_client import models
from .base_loader import BaseLoader

class ClauseLoader(BaseLoader):
    def run(self, file_path, target_mode="all"):
        data = self.load_json(file_path)
        print(f"⚖️ [Clause] Processing {len(data)} items... (Target: {target_mode})")
        
        points = []
        debug_list = []  # 디버그용 리스트

        for c in data:
            # 1. GraphDB 적재 (All 또는 Graph일 때)
            if target_mode in ["all", "graph"]:
                # 엣지 타입 결정 (기본값: HAS_CLAUSE)
                edge_type = c.get('relation_type', 'HAS_CLAUSE')
                source_id = c['source_node']
                
                # Source ID가 Rider인지 Benefit인지 식별
                source_label = "Rider" if "RIDER" in source_id else "Benefit"
                source_key = "rider_id" if source_label == "Rider" else "benefit_id"

                cypher = f"""
                MATCH (s:{source_label} {{{source_key}: '{source_id}'}})
                MERGE (cl:Clause {{clause_id: '{c['clause_id']}'}})
                SET cl.article_num = '{c['article_num']}',
                    cl.title = '{c['title']}',
                    cl.content = '{c['content']}',
                    cl.type = '{c['type']}'
                MERGE (s)-[:{edge_type}]->(cl)
                """
                self.ctx.graph.execute_cypher(cypher)

            # 2. VectorDB 준비 (All 또는 Vector일 때)
            if target_mode in ["all", "vector"]:
                # 검색에 사용될 텍스트 청크 생성
                text_chunk = f"{c['article_num']} {c['title']}\n{c['content']}"
                vec = self.embed_text(text_chunk)
                
                payload = {
                    "type": "clause",
                    "tag": c['tag'], # EXCLUSION, CONDITION, GENERAL
                    "node_id": c['clause_id'],
                    # Rider에 속한 경우 rider_id 추가, 아니면 None
                    "rider_id": c.get('rider_id') if "RIDER" in c.get('source_node', '') else None,
                    "related_concepts": c.get('related_concepts', []),
                    "text": text_chunk
                }
                
                points.append(models.PointStruct(
                    id=models.generate_uuid5(c['clause_id']),
                    vector=vec,
                    payload=payload
                ))
                
                # 디버그 아이템 추가
                debug_item = payload.copy()
                debug_item["_vector_text"] = text_chunk
                debug_list.append(debug_item)

        # VectorDB 업로드 및 디버그 파일 저장
        if points and target_mode in ["all", "vector"]:
            self.qdrant.upsert(collection_name="insurance_knowledge", points=points)
            self.save_debug_json(debug_list, "debug_clauses.json")