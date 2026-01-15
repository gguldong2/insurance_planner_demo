from qdrant_client import models
from .base_loader import BaseLoader

class ClauseLoader(BaseLoader):
    def run(self, file_path, target_mode="all"):
        data = self.load_json(file_path)
        print(f"⚖️ [Clause] Processing {len(data)} items... (Target: {target_mode})")
        
        points = []
        for c in data:
            # 1. GraphDB 적재
            if target_mode in ["all", "graph"]:
                edge_type = c.get('relation_type', 'HAS_CLAUSE')
                source_id = c['source_node']
                
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

            # 2. VectorDB 준비
            if target_mode in ["all", "vector"]:
                text_chunk = f"{c['article_num']} {c['title']}\n{c['content']}"
                vec = self.embed_text(text_chunk)
                
                payload = {
                    "type": "clause",
                    "tag": c['tag'],
                    "node_id": c['clause_id'],
                    "rider_id": c.get('rider_id') if "RIDER" in c.get('source_node', '') else None,
                    "related_concepts": c.get('related_concepts', []),
                    "text": text_chunk
                }
                
                points.append(models.PointStruct(
                    id=models.generate_uuid5(c['clause_id']),
                    vector=vec,
                    payload=payload
                ))

        # VectorDB 업로드
        if points and target_mode in ["all", "vector"]:
            self.qdrant.upsert(collection_name="insurance_knowledge", points=points)
            print(f"   -> Upserted {len(points)} vectors.")