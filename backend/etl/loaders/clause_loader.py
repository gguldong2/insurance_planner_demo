from qdrant_client import models
from .base_loader import BaseLoader

class ClauseLoader(BaseLoader):
    def run(self, file_path):
        data = self.load_json(file_path)
        print(f"⚖️ [Clause] Loading {len(data)} items...")
        
        points = []
        for c in data:
            # 1. GraphDB: 동적 엣지 처리
            edge_type = c.get('relation_type', 'HAS_CLAUSE')
            source_id = c['source_node']
            
            # Source가 Rider인지 Benefit인지 식별
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

            # 2. VectorDB (Type B: Clause)
            text_chunk = f"{c['article_num']} {c['title']}\n{c['content']}"
            vec = self.embed_text(text_chunk)
            
            payload = {
                "type": "clause",
                "tag": c['tag'],
                "node_id": c['clause_id'],
                "rider_id": c.get('rider_id') if "RIDER" in source_id else None,
                "related_concepts": c.get('related_concepts', []),
                "text": text_chunk
            }
            
            points.append(models.PointStruct(
                id=models.generate_uuid5(c['clause_id']),
                vector=vec,
                payload=payload
            ))

        if points:
            self.qdrant.upsert(collection_name="insurance_knowledge", points=points)