from qdrant_client import models
from .base_loader import BaseLoader

class BenefitLoader(BaseLoader):
    def run(self, file_path):
        data = self.load_json(file_path)
        print(f"💎 [Benefit] Loading {len(data)} items...")
        
        points = []
        for b in data:
            # 1. GraphDB (Rider와 Concept 연결)
            cypher = f"""
            MATCH (r:Rider {{rider_id: '{b['rider_id']}'}})
            MATCH (c:Concept {{concept_id: '{b['related_concept']}'}})
            MERGE (b:Benefit {{benefit_id: '{b['benefit_id']}'}})
            SET b.name = '{b['name']}',
                b.amount_value = {b['amount_value']},
                b.amount_text = '{b['amount_text']}',
                b.condition_summary = '{b['condition_summary']}'
            MERGE (r)-[:HAS_BENEFIT]->(b)
            MERGE (b)-[:RELATED_TO]->(c)
            """
            self.ctx.graph.execute_cypher(cypher)

            # 2. VectorDB (Type A: Benefit)
            text_chunk = f"{b['name']} | {b['condition_summary']} | {b['amount_text']}"
            vec = self.embed_text(text_chunk)
            
            payload = {
                "type": "benefit",
                "node_id": b['benefit_id'],
                "rider_id": b['rider_id'],
                "text": text_chunk,
                "concept_ids": [b['related_concept']]
            }
            
            points.append(models.PointStruct(
                id=models.generate_uuid5(b['benefit_id']),
                vector=vec,
                payload=payload
            ))
        
        if points:
            self.qdrant.upsert(collection_name="insurance_knowledge", points=points)