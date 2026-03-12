import logging
from qdrant_client import models
from .base_loader import BaseLoader

logger = logging.getLogger(__name__)

class BenefitLoader(BaseLoader):
    def run(self, file_path, target_mode="all"):
        data = self.load_json(file_path)
        # print(f"💎 [Benefit] Processing {len(data)} items... (Target: {target_mode})")
        
        points = []
        debug_list = []

        for b in data:
            limit_val = b.get("limit_count", None)
            limit_cy = "NULL" if limit_val is None else int(limit_val)
            
            if target_mode in ["all", "graph"]:
                cypher = f"""
                MATCH (r:Rider {{rider_id: '{b['rider_id']}'}})
                MATCH (c:Concept {{concept_id: '{b['related_concept']}'}})
                MERGE (b:Benefit {{benefit_id: '{b['benefit_id']}'}})
                SET b.name = '{b['name']}',
                    b.amount_value = {b['amount_value']},
                    b.amount_text = '{b['amount_text']}',
                    b.condition_summary = '{b['condition_summary']}',
                    b.limit_count = {limit_cy}
                MERGE (r)-[:HAS_BENEFIT]->(b)
                MERGE (b)-[:RELATED_TO]->(c)
                """
                self.ctx.graph.execute_cypher(cypher)

            if target_mode in ["all", "vector"]:
                text_chunk = f"{b['name']} | {b['condition_summary']} | {b['amount_text']}"
                vector = self.embed_text(text_chunk)
                
                payload = {
                    "type": "benefit",
                    "node_id": b['benefit_id'],
                    "rider_id": b['rider_id'],
                    "text": text_chunk,
                    "concept_ids": [b['related_concept']],
                    "limit_count": limit_val
                }
                
                points.append(models.PointStruct(
                    id=models.generate_uuid5(b['benefit_id']),
                    vector=vector,
                    payload=payload
                ))
                self._log_vector_payload_preview(collection="insurance_knowledge", data_type="benefit", data_id=b["benefit_id"], vector_text=text_chunk, payload=payload)
                
                debug_item = payload.copy()
                debug_item["_vector_text"] = text_chunk
                debug_list.append(debug_item)
        
        if points and target_mode in ["all", "vector"]:
            self.qdrant.upsert(collection_name="insurance_knowledge", points=points)
            self._log_upsert_summary(collection="insurance_knowledge", data_type="benefit", item_count=len(points))
            self.save_debug_json(debug_list, "debug_benefits.json")