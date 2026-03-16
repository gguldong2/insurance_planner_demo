import logging
import uuid
from qdrant_client import models
from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class BenefitLoader(BaseLoader):
    def run(self, file_path, target_mode="all"):
        data = self.load_json(file_path)

        points = []
        debug_list = []

        for b in data:
            limit_val = b.get("limit_count", None)

            if target_mode in ["all", "graph"]:
                cypher = """
                MATCH (r:Rider {rider_id: %(rider_id)s})
                MATCH (c:Concept {concept_id: %(related_concept)s})
                MERGE (bn:Benefit {benefit_id: %(benefit_id)s})
                SET bn.name = %(name)s,
                    bn.amount_value = %(amount_value)s,
                    bn.amount_text = %(amount_text)s,
                    bn.condition_summary = %(condition_summary)s,
                    bn.limit_count = %(limit_count)s
                MERGE (r)-[:HAS_BENEFIT]->(bn)
                MERGE (bn)-[:RELATED_TO]->(c)
                """
                params = {
                    "rider_id": b["rider_id"],
                    "related_concept": b["related_concept"],
                    "benefit_id": b["benefit_id"],
                    "name": b["name"],
                    "amount_value": b["amount_value"],
                    "amount_text": b["amount_text"],
                    "condition_summary": b["condition_summary"],
                    "limit_count": limit_val,
                }
                self.ctx.graph.execute_cypher(cypher, params)

            if target_mode in ["all", "vector"]:
                text_chunk = f"{b['name']} | {b['condition_summary']} | {b['amount_text']}"
                vector = self.embed_text(text_chunk)

                payload = {
                    "type": "benefit",
                    "node_id": b["benefit_id"],
                    "rider_id": b["rider_id"],
                    "product_id": b.get("product_id"),
                    "company": b.get("company"),
                    "product_name": b.get("product_name"),
                    "text": text_chunk,
                    "concept_ids": [b["related_concept"]],
                    "limit_count": limit_val,
                }

                points.append(models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, b["benefit_id"])),
                    vector=vector,
                    payload=payload,
                ))
                self._log_vector_payload_preview(collection="insurance_knowledge", data_type="benefit", data_id=b["benefit_id"], vector_text=text_chunk, payload=payload)

                debug_item = payload.copy()
                debug_item["_vector_text"] = text_chunk
                debug_list.append(debug_item)

        if points and target_mode in ["all", "vector"]:
            self.qdrant.upsert(collection_name="insurance_knowledge", points=points)
            self._log_upsert_summary(collection="insurance_knowledge", data_type="benefit", item_count=len(points))
            self.save_debug_json(debug_list, "debug_benefits.json")
