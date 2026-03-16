import logging
import uuid
from qdrant_client import models
from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class ConceptLoader(BaseLoader):
    def run(self, file_path, target_mode="all"):
        data = self.load_json(file_path)

        points = []
        debug_list = []

        for cpt in data:
            if target_mode in ["all", "graph"]:
                cypher = """
                MERGE (c:Concept {concept_id: %(concept_id)s})
                SET c.label_ko = %(label_ko)s,
                    c.category = %(category)s,
                    c.description = %(description)s
                """
                params = {
                    "concept_id": cpt["concept_id"],
                    "label_ko": cpt["label_ko"],
                    "category": cpt["category"],
                    "description": cpt["description"],
                }
                self.ctx.graph.execute_cypher(cypher, params)

            if target_mode in ["all", "vector"]:
                text = f"{cpt['label_ko']} ({cpt['category']}): {cpt['description']}"
                vec = self.embed_text(text)

                points.append(models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, cpt["concept_id"])),
                    vector=vec,
                    payload=cpt,
                ))
                self._log_vector_payload_preview(collection="concepts", data_type="concept", data_id=cpt["concept_id"], vector_text=text, payload=cpt)

                debug_item = cpt.copy()
                debug_item["_vector_text"] = text
                debug_list.append(debug_item)

        if points and target_mode in ["all", "vector"]:
            self.qdrant.upsert(collection_name="concepts", points=points)
            self._log_upsert_summary(collection="concepts", data_type="concept", item_count=len(points))
            self.save_debug_json(debug_list, "debug_concepts.json")
