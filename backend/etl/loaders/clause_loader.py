import logging
import uuid
from qdrant_client import models
from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class ClauseLoader(BaseLoader):
    def run(self, file_path, target_mode="all"):
        data = self.load_json(file_path)

        points = []
        debug_list = []

        for c in data:
            clause_tag = c.get("tag", "GENERAL").upper()
            rel_type = c.get("relation_type", "HAS_CLAUSE")

            if target_mode in ["all", "graph"]:
                source_id = c["source_node"]
                source_label = "Rider" if "RIDER" in source_id else "Benefit"
                source_key = "rider_id" if source_label == "Rider" else "benefit_id"

                cypher = f"""
                MATCH (s:{source_label} {{{source_key}: %(source_id)s}})
                MERGE (cl:Clause {{clause_id: %(clause_id)s}})
                SET cl.article_num = %(article_num)s,
                    cl.title = %(title)s,
                    cl.content = %(content)s,
                    cl.type = 'clause',
                    cl.tag = %(clause_tag)s
                MERGE (s)-[:{rel_type}]->(cl)
                """
                params = {
                    "source_id": source_id,
                    "clause_id": c["clause_id"],
                    "article_num": c["article_num"],
                    "title": c["title"],
                    "content": c["content"],
                    "clause_tag": clause_tag,
                }
                self.ctx.graph.execute_cypher(cypher, params)

            if target_mode in ["all", "vector"]:
                text_chunk = f"{c['article_num']} {c['title']}\n{c['content']}"
                vec = self.embed_text(text_chunk)

                source_node = c.get("source_node", "")
                rider_id_for_payload = source_node if "RIDER" in source_node else None

                payload = {
                    "type": "clause",
                    "tag": clause_tag,
                    "node_id": c["clause_id"],
                    "rider_id": rider_id_for_payload,
                    "product_id": c.get("product_id"),
                    "company": c.get("company"),
                    "product_name": c.get("product_name"),
                    "article_num": c["article_num"],
                    "related_concepts": c.get("related_concepts", []),
                    "text": text_chunk,
                }

                points.append(models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, c["clause_id"])),
                    vector=vec,
                    payload=payload,
                ))
                self._log_vector_payload_preview(collection="insurance_knowledge", data_type="clause", data_id=c["clause_id"], vector_text=text_chunk, payload=payload)

                debug_item = payload.copy()
                debug_item["_vector_text"] = text_chunk
                debug_list.append(debug_item)

        if points and target_mode in ["all", "vector"]:
            self.ctx.qdrant.upsert(
                collection_name="insurance_knowledge",
                points=points,
            )
            self._log_upsert_summary(collection="insurance_knowledge", data_type="clause", item_count=len(points))
            self.save_debug_json(debug_list, "debug_clauses.json")
