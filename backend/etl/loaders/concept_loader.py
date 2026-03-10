from qdrant_client import models
from .base_loader import BaseLoader

class ConceptLoader(BaseLoader):
    def run(self, file_path, target_mode="all"):
        data = self.load_json(file_path)
        self.logger.info("loader processing", extra={"loader": "concept", "count": len(data), "target_mode": target_mode})
        
        points = []
        debug_list = []

        for cpt in data:
            # 1. GraphDB
            if target_mode in ["all", "graph"]:
                cypher = f"""
                MERGE (c:Concept {{concept_id: '{cpt['concept_id']}'}})
                SET c.label_ko = '{cpt['label_ko']}',
                    c.category = '{cpt['category']}',
                    c.description = '{cpt['description']}'
                """
                self.ctx.graph.execute_cypher(cypher)

            # 2. VectorDB
            if target_mode in ["all", "vector"]:
                text = f"{cpt['label_ko']} ({cpt['category']}): {cpt['description']}"
                vec = self.embed_text(text)
                
                points.append(models.PointStruct(
                    id=models.generate_uuid5(cpt['concept_id']),
                    vector=vec,
                    payload=cpt
                ))
                
                # 디버그용 (벡터 텍스트 포함)
                debug_item = cpt.copy()
                debug_item["_vector_text"] = text
                debug_list.append(debug_item)
                self._log_vector_payload_preview(collection="concepts", data_type="concept", data_id=cpt["concept_id"], vector_text=text, payload=cpt)
            
        if points and target_mode in ["all", "vector"]:
            self.qdrant.upsert(collection_name="concepts", points=points)
            self.save_debug_json(debug_list, "debug_concepts.json")