from qdrant_client import models
from .base_loader import BaseLoader

class TermLoader(BaseLoader):
    def run(self, file_path, target_mode="all"):
        if target_mode == "graph":
            self.logger.info("loader skipped", extra={"loader": "term", "reason": "graph target only"})
            return

        data = self.load_json(file_path)
        self.logger.info("loader processing", extra={"loader": "term", "count": len(data), "target_mode": target_mode})
        
        points = []
        debug_list = []

        for t in data:
            text_chunk = f"용어: {t['term_name']}\n정의: {t['definition']}\n유의어: {', '.join(t['synonyms'])}"
            vec = self.embed_text(text_chunk)
            
            payload = {
                "type": "term",
                "term_name": t['term_name'],
                "category": t['category'],
                "definition": t['definition'],
                "synonyms": t['synonyms']
            }
            
            points.append(models.PointStruct(
                id=models.generate_uuid5(t['term_name']),
                vector=vec,
                payload=payload
            ))
            
            debug_item = payload.copy()
            debug_item["_vector_text"] = text_chunk
            debug_list.append(debug_item)
            self._log_vector_payload_preview(collection="glossary", data_type="term", data_id=t["term_name"], vector_text=text_chunk, payload=payload)
            
        if points:
            self.qdrant.upsert(collection_name="glossary", points=points)
            self.save_debug_json(debug_list, "debug_terms.json")