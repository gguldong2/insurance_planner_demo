from qdrant_client import models
from .base_loader import BaseLoader

class ConceptLoader(BaseLoader):
    def run(self, file_path):
        data = self.load_json(file_path)
        print(f"🧠 [Concept] Loading {len(data)} items...")
        
        points = []
        for cpt in data:
            # 1. GraphDB
            cypher = f"""
            MERGE (c:Concept {{concept_id: '{cpt['concept_id']}'}})
            SET c.label_ko = '{cpt['label_ko']}',
                c.category = '{cpt['category']}',
                c.description = '{cpt['description']}'
            """
            self.ctx.graph.execute_cypher(cypher)

            # 2. VectorDB (Linker용)
            text = f"{cpt['label_ko']} ({cpt['category']}): {cpt['description']}"
            vec = self.embed_text(text)
            
            points.append(models.PointStruct(
                id=models.generate_uuid5(cpt['concept_id']),
                vector=vec,
                payload=cpt
            ))
            
        if points:
            self.qdrant.upsert(collection_name="concepts", points=points)