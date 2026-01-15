from qdrant_client import models
from .base_loader import BaseLoader

class ConceptLoader(BaseLoader):
    def run(self, file_path, target_mode="all"):
        data = self.load_json(file_path)
        print(f"🧠 [Concept] Processing {len(data)} items... (Target: {target_mode})")
        
        points = []
        for cpt in data:
            # 1. GraphDB 적재 (All 또는 Graph일 때)
            if target_mode in ["all", "graph"]:
                cypher = f"""
                MERGE (c:Concept {{concept_id: '{cpt['concept_id']}'}})
                SET c.label_ko = '{cpt['label_ko']}',
                    c.category = '{cpt['category']}',
                    c.description = '{cpt['description']}'
                """
                self.ctx.graph.execute_cypher(cypher)

            # 2. VectorDB 준비 (All 또는 Vector일 때)
            if target_mode in ["all", "vector"]:
                text = f"{cpt['label_ko']} ({cpt['category']}): {cpt['description']}"
                vec = self.embed_text(text)
                
                points.append(models.PointStruct(
                    id=models.generate_uuid5(cpt['concept_id']),
                    vector=vec,
                    payload=cpt
                ))
            
        # VectorDB 업로드
        if points and target_mode in ["all", "vector"]:
            self.qdrant.upsert(collection_name="concepts", points=points)
            print(f"   -> Upserted {len(points)} vectors.")