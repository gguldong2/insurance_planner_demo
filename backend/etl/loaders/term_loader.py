#(Vector Only)
from qdrant_client import models
from .base_loader import BaseLoader

class TermLoader(BaseLoader):
    def run(self, file_path, target_mode="all"):
        if target_mode == "graph":
            print(f"📖 [Term] Skipping (Target is graph only)")
            return

        data = self.load_json(file_path)
        print(f"📖 [Term] Loading {len(data)} items...")
        
        points = []
        for t in data:
            text_chunk = f"용어: {t['term_name']}\n정의: {t['definition']}\n유의어: {', '.join(t['synonyms'])}"
            vec = self.embed_text(text_chunk)
            # ... (Point 생성 로직 동일) ...
            points.append(...)
            
        if points:
            self.qdrant.upsert(collection_name="glossary", points=points)