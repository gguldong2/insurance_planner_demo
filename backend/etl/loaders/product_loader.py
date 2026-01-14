from .base_loader import BaseLoader

class ProductLoader(BaseLoader):
    def run(self, file_path):
        data = self.load_json(file_path)
        print(f"📦 [Product] Loading {len(data)} items...")
        
        for p in data:
            cypher = f"""
            MERGE (p:Product {{product_id: '{p['product_id']}'}})
            SET p.name = '{p['name']}',
                p.company = '{p['company']}',
                p.is_active = {str(p['is_active']).lower()}
            """
            self.ctx.graph.execute_cypher(cypher)
