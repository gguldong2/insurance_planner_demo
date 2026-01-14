from .base_loader import BaseLoader

class RiderLoader(BaseLoader):
    def run(self, file_path):
        data = self.load_json(file_path)
        print(f"📜 [Rider] Loading {len(data)} items...")
        
        for r in data:
            # Rider 생성 및 Product와 연결
            cypher = f"""
            MATCH (p:Product {{product_id: '{r['product_id']}'}})
            MERGE (r:Rider {{rider_id: '{r['rider_id']}'}})
            SET r.name = '{r['name']}',
                r.type = '{r['type']}',
                r.renewal_type = '{r['renewal_type']}',
                r.insurance_period = '{r['insurance_period']}'
            MERGE (p)-[:HAS_RIDER]->(r)
            """
            self.ctx.graph.execute_cypher(cypher)