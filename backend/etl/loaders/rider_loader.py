from .base_loader import BaseLoader

class RiderLoader(BaseLoader):
    def run(self, file_path, target_mode="all"):
        # [수정] target_mode가 vector면 실행하지 않음
        if target_mode == "vector":
            print(f"📜 [Rider] Skipping (Target is vector only)")
            return

        data = self.load_json(file_path)
        print(f"📜 [Rider] Loading {len(data)} items...")
        
        debug_list = []

        for r in data:
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
            debug_list.append(r)

        self.save_debug_json(debug_list, "debug_riders.json")