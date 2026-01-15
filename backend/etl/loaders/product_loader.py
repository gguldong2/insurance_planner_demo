from .base_loader import BaseLoader

class ProductLoader(BaseLoader):
    def run(self, file_path, target_mode="all"):
        if target_mode == "vector":
            print(f"📦 [Product] Skipping (Target is vector only)")
            return

        data = self.load_json(file_path)
        print(f"📦 [Product] Loading {len(data)} items...")
        
        # [NEW] 디버그 리스트
        debug_list = []
        
        for p in data:
            cypher = f"""
            MERGE (p:Product {{product_id: '{p['product_id']}'}})
            SET p.name = '{p['name']}',
                p.company = '{p['company']}',
                p.is_active = {str(p.get('is_active', True)).lower()}
            """
            self.ctx.graph.execute_cypher(cypher)
            
            # [NEW] 디버그 데이터 추가
            debug_list.append(p)

        # [NEW] 디버그 파일 저장
        self.save_debug_json(debug_list, "debug_products.json")