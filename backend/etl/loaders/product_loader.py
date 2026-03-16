import logging
from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class ProductLoader(BaseLoader):
    def run(self, file_path, target_mode="all"):
        if target_mode == "vector":
            logger.info("product loader skipped", extra={"reason": "vector_only_target"})
            return

        data = self.load_json(file_path)
        logger.info("product loader processing", extra={"item_count": len(data), "target": target_mode})

        debug_list = []

        for p in data:
            cypher = """
            MERGE (p:Product {product_id: %(product_id)s})
            SET p.name = %(name)s,
                p.company = %(company)s,
                p.is_active = %(is_active)s
            """
            params = {
                "product_id": p["product_id"],
                "name": p["name"],
                "company": p["company"],
                "is_active": bool(p.get("is_active", True)),
            }
            self.ctx.graph.execute_cypher(cypher, params)
            debug_list.append(p)

        self.save_debug_json(debug_list, "debug_products.json")
