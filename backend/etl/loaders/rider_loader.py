import logging
from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class RiderLoader(BaseLoader):
    def run(self, file_path, target_mode="all"):
        if target_mode == "vector":
            logger.info("rider loader skipped", extra={"reason": "vector_only_target"})
            return

        data = self.load_json(file_path)
        logger.info("rider loader processing", extra={"item_count": len(data), "target": target_mode})

        debug_list = []

        for r in data:
            cypher = """
            MATCH (p:Product {product_id: %(product_id)s})
            MERGE (rd:Rider {rider_id: %(rider_id)s})
            SET rd.name = %(name)s,
                rd.type = %(type)s,
                rd.renewal_type = %(renewal_type)s,
                rd.insurance_period = %(insurance_period)s
            MERGE (p)-[:HAS_RIDER]->(rd)
            """
            params = {
                "product_id": r["product_id"],
                "rider_id": r["rider_id"],
                "name": r["name"],
                "type": r["type"],
                "renewal_type": r["renewal_type"],
                "insurance_period": r["insurance_period"],
            }
            self.ctx.graph.execute_cypher(cypher, params)
            debug_list.append(r)

        self.save_debug_json(debug_list, "debug_riders.json")
