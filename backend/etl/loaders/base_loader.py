import json
import logging
import os
from abc import ABC, abstractmethod

from backend.etl.common import DBContext

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    def __init__(self, context: DBContext):
        self.ctx = context
        self.qdrant = context.qdrant
        self.graph = context.graph
        self.model = context.embed_model

    def load_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def embed_text(self, text):
        return self.model.encode(text, return_dense=True)['dense_vecs']

    def save_debug_json(self, data_list, file_name):
        debug_dir = "backend/data/debug"
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, file_name)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        logger.info("etl debug json saved", extra={"file_path": path, "item_count": len(data_list)})

    def _preview_text(self, text: str, limit: int = 50) -> str:
        text = (text or "").replace("\n", " ").strip()
        return text[:limit]

    def _log_vector_payload_preview(self, *, collection: str, data_type: str, data_id: str, vector_text: str, payload: dict):
        payload_preview = {k: v for k, v in payload.items() if k != "text"}
        logger.info(
            "vector payload preview",
            extra={
                "collection": collection,
                "data_type": data_type,
                "data_id": data_id,
                "vector_text_preview": self._preview_text(vector_text, 50),
                "vector_text_length": len(vector_text or ""),
                "payload_keys": sorted(list(payload.keys())),
                "payload_preview": payload_preview,
            },
        )

    def _log_upsert_summary(self, *, collection: str, data_type: str, item_count: int):
        logger.info("vector upsert finished", extra={"collection": collection, "data_type": data_type, "item_count": item_count})

    @abstractmethod
    def run(self, file_path: str, target_mode: str = "all"):
        pass
