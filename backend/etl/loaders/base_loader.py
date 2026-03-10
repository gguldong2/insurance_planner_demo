import os
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from backend.etl.common import DBContext
from backend.logging_utils import get_logger

class BaseLoader(ABC):
    def __init__(self, context: DBContext):
        self.ctx = context
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.qdrant = context.qdrant
        self.graph = context.graph
        self.model = context.embed_model

    def load_json(self, file_path):
        """JSON 파일 로드 유틸"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def embed_text(self, text):
        """텍스트 임베딩 유틸"""
        return self.model.encode(text, return_dense=True)['dense_vecs']
    
    # 디버그 파일 저장 함수 추가
    def save_debug_json(self, data_list, file_name):
        debug_dir = "backend/data/debug"
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, file_name)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        print(f"🐞 [Debug] Saved processed data to {path}")


    def _preview_text(self, text: Optional[str], limit: int = 50) -> str:
        if not text:
            return ""
        normalized = str(text).replace("\n", " ").strip()
        return normalized[:limit]

    def _log_vector_payload_preview(self, *, collection: str, data_type: str, data_id: str, vector_text: str, payload: Dict[str, Any]):
        payload_preview = {
            key: value
            for key, value in payload.items()
            if key != "text"
        }
        self.logger.info(
            "vector payload preview",
            extra={
                "collection": collection,
                "data_type": data_type,
                "data_id": data_id,
                "vector_text_preview": self._preview_text(vector_text),
                "vector_text_length": len(vector_text or ""),
                "payload_keys": sorted(list(payload.keys())),
                "payload_preview": payload_preview,
            },
        )

    def _log_upsert_summary(self, *, collection: str, data_type: str, count: int):
        self.logger.info(
            "vector upsert finished",
            extra={
                "collection": collection,
                "data_type": data_type,
                "count": count,
            },
        )

    @abstractmethod
    def run(self, file_path: str, target_mode: str = "all"):  # <--- [수정] 인자 추가
        """
        ETL 실행 추상 메서드
        :param file_path: 데이터 파일 경로
        :param target_mode: 'all' | 'graph' | 'vector'
        """
        pass