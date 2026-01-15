import json
from abc import ABC, abstractmethod
from backend.etl.common import DBContext

class BaseLoader(ABC):
    def __init__(self, context: DBContext):
        self.ctx = context
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

    @abstractmethod
    def run(self, file_path: str, target_mode: str = "all"):  # <--- [수정] 인자 추가
        """
        ETL 실행 추상 메서드
        :param file_path: 데이터 파일 경로
        :param target_mode: 'all' | 'graph' | 'vector'
        """
        pass