"""engine/engine.py — Insurance RAG query engine (singleton).

이 파일이 있는 engine/ 디렉토리 하나만 다른 프로젝트로 복사해 사용할 수 있습니다.
원본 프로젝트 구조(backend/ 등)에 대한 의존성이 없습니다.

외부 호출 방법:
    from engine.engine import query_engine
    result = await query_engine("질문")   # str (markdown) 반환

초기화:
    - 최초 호출 시 1회만 수행 (LLM 클라이언트, 임베딩 모델, DB 연결)
    - 이후 호출은 초기화된 객체를 재사용 (싱글턴)

설정:
    - engine/.env.serving 파일에서 접속 정보를 읽음
    - .env.serving.example 을 복사해서 값을 채워 사용
"""
from __future__ import annotations

from pathlib import Path

# ── .env.serving 로드 (모든 import보다 먼저 실행해야 DB/LLM 접속 정보가 설정됨) ──
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).parent / ".env.serving"
if not _ENV_PATH.exists():
    raise FileNotFoundError(
        f"[engine] .env.serving 파일이 없습니다: {_ENV_PATH}\n"
        "engine/.env.serving.example 을 복사해서 .env.serving 으로 저장한 뒤 값을 채워 주세요."
    )
# override=True: 이미 세팅된 환경변수도 .env.serving 값으로 덮어씀
load_dotenv(dotenv_path=_ENV_PATH, override=True)

# ── 싱글턴 그래프 객체 (최초 query_engine 호출 시 1회 초기화) ─────────────────
_app_graph = None


def _init_graph():
    """engine.graph 를 임포트해 LLM·임베딩 모델·DB를 초기화한다.

    graph 모듈 최상위에서 embed_model, RuntimeDB(), app_graph 가 생성되므로
    import 자체가 곧 초기화다. 실패 시 명확한 오류 메시지를 출력한다.
    """
    global _app_graph
    if _app_graph is not None:
        return _app_graph
    try:
        from .graph import app_graph  # noqa: PLC0415
        _app_graph = app_graph
        return _app_graph
    except Exception as exc:
        raise RuntimeError(
            f"[engine] RAG 파이프라인 초기화 실패: {exc}\n\n"
            "체크리스트:\n"
            "  1) engine/.env.serving 의 DB_HOST / DB_PORT / DB_USER / DB_PASSWORD / DB_NAME 확인\n"
            "  2) AgensGraph(PostgreSQL) 컨테이너 실행 여부 확인\n"
            "  3) QDRANT_URL 주소에 Qdrant 실행 여부 확인\n"
            "  4) LLM_API_BASE 주소에 vLLM 서버 실행 여부 확인\n"
            "  5) BAAI/bge-m3 임베딩 모델 다운로드 여부 확인\n"
            "     → python -c \"from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3')\""
        ) from exc


async def query_engine(question: str) -> str:
    """Insurance RAG 파이프라인을 실행하고 markdown 형식의 답변 문자열을 반환한다.

    Args:
        question: 사용자 질문 (한국어)

    Returns:
        LangGraph 7-node 파이프라인(Analyzer→Grounder→Planner→Executor→
        Composer→Guard→Generator)이 생성한 markdown 답변 문자열.
        오류 발생 시 오류 설명 문자열 반환.

    Example:
        from engine.engine import query_engine
        result = await query_engine("암 진단비 보장 조건이 뭐야?")
    """
    if not question or not question.strip():
        return "질문을 입력해 주세요."

    from uuid import uuid4

    graph = _init_graph()

    initial_state = {
        "question": question,
        "request_id": uuid4().hex[:12],
        "intent": "",
        "tasks": [],
        "task_candidates": [],
        "required_tasks": [],
        "concept_keywords": [],
        "product_keywords": [],
        "analysis_notes": [],
        "user_filters": {},
        "resolved_concepts": [],
        "task_plan": [],
        "task_results": [],
        "response_sections": [],
        "guarded_sections": [],
        "plan_candidates": [],
        "allowed_entities": {},
        "final_answer": "",
        "trace_log": [],
        "node_models": {},
        "execution_metrics": {},
    }

    try:
        result = await graph.ainvoke(initial_state)
        return result.get("final_answer") or "답변을 생성하지 못했습니다."
    except Exception as exc:
        return f"[engine] 쿼리 처리 중 오류 발생: {exc}"
