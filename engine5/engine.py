"""engine/engine.py — Insurance RAG query engine (singleton).

초기화:
    - 최초 호출 시 1회만 수행 (LLM 클라이언트, 임베딩 모델, DB 연결)
    - 이후 호출은 초기화된 객체를 재사용 (싱글턴)

설정:
    - engine/.env.serving 파일에서 접속 정보를 읽음
    - .env.serving.example 을 복사해서 값을 채워 사용
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator

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


async def query_engine_stream(question: str) -> AsyncIterator[str]:
    """Insurance RAG 파이프라인을 스트리밍으로 실행하고 토큰을 하나씩 yield한다.

    Example:
        from api.engine5.engine import query_engine_stream

        # engine4의 query_lightrag_streaming과 동일한 방식으로 사용
        async for chunk in query_engine_stream("암 진단비 보장 조건이 뭐야?"):
            # chunk: str (토큰 하나씩)
            print(chunk, end="", flush=True)
    """
    _, stream = await query_engine_stream_with_metadata(question)
    async for chunk in stream:
        yield chunk


async def query_engine_stream_with_metadata(
    question: str,
) -> tuple[dict[str, Any], AsyncIterator[str]]:
    """Insurance RAG 파이프라인을 스트리밍으로 실행하고 메타데이터도 함께 반환한다.

    Returns:
        (meta_ref, stream) 튜플
        - meta_ref: 처음엔 {"status": "running", "metadata": {}}
                    스트림 소진 후 {"status": "success/failure", "metadata": {...}} 로 갱신됨
                    metadata 키: intent, tasks, task_statuses, plan_candidates, final_answer
        - stream: 생성 중인 토큰을 하나씩 yield하는 AsyncIterator[str]

    Example:
        from api.engine5.engine import query_engine_stream_with_metadata
        meta, stream = await query_engine_stream_with_metadata("암 진단비 보장 조건이 뭐야?")
        async for chunk in stream:
            print(chunk, end="", flush=True)
        print(meta)  # 스트림 소진 후 메타데이터 확인
    """
    from uuid import uuid4

    if not question or not question.strip():
        async def _empty() -> AsyncIterator[str]:
            yield "질문을 입력해 주세요."
        return {"status": "failure", "metadata": {}}, _empty()

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

    # 스트림 소진 후 호출자가 읽을 수 있도록 mutable dict로 반환
    meta_ref: dict[str, Any] = {"status": "running", "metadata": {}}

    async def _stream() -> AsyncIterator[str]:
        try:
            async for event in graph.astream_events(initial_state, version="v2"):
                etype = event["event"]

                # generator 노드에서 LLM이 토큰을 생성할 때마다 yield
                if (
                    etype == "on_chat_model_stream"
                    and event.get("metadata", {}).get("langgraph_node") == "generator"
                ):
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        yield chunk.content

                # 그래프 전체 완료 시 메타데이터 수집
                elif etype == "on_chain_end" and not event.get("metadata", {}).get("langgraph_node"):
                    output = event["data"].get("output", {})
                    if isinstance(output, dict) and "final_answer" in output:
                        task_results = output.get("task_results", []) or []
                        meta_ref["status"] = "success"
                        meta_ref["metadata"] = {
                            "intent": output.get("intent"),
                            "tasks": output.get("tasks", []) or [],
                            "task_statuses": [
                                {
                                    "task_type": x.get("task_type"),
                                    "status": x.get("status"),
                                    "evidence_count": x.get("evidence_count", len(x.get("evidence", []))),
                                    "duration_ms": x.get("duration_ms"),
                                }
                                for x in task_results
                            ],
                            "plan_candidates": output.get("plan_candidates", []) or [],
                            "final_answer": output.get("final_answer", ""),
                        }
        except Exception as exc:
            meta_ref["status"] = "failure"
            meta_ref["metadata"] = {"error": str(exc)}

    return meta_ref, _stream()
