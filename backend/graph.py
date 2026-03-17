"""LangGraph workflow for the insurance QA agent.

This module wires the end-to-end execution graph used by the FastAPI app.
The pipeline is intentionally split into small nodes so that each phase can be
observed in logs and tightened independently.

Execution flow
--------------
Analyzer -> Grounder -> Planner -> Executor -> Composer -> Guard -> Generator

Key design goals
----------------
1. Keep the final answer grounded in retrieved evidence only.
2. Preserve product/rider metadata all the way to the final answer.
3. Keep the flow easy to debug with explicit node-level trace messages.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from backend.logging_utils import setup_logging
from backend.logic.retrievers import (
    link_concept_candidates,
    retrieve_benefit,
    retrieve_comparison,
    retrieve_condition,
    retrieve_exclusion,
    retrieve_term,
)

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

llm = ChatOpenAI(
    model="Qwen/Qwen3.5-9B",
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    temperature=0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)


class AgentState(TypedDict):
    """Mutable state shared across LangGraph nodes.

    The state stores both user-facing outputs and internal orchestration
    artifacts. The important fields for grounding are:
    - resolved_concepts: normalized concept matches from the grounding step
    - task_results: raw retrieval outputs per task
    - response_sections: structured sections prepared for answer generation
    """

    question: str
    request_id: str

    tasks: List[str]
    concept_keywords: List[str]
    product_keywords: List[str]
    analysis_notes: List[str]

    resolved_concepts: List[Dict[str, Any]]
    task_plan: List[Dict[str, Any]]
    task_results: List[Dict[str, Any]]
    response_sections: List[Dict[str, Any]]
    guarded_sections: List[Dict[str, Any]]

    final_answer: str
    trace_log: List[str]
    node_models: Dict[str, str]
    execution_metrics: Dict[str, Any]


TASK_TITLES = {
    "DEFINE_TERM": "용어 설명",
    "GET_BENEFIT": "보장 금액/항목",
    "GET_CONDITION": "지급 조건",
    "GET_EXCLUSION": "면책/제한",
    "COMPARE_PRODUCTS": "상품 비교",
    "CHIT_CHAT": "일반 대화",
}

SECTION_INSTRUCTIONS = {
    "DEFINE_TERM": "용어의 의미를 먼저 간단하고 명확하게 설명하세요.",
    "GET_BENEFIT": "보장 항목과 금액을 정확하게 정리하세요.",
    "GET_CONDITION": "지급 요건과 시점을 조건 중심으로 설명하세요.",
    "GET_EXCLUSION": "면책 및 제한 사항을 주의사항처럼 분명하게 설명하세요.",
    "COMPARE_PRODUCTS": "상품별 차이를 비교 형태로 정리하세요.",
}

ANALYZER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 보험 QA 분석기다.
질문을 아래 제한된 task taxonomy 안에서만 분류하라.

허용 task:
- DEFINE_TERM: 용어 뜻, 정의, 설명
- GET_BENEFIT: 보장 금액, 얼마, 지급 항목
- GET_CONDITION: 조건, 시점, 언제부터, 요건
- GET_EXCLUSION: 면책, 제한, 보장하지 않는 경우
- COMPARE_PRODUCTS: A와 B 비교, 차이점
- CHIT_CHAT: 인사, 잡담

규칙:
1. 최대 3개 task만 반환하라.
2. 중복 task는 제거하라.
3. 질문에 직접 드러난 요구만 task로 넣어라.
4. 내부 구현 단계(grounding, search 등)는 task로 만들지 마라.
5. 상품명/플랜명/브랜드명/시리즈명처럼 보이는 표현은 product_keywords에 넣어라.
6. 보장 개념, 질병, 치료, 약관 용어처럼 보이는 표현은 concept_keywords에 넣어라.
7. JSON만 출력하라.

출력 스키마:
{
  "tasks": ["..."],
  "concept_keywords": ["..."],
  "product_keywords": ["..."],
  "notes": ["..."]
}
""",
        ),
        ("user", "질문: {question}"),
    ]
)

GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 보험 QA 어시스턴트다.
최종 답변은 반드시 제공된 response_sections의 evidence만 사용해 작성하라.
배경지식, 추측, 일반 상식으로 빈칸을 메우지 마라.

강제 규칙:
1. response_sections의 순서를 유지하라.
2. 각 섹션은 해당 섹션의 evidence만 사용하라.
3. evidence에 없는 상품명, 특약명, 보장명은 새로 만들지 마라.
4. 상품 또는 특약을 설명할 때 evidence 안에 product_name, rider_name이 있으면 반드시 함께 드러나게 써라.
5. 비교/추천은 response_sections의 evidence에 실제 등장한 상품/특약만 대상으로 하라.
6. evidence가 부족한 섹션은 '확인된 근거가 부족하다'고 명시하라.
7. 마지막에 짧은 요약을 추가하라.
""",
        ),
        (
            "user",
            """[질문]
{question}

[실행된 task]
{tasks}

[응답 섹션]
{response_sections}

위 정보를 바탕으로 한국어로 답하라.
""",
        ),
    ]
)


def update_trace(state: AgentState, node: str, msg: str) -> List[str]:
    """Append a compact trace line while keeping the trace window bounded."""
    trace = list(state.get("trace_log", []) or [])
    trace.append(f"[{node}] {msg}")
    return trace[-50:]



def _req(state: AgentState) -> str:
    return state.get("request_id", "-")



def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)



def remove_think_tag(text: str) -> str:
    """Strip hidden thinking blocks that some serving templates may emit."""
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()



def _needs_product_rider(task_type: str) -> bool:
    """Return True when the final answer should carry product/rider identity."""
    return task_type in {"GET_BENEFIT", "GET_CONDITION", "GET_EXCLUSION", "COMPARE_PRODUCTS"}



def _has_product_rider(data: Any) -> bool:
    """Check whether a piece of evidence includes usable product/rider metadata."""
    if isinstance(data, dict):
        if data.get("product_name") or data.get("rider_name"):
            return True
        return any(_has_product_rider(v) for v in data.values())
    if isinstance(data, list):
        return any(_has_product_rider(v) for v in data)
    return False


async def node_analyzer(state: AgentState) -> Dict[str, Any]:
    """Classify the user request into a small, controlled task taxonomy."""
    started = time.perf_counter()
    logger.info("analyzer started", extra={"request_id": _req(state), "question": state["question"]})
    tasks: List[str] = ["CHIT_CHAT"]
    concept_keywords: List[str] = []
    product_keywords: List[str] = []
    notes: List[str] = []

    try:
        res = await (ANALYZER_PROMPT | llm).ainvoke({"question": state["question"]})
        clean_json = remove_think_tag(res.content).replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_json)
        tasks = parsed.get("tasks", ["CHIT_CHAT"]) or ["CHIT_CHAT"]
        concept_keywords = parsed.get("concept_keywords", []) or []
        product_keywords = parsed.get("product_keywords", []) or []
        notes = parsed.get("notes", []) or []
    except Exception as exc:
        logger.exception("analyzer failed", extra={"request_id": _req(state)})
        notes = [f"analyzer_error: {exc}"]

    tasks = _normalize_tasks(tasks)
    duration_ms = _ms(started)
    logger.info(
        "analyzer finished",
        extra={
            "request_id": _req(state),
            "tasks": tasks,
            "concept_keywords": concept_keywords,
            "product_keywords": product_keywords,
            "duration_ms": duration_ms,
        },
    )
    return {
        "tasks": tasks,
        "concept_keywords": concept_keywords,
        "product_keywords": product_keywords,
        "analysis_notes": notes,
        "trace_log": update_trace(
            state,
            "Analyzer",
            f"tasks={tasks}, concept_keywords={concept_keywords}, product_keywords={product_keywords}, duration_ms={duration_ms}",
        ),
    }


async def node_grounder(state: AgentState) -> Dict[str, Any]:
    """Resolve user concept keywords into known concept nodes."""
    started = time.perf_counter()
    keywords = list(dict.fromkeys(state.get("concept_keywords", []) or []))
    logger.info("grounder started", extra={"request_id": _req(state), "keywords": keywords, "keyword_count": len(keywords)})
    resolved_concepts: List[Dict[str, Any]] = []

    async def _resolve(keyword: str) -> Optional[Dict[str, Any]]:
        candidates = await link_concept_candidates(keyword, limit=3)
        if not candidates:
            return None
        top = candidates[0]
        return {
            "keyword": keyword,
            "concept_id": top.get("concept_id"),
            "label_ko": top.get("label_ko"),
            "category": top.get("category"),
            "score": round(float(top.get("score", 0.0) or 0.0), 4),
            "matched_text": top.get("matched_text"),
            "candidates": candidates,
        }

    if keywords:
        resolved = await asyncio.gather(*[_resolve(k) for k in keywords], return_exceptions=True)
        for item in resolved:
            if isinstance(item, Exception):
                logger.exception("grounder candidate resolution failed", extra={"request_id": _req(state)})
                continue
            if item:
                resolved_concepts.append(item)

    duration_ms = _ms(started)
    logger.info(
        "grounder finished",
        extra={
            "request_id": _req(state),
            "resolved_concept_ids": [x.get("concept_id") for x in resolved_concepts],
            "resolved_count": len(resolved_concepts),
            "duration_ms": duration_ms,
        },
    )
    return {
        "resolved_concepts": resolved_concepts,
        "trace_log": update_trace(state, "Grounder", f"resolved={len(resolved_concepts)} concepts, duration_ms={duration_ms}"),
    }


async def node_planner(state: AgentState) -> Dict[str, Any]:
    """Build a minimal task plan from analyzer output and grounding results."""
    started = time.perf_counter()
    tasks = _normalize_tasks(state.get("tasks", []))
    resolved_concepts = state.get("resolved_concepts", []) or []
    product_keywords = state.get("product_keywords", []) or []
    concept_keywords = state.get("concept_keywords", []) or []

    if tasks == ["CHIT_CHAT"]:
        plan = [{
            "task_id": "task_1",
            "task_type": "CHIT_CHAT",
            "title": TASK_TITLES["CHIT_CHAT"],
            "inputs": {},
            "depends_on": [],
            "priority": 1,
        }]
    else:
        plan = []
        for idx, task_type in enumerate(tasks[:3], start=1):
            inputs: Dict[str, Any] = {}
            if task_type == "COMPARE_PRODUCTS":
                inputs = {
                    "concept_id": resolved_concepts[0].get("concept_id") if resolved_concepts else None,
                    "product_keywords": product_keywords,
                }
            elif task_type == "DEFINE_TERM":
                inputs = {
                    "keyword": concept_keywords[0] if concept_keywords else "",
                    "concept_id": resolved_concepts[0].get("concept_id") if resolved_concepts else None,
                }
            else:
                inputs = {"concept_id": resolved_concepts[0].get("concept_id") if resolved_concepts else None}
            plan.append(
                {
                    "task_id": f"task_{idx}",
                    "task_type": task_type,
                    "title": TASK_TITLES.get(task_type, task_type),
                    "inputs": inputs,
                    "depends_on": ["grounding"] if task_type not in {"CHIT_CHAT", "DEFINE_TERM"} else [],
                    "priority": idx,
                }
            )

    duration_ms = _ms(started)
    logger.info("planner finished", extra={"request_id": _req(state), "task_plan": plan, "task_count": len(plan), "duration_ms": duration_ms})
    return {
        "task_plan": plan,
        "trace_log": update_trace(state, "Planner", f"planned={len(plan)} tasks, duration_ms={duration_ms}"),
    }


async def _execute_task(plan_item: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
    """Execute a single retrieval task and normalize its result shape."""
    started = time.perf_counter()
    task_type = plan_item["task_type"]
    inputs = plan_item.get("inputs", {})
    resolved_concepts = state.get("resolved_concepts", []) or []
    logger.info("executor task started", extra={"request_id": _req(state), "task_type": task_type, "inputs": inputs})

    result: Dict[str, Any] = {
        "task_id": plan_item["task_id"],
        "task_type": task_type,
        "title": plan_item["title"],
        "status": "success",
        "resolved_concepts": resolved_concepts,
        "evidence": [],
        "summary": "",
        "error": None,
        "duration_ms": None,
        "evidence_count": 0,
    }

    try:
        if task_type == "DEFINE_TERM":
            keyword = inputs.get("keyword") or (state.get("concept_keywords") or [""])[0]
            res = await retrieve_term(keyword)
            if res:
                result["evidence"] = [res]
                result["summary"] = f"용어 정의 1건 확보 ({res.get('source')})"
            else:
                result["status"] = "no_evidence"
                result["summary"] = "용어 정의를 찾지 못함"

        elif task_type == "GET_BENEFIT":
            concept_id = inputs.get("concept_id")
            if concept_id:
                res = await retrieve_benefit(concept_id)
                result["evidence"] = res
                result["status"] = "success" if res else "no_evidence"
                result["summary"] = f"보장 정보 {len(res)}건 확보"
            else:
                result["status"] = "no_evidence"
                result["summary"] = "개념 grounding 실패로 보장 조회 불가"

        elif task_type == "GET_CONDITION":
            concept_id = inputs.get("concept_id")
            if concept_id:
                res = await retrieve_condition(concept_id)
                result["evidence"] = res
                result["status"] = "success" if res else "no_evidence"
                result["summary"] = f"지급 조건 {len(res)}건 확보"
            else:
                result["status"] = "no_evidence"
                result["summary"] = "개념 grounding 실패로 조건 조회 불가"

        elif task_type == "GET_EXCLUSION":
            concept_id = inputs.get("concept_id")
            if concept_id:
                res = await retrieve_exclusion(concept_id)
                result["evidence"] = res
                result["status"] = "success" if res else "no_evidence"
                result["summary"] = f"면책/제한 {len(res)}건 확보"
            else:
                result["status"] = "no_evidence"
                result["summary"] = "개념 grounding 실패로 면책 조회 불가"

        elif task_type == "COMPARE_PRODUCTS":
            concept_id = inputs.get("concept_id")
            product_keywords = inputs.get("product_keywords") or []
            if concept_id and len(product_keywords) >= 1:
                res = await retrieve_comparison(concept_id, product_keywords)
                result["evidence"] = [res]
                result["status"] = "success" if res else "no_evidence"
                result["summary"] = f"상품 비교 {len(product_keywords)}개 키워드 처리"
            else:
                result["status"] = "no_evidence"
                result["summary"] = "비교 기준 concept 또는 상품 키워드 부족"

        elif task_type == "CHIT_CHAT":
            result["summary"] = "일반 대화 - retrieval 생략"

    except Exception as exc:
        logger.exception("executor task failed", extra={"request_id": _req(state), "task_type": task_type})
        result["status"] = "error"
        result["error"] = str(exc)
        result["summary"] = f"task 실행 중 오류: {exc}"

    result["duration_ms"] = _ms(started)
    result["evidence_count"] = len(result.get("evidence", []))
    logger.info(
        "executor task finished",
        extra={
            "request_id": _req(state),
            "task_type": task_type,
            "status": result["status"],
            "evidence_count": result["evidence_count"],
            "duration_ms": result["duration_ms"],
        },
    )
    return result


async def node_executor(state: AgentState) -> Dict[str, Any]:
    """Run planned retrieval tasks, mostly in parallel, and collect results."""
    started = time.perf_counter()
    task_plan = state.get("task_plan", []) or []
    if not task_plan:
        return {"task_results": [], "trace_log": update_trace(state, "Executor", "planned tasks not found")}

    runnable = [item for item in task_plan if item["task_type"] != "CHIT_CHAT"]
    results: List[Dict[str, Any]] = []

    if runnable:
        gathered = await asyncio.gather(*[_execute_task(item, state) for item in runnable], return_exceptions=True)
        for item, output in zip(runnable, gathered):
            if isinstance(output, Exception):
                logger.exception("executor task crashed", extra={"request_id": _req(state), "task_type": item["task_type"]})
                results.append(
                    {
                        "task_id": item["task_id"],
                        "task_type": item["task_type"],
                        "title": item["title"],
                        "status": "error",
                        "resolved_concepts": state.get("resolved_concepts", []),
                        "evidence": [],
                        "summary": f"task crash: {output}",
                        "error": str(output),
                    }
                )
            else:
                results.append(output)
    else:
        results.append(
            {
                "task_id": "task_1",
                "task_type": "CHIT_CHAT",
                "title": TASK_TITLES["CHIT_CHAT"],
                "status": "success",
                "resolved_concepts": [],
                "evidence": [],
                "summary": "일반 대화 - retrieval 생략",
                "error": None,
            }
        )

    duration_ms = _ms(started)
    trace_msg = ", ".join([f"{r['task_type']}:{r['status']}:{r.get('duration_ms',0)}ms" for r in results])
    logger.info("executor finished", extra={"request_id": _req(state), "task_count": len(results), "duration_ms": duration_ms})
    return {
        "task_results": results,
        "trace_log": update_trace(state, "Executor", f"{trace_msg} | duration_ms={duration_ms}"),
    }


async def node_composer(state: AgentState) -> Dict[str, Any]:
    """Convert raw task outputs into stable response sections for generation."""
    started = time.perf_counter()
    task_results = state.get("task_results", []) or []
    sections: List[Dict[str, Any]] = []

    for item in task_results:
        if item["task_type"] == "CHIT_CHAT":
            continue
        sections.append(
            {
                "title": item["title"],
                "task_id": item["task_id"],
                "task_type": item["task_type"],
                "instruction": SECTION_INSTRUCTIONS.get(item["task_type"], "근거 중심으로 설명하세요."),
                "status": item["status"],
                "summary": item.get("summary", ""),
                "evidence": item.get("evidence", []),
                "evidence_count": item.get("evidence_count", len(item.get("evidence", []))),
                "duration_ms": item.get("duration_ms"),
            }
        )

    duration_ms = _ms(started)
    logger.info("composer finished", extra={"request_id": _req(state), "section_count": len(sections), "duration_ms": duration_ms})
    return {
        "response_sections": sections,
        "trace_log": update_trace(state, "Composer", f"sections={len(sections)}, duration_ms={duration_ms}"),
    }


async def node_guard(state: AgentState) -> Dict[str, Any]:
    """Apply a lightweight evidence guard before answer generation.

    This node is intentionally code-based, not model-based, to keep latency low.
    It rejects sections that need product/rider identity but do not carry it in
    evidence, and it annotates thin sections so the generator does not overclaim.
    """
    started = time.perf_counter()
    sections = state.get("response_sections", []) or []
    guarded_sections: List[Dict[str, Any]] = []

    for section in sections:
        guarded = dict(section)
        evidence = guarded.get("evidence", []) or []
        needs_identity = _needs_product_rider(guarded.get("task_type", ""))

        if not evidence:
            guarded["guard_status"] = "no_evidence"
            guarded["guard_note"] = "검색된 근거가 없어 확정 답변을 생성하지 않음"
        elif needs_identity and not _has_product_rider(evidence):
            guarded["guard_status"] = "identity_missing"
            guarded["guard_note"] = "상품명 또는 특약명이 확인되지 않아 근거 부족으로 처리"
            guarded["status"] = "no_evidence"
            guarded["summary"] = "상품/특약 식별 근거 부족"
        else:
            guarded["guard_status"] = "passed"
            guarded["guard_note"] = "근거 검증 통과"

        guarded_sections.append(guarded)

    duration_ms = _ms(started)
    logger.info("guard finished", extra={"request_id": _req(state), "section_count": len(guarded_sections), "duration_ms": duration_ms})
    return {
        "guarded_sections": guarded_sections,
        "trace_log": update_trace(state, "Guard", f"sections={len(guarded_sections)}, duration_ms={duration_ms}"),
    }


async def node_generator(state: AgentState) -> Dict[str, Any]:
    """Generate the final user-facing answer from guarded response sections."""
    started = time.perf_counter()
    task_types = state.get("tasks", []) or []

    if task_types == ["CHIT_CHAT"]:
        logger.info("generator started", extra={"request_id": _req(state), "mode": "chit_chat"})
        res = await llm.ainvoke(state["question"])
        return {
            "final_answer": remove_think_tag(res.content),
            "trace_log": update_trace(state, "Generator", f"chit-chat response, duration_ms={_ms(started)}"),
        }

    sections = state.get("guarded_sections", []) or state.get("response_sections", []) or []
    if not sections:
        sections = [
            {
                "title": "확인 결과",
                "task_type": "NONE",
                "instruction": "관련 근거 부족을 명시하세요.",
                "status": "no_evidence",
                "summary": "검색된 근거가 없습니다.",
                "evidence": [],
            }
        ]

    payload_sections = json.dumps(sections, ensure_ascii=False, indent=2)
    payload_tasks = json.dumps(task_types, ensure_ascii=False)
    logger.info("generator started", extra={"request_id": _req(state), "tasks": task_types, "section_count": len(sections)})

    res = await (GENERATOR_PROMPT | llm).ainvoke(
        {"question": state["question"], "tasks": payload_tasks, "response_sections": payload_sections}
    )

    return {
        "final_answer": remove_think_tag(res.content),
        "trace_log": update_trace(state, "Generator", f"response generated, duration_ms={_ms(started)}"),
    }



def _normalize_tasks(tasks: List[str]) -> List[str]:
    """Normalize LLM-produced tasks into the fixed supported task set."""
    allowed = {"DEFINE_TERM", "GET_BENEFIT", "GET_CONDITION", "GET_EXCLUSION", "COMPARE_PRODUCTS", "CHIT_CHAT"}
    normalized: List[str] = []
    for task in tasks or []:
        if task in allowed and task not in normalized:
            normalized.append(task)
    if not normalized:
        normalized = ["CHIT_CHAT"]
    if "CHIT_CHAT" in normalized and len(normalized) > 1:
        normalized = [t for t in normalized if t != "CHIT_CHAT"]
    return normalized[:3]


workflow = StateGraph(AgentState)
workflow.add_node("analyzer", node_analyzer)
workflow.add_node("grounder", node_grounder)
workflow.add_node("planner", node_planner)
workflow.add_node("executor", node_executor)
workflow.add_node("composer", node_composer)
workflow.add_node("guard", node_guard)
workflow.add_node("generator", node_generator)

workflow.add_edge(START, "analyzer")
workflow.add_edge("analyzer", "grounder")
workflow.add_edge("grounder", "planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "composer")
workflow.add_edge("composer", "guard")
workflow.add_edge("guard", "generator")
workflow.add_edge("generator", END)

app_graph = workflow.compile()
