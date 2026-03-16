# backend/graph.py
import asyncio
import json
import logging
import os
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
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    },
)


class AgentState(TypedDict):
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
반드시 제공된 섹션 근거만 사용해 답하라.
추측하거나 빈칸을 채우지 마라.

규칙:
1. response_sections의 순서를 유지하라.
2. 각 섹션은 해당 섹션의 evidence만 사용하라.
3. 섹션 제목을 살려서 답하라.
4. evidence가 부족한 섹션은 '확인된 근거가 부족하다'고 명시하라.
5. 마지막에 짧은 요약을 추가하라.
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
    trace = list(state.get("trace_log", []) or [])
    trace.append(f"[{node}] {msg}")
    return trace[-50:]




def _req(state: AgentState) -> str:
    return state.get("request_id", "-")


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)

def remove_think_tag(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


async def node_analyzer(state: AgentState) -> Dict[str, Any]:
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
    logger.info("analyzer finished", extra={"request_id": _req(state), "tasks": tasks, "concept_keywords": concept_keywords, "product_keywords": product_keywords, "duration_ms": duration_ms})
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
    started = time.perf_counter()
    keywords = list(dict.fromkeys(state.get("concept_keywords", []) or []))
    logger.info("grounder started", extra={"request_id": _req(state), "keywords": keywords, "keyword_count": len(keywords)})
    resolved_concepts: List[Dict[str, Any]] = []

    async def _resolve(keyword: str) -> Optional[Dict[str, Any]]:
        candidates = await link_concept_candidates(keyword, limit=3, request_id=_req(state))
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
    logger.info("grounder finished", extra={"request_id": _req(state), "resolved_concept_ids": [x.get("concept_id") for x in resolved_concepts], "resolved_count": len(resolved_concepts), "duration_ms": duration_ms})
    return {
        "resolved_concepts": resolved_concepts,
        "trace_log": update_trace(
            state,
            "Grounder",
            f"resolved={len(resolved_concepts)} concepts, duration_ms={duration_ms}",
        ),
    }


async def node_planner(state: AgentState) -> Dict[str, Any]:
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
                inputs = {
                    "concept_id": resolved_concepts[0].get("concept_id") if resolved_concepts else None,
                }
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

    logger.info("planner finished", extra={"request_id": _req(state), "task_plan": plan, "task_count": len(plan), "duration_ms": _ms(started)})
    return {
        "task_plan": plan,
        "trace_log": update_trace(state, "Planner", f"planned={len(plan)} tasks, duration_ms={_ms(started)}"),
    }


async def _execute_task(plan_item: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
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
    logger.info("executor task finished", extra={"request_id": _req(state), "task_type": task_type, "status": result["status"], "evidence_count": result["evidence_count"], "duration_ms": result["duration_ms"]})
    return result


async def node_executor(state: AgentState) -> Dict[str, Any]:
    started = time.perf_counter()
    task_plan = state.get("task_plan", []) or []
    if not task_plan:
        return {
            "task_results": [],
            "trace_log": update_trace(state, "Executor", "planned tasks not found"),
        }

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

    logger.info("composer finished", extra={"request_id": _req(state), "section_count": len(sections), "duration_ms": _ms(started)})
    return {
        "response_sections": sections,
        "trace_log": update_trace(state, "Composer", f"sections={len(sections)}, duration_ms={_ms(started)}"),
    }


async def node_generator(state: AgentState) -> Dict[str, Any]:
    started = time.perf_counter()
    task_types = state.get("tasks", []) or []

    if task_types == ["CHIT_CHAT"]:
        logger.info("generator started", extra={"request_id": _req(state), "mode": "chit_chat"})
        res = await llm.ainvoke(state["question"])
        return {
            "final_answer": remove_think_tag(res.content),
            "trace_log": update_trace(state, "Generator", f"chit-chat response, duration_ms={_ms(started)}"),
        }

    sections = state.get("response_sections", []) or []
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
        {
            "question": state["question"],
            "tasks": payload_tasks,
            "response_sections": payload_sections,
        }
    )

    return {
        "final_answer": remove_think_tag(res.content),
        "trace_log": update_trace(state, "Generator", f"response generated, duration_ms={_ms(started)}"),
    }


def _normalize_tasks(tasks: List[str]) -> List[str]:
    allowed = {
        "DEFINE_TERM",
        "GET_BENEFIT",
        "GET_CONDITION",
        "GET_EXCLUSION",
        "COMPARE_PRODUCTS",
        "CHIT_CHAT",
    }
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
workflow.add_node("generator", node_generator)

workflow.add_edge(START, "analyzer")
workflow.add_edge("analyzer", "grounder")
workflow.add_edge("grounder", "planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "composer")
workflow.add_edge("composer", "generator")
workflow.add_edge("generator", END)

app_graph = workflow.compile()
