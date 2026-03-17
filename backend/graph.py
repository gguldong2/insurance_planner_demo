"""LangGraph workflow for the insurance QA agent.

Execution flow
--------------
Analyzer -> Grounder -> Planner -> Executor -> Composer -> Guard -> Generator

This version is intentionally more explicit about two requirements:
1. Insurance-domain answers must stay inside retrieved evidence.
2. Recommendation/comparison should operate on company-product-rider plan units.
"""

from __future__ import annotations

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
    retrieve_plan_catalog,
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


class AgentState(TypedDict, total=False):
    """Mutable state shared across LangGraph nodes."""

    question: str
    request_id: str

    intent: str
    tasks: List[str]
    task_candidates: List[str]
    required_tasks: List[str]
    concept_keywords: List[str]
    product_keywords: List[str]
    analysis_notes: List[str]
    user_filters: Dict[str, Any]

    resolved_concepts: List[Dict[str, Any]]
    task_plan: List[Dict[str, Any]]
    task_results: List[Dict[str, Any]]
    response_sections: List[Dict[str, Any]]
    guarded_sections: List[Dict[str, Any]]
    plan_candidates: List[Dict[str, Any]]
    allowed_entities: Dict[str, Any]

    final_answer: str
    trace_log: List[str]
    node_models: Dict[str, str]
    execution_metrics: Dict[str, Any]


TASK_TITLES = {
    "DEFINE_TERM": "용어 설명",
    "GET_BENEFIT": "보장 금액/항목",
    "GET_CONDITION": "지급 조건",
    "GET_EXCLUSION": "면책/제한",
    "COMPARE_PLANS": "상품-특약 비교",
    "RECOMMEND_PLANS": "상품-특약 추천",
    "CHIT_CHAT": "일반 대화",
}

SECTION_INSTRUCTIONS = {
    "DEFINE_TERM": "용어의 의미를 먼저 간단하고 명확하게 설명하세요.",
    "GET_BENEFIT": "보장 항목과 금액을 정확하게 정리하세요.",
    "GET_CONDITION": "지급 요건과 시점을 조건 중심으로 설명하세요.",
    "GET_EXCLUSION": "면책 및 제한 사항을 주의사항처럼 분명하게 설명하세요.",
    "COMPARE_PLANS": "회사명-상품명-특약명 세트 기준으로 차이를 비교하세요.",
    "RECOMMEND_PLANS": "회사명-상품명-특약명 세트 기준으로 추천 이유와 유의점을 정리하세요.",
}

ANALYZER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 보험 QA 분석기다.
질문을 해석해서 상위 intent와 task 후보를 JSON으로 반환하라.

허용 intent:
- chit_chat: 인사, 감사, 짧은 사용법 질문
- define_term: 용어 뜻/정의
- explain: 보장/조건/면책 등 설명형 질문
- compare: 상품-특약 비교 질문
- recommend: 상품-특약 추천 질문

허용 task:
- DEFINE_TERM
- GET_BENEFIT
- GET_CONDITION
- GET_EXCLUSION
- COMPARE_PLANS
- RECOMMEND_PLANS
- CHIT_CHAT

중요 규칙:
1. 보험, 상품, 특약, 보장, 조건, 면책, 지급, 추천, 비교 관련 질문은 CHIT_CHAT으로 분류하지 마라.
2. 추천 intent면 RECOMMEND_PLANS를 포함하라.
3. 비교 intent면 COMPARE_PLANS를 포함하라.
4. 추천/비교 intent에서는 필요한 설명 task도 함께 후보에 넣어라.
5. 추후 tool 확장을 고려해 task_candidates는 동적으로 선택하되, 현재 질문 해결에 꼭 필요한 task는 required_tasks에 넣어라.
6. 상품명/플랜명/브랜드명/시리즈명처럼 보이는 표현은 product_keywords에 넣어라.
7. 보장 개념, 질병, 치료, 약관 용어처럼 보이는 표현은 concept_keywords에 넣어라.
8. 나이/성별/질병 이력처럼 보이는 정보는 user_filters에 구조화하라.
9. JSON만 출력하라.

few-shot 예시 1:
질문: 질병 없는 30세 남성이 들면 좋을 특약명과 상품을 추천해줘
출력:
{
  "intent": "recommend",
  "task_candidates": ["GET_BENEFIT", "GET_CONDITION", "GET_EXCLUSION", "RECOMMEND_PLANS"],
  "required_tasks": ["GET_BENEFIT", "GET_EXCLUSION", "RECOMMEND_PLANS"],
  "concept_keywords": [],
  "product_keywords": [],
  "user_filters": {"age": 30, "gender": "male", "disease_history": "none"},
  "notes": ["보험 추천 질문"]
}

few-shot 예시 2:
질문: 안녕, 이 서비스 뭐야?
출력:
{
  "intent": "chit_chat",
  "task_candidates": ["CHIT_CHAT"],
  "required_tasks": ["CHIT_CHAT"],
  "concept_keywords": [],
  "product_keywords": [],
  "user_filters": {},
  "notes": ["가벼운 대화"]
}

출력 스키마:
{
  "intent": "recommend|compare|explain|define_term|chit_chat",
  "task_candidates": ["..."],
  "required_tasks": ["..."],
  "concept_keywords": ["..."],
  "product_keywords": ["..."],
  "user_filters": {"age": null, "gender": null, "disease_history": null, "coverage_focus": []},
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
최종 답변은 반드시 제공된 response_sections의 evidence와 allowed_entities만 사용해 작성하라.
배경지식, 추측, 일반 상식으로 빈칸을 메우지 마라.

강제 규칙:
1. response_sections의 순서를 유지하라.
2. 각 섹션은 해당 섹션의 evidence만 사용하라.
3. allowed_entities에 없는 회사명, 상품명, 특약명은 새로 만들지 마라.
4. 추천/비교는 반드시 회사명 / 상품명 / 특약명 세트를 같이 쓰라.
5. 각 추천/비교 항목에는 추천 이유 또는 비교 포인트를 근거 중심으로 적어라.
6. evidence가 부족한 섹션은 '확인된 근거가 부족하다'고 명시하라.
7. 마지막에 짧은 요약을 추가하라.

few-shot 예시:
입력 섹션에 한화생명 / Need AI 암보험 / 특정 특약 evidence가 있으면,
답변은 반드시 "한화생명 / Need AI 암보험 / 특정 특약" 형태로 서술해야 한다.
""",
        ),
        (
            "user",
            """[질문]
{question}

[실행된 task]
{tasks}

[허용 엔티티]
{allowed_entities}

[응답 섹션]
{response_sections}

위 정보를 바탕으로 한국어로 답하라.
""",
        ),
    ]
)


INSURANCE_HINTS = (
    "보험",
    "특약",
    "상품",
    "약관",
    "보장",
    "면책",
    "제외",
    "추천",
    "비교",
    "진단금",
    "입원",
    "수술",
    "지급",
    "암",
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def update_trace(state: AgentState, node: str, msg: str) -> List[str]:
    trace = list(state.get("trace_log", []) or [])
    trace.append(f"[{node}] {msg}")
    return trace[-80:]


def _req(state: AgentState) -> str:
    return state.get("request_id", "-")


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def remove_think_tag(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _is_insurance_question(question: str) -> bool:
    q = question or ""
    return any(token in q for token in INSURANCE_HINTS)


def _normalize_task_name(task: str) -> Optional[str]:
    aliases = {
        "COMPARE_PRODUCTS": "COMPARE_PLANS",
        "RECOMMEND_PRODUCTS": "RECOMMEND_PLANS",
        "RECOMMEND_RIDERS": "RECOMMEND_PLANS",
        "COMPARE_RIDERS": "COMPARE_PLANS",
    }
    task = aliases.get(task, task)
    allowed = {
        "DEFINE_TERM",
        "GET_BENEFIT",
        "GET_CONDITION",
        "GET_EXCLUSION",
        "COMPARE_PLANS",
        "RECOMMEND_PLANS",
        "CHIT_CHAT",
    }
    return task if task in allowed else None


def _normalize_tasks(tasks: List[str]) -> List[str]:
    normalized: List[str] = []
    for task in tasks or []:
        name = _normalize_task_name(task)
        if name and name not in normalized:
            normalized.append(name)
    if not normalized:
        return ["CHIT_CHAT"]
    if "CHIT_CHAT" in normalized and len(normalized) > 1:
        normalized = [t for t in normalized if t != "CHIT_CHAT"]
    return normalized[:6]


def _derive_intent_from_tasks(tasks: List[str], question: str) -> str:
    if "RECOMMEND_PLANS" in tasks:
        return "recommend"
    if "COMPARE_PLANS" in tasks:
        return "compare"
    if "DEFINE_TERM" in tasks:
        return "define_term"
    if tasks == ["CHIT_CHAT"] and not _is_insurance_question(question):
        return "chit_chat"
    return "explain"


def _extract_inline_filters(question: str, existing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    filters = dict(existing or {})
    q = question or ""
    age_match = re.search(r"(\d{1,2})\s*세", q)
    if age_match:
        filters["age"] = int(age_match.group(1))
    if any(token in q for token in ["남성", "남자"]):
        filters["gender"] = "male"
    elif any(token in q for token in ["여성", "여자"]):
        filters["gender"] = "female"
    if any(token in q for token in ["질병 없는", "질병없", "병력 없음", "질병 없음"]):
        filters["disease_history"] = "none"
    coverage_focus = list(filters.get("coverage_focus") or [])
    for token in ["암", "진단금", "수술비", "입원비", "통원"]:
        if token in q and token not in coverage_focus:
            coverage_focus.append(token)
    if coverage_focus:
        filters["coverage_focus"] = coverage_focus
    return filters


def _needs_product_rider(task_type: str) -> bool:
    return task_type in {"GET_BENEFIT", "GET_CONDITION", "GET_EXCLUSION", "COMPARE_PLANS", "RECOMMEND_PLANS"}


def _has_product_rider(data: Any) -> bool:
    if isinstance(data, dict):
        if data.get("product_name") or data.get("rider_name"):
            return True
        return any(_has_product_rider(v) for v in data.values())
    if isinstance(data, list):
        return any(_has_product_rider(v) for v in data)
    return False


def _flatten_allowed_entities(sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    companies, products, riders, pairs = set(), set(), set(), set()
    for section in sections:
        for item in section.get("evidence", []) or []:
            if not isinstance(item, dict):
                continue
            company = item.get("company")
            product_name = item.get("product_name")
            rider_name = item.get("rider_name")
            if company:
                companies.add(company)
            if product_name:
                products.add(product_name)
            if rider_name:
                riders.add(rider_name)
            if product_name or rider_name:
                pairs.add(f"{company or '-'}::{product_name or '-'}::{rider_name or '-'}")
    return {
        "companies": sorted(companies),
        "products": sorted(products),
        "riders": sorted(riders),
        "product_rider_pairs": sorted(pairs),
    }


def _build_candidate_key(item: Dict[str, Any]) -> str:
    return "::".join(
        [
            str(item.get("company") or "-"),
            str(item.get("product_id") or item.get("product_name") or "-"),
            str(item.get("rider_id") or item.get("rider_name") or "-"),
        ]
    )


def _text_len_score(text: str) -> int:
    length = len((text or "").strip())
    if length >= 80:
        return 100
    if length >= 40:
        return 80
    if length >= 20:
        return 60
    if length >= 8:
        return 35
    return 10 if length else 0


def _condition_clarity_score(candidate: Dict[str, Any]) -> int:
    conditions = [x for x in (candidate.get("conditions") or []) if x]
    if not conditions:
        return 20
    base = sum(_text_len_score(x) for x in conditions) / max(len(conditions), 1)
    penalty = max(len(conditions) - 3, 0) * 8
    return max(0, min(100, int(base - penalty)))


def _benefit_match_score(candidate: Dict[str, Any], user_filters: Dict[str, Any]) -> int:
    benefits = candidate.get("benefits") or []
    if not benefits:
        return 0
    focus_tokens = list(user_filters.get("coverage_focus") or [])
    if not focus_tokens:
        focus_tokens = []
    score = 20
    benefit_text = " ".join(
        f"{b.get('benefit_name', '')} {b.get('amount_text', '')} {b.get('condition_summary', '')}" for b in benefits
    )
    if focus_tokens:
        matched = sum(1 for token in focus_tokens if token and token in benefit_text)
        score += min(50, matched * 20)
    score += min(30, len(benefits) * 8)
    if any(b.get("amount_text") for b in benefits):
        score += 10
    return min(100, score)


def _coverage_breadth_score(candidate: Dict[str, Any]) -> int:
    benefits = candidate.get("benefits") or []
    concept_ids = {b.get("concept_id") for b in benefits if b.get("concept_id")}
    score = min(60, len(benefits) * 12) + min(40, len(concept_ids) * 20)
    return min(100, score)


def _exclusion_penalty(candidate: Dict[str, Any]) -> int:
    clauses = candidate.get("clauses") or []
    restrictive = [c for c in clauses if c.get("relation_type") == "RESTRICTS" or c.get("tag") in {"EXCLUSION", "LIMIT", "RESTRICTION"}]
    penalty = len(restrictive) * 12
    for clause in restrictive:
        text = (clause.get("content") or "") + " " + (clause.get("title") or "")
        if any(token in text for token in ["제외", "면책", "지급하지", "보장하지", "한도"]):
            penalty += 8
    return min(80, penalty)


def _user_filter_match_score(candidate: Dict[str, Any], user_filters: Dict[str, Any], question: str) -> int:
    score = 50
    q = question or ""
    rider_text = f"{candidate.get('product_name', '')} {candidate.get('rider_name', '')}"
    if user_filters.get("age") is not None:
        score += 10
    if user_filters.get("gender"):
        score += 10
    if user_filters.get("disease_history") == "none":
        score += 10
    if "암" in q and "암" in rider_text:
        score += 10
    focus_tokens = list(user_filters.get("coverage_focus") or [])
    if any(token in rider_text for token in focus_tokens):
        score += 10
    return min(100, score)


def _score_plan_candidates(candidates: List[Dict[str, Any]], user_filters: Dict[str, Any], question: str) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for candidate in candidates:
        benefit_match = _benefit_match_score(candidate, user_filters)
        condition_clarity = _condition_clarity_score(candidate)
        exclusion_penalty = _exclusion_penalty(candidate)
        coverage_breadth = _coverage_breadth_score(candidate)
        user_filter_match = _user_filter_match_score(candidate, user_filters, question)

        is_eligible = bool(candidate.get("product_name") and candidate.get("rider_name"))
        exclusion_block = exclusion_penalty >= 70
        if exclusion_block:
            is_eligible = False

        final_score = round(
            benefit_match * 0.35
            + condition_clarity * 0.20
            - exclusion_penalty * 0.20
            + coverage_breadth * 0.10
            + user_filter_match * 0.35,
            2,
        )

        scored.append(
            {
                **candidate,
                "score_breakdown": {
                    "benefit_match_score": benefit_match,
                    "condition_clarity_score": condition_clarity,
                    "exclusion_penalty": exclusion_penalty,
                    "coverage_breadth_score": coverage_breadth,
                    "user_filter_match_score": user_filter_match,
                    "final_score": final_score,
                },
                "is_eligible": is_eligible,
                "ineligible_reason": "exclusion_penalty_too_high" if exclusion_block else (None if is_eligible else "identity_missing"),
            }
        )

    scored.sort(key=lambda x: (x.get("is_eligible", False), x["score_breakdown"]["final_score"]), reverse=True)
    return scored


def _candidate_to_evidence(candidate: Dict[str, Any]) -> Dict[str, Any]:
    benefits = candidate.get("benefits") or []
    clauses = candidate.get("clauses") or []
    top_benefits = benefits[:3]
    top_clauses = clauses[:2]
    recommend_reason = []
    if top_benefits:
        recommend_reason.append(f"관련 보장 {len(benefits)}건이 확인됨")
    if candidate["score_breakdown"].get("condition_clarity_score", 0) >= 70:
        recommend_reason.append("조건 설명이 비교적 명확함")
    if candidate["score_breakdown"].get("exclusion_penalty", 0) >= 20:
        recommend_reason.append("제한/면책 조항 확인 필요")
    return {
        "company": candidate.get("company"),
        "product_id": candidate.get("product_id"),
        "product_name": candidate.get("product_name"),
        "rider_id": candidate.get("rider_id"),
        "rider_name": candidate.get("rider_name"),
        "renewal_type": candidate.get("renewal_type"),
        "benefits": top_benefits,
        "clauses": top_clauses,
        "score_breakdown": candidate.get("score_breakdown"),
        "is_eligible": candidate.get("is_eligible"),
        "ineligible_reason": candidate.get("ineligible_reason"),
        "recommend_reason": recommend_reason,
    }


def _is_compare_question(question: str) -> bool:
    return any(token in (question or "") for token in ["비교", "차이", "더 나은", "무엇이 더"])


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------
async def node_analyzer(state: AgentState) -> Dict[str, Any]:
    """Interpret the user question into intent, task candidates, and filters."""
    started = time.perf_counter()
    logger.info("analyzer started", extra={"request_id": _req(state), "question": state["question"]})

    payload: Dict[str, Any] = {
        "intent": "chit_chat",
        "task_candidates": ["CHIT_CHAT"],
        "required_tasks": ["CHIT_CHAT"],
        "concept_keywords": [],
        "product_keywords": [],
        "user_filters": {},
        "notes": [],
    }

    try:
        res = await (ANALYZER_PROMPT | llm).ainvoke({"question": state["question"]})
        clean_json = remove_think_tag(res.content).replace("```json", "").replace("```", "").strip()
        payload = json.loads(clean_json)
    except Exception as exc:
        logger.exception("analyzer failed", extra={"request_id": _req(state)})
        payload["notes"] = [f"analyzer_error: {exc}"]

    question = state["question"]
    task_candidates = _normalize_tasks(payload.get("task_candidates", []))
    required_tasks = _normalize_tasks(payload.get("required_tasks", []))
    intent = payload.get("intent") or _derive_intent_from_tasks(task_candidates, question)
    concept_keywords = list(dict.fromkeys(payload.get("concept_keywords", []) or []))
    product_keywords = list(dict.fromkeys(payload.get("product_keywords", []) or []))
    user_filters = _extract_inline_filters(question, payload.get("user_filters") or {})
    notes = list(payload.get("notes", []) or [])

    if _is_insurance_question(question) and task_candidates == ["CHIT_CHAT"]:
        task_candidates = ["GET_BENEFIT", "GET_EXCLUSION"]
        required_tasks = ["GET_BENEFIT"]
        intent = "explain"
        notes.append("보험 질문이어서 CHIT_CHAT 우회를 차단하고 설명형 task로 보정함")

    if intent == "recommend" or any(token in question for token in ["추천", "어울리는", "좋은", "유리한"]):
        task_candidates = _normalize_tasks(task_candidates + ["GET_BENEFIT", "GET_CONDITION", "GET_EXCLUSION", "RECOMMEND_PLANS"])
        required_tasks = _normalize_tasks(required_tasks + ["GET_BENEFIT", "GET_EXCLUSION", "RECOMMEND_PLANS"])
        intent = "recommend"
    elif intent == "compare" or _is_compare_question(question):
        task_candidates = _normalize_tasks(task_candidates + ["GET_BENEFIT", "GET_CONDITION", "GET_EXCLUSION", "COMPARE_PLANS"])
        required_tasks = _normalize_tasks(required_tasks + ["GET_BENEFIT", "GET_EXCLUSION", "COMPARE_PLANS"])
        intent = "compare"
    elif intent == "define_term":
        task_candidates = _normalize_tasks(task_candidates + ["DEFINE_TERM"])
        required_tasks = _normalize_tasks(required_tasks + ["DEFINE_TERM"])
    elif intent == "explain":
        task_candidates = _normalize_tasks(task_candidates or ["GET_BENEFIT"])
        required_tasks = _normalize_tasks(required_tasks or [task_candidates[0]])

    tasks = _normalize_tasks(required_tasks + task_candidates)
    if intent == "recommend":
        ordered = ["GET_BENEFIT", "GET_CONDITION", "GET_EXCLUSION", "RECOMMEND_PLANS"]
        tasks = [t for t in ordered if t in tasks]
    elif intent == "compare":
        ordered = ["GET_BENEFIT", "GET_CONDITION", "GET_EXCLUSION", "COMPARE_PLANS"]
        tasks = [t for t in ordered if t in tasks]

    duration_ms = _ms(started)
    logger.info(
        "analyzer finished",
        extra={
            "request_id": _req(state),
            "intent": intent,
            "tasks": tasks,
            "task_candidates": task_candidates,
            "required_tasks": required_tasks,
            "concept_keywords": concept_keywords,
            "product_keywords": product_keywords,
            "user_filters": user_filters,
            "duration_ms": duration_ms,
        },
    )
    return {
        "intent": intent,
        "tasks": tasks,
        "task_candidates": task_candidates,
        "required_tasks": required_tasks,
        "concept_keywords": concept_keywords,
        "product_keywords": product_keywords,
        "analysis_notes": notes,
        "user_filters": user_filters,
        "trace_log": update_trace(state, "Analyzer", f"intent={intent}, tasks={tasks}, duration_ms={duration_ms}"),
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
    """Transform analyzer output into an ordered execution plan."""
    started = time.perf_counter()
    tasks = _normalize_tasks(state.get("tasks", []))
    resolved_concepts = state.get("resolved_concepts", []) or []
    product_keywords = state.get("product_keywords", []) or []
    concept_keywords = state.get("concept_keywords", []) or []
    intent = state.get("intent", _derive_intent_from_tasks(tasks, state.get("question", "")))

    plan: List[Dict[str, Any]] = []
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
        for idx, task_type in enumerate(tasks, start=1):
            inputs: Dict[str, Any] = {
                "concept_id": resolved_concepts[0].get("concept_id") if resolved_concepts else None,
                "product_keywords": product_keywords,
                "user_filters": state.get("user_filters", {}),
                "intent": intent,
            }
            if task_type == "DEFINE_TERM":
                inputs.update({
                    "keyword": concept_keywords[0] if concept_keywords else "",
                })
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
    logger.info(
        "planner finished",
        extra={
            "request_id": _req(state),
            "intent": intent,
            "task_plan": plan,
            "task_count": len(plan),
            "duration_ms": duration_ms,
        },
    )
    return {
        "task_plan": plan,
        "trace_log": update_trace(state, "Planner", f"planned={len(plan)} tasks, duration_ms={duration_ms}"),
    }


async def _execute_task(plan_item: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
    """Execute a single retrieval or scoring task and normalize its result shape."""
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
            else:
                catalog = await retrieve_plan_catalog(concept_id=None, product_keywords=inputs.get("product_keywords"), limit=12)
                res = [
                    {
                        "company": item.get("company"),
                        "product_name": item.get("product_name"),
                        "rider_name": item.get("rider_name"),
                        "benefit_name": (item.get("benefits") or [{}])[0].get("benefit_name"),
                        "amount": (item.get("benefits") or [{}])[0].get("amount_text"),
                        "condition": (item.get("benefits") or [{}])[0].get("condition_summary"),
                    }
                    for item in catalog
                ]
            result["evidence"] = res
            result["status"] = "success" if res else "no_evidence"
            result["summary"] = f"보장 정보 {len(res)}건 확보"

        elif task_type == "GET_CONDITION":
            concept_id = inputs.get("concept_id")
            if concept_id:
                res = await retrieve_condition(concept_id)
            else:
                catalog = await retrieve_plan_catalog(concept_id=None, product_keywords=inputs.get("product_keywords"), limit=12)
                res = [
                    {
                        "company": item.get("company"),
                        "product_name": item.get("product_name"),
                        "rider_name": item.get("rider_name"),
                        "benefit": (item.get("benefits") or [{}])[0].get("benefit_name"),
                        "condition_detail": (item.get("benefits") or [{}])[0].get("condition_summary"),
                    }
                    for item in catalog
                ]
            result["evidence"] = res
            result["status"] = "success" if res else "no_evidence"
            result["summary"] = f"지급 조건 {len(res)}건 확보"

        elif task_type == "GET_EXCLUSION":
            concept_id = inputs.get("concept_id")
            if concept_id:
                res = await retrieve_exclusion(concept_id)
            else:
                catalog = await retrieve_plan_catalog(concept_id=None, product_keywords=inputs.get("product_keywords"), limit=12)
                res = []
                for item in catalog:
                    for clause in item.get("clauses") or []:
                        if clause.get("relation_type") == "RESTRICTS" or clause.get("tag") in {"EXCLUSION", "LIMIT", "RESTRICTION"}:
                            res.append({
                                "company": item.get("company"),
                                "product_name": item.get("product_name"),
                                "rider_name": item.get("rider_name"),
                                "text": clause.get("content"),
                                "clause_title": clause.get("title"),
                                "relation_type": clause.get("relation_type"),
                            })
            result["evidence"] = res
            result["status"] = "success" if res else "no_evidence"
            result["summary"] = f"면책/제한 {len(res)}건 확보"

        elif task_type in {"RECOMMEND_PLANS", "COMPARE_PLANS"}:
            concept_id = inputs.get("concept_id")
            product_keywords = inputs.get("product_keywords") or []
            raw_candidates = await retrieve_plan_catalog(concept_id=concept_id, product_keywords=product_keywords, limit=12)
            scored_candidates = _score_plan_candidates(raw_candidates, inputs.get("user_filters", {}), state.get("question", ""))
            if task_type == "COMPARE_PLANS":
                selected = scored_candidates[:6]
                evidence = [_candidate_to_evidence(x) for x in selected]
                status = "success" if len(selected) >= 2 else "no_evidence"
                summary = f"비교 후보 {len(selected)}건 확보"
            else:
                selected = [x for x in scored_candidates if x.get("is_eligible")][:6] or scored_candidates[:6]
                evidence = [_candidate_to_evidence(x) for x in selected]
                status = "success" if selected else "no_evidence"
                summary = f"추천 후보 {len(selected)}건 확보"
            result["evidence"] = evidence
            result["status"] = status
            result["summary"] = summary
            result["score_breakdown"] = [x.get("score_breakdown") for x in selected]
            result["raw_candidates"] = scored_candidates[:6]

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
    """Run planned tasks, mostly in parallel, and collect normalized outputs."""
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
                "duration_ms": 0,
                "evidence_count": 0,
            }
        )

    plan_candidates: List[Dict[str, Any]] = []
    for item in results:
        if item.get("task_type") in {"RECOMMEND_PLANS", "COMPARE_PLANS"}:
            plan_candidates.extend(item.get("raw_candidates", []) or [])
    dedup: Dict[str, Dict[str, Any]] = {}
    for candidate in plan_candidates:
        key = _build_candidate_key(candidate)
        if key not in dedup or candidate.get("score_breakdown", {}).get("final_score", 0) > dedup[key].get("score_breakdown", {}).get("final_score", 0):
            dedup[key] = candidate
    plan_candidates = sorted(dedup.values(), key=lambda x: x.get("score_breakdown", {}).get("final_score", 0), reverse=True)[:6]

    duration_ms = _ms(started)
    logger.info("executor finished", extra={"request_id": _req(state), "task_result_count": len(results), "plan_candidate_count": len(plan_candidates), "duration_ms": duration_ms})
    return {
        "task_results": results,
        "plan_candidates": plan_candidates,
        "trace_log": update_trace(state, "Executor", f"results={len(results)}, plan_candidates={len(plan_candidates)}, duration_ms={duration_ms}"),
    }


async def node_composer(state: AgentState) -> Dict[str, Any]:
    """Turn task outputs into response sections for final answer generation."""
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
    """Apply a fast code-based evidence guard before answer generation."""
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
        elif guarded.get("task_type") == "COMPARE_PLANS" and len(evidence) < 2:
            guarded["guard_status"] = "insufficient_compare_candidates"
            guarded["guard_note"] = "비교 대상이 2개 미만이라 비교 설명을 제한함"
            guarded["status"] = "no_evidence"
        else:
            guarded["guard_status"] = "passed"
            guarded["guard_note"] = "근거 검증 통과"

        guarded_sections.append(guarded)

    allowed_entities = _flatten_allowed_entities(guarded_sections)
    duration_ms = _ms(started)
    logger.info("guard finished", extra={"request_id": _req(state), "section_count": len(guarded_sections), "allowed_entities": allowed_entities, "duration_ms": duration_ms})
    return {
        "guarded_sections": guarded_sections,
        "allowed_entities": allowed_entities,
        "trace_log": update_trace(state, "Guard", f"sections={len(guarded_sections)}, duration_ms={duration_ms}"),
    }


async def node_generator(state: AgentState) -> Dict[str, Any]:
    """Generate the final answer while staying inside guarded evidence."""
    started = time.perf_counter()
    task_types = state.get("tasks", []) or []
    question = state.get("question", "")

    if task_types == ["CHIT_CHAT"] and not _is_insurance_question(question):
        logger.info("generator started", extra={"request_id": _req(state), "mode": "chit_chat"})
        res = await llm.ainvoke(question)
        return {
            "final_answer": remove_think_tag(res.content),
            "trace_log": update_trace(state, "Generator", f"chit-chat response, duration_ms={_ms(started)}"),
        }

    sections = state.get("guarded_sections", []) or state.get("response_sections", []) or []
    if not sections:
        sections = [{
            "title": "확인 결과",
            "task_type": "NONE",
            "instruction": "관련 근거 부족을 명시하세요.",
            "status": "no_evidence",
            "summary": "검색된 근거가 없습니다.",
            "evidence": [],
        }]

    non_empty_sections = [s for s in sections if s.get("status") != "no_evidence" and (s.get("evidence") or [])]
    if not non_empty_sections and _is_insurance_question(question):
        return {
            "final_answer": "현재 검색된 근거 내에서 확인 가능한 상품/특약 정보를 찾지 못했습니다. 실제 검색 결과에 있는 회사명·상품명·특약명만 답변하도록 제한되어 있어, 근거가 확보되면 다시 추천 또는 비교할 수 있습니다.",
            "trace_log": update_trace(state, "Generator", f"grounded fallback, duration_ms={_ms(started)}"),
        }

    payload_sections = json.dumps(sections, ensure_ascii=False, indent=2)
    payload_tasks = json.dumps(task_types, ensure_ascii=False)
    payload_entities = json.dumps(state.get("allowed_entities", {}), ensure_ascii=False)
    logger.info("generator started", extra={"request_id": _req(state), "tasks": task_types, "section_count": len(sections)})

    res = await (GENERATOR_PROMPT | llm).ainvoke(
        {
            "question": question,
            "tasks": payload_tasks,
            "allowed_entities": payload_entities,
            "response_sections": payload_sections,
        }
    )
    answer = remove_think_tag(res.content)

    # Simple entity guard: if unknown company/product/rider slips in, fall back.
    allowed = state.get("allowed_entities", {}) or {}
    flattened_allowed = " ".join(allowed.get("companies", []) + allowed.get("products", []) + allowed.get("riders", []))
    for suspicious in ["삼성", "KB", "교보", "DB손해", "메리츠"]:
        if suspicious in answer and suspicious not in flattened_allowed:
            answer = "현재 검색된 근거 내에서 확인 가능한 회사·상품·특약만 답변하도록 제한되어 있습니다. 검색 결과에 없는 회사명이나 상품명은 제외하고 다시 확인해 주세요."
            break

    return {
        "final_answer": answer,
        "trace_log": update_trace(state, "Generator", f"response generated, duration_ms={_ms(started)}"),
    }


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
