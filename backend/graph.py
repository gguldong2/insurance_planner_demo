"""LangGraph workflow for the insurance QA agent.

Execution flow
--------------
Analyzer -> Grounder -> Planner -> Executor -> Composer -> Guard -> Generator

Design goals
------------
1. Insurance-domain answers must stay inside retrieved evidence.
2. Recommendation/comparison must operate on company-product-rider plan units.
3. Full names from retrieved data must be preserved without shortening.
4. Limits for retrieval, final candidates, and answer display are explicit.
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, TypedDict

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from backend.logging_utils import setup_logging
from backend.logic.retrievers import (
    _normalize_keyword_text,
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
    model=os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3.5-9B"),
    base_url=os.getenv("LLM_API_BASE", "http://localhost:8000/v1"),
    api_key="dummy",
    temperature=0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

# Explicit limits for retrieval / final candidates / answer display.
INITIAL_RETRIEVAL_LIMIT = 12
FINAL_CANDIDATE_LIMIT = 6
RECOMMEND_ANSWER_TOP_N = 6
COMPARE_ANSWER_TOP_N = 2
MAX_EVIDENCE_PER_CANDIDATE = 3
MAX_SECTION_EVIDENCE = 6


class AgentState(TypedDict, total=False):
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
    answer_skeleton: Dict[str, Any]

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
    "DEFINE_TERM": "용어의 의미를 간단하고 명확하게 설명하세요.",
    "GET_BENEFIT": "보장 항목과 금액을 정확하게 정리하세요.",
    "GET_CONDITION": "지급조건, 가입조건, 유지조건을 구분해서 설명하세요.",
    "GET_EXCLUSION": "면책, 횟수제한, 금액제한, 특정질환 제외를 구분해서 설명하세요.",
    "COMPARE_PLANS": "회사명-상품명-특약명 세트 기준으로 보장 항목, 지급 조건, 제한/면책, 사용자 적합성을 비교하세요.",
    "RECOMMEND_PLANS": "회사명-상품명-특약명 세트 기준으로 추천 이유, 보장 항목, 지급 조건, 제한/면책, 유의사항을 정리하세요.",
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
5. task_candidates는 동적으로 선택하되, 현재 질문 해결에 꼭 필요한 task는 required_tasks에 넣어라.
6. 회사명/상품명/특약명처럼 보이는 표현은 product_keywords에 넣어라.
7. 보장 개념, 질병, 치료, 약관 용어처럼 보이는 표현은 concept_keywords에 넣어라.
8. 나이/성별/질병 이력처럼 보이는 정보는 user_filters에 구조화하라.
9. JSON만 출력하라.

few-shot 예시 1:
질문: 질병 없는 30세 남성이 들면 좋을 특약명과 상품을 추천해줘
출력:
{{
  "intent": "recommend",
  "task_candidates": ["GET_BENEFIT", "GET_CONDITION", "GET_EXCLUSION", "RECOMMEND_PLANS"],
  "required_tasks": ["GET_BENEFIT", "GET_EXCLUSION", "RECOMMEND_PLANS"],
  "concept_keywords": [],
  "product_keywords": [],
  "user_filters": {{"age": 30, "gender": "male", "disease_history": "none"}},
  "notes": ["보험 추천 질문"]
}}

few-shot 예시 2:
질문: 안녕, 이 서비스 뭐야?
출력:
{{
  "intent": "chit_chat",
  "task_candidates": ["CHIT_CHAT"],
  "required_tasks": ["CHIT_CHAT"],
  "concept_keywords": [],
  "product_keywords": [],
  "user_filters": {{}},
  "notes": ["가벼운 대화"]
}}

출력 스키마:
{{
  "intent": "recommend|compare|explain|define_term|chit_chat",
  "task_candidates": ["..."],
  "required_tasks": ["..."],
  "concept_keywords": ["..."],
  "product_keywords": ["..."],
  "user_filters": {{"age": null, "gender": null, "disease_history": null, "coverage_focus": []}},
  "notes": ["..."]
}}
""",
        ),
        ("user", "질문: {question}"),
    ]
)

RECOMMEND_GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 보험 추천 답변 작성기다.
최종 답변은 반드시 제공된 answer_skeleton, allowed_entities, user_filters만 사용해 작성하라.
배경지식, 추측, 일반 상식으로 빈칸을 메우지 마라.

강제 규칙:
1. 회사명 / 상품명 / 특약명은 전달된 full name 그대로 사용하라. 축약, 생략, 재작성 금지.
2. 추천 intent에서는 비교 섹션을 만들지 마라. "비교 대상", "후보 A", "후보 B" 같은 표현 금지.
3. 답변 첫 부분에는 반드시 `## 추천 상품 요약표` 제목과, answer_skeleton.summary_table_rows를 그대로 사용한 markdown 표를 작성하라.
4. 표의 컬럼 순서는 `순위 | 특약명 | 핵심 보장 | 지급 조건 요약 | 제한/면책`으로 유지하라. 특약명 셀 안에는 특약명 다음 줄에 `(회사명 / 상품명)`을 그대로 넣어라.
5. 표 아래 상세 설명은 회사명 / 상품명 단위로 묶고, 그 아래에 `A. 특약명 (N위)` 형태로 정리하라.
6. 각 특약 상세 설명에서는 값이 있는 항목만 작성하라. `a. 추천 이유`는 항상 작성하되, `b. 확인된 보장`, `c. 지급 조건`, `d. 제한/면책`은 해당 값이 있을 때만 작성하라.
7. benefits가 하나라도 있으면 `정보가 제공되지 않았습니다` 같은 문장을 쓰지 마라. conditions나 exclusions도 값이 있으면 마찬가지다.
8. 추천 이유는 점수 숫자 대신, 전달된 recommend_reasons와 score_breakdown 근거를 자연어로 설명하라.
9. 질문 조건(user_filters)은 실제 근거와 연결될 때만 언급하라.
   - age는 지급 조건/가입 조건/보장 조건에 연령 관련 근거가 있을 때만 언급
   - gender는 성별 관련 근거가 있을 때만 언급
   - disease_history는 기존 진단/병력/보장개시일 이전 진단 관련 제한·면책 근거가 있을 때만 언급
   - 근거가 없으면 질문 조건을 억지로 추천 이유에 넣지 마라.
10. condition은 지급조건/가입조건/유지조건으로, exclusion은 면책/횟수제한/금액제한/특정질환 제외로 구분해 설명하라.
11. 면책/제한 내용은 특약 단위 설명에 포함하라. 특약 추천이면 관련 조건과 제한/면책은 함께 제시하는 방향으로 작성하라.
""",
        ),
        (
            "user",
            """[질문]
{question}

[사용자 필터]
{user_filters}

[실행된 task]
{tasks}

[허용 엔티티]
{allowed_entities}

[답변 뼈대]
{answer_skeleton}

위 정보를 바탕으로 한국어 markdown 답변을 작성하라.
""",
        ),
    ]
)

COMPARE_GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 보험 비교 답변 작성기다.
최종 답변은 반드시 제공된 answer_skeleton과 allowed_entities만 사용해 작성하라.
배경지식, 추측, 일반 상식으로 빈칸을 메우지 마라.

강제 규칙:
1. 회사명 / 상품명 / 특약명은 전달된 full name 그대로 사용하라. 축약, 생략, 재작성 금지.
2. 비교 답변은 answer_skeleton의 comparison_axes 기준으로만 정리하라.
3. 추천, 순위, 추천 상품 요약표 같은 표현은 쓰지 마라.
4. 값이 없는 항목은 생략하되, 비교가 불완전한 경우에만 `확인 가능한 근거 범위에서만 비교함` 정도로 짧게 덧붙여라.
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

[답변 뼈대]
{answer_skeleton}

위 정보를 바탕으로 한국어 markdown 답변을 작성하라.
""",
        ),
    ]
)

DEFAULT_GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 보험 QA 답변 작성기다.
최종 답변은 반드시 제공된 answer_skeleton과 allowed_entities만 사용해 작성하라.
배경지식, 추측, 일반 상식으로 빈칸을 메우지 마라.
회사명 / 상품명 / 특약명은 전달된 full name 그대로 사용하라.
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

[답변 뼈대]
{answer_skeleton}

위 정보를 바탕으로 한국어 markdown 답변을 작성하라.
""",
        ),
    ]
)

INSURANCE_HINTS = (
    "보험", "특약", "상품", "약관", "보장", "면책", "제외", "추천", "비교",
    "진단금", "입원", "수술", "지급", "암",
)

# 약어 매핑: 회사 full name → LLM이 답변에서 쓸 수 있는 표현들(공식 전체명·축약어 포함)
# backend/data/products 아래 실제 존재하는 회사만 관리한다.
_COMPANY_ABBREV_MAP: Dict[str, List[str]] = {
    "삼성화재": ["삼성화재", "삼성화재해상보험", "삼성화재보험"],
    "삼성생명": ["삼성생명", "삼성생명보험"],
    "교보라이프플래닛": ["교보라이프플래닛", "교보라이프플래닛생명", "교보라이프", "교보"],
    "NH농협손해보험": ["NH농협손해보험", "NH농협", "농협손해보험", "농협", "NH손해보험"],
    "한화생명": ["한화생명", "한화생명보험", "한화"],
}


def _normalize_for_guard(text: str) -> str:
    """guard 비교용 정규화: 띄어쓰기·괄호 제거 후 소문자로 변환.

    예) 'NH농협 손해보험' → 'nh농협손해보험'
        '교보라이프플래닛 생명' → '교보라이프플래닛생명'
    """
    text = (text or "").replace(" ", "")
    text = re.sub(r"[()（）\[\]]", "", text)
    return text.lower()


def _load_known_company_tokens() -> List[str]:
    """products 폴더의 01_products.json을 읽어 회사명 토큰 집합을 반환한다.

    새 회사 데이터를 추가하기만 하면 자동으로 guard에 반영된다.
    """
    _data_dir = os.path.join(os.path.dirname(__file__), "data", "products")
    found_companies: set[str] = set()
    for path in glob.glob(os.path.join(_data_dir, "*", "01_products.json")):
        try:
            for prod in json.load(open(path, encoding="utf-8")):
                company = prod.get("company")
                if company:
                    found_companies.add(company)
        except Exception:
            logger.warning("company token load failed: %s", path)

    tokens: set[str] = set()
    for company in found_companies:
        tokens.add(company)
        for abbrev in _COMPANY_ABBREV_MAP.get(company, []):
            tokens.add(abbrev)

    return sorted(tokens)


HALLUCINATION_GUARD_TOKENS: List[str] = _load_known_company_tokens()
logger_startup = logging.getLogger(__name__)
logger_startup.debug("hallucination guard tokens loaded: %s", HALLUCINATION_GUARD_TOKENS)


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
        "DEFINE_TERM", "GET_BENEFIT", "GET_CONDITION", "GET_EXCLUSION",
        "COMPARE_PLANS", "RECOMMEND_PLANS", "CHIT_CHAT",
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
    return normalized[:8]


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
    for token in ["암", "진단금", "수술", "입원", "통원", "항암", "방사선"]:
        if token in q and token not in coverage_focus:
            coverage_focus.append(token)
    if coverage_focus:
        filters["coverage_focus"] = coverage_focus
    return filters


def _parse_requested_answer_count(question: str, intent: str) -> int:
    q = question or ""
    m = re.search(r"(\d+)개", q)
    requested = int(m.group(1)) if m else 0
    if intent == "compare":
        return max(COMPARE_ANSWER_TOP_N, min(requested or COMPARE_ANSWER_TOP_N, FINAL_CANDIDATE_LIMIT))
    return max(1, min(requested or RECOMMEND_ANSWER_TOP_N, FINAL_CANDIDATE_LIMIT))


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


def _flatten_allowed_entities_from_candidates(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    companies, products, riders, pairs = set(), set(), set(), set()
    for item in candidates or []:
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
    return "::".join([
        str(item.get("company") or "-"),
        str(item.get("product_id") or item.get("product_name") or "-"),
        str(item.get("rider_id") or item.get("rider_name") or "-"),
    ])


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


def _classify_condition_type(text: str) -> str:
    t = text or ""
    if any(x in t for x in ["가입", "계약"]):
        return "가입조건"
    if any(x in t for x in ["유지", "갱신"]):
        return "유지조건"
    return "지급조건"


def _classify_exclusion_type(text: str, title: str = "") -> str:
    blob = f"{title} {text}".strip()
    if any(x in blob for x in ["횟수", "연 ", "회 한도", "최대"]):
        return "횟수제한"
    if any(x in blob for x in ["금액", "감액", "절반", "1년 미만"]):
        return "금액제한"
    if any(x in blob for x in ["기타피부암", "갑상선암", "제외", "특정질환"]):
        return "특정질환 제외"
    return "면책"


def _condition_clarity_score(candidate: Dict[str, Any]) -> int:
    conditions = [x.get("condition_summary") for x in (candidate.get("benefits") or []) if x.get("condition_summary")]
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
    restrictive = [
        c for c in clauses
        if c.get("relation_type") == "RESTRICTS" or c.get("tag") in {"EXCLUSION", "LIMIT", "RESTRICTION"}
    ]
    penalty = len(restrictive) * 12
    for clause in restrictive:
        text = f"{clause.get('content', '')} {clause.get('title', '')}".strip()
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
        reason_codes = []
        if benefit_match >= 60:
            reason_codes.append("BENEFIT_MATCH_HIGH")
        if condition_clarity >= 60:
            reason_codes.append("CONDITION_CLEAR")
        if exclusion_penalty >= 30:
            reason_codes.append("EXCLUSION_PRESENT")
        if user_filter_match >= 70:
            reason_codes.append("USER_FIT_HIGH")
        logger.info(
            "recommend candidate scoring",
            extra={
                "company": candidate.get("company"),
                "product_name": candidate.get("product_name"),
                "rider_name": candidate.get("rider_name"),
                "benefit_count": len(candidate.get("benefits") or []),
                "clause_count": len(candidate.get("clauses") or []),
                "restrict_clause_count": len([
                    x for x in (candidate.get("clauses") or [])
                    if x.get("relation_type") == "RESTRICTS"
                    or x.get("tag") in {"EXCLUSION", "LIMIT", "RESTRICTION"}
                ]),
                "benefit_match_score": benefit_match,
                "condition_clarity_score": condition_clarity,
                "exclusion_penalty": exclusion_penalty,
                "coverage_breadth_score": coverage_breadth,
                "user_filter_match_score": user_filter_match,
                "final_score": final_score,
                "is_eligible": is_eligible,
                "ineligible_reason": "exclusion_penalty_too_high" if exclusion_block else (None if is_eligible else "identity_missing"),
            },
        )
        scored.append({
            **candidate,
            "score_breakdown": {
                "benefit_match_score": benefit_match,
                "condition_clarity_score": condition_clarity,
                "exclusion_penalty": exclusion_penalty,
                "coverage_breadth_score": coverage_breadth,
                "user_filter_match_score": user_filter_match,
                "final_score": final_score,
            },
            "reason_codes": reason_codes,
            "is_eligible": is_eligible,
            "ineligible_reason": "exclusion_penalty_too_high" if exclusion_block else (None if is_eligible else "identity_missing"),
        })
    scored.sort(key=lambda x: (x.get("is_eligible", False), x["score_breakdown"]["final_score"]), reverse=True)
    return scored


def _prepare_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    benefits = [b for b in (candidate.get("benefits") or []) if isinstance(b, dict)]
    clauses = [c for c in (candidate.get("clauses") or []) if isinstance(c, dict)]
    own_conditions = []
    for benefit in benefits:
        condition_text = benefit.get("condition_summary") or ""
        if condition_text:
            own_conditions.append({
                "benefit_name": benefit.get("benefit_name"),
                "condition_summary": condition_text,
                "condition_type": _classify_condition_type(condition_text),
            })
    own_exclusions = []
    own_general_clauses = []
    for clause in clauses:
        item = {
            "clause_id": clause.get("clause_id"),
            "title": clause.get("title"),
            "content": clause.get("content"),
            "relation_type": clause.get("relation_type"),
            "tag": clause.get("tag"),
        }
        if clause.get("relation_type") == "RESTRICTS" or clause.get("tag") in {"EXCLUSION", "LIMIT", "RESTRICTION"}:
            item["exclusion_type"] = _classify_exclusion_type(clause.get("content", ""), clause.get("title", ""))
            own_exclusions.append(item)
        else:
            own_general_clauses.append(item)
    score = candidate.get("score_breakdown", {})
    reasons = []
    if score.get("benefit_match_score", 0) >= 60:
        reasons.append("관련 보장 항목이 비교적 풍부합니다.")
    if score.get("condition_clarity_score", 0) >= 60:
        reasons.append("지급 조건 설명이 비교적 명확합니다.")
    if score.get("user_filter_match_score", 0) >= 70:
        reasons.append("질문에서 제시한 사용자 조건과의 적합성이 높게 평가되었습니다.")
    cautions = []
    if own_exclusions:
        cautions.append("제한/면책 조항이 있어 세부 조건 확인이 필요합니다.")
    if score.get("condition_clarity_score", 0) < 40:
        cautions.append("지급 조건 설명이 단순하지 않아 약관 세부 확인이 필요합니다.")
    return {
        "company": candidate.get("company"),
        "product_id": candidate.get("product_id"),
        "product_name": candidate.get("product_name"),
        "rider_id": candidate.get("rider_id"),
        "rider_name": candidate.get("rider_name"),
        "renewal_type": candidate.get("renewal_type"),
        "benefits": benefits,
        "own_conditions": own_conditions,
        "own_exclusions": own_exclusions,
        "own_general_clauses": own_general_clauses,
        "score_breakdown": score,
        "reason_codes": candidate.get("reason_codes", []),
        "recommend_reasons": reasons,
        "cautions": cautions,
        "is_eligible": candidate.get("is_eligible"),
        "ineligible_reason": candidate.get("ineligible_reason"),
    }


def _compact_text(text: str, limit: int = 220) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _first_nonempty(items: List[str]) -> str:
    for item in items or []:
        if (item or "").strip():
            return item.strip()
    return ""


def _summarize_benefit_row(benefits: List[Dict[str, Any]]) -> str:
    if not benefits:
        return "-"
    parts = []
    for benefit in benefits[:2]:
        name = (benefit.get("benefit_name") or "").strip()
        amount = (benefit.get("amount_text") or "").strip()
        chunk = f"{name} {amount}".strip()
        if chunk:
            parts.append(chunk)
    return " / ".join(parts) if parts else "-"


def _summarize_condition_row(conditions: List[Dict[str, Any]]) -> str:
    first = conditions[0] if conditions else {}
    summary = (first.get("condition_summary") or "").strip()
    if not summary:
        return "-"
    label = (first.get("condition_type") or "").strip()
    return _compact_text(f"{label}: {summary}" if label else summary, 90)


def _summarize_exclusion_row(exclusions: List[Dict[str, Any]]) -> str:
    first = exclusions[0] if exclusions else {}
    content = (first.get("content") or "").strip()
    if not content:
        return "-"
    label = (first.get("exclusion_type") or "").strip()
    return _compact_text(f"{label}: {content}" if label else content, 90)


def _compact_candidate_for_answer(candidate: Dict[str, Any]) -> Dict[str, Any]:
    prepared = _prepare_candidate(candidate)
    logger.info(
        "recommend compact candidate stats",
        extra={
            "company": prepared.get("company"),
            "product_name": prepared.get("product_name"),
            "rider_name": prepared.get("rider_name"),
            "prepared_benefit_count": len(prepared.get("benefits") or []),
            "prepared_condition_count": len(prepared.get("own_conditions") or []),
            "prepared_exclusion_count": len(prepared.get("own_exclusions") or []),
            "prepared_general_clause_count": len(prepared.get("own_general_clauses") or []),
        },
    )
    benefits = []
    for benefit in prepared["benefits"][:MAX_EVIDENCE_PER_CANDIDATE]:
        benefits.append({
            "benefit_name": benefit.get("benefit_name"),  # full name preserved
            "amount_text": benefit.get("amount_text"),
            "condition_summary": _compact_text(benefit.get("condition_summary", ""), 180),
        })
    conditions = []
    for condition in prepared["own_conditions"][:MAX_EVIDENCE_PER_CANDIDATE]:
        conditions.append({
            "benefit_name": condition.get("benefit_name"),
            "condition_type": condition.get("condition_type"),
            "condition_summary": _compact_text(condition.get("condition_summary", ""), 180),
        })
    exclusions = []
    for exclusion in prepared["own_exclusions"][:MAX_EVIDENCE_PER_CANDIDATE]:
        exclusions.append({
            "title": exclusion.get("title"),
            "exclusion_type": exclusion.get("exclusion_type"),
            "content": _compact_text(exclusion.get("content", ""), 180),
        })
    return {
        "company": prepared.get("company"),
        "product_name": prepared.get("product_name"),
        "rider_name": prepared.get("rider_name"),
        "renewal_type": prepared.get("renewal_type"),
        "benefits": benefits,
        "conditions": conditions,
        "exclusions": exclusions,
        "general_clauses": [
            {
                "title": clause.get("title"),
                "content": _compact_text(clause.get("content", ""), 180),
            }
            for clause in prepared.get("own_general_clauses", [])[:MAX_EVIDENCE_PER_CANDIDATE]
        ],
        "recommend_reasons": prepared.get("recommend_reasons", []),
        "cautions": prepared.get("cautions", []),
        "score_breakdown": prepared.get("score_breakdown", {}),
        "is_eligible": prepared.get("is_eligible"),
        "summary_core_benefit": _summarize_benefit_row(benefits),
        "summary_condition": _summarize_condition_row(conditions),
        "summary_exclusion": _summarize_exclusion_row(exclusions),
    }


def _build_answer_skeleton(state: AgentState) -> Dict[str, Any]:
    intent = state.get("intent", "explain")
    question = state.get("question", "")
    plan_candidates = state.get("plan_candidates", []) or []
    answer_top_n = _parse_requested_answer_count(question, intent)
    selected = plan_candidates[:answer_top_n]
    selected_compact = [_compact_candidate_for_answer(c) for c in selected]

    if intent == "recommend":
        summary_table_rows = []
        grouped_details: List[Dict[str, Any]] = []
        grouped_map: Dict[str, Dict[str, Any]] = {}
        for idx, candidate in enumerate(selected_compact, start=1):
            company = candidate.get("company") or "-"
            product_name = candidate.get("product_name") or "-"
            rider_name = candidate.get("rider_name") or "-"
            summary_table_rows.append({
                "rank": idx,
                "rider_name": rider_name,
                "company": company,
                "product_name": product_name,
                "display_name": f"{rider_name}\n({company} / {product_name})",
                "core_benefit": candidate.get("summary_core_benefit") or "-",
                "condition_summary": candidate.get("summary_condition") or "-",
                "exclusion_summary": candidate.get("summary_exclusion") or "-",
            })
            group_key = f"{company}::{product_name}"
            if group_key not in grouped_map:
                grouped_map[group_key] = {
                    "company": company,
                    "product_name": product_name,
                    "items": [],
                }
                grouped_details.append(grouped_map[group_key])
            alpha = chr(64 + len(grouped_map[group_key]["items"]) + 1)
            grouped_map[group_key]["items"].append({
                **candidate,
                "rank": idx,
                "alpha": alpha,
            })
        return {
            "intent": "recommend",
            "answer_top_n": answer_top_n,
            "summary_table_title": "추천 상품 요약표",
            "summary_table_columns": ["순위", "특약명", "핵심 보장", "지급 조건 요약", "제한/면책"],
            "summary_table_rows": summary_table_rows,
            "grouped_details": grouped_details,
            "candidates": selected_compact,
        }
    if intent == "compare":
        compare_candidates = selected_compact[: max(COMPARE_ANSWER_TOP_N, answer_top_n)]
        return {
            "intent": "compare",
            "answer_top_n": len(compare_candidates),
            "candidates": compare_candidates,
            "comparison_axes": ["보장 항목", "지급 조건", "제한/면책", "사용자 적합성"],
        }
    sections = []
    for section in state.get("guarded_sections", []) or []:
        evidence = (section.get("evidence") or [])[:MAX_SECTION_EVIDENCE]
        compact = []
        for item in evidence:
            if isinstance(item, dict):
                copied = dict(item)
                for key in ["text", "condition", "condition_detail", "content", "definition"]:
                    if key in copied:
                        copied[key] = _compact_text(str(copied.get(key) or ""), 180)
                compact.append(copied)
        sections.append({
            "title": section.get("title"),
            "task_type": section.get("task_type"),
            "instruction": section.get("instruction"),
            "summary": section.get("summary"),
            "status": section.get("status"),
            "evidence": compact,
        })
    return {"intent": intent, "sections": sections}


async def node_analyzer(state: AgentState) -> Dict[str, Any]:
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
    elif intent == "compare" or any(token in question for token in ["비교", "차이", "더 나은"]):
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
    logger.info("analyzer finished", extra={
        "request_id": _req(state), "intent": intent, "tasks": tasks,
        "task_candidates": task_candidates, "required_tasks": required_tasks,
        "concept_keywords": concept_keywords, "product_keywords": product_keywords,
        "user_filters": user_filters, "duration_ms": duration_ms,
    })
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
    logger.info("grounder finished", extra={
        "request_id": _req(state),
        "resolved_concept_ids": [x.get("concept_id") for x in resolved_concepts],
        "resolved_count": len(resolved_concepts),
        "duration_ms": duration_ms,
    })
    return {
        "resolved_concepts": resolved_concepts,
        "trace_log": update_trace(state, "Grounder", f"resolved={len(resolved_concepts)} concepts, duration_ms={duration_ms}"),
    }


async def node_planner(state: AgentState) -> Dict[str, Any]:
    started = time.perf_counter()
    tasks = _normalize_tasks(state.get("tasks", []))
    resolved_concepts = state.get("resolved_concepts", []) or []
    product_keywords = state.get("product_keywords", []) or []
    concept_keywords = state.get("concept_keywords", []) or []
    intent = state.get("intent", _derive_intent_from_tasks(tasks, state.get("question", "")))
    plan: List[Dict[str, Any]] = []
    if tasks == ["CHIT_CHAT"]:
        plan = [{"task_id": "task_1", "task_type": "CHIT_CHAT", "title": TASK_TITLES["CHIT_CHAT"], "inputs": {}, "depends_on": [], "priority": 1}]
    else:
        for idx, task_type in enumerate(tasks, start=1):
            inputs: Dict[str, Any] = {
                "concept_id": resolved_concepts[0].get("concept_id") if resolved_concepts else None,
                "concept_ids": [x.get("concept_id") for x in resolved_concepts if x.get("concept_id")],
                "product_keywords": product_keywords,
                "user_filters": state.get("user_filters", {}),
                "intent": intent,
                "retrieval_limit": INITIAL_RETRIEVAL_LIMIT,
                "final_candidate_limit": FINAL_CANDIDATE_LIMIT,
                "answer_top_n": _parse_requested_answer_count(state.get("question", ""), intent),
            }
            if task_type == "DEFINE_TERM":
                inputs["keyword"] = concept_keywords[0] if concept_keywords else ""
            plan.append({
                "task_id": f"task_{idx}",
                "task_type": task_type,
                "title": TASK_TITLES.get(task_type, task_type),
                "inputs": inputs,
                "depends_on": ["grounding"] if task_type not in {"CHIT_CHAT", "DEFINE_TERM"} else [],
                "priority": idx,
            })
    duration_ms = _ms(started)
    logger.info("planner finished", extra={"request_id": _req(state), "intent": intent, "task_plan": plan, "task_count": len(plan), "duration_ms": duration_ms})
    return {"task_plan": plan, "trace_log": update_trace(state, "Planner", f"planned={len(plan)} tasks, duration_ms={duration_ms}")}


def _benefit_evidence_from_catalog(catalog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = []
    for candidate in catalog:
        for benefit in (candidate.get("benefits") or [])[:MAX_EVIDENCE_PER_CANDIDATE]:
            items.append({
                "company": candidate.get("company"),
                "product_name": candidate.get("product_name"),
                "rider_name": candidate.get("rider_name"),
                "benefit_name": benefit.get("benefit_name"),
                "amount": benefit.get("amount_text"),
                "condition": benefit.get("condition_summary"),
            })
    return items


def _condition_evidence_from_catalog(catalog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = []
    for candidate in catalog:
        for benefit in (candidate.get("benefits") or [])[:MAX_EVIDENCE_PER_CANDIDATE]:
            condition = benefit.get("condition_summary") or ""
            if condition:
                items.append({
                    "company": candidate.get("company"),
                    "product_name": candidate.get("product_name"),
                    "rider_name": candidate.get("rider_name"),
                    "benefit_name": benefit.get("benefit_name"),
                    "condition_type": _classify_condition_type(condition),
                    "condition_detail": condition,
                })
    return items


def _exclusion_evidence_from_catalog(catalog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = []
    for candidate in catalog:
        for clause in (candidate.get("clauses") or []):
            if clause.get("relation_type") == "RESTRICTS" or clause.get("tag") in {"EXCLUSION", "LIMIT", "RESTRICTION"}:
                items.append({
                    "company": candidate.get("company"),
                    "product_name": candidate.get("product_name"),
                    "rider_name": candidate.get("rider_name"),
                    "clause_title": clause.get("title"),
                    "exclusion_type": _classify_exclusion_type(clause.get("content", ""), clause.get("title", "")),
                    "text": clause.get("content"),
                    "relation_type": clause.get("relation_type"),
                })
    return items


async def _execute_task(plan_item: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
    started = time.perf_counter()
    task_type = plan_item["task_type"]
    inputs = plan_item.get("inputs", {})
    resolved_concepts = state.get("resolved_concepts", []) or []
    logger.info("executor task started", extra={"request_id": _req(state), "task_type": task_type, "inputs": inputs})
    result: Dict[str, Any] = {
        "task_id": plan_item["task_id"], "task_type": task_type, "title": plan_item["title"],
        "status": "success", "resolved_concepts": resolved_concepts, "evidence": [], "summary": "",
        "error": None, "duration_ms": None, "evidence_count": 0,
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
                catalog = await retrieve_plan_catalog(concept_id=None, product_keywords=inputs.get("product_keywords"), limit=inputs.get("retrieval_limit", INITIAL_RETRIEVAL_LIMIT))
                res = _benefit_evidence_from_catalog(catalog)
            result["evidence"] = res[:MAX_SECTION_EVIDENCE * 2]
            result["status"] = "success" if res else "no_evidence"
            result["summary"] = f"보장 정보 {len(res)}건 확보"
        elif task_type == "GET_CONDITION":
            concept_id = inputs.get("concept_id")
            if concept_id:
                res = await retrieve_condition(concept_id)
            else:
                catalog = await retrieve_plan_catalog(concept_id=None, product_keywords=inputs.get("product_keywords"), limit=inputs.get("retrieval_limit", INITIAL_RETRIEVAL_LIMIT))
                res = _condition_evidence_from_catalog(catalog)
            result["evidence"] = res[:MAX_SECTION_EVIDENCE * 2]
            result["status"] = "success" if res else "no_evidence"
            result["summary"] = f"지급 조건 {len(res)}건 확보"
        elif task_type == "GET_EXCLUSION":
            concept_id = inputs.get("concept_id")
            if concept_id:
                res = await retrieve_exclusion(concept_id)
            else:
                catalog = await retrieve_plan_catalog(concept_id=None, product_keywords=inputs.get("product_keywords"), limit=inputs.get("retrieval_limit", INITIAL_RETRIEVAL_LIMIT))
                res = _exclusion_evidence_from_catalog(catalog)
            result["evidence"] = res[:MAX_SECTION_EVIDENCE * 3]
            result["status"] = "success" if res else "no_evidence"
            result["summary"] = f"면책/제한 {len(res)}건 확보"
        elif task_type in {"RECOMMEND_PLANS", "COMPARE_PLANS"}:
            concept_id = inputs.get("concept_id")
            product_keywords = inputs.get("product_keywords") or []
            raw_candidates = await retrieve_plan_catalog(
                concept_id=concept_id,
                product_keywords=product_keywords,
                limit=inputs.get("retrieval_limit", INITIAL_RETRIEVAL_LIMIT),
            )
            for c in raw_candidates:
                logger.info(
                    "recommend raw candidate stats",
                    extra={
                        "request_id": _req(state),
                        "company": c.get("company"),
                        "product_name": c.get("product_name"),
                        "rider_name": c.get("rider_name"),
                        "benefit_count": len(c.get("benefits") or []),
                        "clause_count": len(c.get("clauses") or []),
                        "restrict_clause_count": len([
                            x for x in (c.get("clauses") or [])
                            if x.get("relation_type") == "RESTRICTS"
                            or x.get("tag") in {"EXCLUSION", "LIMIT", "RESTRICTION"}
                        ]),
                    },
                )
            scored_candidates = _score_plan_candidates(raw_candidates, inputs.get("user_filters", {}), state.get("question", ""))
            final_limit = inputs.get("final_candidate_limit", FINAL_CANDIDATE_LIMIT)
            if task_type == "COMPARE_PLANS":
                # Diversity pass: ensure at least 1 candidate per requested company/keyword
                # so that explicit comparison targets are never dropped by the final limit cut.
                if product_keywords:
                    diversity_slots: List[Dict[str, Any]] = []
                    remaining = list(scored_candidates)
                    for kwd in product_keywords:
                        norm_kwd = _normalize_keyword_text(kwd)
                        for i, cand in enumerate(remaining):
                            haystack = _normalize_keyword_text(
                                f"{cand.get('company', '')} {cand.get('product_name', '')} {cand.get('rider_name', '')}"
                            )
                            if norm_kwd in haystack:
                                diversity_slots.append(remaining.pop(i))
                                break
                    # Fill remaining slots up to final_limit with highest-scored candidates
                    fill_count = max(0, final_limit - len(diversity_slots))
                    selected = diversity_slots + remaining[:fill_count]
                else:
                    selected = scored_candidates[:final_limit]
                status = "success" if len(selected) >= 2 else "no_evidence"
                summary = f"비교 후보 {len(selected)}건 확보"
            else:
                selected = [x for x in scored_candidates if x.get("is_eligible")][:final_limit] or scored_candidates[:final_limit]
                status = "success" if selected else "no_evidence"
                summary = f"추천 후보 {len(selected)}건 확보"
            evidence = [_compact_candidate_for_answer(x) for x in selected]
            result.update({
                "evidence": evidence,
                "status": status,
                "summary": summary,
                "raw_candidates": selected,
            })
        elif task_type == "CHIT_CHAT":
            result["status"] = "success"
            result["summary"] = "일반 대화"
        else:
            result["status"] = "error"
            result["summary"] = f"알 수 없는 task: {task_type}"
            result["error"] = "unknown_task"
    except Exception as exc:
        logger.exception("executor task failed", extra={"request_id": _req(state), "task_type": task_type})
        result["status"] = "error"
        result["summary"] = f"task error: {exc}"
        result["error"] = str(exc)
    result["duration_ms"] = _ms(started)
    result["evidence_count"] = len(result.get("evidence", []))
    logger.info("executor task finished", extra={
        "request_id": _req(state), "task_type": task_type, "status": result["status"],
        "evidence_count": result["evidence_count"], "duration_ms": result["duration_ms"],
    })
    return result


async def node_executor(state: AgentState) -> Dict[str, Any]:
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
                results.append({
                    "task_id": item["task_id"], "task_type": item["task_type"], "title": item["title"],
                    "status": "error", "resolved_concepts": state.get("resolved_concepts", []),
                    "evidence": [], "summary": f"task crash: {output}", "error": str(output),
                })
            else:
                results.append(output)
    else:
        results.append({
            "task_id": "task_1", "task_type": "CHIT_CHAT", "title": TASK_TITLES["CHIT_CHAT"],
            "status": "success", "resolved_concepts": [], "evidence": [], "summary": "일반 대화 - retrieval 생략",
            "error": None, "duration_ms": 0, "evidence_count": 0,
        })
    plan_candidates: List[Dict[str, Any]] = []
    for item in results:
        if item.get("task_type") in {"RECOMMEND_PLANS", "COMPARE_PLANS"}:
            plan_candidates.extend(item.get("raw_candidates", []) or [])
    dedup: Dict[str, Dict[str, Any]] = {}
    for candidate in plan_candidates:
        key = _build_candidate_key(candidate)
        if key not in dedup or candidate.get("score_breakdown", {}).get("final_score", 0) > dedup[key].get("score_breakdown", {}).get("final_score", 0):
            dedup[key] = candidate
    plan_candidates = sorted(dedup.values(), key=lambda x: x.get("score_breakdown", {}).get("final_score", 0), reverse=True)[:FINAL_CANDIDATE_LIMIT]
    duration_ms = _ms(started)
    logger.info("executor finished", extra={"request_id": _req(state), "task_result_count": len(results), "plan_candidate_count": len(plan_candidates), "duration_ms": duration_ms})
    return {
        "task_results": results,
        "plan_candidates": plan_candidates,
        "trace_log": update_trace(state, "Executor", f"results={len(results)}, plan_candidates={len(plan_candidates)}, duration_ms={duration_ms}"),
    }


async def node_composer(state: AgentState) -> Dict[str, Any]:
    started = time.perf_counter()
    task_results = state.get("task_results", []) or []
    sections: List[Dict[str, Any]] = []
    for item in task_results:
        if item["task_type"] == "CHIT_CHAT":
            continue
        sections.append({
            "title": item["title"],
            "task_id": item["task_id"],
            "task_type": item["task_type"],
            "instruction": SECTION_INSTRUCTIONS.get(item["task_type"], "근거 중심으로 설명하세요."),
            "status": item["status"],
            "summary": item.get("summary", ""),
            "evidence": item.get("evidence", []),
            "evidence_count": item.get("evidence_count", len(item.get("evidence", []))),
            "duration_ms": item.get("duration_ms"),
        })
    duration_ms = _ms(started)
    logger.info("composer finished", extra={"request_id": _req(state), "section_count": len(sections), "duration_ms": duration_ms})
    return {
        "response_sections": sections,
        "trace_log": update_trace(state, "Composer", f"sections={len(sections)}, duration_ms={duration_ms}"),
    }


async def node_guard(state: AgentState) -> Dict[str, Any]:
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
    allowed_entities = _flatten_allowed_entities_from_candidates(state.get("plan_candidates", []) or [])
    answer_skeleton = _build_answer_skeleton({**state, "guarded_sections": guarded_sections})
    duration_ms = _ms(started)
    logger.info("guard finished", extra={"request_id": _req(state), "section_count": len(guarded_sections), "allowed_entities": allowed_entities, "duration_ms": duration_ms})
    return {
        "guarded_sections": guarded_sections,
        "allowed_entities": allowed_entities,
        "answer_skeleton": answer_skeleton,
        "trace_log": update_trace(state, "Guard", f"sections={len(guarded_sections)}, duration_ms={duration_ms}"),
    }


async def node_generator(state: AgentState) -> Dict[str, Any]:
    started = time.perf_counter()
    task_types = state.get("tasks", []) or []
    question = state.get("question", "")
    if task_types == ["CHIT_CHAT"] and not _is_insurance_question(question):
        logger.info("generator started", extra={"request_id": _req(state), "mode": "chit_chat"})
        res = await llm.ainvoke(question)
        return {"final_answer": remove_think_tag(res.content), "trace_log": update_trace(state, "Generator", f"chit-chat response, duration_ms={_ms(started)}")}

    answer_skeleton = state.get("answer_skeleton") or _build_answer_skeleton(state)
    if not answer_skeleton:
        return {"final_answer": "현재 검색된 근거 내에서 확인 가능한 정보를 찾지 못했습니다.", "trace_log": update_trace(state, "Generator", f"grounded fallback, duration_ms={_ms(started)}")}
    if state.get("intent") in {"recommend", "compare"} and not (answer_skeleton.get("candidates") or []):
        return {"final_answer": "현재 검색된 근거 내에서 확인 가능한 상품/특약 후보를 찾지 못했습니다.", "trace_log": update_trace(state, "Generator", f"grounded fallback, duration_ms={_ms(started)}")}

    payload_tasks = json.dumps(task_types, ensure_ascii=False)
    payload_entities = json.dumps(state.get("allowed_entities", {}), ensure_ascii=False)
    payload_skeleton = json.dumps(answer_skeleton, ensure_ascii=False, indent=2)
    payload_user_filters = json.dumps(state.get("user_filters", {}), ensure_ascii=False)
    logger.info("generator started", extra={"request_id": _req(state), "tasks": task_types, "section_count": len((state.get('guarded_sections') or [])), "intent": state.get("intent")})
    intent = state.get("intent")
    if intent == "recommend":
        prompt = RECOMMEND_GENERATOR_PROMPT
    elif intent == "compare":
        prompt = COMPARE_GENERATOR_PROMPT
    else:
        prompt = DEFAULT_GENERATOR_PROMPT
    res = await (prompt | llm).ainvoke({
        "question": question,
        "tasks": payload_tasks,
        "allowed_entities": payload_entities,
        "answer_skeleton": payload_skeleton,
        "user_filters": payload_user_filters,
    })
    answer = remove_think_tag(res.content)
    if intent == "recommend":
        answer = re.sub(r"\n?#+?\s*비교 대상.*", "", answer, flags=re.DOTALL).strip()
    allowed = state.get("allowed_entities", {}) or {}
    allowed_parts = allowed.get("companies", []) + allowed.get("products", []) + allowed.get("riders", [])
    # allowed 회사의 모든 약어·전체명도 허용 범위에 포함 (오탐 방지)
    expanded_allowed: List[str] = list(allowed_parts)
    for company in allowed.get("companies", []):
        expanded_allowed.extend(_COMPANY_ABBREV_MAP.get(company, []))
    norm_answer = _normalize_for_guard(" ".join([answer]))
    norm_allowed = _normalize_for_guard(" ".join(expanded_allowed))
    for suspicious in HALLUCINATION_GUARD_TOKENS:
        norm_suspicious = _normalize_for_guard(suspicious)
        if norm_suspicious in norm_answer and norm_suspicious not in norm_allowed:
            answer = "현재 검색된 근거 내에서 확인 가능한 회사·상품·특약만 답변하도록 제한되어 있습니다. 검색 결과에 없는 회사명이나 상품명은 제외하고 다시 확인해 주세요."
            break
    if intent == "recommend" and answer_skeleton.get("summary_table_rows") and "추천 상품 요약표" not in answer:
        rows = answer_skeleton.get("summary_table_rows", [])
        lines = ["## 추천 상품 요약표", "| 순위 | 특약명 | 핵심 보장 | 지급 조건 요약 | 제한/면책 |", "|---|---|---|---|---|"]
        for row in rows:
            name = str(row.get("display_name", "-")).replace("\n", "<br>")
            lines.append(f"| {row.get('rank','-')} | {name} | {row.get('core_benefit','-')} | {row.get('condition_summary','-')} | {row.get('exclusion_summary','-')} |")
        answer = "\n".join(lines) + "\n\n" + answer
    return {"final_answer": answer, "trace_log": update_trace(state, "Generator", f"response generated, duration_ms={_ms(started)}")}


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
