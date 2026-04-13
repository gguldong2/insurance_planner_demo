"""Retrieval helpers for the insurance QA workflow.

These helpers keep evidence compact and metadata-rich so the graph layer can
stay deterministic. The most important guarantee is that product/company/rider
identity survives retrieval whenever possible.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import os

from openai import AsyncOpenAI
from qdrant_client import models

from .runtime_conn import RuntimeDB  # ← 상대 import (engine 패키지 내부)

logger = logging.getLogger(__name__)


def _normalize_keyword_text(text: str) -> str:
    text = (text or "").lower()
    text = text.replace(" ", "")
    text = text.replace("-", "")
    text = text.replace("_", "")
    text = text.replace("(", "").replace(")", "")
    text = text.replace("[", "").replace("]", "")
    return text


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


_embed_client = AsyncOpenAI(
    base_url=os.getenv("EMBED_API_BASE", "http://localhost:8001/v1"),
    api_key="dummy",
)
_embed_model = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
db = RuntimeDB()


async def get_embedding(text: str):
    """Encode text via remote vLLM embedding endpoint."""
    resp = await _embed_client.embeddings.create(model=_embed_model, input=text)
    return resp.data[0].embedding


async def link_concept_candidates(keyword: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Return candidate concept nodes for a user keyword."""
    if not keyword:
        return []

    logger.info("[Retriever:Grounding] keyword=%s", keyword)
    vec = await get_embedding(keyword)
    hits = await db.search_vector(collection="concepts", vector=vec, limit=limit)

    candidates: List[Dict[str, Any]] = []
    for hit in hits or []:
        payload = dict(hit.payload or {})
        payload.update(
            {
                "keyword": keyword,
                "score": float(getattr(hit, "score", 0.0) or 0.0),
                "matched_text": payload.get("text") or f"{payload.get('label_ko', '')} ({payload.get('category', '')})",
            }
        )
        candidates.append(payload)
    return candidates


async def link_concept(keyword: str) -> Optional[Dict[str, Any]]:
    """Convenience wrapper that returns only the top concept candidate."""
    candidates = await link_concept_candidates(keyword, limit=1)
    return candidates[0] if candidates else None


async def retrieve_benefit(concept_id: str) -> List[Dict[str, Any]]:
    """Fetch benefits tied to a concept with product/rider identity preserved."""
    started = time.perf_counter()
    logger.info("benefit retrieval started", extra={"concept_id": concept_id})

    query = f"""
    MATCH (p:Product)-[:HAS_RIDER]->(r:Rider)-[:HAS_BENEFIT]->(b:Benefit)-[:RELATED_TO]->(c:Concept {{concept_id: '{concept_id}'}})
    RETURN {{
        company: p.company,
        product_id: p.product_id,
        product_name: p.name,
        rider_id: r.rider_id,
        rider_name: r.name,
        benefit_name: b.name,
        amount: b.amount_text,
        condition: b.condition_summary,
        concept_id: c.concept_id,
        concept_label: c.label_ko
    }}
    """
    results = await db.execute_cypher(query)
    logger.info("benefit retrieval finished", extra={"concept_id": concept_id, "result_count": len(results), "duration_ms": _ms(started)})
    return results or []


async def retrieve_exclusion(concept_id: str) -> List[Dict[str, Any]]:
    """Fetch exclusion/limitation clauses relevant to a concept."""
    started = time.perf_counter()
    logger.info("exclusion retrieval started", extra={"concept_id": concept_id})
    vec = await get_embedding(f"{concept_id} 면책 또는 제한")

    filter_cond = models.Filter(
        must=[
            models.FieldCondition(key="type", match=models.MatchValue(value="clause")),
            models.FieldCondition(key="related_concepts", match=models.MatchAny(any=[concept_id])),
        ]
    )
    hits = await db.search_vector("insurance_knowledge", vec, filter=filter_cond, limit=5)

    valid_results = []
    for hit in hits or []:
        payload = hit.payload or {}
        clause_id = payload.get("node_id")
        rider_id = payload.get("rider_id")

        verify_query = f"""
        MATCH (p:Product)-[:HAS_RIDER]->(r:Rider {{rider_id: '{rider_id}'}})-[:RESTRICTS]->(c:Clause {{clause_id: '{clause_id}'}})
        RETURN {{
            company: p.company,
            product_id: p.product_id,
            product_name: p.name,
            rider_id: r.rider_id,
            rider_name: r.name,
            clause_id: c.clause_id,
            clause_title: c.title,
            relation_type: 'RESTRICTS',
            tag: c.tag
        }} AS row
        UNION ALL
        MATCH (p:Product)-[:HAS_RIDER]->(r:Rider {{rider_id: '{rider_id}'}})-[:HAS_CLAUSE]->(c:Clause {{clause_id: '{clause_id}'}})
        RETURN {{
            company: p.company,
            product_id: p.product_id,
            product_name: p.name,
            rider_id: r.rider_id,
            rider_name: r.name,
            clause_id: c.clause_id,
            clause_title: c.title,
            relation_type: 'HAS_CLAUSE',
            tag: c.tag
        }} AS row
        """
        graph_check = await db.execute_cypher(verify_query)

        if graph_check:
            meta = graph_check[0].get("row", graph_check[0]) if isinstance(graph_check[0], dict) else graph_check[0]
            valid_results.append(
                {
                    "company": meta.get("company"),
                    "product_id": meta.get("product_id"),
                    "product_name": meta.get("product_name"),
                    "rider_id": meta.get("rider_id") or rider_id,
                    "rider_name": meta.get("rider_name"),
                    "text": payload.get("text"),
                    "score": float(getattr(hit, "score", 0.0) or 0.0),
                    "verified": True,
                    "clause_id": clause_id,
                    "clause_title": meta.get("clause_title"),
                    "relation_type": meta.get("relation_type"),
                    "tag": meta.get("tag"),
                }
            )

    logger.info("exclusion retrieval finished", extra={"concept_id": concept_id, "result_count": len(valid_results), "duration_ms": _ms(started)})
    return valid_results


async def retrieve_condition(concept_id: str) -> List[Dict[str, Any]]:
    """Return condition-focused evidence, with vector fallback if needed."""
    started = time.perf_counter()
    logger.info("condition retrieval started", extra={"concept_id": concept_id})
    graph_data = await retrieve_benefit(concept_id)

    final_results = []
    for item in graph_data:
        condition_text = item.get("condition", "") or ""

        if len(condition_text) < 10 or "참조" in condition_text:
            logger.info("condition fallback to vector", extra={"concept_id": concept_id, "benefit_name": item.get("benefit_name")})
            vec = await get_embedding(f"{item['benefit_name']} 지급 조건 상세")
            filter_cond = models.Filter(
                must=[
                    models.FieldCondition(key="type", match=models.MatchValue(value="clause")),
                    models.FieldCondition(key="tag", match=models.MatchValue(value="CONDITION")),
                    models.FieldCondition(key="related_concepts", match=models.MatchAny(any=[concept_id])),
                ]
            )
            hits = await db.search_vector("insurance_knowledge", vec, filter=filter_cond, limit=1)
            if hits:
                condition_text = hits[0].payload.get("text")

        final_results.append(
            {
                "company": item.get("company"),
                "product_id": item.get("product_id"),
                "product_name": item.get("product_name"),
                "rider_id": item.get("rider_id"),
                "rider_name": item.get("rider_name"),
                "benefit": item.get("benefit_name"),
                "condition_detail": condition_text,
            }
        )

    logger.info("condition retrieval finished", extra={"concept_id": concept_id, "result_count": len(final_results), "duration_ms": _ms(started)})
    return final_results


async def retrieve_term(keyword: str) -> Optional[Dict[str, Any]]:
    """Search glossary first, then fall back to clause text search."""
    started = time.perf_counter()
    logger.info("term retrieval started", extra={"keyword": keyword})
    if not keyword:
        return None

    vec = await get_embedding(keyword)

    try:
        glossary_hits = await db.search_vector(collection="glossary", vector=vec, limit=1)
        if glossary_hits:
            payload = glossary_hits[0].payload or {}
            result = {
                "definition": payload.get("definition", ""),
                "category": payload.get("category", "TERM"),
                "term_name": payload.get("term_name", keyword),
                "source": "glossary",
                "score": float(getattr(glossary_hits[0], "score", 0.0) or 0.0),
                "mapped_concept_id": payload.get("mapped_concept_id"),
            }
            logger.info("term retrieval finished", extra={"keyword": keyword, "source": "glossary", "duration_ms": _ms(started)})
            return result
    except Exception:
        logger.warning("term glossary search failed", extra={"keyword": keyword})

    filter_cond = models.Filter(must=[models.FieldCondition(key="type", match=models.MatchValue(value="clause"))])
    hits = await db.search_vector(collection="insurance_knowledge", vector=vec, filter=filter_cond, limit=1)

    if hits:
        payload = hits[0].payload or {}
        result = {
            "definition": payload.get("text", ""),
            "category": payload.get("tag", "GENERAL"),
            "term_name": keyword,
            "source": "insurance_knowledge",
            "score": float(getattr(hits[0], "score", 0.0) or 0.0),
            "mapped_concept_id": None,
            "company": payload.get("company"),
            "product_name": payload.get("product_name"),
            "rider_name": payload.get("rider_name"),
        }
        logger.info("term retrieval finished", extra={"keyword": keyword, "source": "insurance_knowledge", "duration_ms": _ms(started)})
        return result

    logger.info("term retrieval finished", extra={"keyword": keyword, "source": None, "duration_ms": _ms(started)})
    return None


async def retrieve_comparison(concept_id: str, product_keywords: list) -> Dict[str, Any]:
    """Return comparison evidence for product keywords under one concept."""
    started = time.perf_counter()
    logger.info("comparison retrieval started", extra={"concept_id": concept_id, "product_keywords": product_keywords})

    comparison_data = {}
    for prod_kwd in product_keywords:
        query = f"""
        MATCH (p:Product)-[:HAS_RIDER]->(r:Rider)-[:HAS_BENEFIT]->(b:Benefit)-[:RELATED_TO]->(c:Concept {{concept_id: '{concept_id}'}})
        WHERE p.name CONTAINS '{prod_kwd}'
        RETURN {{
            company: p.company,
            product_id: p.product_id,
            product_name: p.name,
            rider_id: r.rider_id,
            rider_name: r.name,
            renewal_type: r.renewal_type,
            benefit_name: b.name,
            amount: b.amount_text,
            condition: b.condition_summary
        }}
        """
        results = await db.execute_cypher(query)

        if results:
            comparison_data[prod_kwd] = results[0]
        else:
            comparison_data[prod_kwd] = {
                "product_name": prod_kwd,
                "message": "해당 상품에서 관련 보장을 찾을 수 없습니다.",
            }

    logger.info("comparison retrieval finished", extra={"concept_id": concept_id, "product_keyword_count": len(product_keywords), "result_count": len(comparison_data), "duration_ms": _ms(started)})
    return comparison_data


async def retrieve_plan_catalog(
    concept_id: Optional[str] = None,
    product_keywords: Optional[List[str]] = None,
    limit: int = 12,
) -> List[Dict[str, Any]]:
    """Return plan-unit candidates grouped by company/product/rider.

    The returned objects are intentionally rich enough for scoring:
    - benefits: small list of benefit summaries
    - clauses: small list of clause summaries, including RESTRICTS edges
    - company/product/rider identity
    """
    started = time.perf_counter()
    logger.info("plan catalog retrieval started", extra={"concept_id": concept_id, "product_keywords": product_keywords, "limit": limit})

    query = """
    MATCH (p:Product)-[:HAS_RIDER]->(r:Rider)
    OPTIONAL MATCH (r)-[:HAS_BENEFIT]->(b:Benefit)
    OPTIONAL MATCH (b)-[:RELATED_TO]->(c:Concept)
    WITH p, r, collect(DISTINCT {
        benefit_name: b.name,
        amount_text: b.amount_text,
        condition_summary: b.condition_summary,
        concept_id: c.concept_id,
        concept_label: c.label_ko
    }) AS benefits

    OPTIONAL MATCH (r)-[:HAS_CLAUSE]->(cl_general:Clause)
    WITH p, r, benefits, collect(DISTINCT {
        clause_id: cl_general.clause_id,
        title: cl_general.title,
        content: cl_general.content,
        relation_type: 'HAS_CLAUSE',
        tag: cl_general.tag
    }) AS general_clauses

    OPTIONAL MATCH (r)-[:RESTRICTS]->(cl_restrict:Clause)
    WITH p, r, benefits, general_clauses, collect(DISTINCT {
        clause_id: cl_restrict.clause_id,
        title: cl_restrict.title,
        content: cl_restrict.content,
        relation_type: 'RESTRICTS',
        tag: cl_restrict.tag
    }) AS restrict_clauses

    RETURN {
        company: p.company,
        product_id: p.product_id,
        product_name: p.name,
        rider_id: r.rider_id,
        rider_name: r.name,
        renewal_type: r.renewal_type,
        benefits: benefits,
        general_clauses: general_clauses,
        restrict_clauses: restrict_clauses
    } AS row
    """
    rows = await db.execute_cypher(query)
    rows = rows or []

    cleaned: List[Dict[str, Any]] = []
    product_keywords = [x for x in (product_keywords or []) if x]
    for raw in rows:
        row = raw.get("row", raw) if isinstance(raw, dict) else raw
        if not isinstance(row, dict):
            continue
        benefits = [b for b in (row.get("benefits") or []) if isinstance(b, dict) and any(b.values())]
        general_clauses = [c for c in (row.get("general_clauses") or []) if isinstance(c, dict) and any(c.values())]
        restrict_clauses = [c for c in (row.get("restrict_clauses") or []) if isinstance(c, dict) and any(c.values())]
        clauses = general_clauses + restrict_clauses
        item = {**row, "benefits": benefits, "clauses": clauses}

        if concept_id:
            concept_filtered = [b for b in benefits if b.get("concept_id") == concept_id]
            if concept_filtered:
                item["benefits"] = concept_filtered
            elif not product_keywords:
                continue

        if product_keywords:
            haystack = _normalize_keyword_text(f"{item.get('company', '')} {item.get('product_name', '')} {item.get('rider_name', '')}")
            normalized_keywords = [_normalize_keyword_text(k) for k in product_keywords if k]
            if normalized_keywords and not any(keyword in haystack for keyword in normalized_keywords):
                continue

        cleaned.append(item)

    cleaned.sort(key=lambda x: (len(x.get("benefits") or []), len(x.get("clauses") or [])), reverse=True)
    cleaned = cleaned[:limit]
    logger.info("plan catalog retrieval finished", extra={"concept_id": concept_id, "result_count": len(cleaned), "duration_ms": _ms(started)})
    return cleaned
