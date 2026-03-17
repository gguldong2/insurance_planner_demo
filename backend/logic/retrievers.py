"""Retrieval helpers for the insurance QA workflow.

These functions are intentionally thin and deterministic. Their main job is to
return compact evidence objects with enough metadata for grounded final answers.
In particular, product_name and rider_name are preserved whenever possible so
that the generator can explicitly mention which product/rider the evidence came
from.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import models
import torch

from backend.db.runtime_conn import RuntimeDB

logger = logging.getLogger(__name__)


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


# Single shared embedding model instance.
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=device)
db = RuntimeDB()



def get_embedding(text: str):
    """Encode text into the dense vector used by Qdrant searches."""
    return embed_model.encode(text, return_dense=True)["dense_vecs"]


async def link_concept_candidates(keyword: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Return candidate concept nodes for a user keyword."""
    if not keyword:
        return []

    logger.info("[Retriever:Grounding] keyword=%s", keyword)
    vec = get_embedding(keyword)
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


async def retrieve_benefit(concept_id: str):
    """Fetch benefits tied to a concept with product/rider identity preserved."""
    started = time.perf_counter()
    logger.info("benefit retrieval started", extra={"concept_id": concept_id})

    query = f"""
    MATCH (p:Product)-[:HAS_RIDER]->(r:Rider)-[:HAS_BENEFIT]->(b:Benefit)-[:RELATED_TO]->(c:Concept {{concept_id: '{concept_id}'}})
    RETURN {{
        product_name: p.name,
        rider_name: r.name,
        benefit_name: b.name,
        amount: b.amount_text,
        condition: b.condition_summary,
        limit: b.limit_count
    }}
    """
    results = await db.execute_cypher(query)
    logger.info("benefit retrieval finished", extra={"concept_id": concept_id, "result_count": len(results or []), "duration_ms": _ms(started)})
    return results or []


async def retrieve_exclusion(concept_id: str):
    """Find exclusion clauses and verify them against Graph relationships."""
    started = time.perf_counter()
    logger.info("exclusion retrieval started", extra={"concept_id": concept_id})

    filter_condition = models.Filter(
        must=[
            models.FieldCondition(key="tag", match=models.MatchValue(value="EXCLUSION")),
            models.FieldCondition(key="related_concepts", match=models.MatchAny(any=[concept_id])),
        ]
    )

    vec = get_embedding("면책 지급제한 보장하지 않는 경우")
    hits = await db.search_vector(collection="insurance_knowledge", vector=vec, filter=filter_condition, limit=3)

    valid_results = []
    for hit in hits or []:
        payload = hit.payload or {}
        clause_id = payload.get("node_id")
        rider_id = payload.get("rider_id")

        verify_query = f"""
        MATCH (p:Product)-[:HAS_RIDER]->(r:Rider {{rider_id: '{rider_id}'}})-[:RESTRICTS]->(c:Clause {{clause_id: '{clause_id}'}})
        RETURN {{
            product_name: p.name,
            rider_name: r.name,
            clause_id: c.clause_id,
            clause_title: c.title
        }}
        """
        graph_check = await db.execute_cypher(verify_query)

        if graph_check:
            meta = graph_check[0]
            valid_results.append(
                {
                    "text": payload.get("text"),
                    "score": float(getattr(hit, "score", 0.0) or 0.0),
                    "verified": True,
                    "product_name": meta.get("product_name"),
                    "rider_name": meta.get("rider_name"),
                    "rider_id": rider_id,
                    "clause_id": clause_id,
                    "clause_title": meta.get("clause_title"),
                }
            )

    logger.info("exclusion retrieval finished", extra={"concept_id": concept_id, "result_count": len(valid_results), "duration_ms": _ms(started)})
    return valid_results


async def retrieve_condition(concept_id: str):
    """Return condition-focused evidence, with vector fallback if needed."""
    started = time.perf_counter()
    logger.info("condition retrieval started", extra={"concept_id": concept_id})
    graph_data = await retrieve_benefit(concept_id)

    final_results = []
    for item in graph_data:
        condition_text = item.get("condition", "") or ""

        if len(condition_text) < 10 or "참조" in condition_text:
            logger.info("condition fallback to vector", extra={"concept_id": concept_id, "benefit_name": item.get("benefit_name")})
            vec = get_embedding(f"{item['benefit_name']} 지급 조건 상세")
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
                "product_name": item.get("product_name"),
                "rider_name": item.get("rider_name"),
                "benefit": item.get("benefit_name"),
                "condition_detail": condition_text,
            }
        )

    logger.info("condition retrieval finished", extra={"concept_id": concept_id, "result_count": len(final_results), "duration_ms": _ms(started)})
    return final_results


async def retrieve_term(keyword: str):
    """Search glossary first, then fall back to clause text search."""
    started = time.perf_counter()
    logger.info("term retrieval started", extra={"keyword": keyword})
    if not keyword:
        return None

    vec = get_embedding(keyword)

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
            "product_name": payload.get("product_name"),
            "rider_name": payload.get("rider_name"),
        }
        logger.info("term retrieval finished", extra={"keyword": keyword, "source": "insurance_knowledge", "duration_ms": _ms(started)})
        return result

    logger.info("term retrieval finished", extra={"keyword": keyword, "source": None, "duration_ms": _ms(started)})
    return None


async def retrieve_comparison(concept_id: str, product_keywords: list):
    """Return comparison evidence for product keywords under one concept."""
    started = time.perf_counter()
    logger.info("comparison retrieval started", extra={"concept_id": concept_id, "product_keywords": product_keywords})

    comparison_data = {}
    for prod_kwd in product_keywords:
        query = f"""
        MATCH (p:Product)-[:HAS_RIDER]->(r:Rider)-[:HAS_BENEFIT]->(b:Benefit)-[:RELATED_TO]->(c:Concept {{concept_id: '{concept_id}'}})
        WHERE p.name CONTAINS '{prod_kwd}'
        RETURN {{
            product_name: p.name,
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
