import json
from qdrant_client import models
from FlagEmbedding import BGEM3FlagModel
import torch
from backend.db.runtime_conn import RuntimeDB

# 1. 임베딩 모델 로드 (전역/싱글톤)
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=device)
db = RuntimeDB()

def get_embedding(text: str):
    return embed_model.encode(text, return_dense=True)['dense_vecs']

# =========================================================
# [Step 1] Grounding: Entity Linker
# =========================================================
async def link_concept(keyword: str):
    """
    사용자 키워드('표적항암') -> Concept ID('CPT_TARGETED_THERAPY') 변환
    """
    if not keyword: return None
    
    vec = get_embedding(keyword)
    hits = await db.search_vector(
        collection="concepts",
        vector=vec,
        limit=1
    )
    
    if hits:
        # Payload에서 concept_id 반환
        return hits[0].payload
    return None

# =========================================================
# [Step 2] Intent Executors
# =========================================================

# Case 1: CHECK_BENEFIT (Graph 집중)
async def retrieve_benefit(concept_id: str):
    """
    GraphDB에서 해당 Concept과 연결된 Benefit(금액) 조회
    """
    print(f"🔍 [Benefit] Searching for concept: {concept_id}")
    
    query = f"""
    MATCH (c:Concept {{concept_id: '{concept_id}'}})<-[:RELATED_TO]-(b:Benefit)<-[:HAS_BENEFIT]-(r:Rider)
    RETURN {{
        rider_name: r.name,
        benefit_name: b.name,
        amount: b.amount_text,
        condition: b.condition_summary,
        limit: b.limit_count
    }}
    """
    results = await db.execute_cypher(query)
    
    # 결과 포맷팅 (List[Dict])
    # AgensGraph 결과는 JSON 문자열일 수 있으므로 파싱 필요할 수 있음 (환경에 따라 다름)
    # 여기서는 Dict로 가정
    return results

# Case 3: CHECK_EXCLUSION (Vector Filter + Graph Verify)
async def retrieve_exclusion(concept_id: str):
    """
    VectorDB에서 'EXCLUSION' 태그 + Concept 필터링 검색 후
    GraphDB에서 연결 관계(RESTRICTS) 검증
    """
    print(f"🔍 [Exclusion] Searching for concept: {concept_id}")
    
    # 1. Vector Search (Semantic Filtering)
    # 필터 조건: (Tag='EXCLUSION') AND (Concept IN [concept_id])
    filter_condition = models.Filter(
        must=[
            models.FieldCondition(key="tag", match=models.MatchValue(value="EXCLUSION")),
            models.FieldCondition(key="related_concepts", match=models.MatchAny(any=[concept_id])) # 배열 포함 여부 확인
        ]
    )
    
    # 쿼리 벡터: 문맥상 '면책', '제한' 의미를 담은 임의 벡터 or 키워드 벡터 사용
    # 여기서는 Concept 키워드 자체를 쿼리로 사용
    vec = get_embedding("면책 지급제한 보장하지 않는 경우") 
    
    hits = await db.search_vector(
        collection="insurance_knowledge",
        vector=vec,
        filter=filter_condition,
        limit=3
    )
    
    valid_results = []
    
    # 2. Graph Verification
    for hit in hits:
        clause_id = hit.payload.get('node_id')
        rider_id = hit.payload.get('rider_id')
        
        # Graph Query: 진짜 이 Rider가 이 Clause에 RESTRICTS로 연결되어 있는지 확인
        verify_query = f"""
        MATCH (r:Rider {{rider_id: '{rider_id}'}})-[:RESTRICTS]->(c:Clause {{clause_id: '{clause_id}'}})
        RETURN c.clause_id
        """
        graph_check = await db.execute_cypher(verify_query)
        
        if graph_check:
            valid_results.append({
                "text": hit.payload.get('text'),
                "score": hit.score,
                "verified": True
            })
            
    return valid_results

# Case 2: CHECK_CONDITION (Graph -> Vector Fallback)
async def retrieve_condition(concept_id: str):
    """
    Graph에서 condition_summary 조회 -> 내용 부실하면 Vector(Condition Tag) 검색
    """
    # 1. Graph 조회
    graph_data = await retrieve_benefit(concept_id) # Benefit 정보 재활용
    
    final_results = []
    for item in graph_data:
        condition_text = item.get('condition', '')
        
        # 내용이 충분한지 체크 (Rule-based: 길이가 짧거나 '별표 참조'면 Fallback)
        if len(condition_text) < 10 or "참조" in condition_text:
            # 2. Vector Fallback
            print("⚠️ [Condition] Graph content insufficient. Fallback to Vector.")
            vec = get_embedding(f"{item['benefit_name']} 지급 조건 상세")
            
            filter_cond = models.Filter(
                must=[
                    models.FieldCondition(key="type", match=models.MatchValue(value="clause")),
                    models.FieldCondition(key="tag", match=models.MatchValue(value="CONDITION")),
                    models.FieldCondition(key="related_concepts", match=models.MatchAny(any=[concept_id]))
                ]
            )
            hits = await db.search_vector("insurance_knowledge", vec, filter=filter_cond, limit=1)
            
            if hits:
                condition_text = hits[0].payload.get('text')
        
        final_results.append({
            "benefit": item['benefit_name'],
            "condition_detail": condition_text
        })
        
    return final_results

# Case 4: EXPLAIN_TERM (Graph Only / Vector Fallback)
async def retrieve_term(keyword: str):
    # 1. Graph (Ontology)
    # 2. Vector (Glossary)
    # (Turn 1 로직과 유사하므로 간략화)
    pass 




async def retrieve_comparison(concept_id: str, product_keywords: list):
    """
    [COMPARE_PRODUCT]
    여러 상품(키워드)에 대해 특정 Concept(표적항암)의 보장 내용을 비교 조회
    """
    print(f"🔍 [Comparison] Comparing {product_keywords} on {concept_id}")
    
    comparison_data = {}
    
    # 사용자가 언급한 상품 키워드별로 루프 (예: ["시그니처", "실속"])
    # 실무에서는 '상품명 Entity Linking'을 먼저 수행하지만, 여기서는 LIKE 검색으로 약식 구현
    for prod_kwd in product_keywords:
        # 쿼리: 해당 상품명(유사)을 가진 Product 안에서 -> 해당 Concept과 연결된 Benefit 찾기
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
            # 상품명을 키로 데이터 저장
            # 결과가 여러 개일 수 있으나 첫 번째(대표)만 사용하거나 리스트화
            comparison_data[prod_kwd] = results[0] 
        else:
            comparison_data[prod_kwd] = "해당 상품에서 관련 보장을 찾을 수 없습니다."
            
    return comparison_data