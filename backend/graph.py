# backend/graph.py
import os
import json
import re
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# [Turn 3] Logic Layer에서 구현한 검색 함수들 임포트
# (retrieve_comparison 추가됨)
from backend.logic.retrievers import (
    link_concept,
    retrieve_benefit,
    retrieve_exclusion,
    retrieve_condition,
    retrieve_term,
    retrieve_comparison  # ★ 추가: 상품 비교용 함수
)

# -------------------------------------------------------------------------
# 1. 환경 설정 & 모델 초기화
# -------------------------------------------------------------------------
load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
    api_key="EMPTY",
    base_url=os.getenv("LLM_API_BASE", "http://localhost:8000/v1"),
    temperature=0
)

# -------------------------------------------------------------------------
# 2. 상태(State) 정의 - Neuro-Symbolic 전용
# -------------------------------------------------------------------------
class AgentState(TypedDict):
    """
    LangGraph의 전체 상태 스키마.
    각 노드는 이 스키마의 일부(Subset)를 반환하여 상태를 업데이트합니다.
    """
    question: str               # 사용자 입력
    intent: str                 # Router가 분류한 의도
    keywords: List[str]         # Router가 추출한 핵심어
    concept_id: Optional[str]   # Grounding된 Concept ID
    
    context: List[str]          # Retriever가 가져온 증거 텍스트들
    final_answer: str           # LLM이 생성한 최종 답변
    
    trace_log: List[str]        # 디버깅용 로그
    node_models: Dict[str, str] # 사용된 모델 기록

# -------------------------------------------------------------------------
# 3. Helper 함수
# -------------------------------------------------------------------------
def update_trace(state: AgentState, node: str, msg: str) -> List[str]:
    """로그를 누적 업데이트하는 헬퍼 함수"""
    trace = list(state.get("trace_log", []) or [])
    trace.append(f"[{node}] {msg}")
    return trace[-30:] # 최근 30개 유지

def remove_think_tag(text: str) -> str:
    """LLM의 <think> 태그 제거 (Qwen 계열 등)"""
    if not text: return ""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

# -------------------------------------------------------------------------
# 4. 프롬프트 정의
# -------------------------------------------------------------------------

# [Router Prompt]
ROUTER_PROMPT = """# Role
You are an intelligent Insurance Chatbot Router.
Your goal is to analyze the user's input (Korean) and extract structured information.

# Task
Analyze the User Input and generate a JSON output.

## 1. Classify Intents (Select ONE)
- **CHECK_BENEFIT**: Payout amount, eligibility. (e.g., "얼마 나와?", "보장 돼?")
- **CHECK_CONDITION**: Requirements, timing. (e.g., "언제부터?", "수술 꼭 해야 해?")
- **CHECK_EXCLUSION**: NOT covered cases. (e.g., "면책 사유", "지급 제한")
- **EXPLAIN_TERM**: Definitions. (e.g., "표적항암이 뭐야?")
- **COMPARE_PRODUCTSS**: Comparison. (e.g., "A랑 B 차이점")
- **CHIT_CHAT**: Greetings.

## 2. Extract Entities
Extract key insurance terms.

## 3. Output Format
Return **JSON Only**. Keys: "intent", "keywords".
Example:
Input: "표적항암 치료비 얼마야?"
Output: {"intent": "CHECK_BENEFIT", "keywords": ["표적항암"]}

Input: "{question}"
"""

# [Generator Prompt]
GENERATOR_PROMPT = """<|im_start|>system
You are a professional insurance AI assistant.
Answer the user's question based ONLY on the provided [Context].

### Instructions
1. **Fact-based**: Use only the information in [Context]. Do not invent facts.
2. **Persona**:
   - If intent is CHECK_BENEFIT -> Be precise like an accountant. (Focus on Amount)
   - If intent is CHECK_EXCLUSION -> Be strict like a lawyer. (Warn about restrictions)
   - If intent is COMPARE_PRODUCTS -> Be analytical. (Use tables or lists)
   - Otherwise -> Be helpful and clear.
3. **Format**:
   - 결론: (Simple answer)
   - 근거: (Details from context)

### Context
{context}

<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""

# -------------------------------------------------------------------------
# 5. 노드(Node) 정의
# -------------------------------------------------------------------------

async def node_router(state: AgentState) -> Dict[str, Any]:
    """
    [Router] 사용자 의도 및 키워드 추출
    Returns: intent, keywords, trace_log (Partial Update)
    """
    chain = ChatPromptTemplate.from_template(ROUTER_PROMPT) | llm
    
    intent = "CHIT_CHAT"
    keywords = []
    
    try:
        res = await chain.ainvoke({"question": state["question"]})
        clean_json = remove_think_tag(res.content).replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_json)
        
        intent = parsed.get("intent", "CHIT_CHAT")
        if isinstance(intent, list): intent = intent[0]
            
        keywords = parsed.get("keywords", [])
        # [{"term": "암", ...}] 형태라면 문자열 리스트로 변환
        if keywords and isinstance(keywords[0], dict):
            keywords = [k.get("term", "") for k in keywords]
            
    except Exception as e:
        print(f"Router Error: {e}")
    
    return {
        "intent": intent, 
        "keywords": keywords,
        "trace_log": update_trace(state, "Router", f"Intent: {intent}, Keywords: {keywords}")
    }

async def node_retriever(state: AgentState) -> Dict[str, Any]:
    """
    [Retriever] 의도에 따른 Logic 함수 호출 (Dispatcher)
    Returns: concept_id, context, trace_log (Partial Update)
    """
    intent = state["intent"]
    keywords = state["keywords"]
    
    # 1. Entity Linking (Grounding)
    concept_id = None
    product_candidates = [] # 비교 질문용 상품 키워드

    # 키워드 분류: Concept인지 단순 상품명인지 식별
    if keywords:
        for k in keywords:
            # 하나라도 Concept으로 링크되면 그것을 주제로 삼음
            linked = await link_concept(k)
            if linked and not concept_id:
                concept_id = linked.get("concept_id")
            else:
                product_candidates.append(k) # Concept이 아니면 상품명으로 간주
    
    context_data = []
    log_msg = ""
    
    try:
        if intent == "CHIT_CHAT":
            log_msg = "Skipped retrieval"
        elif intent == "COMPARE_PRODUCTS":
            # [추가된 로직] 상품 비교
            if not concept_id:
                # 비교 기준(표적항암 등)이 없으면 에러 메시지
                context_data = ["비교할 기준(예: 표적항암, 수술비)을 찾지 못했습니다."]
                log_msg = "Comparison failed (No concept)"
            else:
                # Logic Layer 호출
                comp_result = await retrieve_comparison(concept_id, product_candidates)
                for prod, info in comp_result.items():
                    if isinstance(info, str):
                        context_data.append(f"상품 '{prod}': {info}")
                    else:
                        context_data.append(
                            f"■ 상품: {info['product_name']}\n"
                            f"  - 특약: {info['rider_name']} ({info['renewal_type']})\n"
                            f"  - 보장: {info['benefit_name']} ({info['amount']})"
                        )
                log_msg = f"Compared {len(product_candidates)} products"

        elif not concept_id:
             # 다른 인텐트인데 Concept을 못 찾음 -> 단순 용어 검색으로 Fallback
             search_term = keywords[0] if keywords else ""
             res = await retrieve_term(search_term)
             if res: 
                 context_data = [f"용어 정의: {res['definition']}"]
             else:
                 context_data = ["관련된 보험 용어나 개념을 찾을 수 없습니다."]
             log_msg = "Fallback to Term Search"
             
        elif intent == "CHECK_BENEFIT":
            res = await retrieve_benefit(concept_id)
            context_data = [f"보장명: {r['benefit_name']} | 금액: {r['amount']} | 한도: {r['limit']}" for r in res]
            log_msg = f"Retrieved {len(res)} benefits"
            
        elif intent == "CHECK_EXCLUSION":
            res = await retrieve_exclusion(concept_id)
            context_data = [f"면책조항: {r['text']} (검증됨)" for r in res]
            log_msg = f"Retrieved {len(res)} exclusions"
            
        elif intent == "CHECK_CONDITION":
            res = await retrieve_condition(concept_id)
            context_data = [f"조건: {r['condition_detail']}" for r in res]
            log_msg = f"Retrieved {len(res)} conditions"
            
        elif intent == "EXPLAIN_TERM":
            res = await retrieve_term(keywords[0] if keywords else "")
            if res: 
                context_data = [f"정의: {res['definition']} (출처: {res['category']})"]
            log_msg = "Term retrieved"
            
    except Exception as e:
        log_msg = f"Error: {str(e)}"
        context_data = [f"시스템 오류 발생: {str(e)}"]

    return {
        "concept_id": concept_id,
        "context": context_data,
        "trace_log": update_trace(state, "Retriever", log_msg)
    }

async def node_generator(state: AgentState) -> Dict[str, Any]:
    """
    [Generator] 최종 답변 생성
    Returns: final_answer, trace_log (Partial Update)
    """
    if state["intent"] == "CHIT_CHAT":
        res = await llm.ainvoke(state["question"])
        return {
            "final_answer": remove_think_tag(res.content),
            "trace_log": update_trace(state, "Generator", "Chit-chat response")
        }

    context_text = "\n\n".join(state["context"])
    if not context_text:
        context_text = "관련된 정보를 찾을 수 없습니다."
        
    chain = ChatPromptTemplate.from_template(GENERATOR_PROMPT) | llm
    
    res = await chain.ainvoke({
        "question": state["question"],
        "context": context_text
    })
    
    return {
        "final_answer": remove_think_tag(res.content),
        "trace_log": update_trace(state, "Generator", "Response generated")
    }

# -------------------------------------------------------------------------
# 6. 그래프 조립
# -------------------------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("router", node_router)
workflow.add_node("retriever", node_retriever)
workflow.add_node("generator", node_generator)

workflow.add_edge(START, "router")
workflow.add_edge("router", "retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END)

app_graph = workflow.compile()