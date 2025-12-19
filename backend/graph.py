import os
import json
import re
import asyncio
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# [변경] async 처리가 된 DB 및 Vector 함수 임포트
from backend.db.db import execute_query, execute_sql_query 
from backend.vector_store import search_documents 
from dotenv import load_dotenv

# -------------------------------------------------------------------------
# 1. 환경 설정
# -------------------------------------------------------------------------
load_dotenv()

# [LLM 설정]
# # Async 환경에서도 ChatOpenAI 객체는 동일하게 생성하되, 호출 시 ainvoke를 사용합니다.
# llm = ChatOpenAI(
#     model="gpt-4o", 
#     temperature=0 
# )

# 로컬 vLLM/Ollama 서버를 바라보도록 설정합니다.
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3-8B"), # .env에서 가져옴
    api_key="EMPTY",      # 로컬은 키 불필요
    base_url=os.getenv("LLM_API_BASE", "http://localhost:8000/v1"), # 로컬 API 주소
    temperature=0
)



# -------------------------------------------------------------------------
# 2. 상태(State) 정의
# -------------------------------------------------------------------------
class AgentState(TypedDict):
    """     
    LangGraph 상태 스키마 (이전과 동일)
    """
    question: str
    graph_schema: str
    sql_schema: str
    mode: str                       # graph | sql | vector | general
    generated_query: str
    query_result: str
    context: List[str]              # RAG용 검색 문서
    evaluation: Dict[str, Any]      # 검증 결과
    final_answer: str
    retry_count: int
    error: Optional[str]            # 실행 에러 메시지
    trace_log: List[str]

# -------------------------------------------------------------------------
# 3. Helper 함수
# -------------------------------------------------------------------------
def update_trace(state: AgentState, step: str, detail: str) -> List[str]:
    """로그 업데이트 (CPU 연산이므로 동기 함수 유지)"""
    log = state.get("trace_log", []) or []
    return list(log) + [f"[{step}] {detail}"]


# -------------------------------------------------------------------------
# [Helper] Think 태그 제거 함수
# -------------------------------------------------------------------------
def remove_think_tag(text: str) -> str:
    """
    <think>...</think> 태그와 그 내용을 제거하여 순수 응답만 반환합니다.
    Flags=re.DOTALL: 줄바꿈이 포함된 내용도 한 번에 매칭
    """
    if not text: return ""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


# -------------------------------------------------------------------------
# 4. 프롬프트 정의 (분리된 구조)
# -------------------------------------------------------------------------
#Qwen3-8B용 쿼리(좀 더 구조화해줘야 잘 알아듣는다(추론형))
ROUTER_PROMPT = """<|im_start|>system
You are a strict classifier. Your job is to route user inputs into exactly one of two categories: 'vector' or 'general'.

### Categories
1. **vector**:
   - ALL questions about information, data, facts, or knowledge.
   - Examples: "What are the scholarship rules?", "How many students?", "Who is the professor?", "Tell me about SeoulTech."
   - Rule: If the user asks for ANY information, choose 'vector'.

2. **general**:
   - ONLY for casual greetings or self-introductions.
   - Examples: "Hi", "Hello", "Who are you?", "Good morning".

### Output Format
- Output ONLY the keyword: `vector` or `general`.
- Do NOT output any other text or punctuation.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""


# ROUTER_PROMPT = """You are a classification assistant. Classify the user question into one of these two categories:

# 1. 'vector': Use this for ANY question requiring information, knowledge, data, regulations, statistics, or relationships.
#    - Includes: University regulations, specific rules, database queries, relationship checks, counting, or raw data retrieval.
#    - If the user asks about SeoulTech rules, relationships, or statistics, ALWAYS output 'vector'.

# 2. 'general': ONLY for greetings (hi, hello), self-introductions, or questions about the AI assistant itself.

# User Question: {question}

# Rule: If it's not a simple greeting, default to 'vector'.
# Output only the keyword (vector, general)."""




# ROUTER_PROMPT = """You are a classification assistant. Classify the user question into one of these three categories:

# 1. 'graph': ONLY for questions about relationships, connections, hierarchy, or network structures between entities.
# 2. 'sql': ONLY for questions asking for aggregations, statistics, counts, raw logs, or table data.
# 3. 'general': For greetings (hi, hello), self-introduction, questions about the AI itself, or vague questions that don't fit graph/sql.

# User Question: {question}

# Rule: If the question is a simple greeting like "hi" or "hello", YOU MUST output 'general'.
# Output only the keyword (graph, sql, general)."""

CYPHER_GEN_PROMPT = """You are a Cypher expert.
Schema: {schema}
Feedback: {feedback}
Task: Convert '{question}' to Cypher.
Output ONLY code."""

SQL_GEN_PROMPT = """You are a SQL expert.
Schema: {schema}
Feedback: {feedback}
Task: Convert '{question}' to SQL.
Output ONLY code."""

CRITIC_PROMPT = """Validate {mode} query.
Schema: {schema}
Query: {query}
Return JSON: {{"valid": bool, "reason": str}}"""

# graph.py 의 SUMMARIZER_PROMPT 변수를 통째로 교체하세요.



#Qwen3-8B용 쿼리(좀 더 구조화해줘야 잘 알아듣는다(추론형))
# graph.py에 덮어씌워 주세요.

SUMMARIZER_PROMPT = """<|im_start|>system
You are a helpful and factual assistant for Seoul National University of Science and Technology (SeoulTech).
Your task is to answer the user's question using ONLY the provided [Context] and [DB Result].

### [Data Structure & Schema]
The provided [Context] consists of university regulations.
1. **Regulation Name (doc_title)**: The title of the regulation.
2. **Hierarchy**: Depending on the regulation, it may follow: Chapter (장) > Section (절) > Article (조).
   *Note: 'Chapter' or 'Section' may be omitted in smaller regulations, but 'Article (조)' is the standard unit.*
3. **Appendix (별표)**: Tables, forms, or detailed lists are contained in '별표'.

### Instructions
1. **Search First:** Look for the answer in the [Context] or [DB Result] below.
2. **Strict Grounding:** If the answer is NOT in the provided text, respond exactly with: "제공된 정보(규정 및 데이터) 내에서 관련 내용을 찾을 수 없습니다."
3. **No Fabrication:** Do NOT make up URLs, phone numbers, or facts. Do NOT use outside knowledge.
4. **Citation Style (Important):**
   - Answer professionally by citing the source.
   - **Format:** "OO규정 제N조(제목)에 따르면..." or "별표 N에 의거하여..."
   - If Chapter/Section is missing, just cite the Regulation Name and Article.
   - Example: "학칙 제5조(자격)에 따르면..."
5. **Language:** Answer in Korean.

### [Context]
{c}

### [DB Result]
{r}
<|im_end|>
<|im_start|>user
{q}
<|im_end|>
<|im_start|>assistant
"""


# SUMMARIZER_PROMPT = """You are a strictly factual assistant for Seoul National University of Science and Technology (SeoulTech).
# Your task is to answer the user's question based **ONLY** on the provided 'Context' and 'DB Result'.

# ### Context (Retrieved Documents):
# {c}

# ### DB Result (Structured Data):
# {r}

# ### Question:
# {q}

# ### *** STRICT RESPONSE RULES ***
# 1. **Grounding:** You must answer using *only* the information explicitly present in the 'Context' or 'DB Result' above.
# 2. **No Hallucination:** - NEVER invent URLs, phone numbers, or email addresses. If it's not in the text, do not generate it. (e.g., do not give a Sungkyunkwan URL for SeoulTech).
#    - If the provided Context/DB Result does not contain enough information to answer the question, you MUST say: "제공된 정보(규정 및 데이터) 내에서 관련 내용을 찾을 수 없습니다."
# 3. **Prioritize Context:** Use the 'Context' primarily for regulations/rules queries. Use 'DB Result' for statistics/counts queries.
# 4. **Tone:** Professional, objective, and concise. Answer in **Korean**.
# """

## -------------------------------------------------------------------------
# 5. [Async] 노드 정의 (Think 태그 제거(llm생성 시) 적용)
# -------------------------------------------------------------------------

async def node_router(state: AgentState):
    """[Router] <think> 제거 후 키워드만 추출"""
    chain = ChatPromptTemplate.from_template(ROUTER_PROMPT) | llm
    try:
        res = await chain.ainvoke({"question": state["question"]})
        # [적용]
        clean_content = remove_think_tag(res.content)
        decision = clean_content.strip().lower()
    except:
        decision = "general"
    
    if "graph" in decision: mode = "graph"
    elif "sql" in decision: mode = "sql"
    elif "vector" in decision: mode = "vector"
    else: mode = "general"
    
    return {"mode": mode, "trace_log": update_trace(state, "Router", f"Mode: {mode}")}

async def node_retriever(state: AgentState):
    """[Retriever] LLM 미사용 -> Think 제거 불필요"""
    query = state["question"]
    docs = await search_documents(query, k=3)
    
    if docs and hasattr(docs[0], 'page_content'):
        doc_preview = ", ".join([d.page_content[:20] + "..." for d in docs])
    else:
        doc_preview = str(docs)[:50]
    
    return {
            "context": docs,
            "query_result": "N/A (Vector Mode)",
            "trace_log": update_trace(state, "Retriever", f"Found {len(docs)} docs: {doc_preview}")
        }

async def node_gen_graph(state: AgentState):
    """[Gen Graph] <think> 제거 후 Cypher 코드만 추출"""
    schema = state["graph_schema"]
    feedback = state.get("evaluation", {}).get("reason", "")
    
    chain = ChatPromptTemplate.from_template(CYPHER_GEN_PROMPT) | llm
    response = await chain.ainvoke({
        "schema": schema, "feedback": feedback, "question": state["question"]
    })
    
    # [적용]
    clean_content = remove_think_tag(response.content)
    query = clean_content.strip().replace("```cypher", "").replace("```", "")
    
    return {
        "generated_query": query,
        "retry_count": state.get("retry_count", 0) + 1,
        "trace_log": update_trace(state, "GenGraph", "Query generated")
    }

async def node_gen_sql(state: AgentState):
    """[Gen SQL] <think> 제거 후 SQL 코드만 추출"""
    schema = state["sql_schema"]
    feedback = state.get("evaluation", {}).get("reason", "")
    
    chain = ChatPromptTemplate.from_template(SQL_GEN_PROMPT) | llm
    response = await chain.ainvoke({
        "schema": schema, "feedback": feedback, "question": state["question"]
    })
    
    # [적용]
    clean_content = remove_think_tag(response.content)
    query = clean_content.strip().replace("```sql", "").replace("```", "")
    
    return {
        "generated_query": query,
        "retry_count": state.get("retry_count", 0) + 1,
        "trace_log": update_trace(state, "GenSQL", "Query generated")
    }

async def node_critic(state: AgentState):
    """[Critic] <think> 제거 후 JSON 파싱"""
    mode = state["mode"]
    schema = state["graph_schema"] if mode == "graph" else state["sql_schema"]
    
    chain = ChatPromptTemplate.from_template(CRITIC_PROMPT) | llm
    try:
        res = await chain.ainvoke({
            "mode": "Cypher" if mode == "graph" else "SQL", 
            "schema": schema, "query": state["generated_query"]
        })
        # [적용]
        clean_content = remove_think_tag(res.content)
        
        # 마크다운 코드블록 제거 (혹시 남아있을 경우)
        if clean_content.startswith("```json"): 
            clean_content = clean_content[7:]
        if clean_content.endswith("```"):
            clean_content = clean_content[:-3]
            
        eval_result = json.loads(clean_content.strip())
    except:
        eval_result = {"valid": True, "reason": "Parsing failed (Default Valid)"}
    
    return {
        "evaluation": eval_result,
        "trace_log": update_trace(state, "Critic", f"Valid: {eval_result.get('valid')}")
    }

async def node_executor(state: AgentState):
    """[Executor] LLM 미사용 -> Think 제거 불필요"""
    try:
        if state["mode"] == "graph":
            res = await execute_query(state["generated_query"])
        else:
            res = await execute_sql_query(state["generated_query"])
        
        return {
            "query_result": str(res),
            "error": None,
            "trace_log": update_trace(state, "Executor", f"Success, Rows: {len(res)}")
        }
    except Exception as e:
        return {
            "error": str(e),
            "trace_log": update_trace(state, "Executor", f"DB Error: {e}")
        }

async def node_summarizer(state: AgentState):
    """[Summarizer] <think> 제거 후 최종 답변 제공"""
    chain = ChatPromptTemplate.from_template(SUMMARIZER_PROMPT) | llm
    
    db_res = state.get("query_result", "N/A")
    docs = state.get("context", [])
    doc_text = "\n".join(docs) if docs else "No docs"
    
    ans = await chain.ainvoke({
        "q": state["question"], "r": db_res, "c": doc_text
    })
    
    # [적용]
    final_clean = remove_think_tag(ans.content)
    
    return {"final_answer": final_clean}

async def node_general(state: AgentState):
    """[General] <think> 제거 후 잡담 응답"""
    ans = await llm.ainvoke(state["question"])
    # [적용]
    final_clean = remove_think_tag(ans.content)
    
    return {"final_answer": final_clean}

async def node_fail(state: AgentState):
    """[Fail] LLM 미사용 -> Think 제거 불필요"""
    error_msg = state.get("error")
    retry_cnt = state.get("retry_count", 0)
    
    if error_msg:
        final_msg = f"죄송합니다. 데이터베이스 오류가 발생했습니다.\n(Error: {error_msg})"
        detail = "Fail: DB Execution Error"
    elif retry_cnt >= 3:
        final_msg = "죄송합니다. 올바른 쿼리를 생성하지 못했습니다. (재시도 초과)"
        detail = "Fail: Max Retries"
    else:
        final_msg = "알 수 없는 오류가 발생했습니다."
        detail = "Fail: Unknown"

    return {
        "final_answer": final_msg,
        "trace_log": update_trace(state, "FailNode", detail)
    }

# -------------------------------------------------------------------------
# 6. 그래프 조립 (Wiring)
# -------------------------------------------------------------------------
# *중요*: 노드 함수들이 async여도 workflow 정의 방식은 동일합니다.
# LangGraph 컴파일러가 실행 시 자동으로 비동기를 감지합니다.

workflow = StateGraph(AgentState)

# 노드 등록
workflow.add_node("router", node_router)
workflow.add_node("retriever", node_retriever)
workflow.add_node("gen_graph", node_gen_graph)
workflow.add_node("gen_sql", node_gen_sql)
workflow.add_node("critic", node_critic)
workflow.add_node("executor", node_executor)
workflow.add_node("summarizer", node_summarizer)
workflow.add_node("general", node_general)
workflow.add_node("fail", node_fail)

# 엣지 연결 (이전과 논리 동일)
workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    lambda x: x["mode"] if x["mode"] != "vector" else "retriever",
    {
        "retriever": "retriever",
        "graph": "gen_graph", # mode 이름과 노드 매핑
        "sql": "gen_sql",
        "general": "general"
    }
)

# Vector Path
workflow.add_edge("retriever", "summarizer")

# DB Generation Path
workflow.add_edge("gen_graph", "critic")
workflow.add_edge("gen_sql", "critic")

# Critic Logic
def critic_logic(state):
    if state["evaluation"].get("valid"): return "executor"
    if state["retry_count"] >= 3: return "fail"
    return "gen_graph" if state["mode"] == "graph" else "gen_sql"

workflow.add_conditional_edges("critic", critic_logic)

# Executor Logic
def exec_logic(state):
    return "fail" if state.get("error") else "summarizer"

workflow.add_conditional_edges("executor", exec_logic)

# Ends
workflow.add_edge("summarizer", END)
workflow.add_edge("general", END)
workflow.add_edge("fail", END)

app_graph = workflow.compile()