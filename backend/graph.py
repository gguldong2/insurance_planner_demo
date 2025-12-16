import os
import json
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
    model=os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3-8B-Instruc"), # .env에서 가져옴
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
# 4. 프롬프트 정의 (분리된 구조)
# -------------------------------------------------------------------------
ROUTER_PROMPT = """You are a classification assistant. Classify the user question into one of these three categories:

1. 'graph': ONLY for questions about relationships, connections, hierarchy, or network structures between entities.
2. 'sql': ONLY for questions asking for aggregations, statistics, counts, raw logs, or table data.
3. 'general': For greetings (hi, hello), self-introduction, questions about the AI itself, or vague questions that don't fit graph/sql.

User Question: {question}

Rule: If the question is a simple greeting like "hi" or "hello", YOU MUST output 'general'.
Output only the keyword (graph, sql, general)."""

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

SUMMARIZER_PROMPT = """Answer based on Context and DB Result.
Context: {c}
DB Result: {r}
Question: {q}
"""

# -------------------------------------------------------------------------
# 5. [Async] 노드 정의
# -------------------------------------------------------------------------
# 모든 노드 함수 앞에 'async' 키워드가 붙습니다.

async def node_router(state: AgentState):
    """[Router] LLM을 비동기 호출하여 모드를 결정합니다."""
    chain = ChatPromptTemplate.from_template(ROUTER_PROMPT) | llm
    try:
        # [변경] invoke -> ainvoke (await 필수)
        res = await chain.ainvoke({"question": state["question"]})
        decision = res.content.strip().lower()
    except:
        decision = "general"
    
    if "graph" in decision: mode = "graph"
    elif "sql" in decision: mode = "sql"
    elif "vector" in decision: mode = "vector"
    else: mode = "general"
    
    return {"mode": mode, "trace_log": update_trace(state, "Router", f"Mode: {mode}")}

async def node_retriever(state: AgentState):
    """[Retriever] 비동기 Vector DB 검색을 수행합니다."""
    query = state["question"]

    # 1. vector_store.py에서 가져온 함수 실행
    # search_documents가 async 함수이므로 await 사용
    docs = await search_documents(query, k=3)
    

    # 2. 로그에 보여줄 내용 정리 (문서 내용이 있으면 앞부분만 잘라서 보여줌)
    # docs가 문자열 리스트인지 Document 객체 리스트인지에 따라 처리
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
    """[Gen Graph] Cypher 쿼리 비동기 생성"""
    schema = state["graph_schema"]
    feedback = state.get("evaluation", {}).get("reason", "")
    
    chain = ChatPromptTemplate.from_template(CYPHER_GEN_PROMPT) | llm
    # [변경] ainvoke
    response = await chain.ainvoke({
        "schema": schema, "feedback": feedback, "question": state["question"]
    })
    
    query = response.content.strip().replace("```cypher", "").replace("```", "")
    return {
        "generated_query": query,
        "retry_count": state.get("retry_count", 0) + 1,
        "trace_log": update_trace(state, "GenGraph", "Query generated")
    }

async def node_gen_sql(state: AgentState):
    """[Gen SQL] SQL 쿼리 비동기 생성"""
    schema = state["sql_schema"]
    feedback = state.get("evaluation", {}).get("reason", "")
    
    chain = ChatPromptTemplate.from_template(SQL_GEN_PROMPT) | llm
    # [변경] ainvoke
    response = await chain.ainvoke({
        "schema": schema, "feedback": feedback, "question": state["question"]
    })
    
    query = response.content.strip().replace("```sql", "").replace("```", "")
    return {
        "generated_query": query,
        "retry_count": state.get("retry_count", 0) + 1,
        "trace_log": update_trace(state, "GenSQL", "Query generated")
    }

async def node_critic(state: AgentState):
    """[Critic] 비동기 검증"""
    mode = state["mode"]
    schema = state["graph_schema"] if mode == "graph" else state["sql_schema"]
    
    chain = ChatPromptTemplate.from_template(CRITIC_PROMPT) | llm
    try:
        # [변경] ainvoke
        res = await chain.ainvoke({
            "mode": "Cypher" if mode == "graph" else "SQL", 
            "schema": schema, "query": state["generated_query"]
        })
        content = res.content.strip()
        if content.startswith("```json"): content = content[7:-3]
        eval_result = json.loads(content)
    except:
        eval_result = {"valid": True, "reason": "Parsing failed"}
    
    return {
        "evaluation": eval_result,
        "trace_log": update_trace(state, "Critic", f"Valid: {eval_result.get('valid')}")
    }

async def node_executor(state: AgentState):
    """[Executor] 비동기 DB 실행 (db.py에서 Thread로 실행됨)"""
    try:
        if state["mode"] == "graph":
            # [변경] await execute_query
            res = await execute_query(state["generated_query"])
        else:
            # [변경] await execute_sql_query
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
    """[Summarizer] 최종 답변 생성"""
    chain = ChatPromptTemplate.from_template(SUMMARIZER_PROMPT) | llm
    
    db_res = state.get("query_result", "N/A")
    docs = state.get("context", [])
    doc_text = "\n".join(docs) if docs else "No docs"
    
    # [변경] ainvoke
    ans = await chain.ainvoke({
        "q": state["question"], "r": db_res, "c": doc_text
    })
    
    return {"final_answer": ans.content}

async def node_general(state: AgentState):
    """[General] 비동기 잡담"""
    # [변경] ainvoke
    ans = await llm.ainvoke(state["question"])
    return {"final_answer": ans.content}

async def node_fail(state: AgentState):
    """
    [Fail] 에러 구분 로직 (요청하신 기능 반영)
    async 함수여야 LangGraph가 await로 호출할 수 있습니다.
    """
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