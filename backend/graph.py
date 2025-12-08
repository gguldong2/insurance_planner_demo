import os
import json
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from backend.db import execute_query, execute_sql_query
from dotenv import load_dotenv

# 환경 변수 로드 (API Key, DB 설정 등)
load_dotenv()

# --- LLM Client 설정 ---
# vLLM 또는 로컬 LLM 서버에 연결 (OpenAI 호환 API 사용)
llm = ChatOpenAI(
    base_url=os.getenv("VLLM_API_BASE"), # vLLM 서버 주소
    api_key="EMPTY", # vLLM은 일반적으로 API Key를 사용하지 않음
    model=os.getenv("VLLM_MODEL_NAME"), # 사용할 모델 이름 지정
    temperature=0.1 # 창의성 조절 (낮을수록 일관된 코드 생성에 유리)
)

# --- State 정의 (AgentState) ---
# LangGraph에서 모든 노드가 공유하고 읽고 쓸 수 있는 상태
class AgentState(TypedDict):
    question: str                   # 1. 사용자 초기 질문
    graph_schema: str               # 2. AgensGraph 스키마 정보 (Cypher 생성용)
    sql_schema: str                 # 3. PostgreSQL 스키마 정보 (SQL 생성용)
    mode: str                       # 4. 라우터가 결정한 실행 모드: "graph" | "sql" | "general"
    generated_query: str            # 5. LLM이 생성한 Cypher 또는 SQL 쿼리
    query_result: str               # 6. DB 실행 결과 (문자열)
    evaluation: Dict[str, Any]      # 7. Critic 노드의 쿼리 유효성 검사 결과
    final_answer: str               # 8. 최종 사용자에게 제공할 답변
    retry_count: int                # 9. 쿼리 재생성 시도 횟수
    error: Optional[str]            # 10. 실행기(Executor)에서 발생한 에러 메시지
    trace_log: List[str]            # 11. 워크플로우 추적 로그 (프론트엔드 출력용)

# --- Prompts 정의 ---
ROUTER_PROMPT = """Classify the question:
1. 'graph': Relationships, hierarchy, connections.
2. 'sql': Statistics, counts, logs, raw data.
3. 'general': Greetings, others.
Output only the keyword."""

GEN_PROMPT = """Convert question to {mode} query.
Schema: {schema}
Feedback: {feedback}
Output code only."""

CRITIC_PROMPT = """Validate {mode} query.
Schema: {schema}
Query: {query}
Return JSON: {{"valid": bool, "reason": str}}"""

# --- Helper 함수 ---
def update_trace(state: AgentState, step: str, detail: str):
    """상태에 실행 로그를 추가하고 업데이트된 로그 리스트를 반환"""
    log = state.get("trace_log", []) or []
    return list(log) + [f"[{step}] {detail}"]

# --- Nodes 정의 (워크플로우 단계) ---
# 

def node_router(state: AgentState):
    """사용자 질문을 분석하여 실행 모드(graph/sql/general)를 결정하는 노드"""
    chain = ChatPromptTemplate.from_template(ROUTER_PROMPT) | llm
    try:
        decision = chain.invoke({"question": state["question"]}).content.strip().lower()
    except:
        # LLM 호출 실패 시 기본값
        decision = "general"
    
    if "graph" in decision: mode = "graph"
    elif "sql" in decision: mode = "sql"
    else: mode = "general"
    
    # mode와 로그를 상태에 업데이트
    return {"mode": mode, "trace_log": update_trace(state, "Router", f"Mode: {mode}")}

def node_generator(state: AgentState):
    """결정된 모드에 따라 Cypher 또는 SQL 쿼리를 생성하는 노드"""
    mode = state["mode"]
    # 모드에 따라 사용할 스키마를 결정
    schema = state["graph_schema"] if mode == "graph" else state["sql_schema"]
    # 이전 Critic 노드에서 받은 피드백 (재시도 시 사용)
    feedback = state.get("evaluation", {}).get("reason", "")
    
    chain = ChatPromptTemplate.from_template(GEN_PROMPT) | llm
    response = chain.invoke({
        "mode": "Cypher" if mode == "graph" else "SQL",
        "schema": schema,
        "feedback": feedback,
        "question": state["question"]
    })
    
    # 응답에서 코드 블록(```cypher, ```sql)을 제거하고 순수 쿼리 추출
    query = response.content.strip().replace("```cypher", "").replace("```sql", "").replace("```", "")
    
    return {
        "generated_query": query,
        "retry_count": state.get("retry_count", 0) + 1, # 재시도 횟수 증가
        "trace_log": update_trace(state, "Generator", f"Query: {query[:80]}...")
    }

def node_critic(state: AgentState):
    """생성된 쿼리가 스키마에 맞고 논리적으로 유효한지 검증하는 노드"""
    mode = state["mode"]
    schema = state["graph_schema"] if mode == "graph" else state["sql_schema"]
    
    chain = ChatPromptTemplate.from_template(CRITIC_PROMPT) | llm

    # LLM 호출 및 JSON 파싱
    try:
        res = chain.invoke({"mode": mode, "schema": schema, "query": state["generated_query"]})
        content = res.content.strip()
        # 코드 블록이 있다면 제거
        if content.startswith("```json"): content = content[7:-3]
        eval_result = json.loads(content)
    except:
        # JSON 파싱 실패나 LLM 호출 실패 시, 안전을 위해 일단 유효하다고 간주하고 다음 단계로 진행
        eval_result = {"valid": True, "reason": "Parsing failed or LLM failed"}
    
    # --- [수정] 로그 메시지는 eval_result가 정의된 후 생성되어야 합니다. ---
    valid = eval_result.get('valid')
    reason = eval_result.get('reason', '')
    detail_log = f"Valid: {valid}, Reason: {reason}"
    # ------------------------------------------------------------------
    
    return {
        "evaluation": eval_result,
        "trace_log": update_trace(state, "Critic", detail_log)
    }

def node_executor(state: AgentState):
    """생성된 쿼리를 DB에 실행하고 결과를 가져오는 노드"""
    try:
        # 모드에 따라 적절한 DB 실행 함수 호출
        if state["mode"] == "graph":
            res = execute_query(state["generated_query"])
        else:
            res = execute_sql_query(state["generated_query"])
        
        # 결과 로그 (결과 행 수 기반)
        detail_log = f"Rows: {len(res)}"

        return {
            "query_result": str(res), # 결과를 문자열로 변환하여 상태에 저장
            "error": None, 
            "trace_log": update_trace(state, "Executor", detail_log)
        }
    except Exception as e:
        # DB 연결 또는 쿼리 실행 실패 시 에러 상태 저장
        return {
            "error": str(e), 
            "trace_log": update_trace(state, "Executor", f"Error: {e}")
        }

def node_summarizer(state: AgentState):
    """DB 실행 결과를 바탕으로 최종 답변을 생성하는 노드"""
    chain = ChatPromptTemplate.from_template("Answer based on result.\nQ: {q}\nResult: {r}") | llm
    ans = chain.invoke({"q": state["question"], "r": state.get("query_result")})
    return {"final_answer": ans.content}

def node_general(state: AgentState):
    """DB와 관련 없는 일반적인 질문(인사, 날씨 등)에 응답하는 노드"""
    ans = llm.invoke(state["question"])
    return {"final_answer": ans.content}

def node_fail(state: AgentState):
    """쿼리 재시도 횟수 초과 또는 치명적인 DB 에러 발생 시 최종 에러 메시지를 반환하는 노드"""
    return {"final_answer": "Error: 데이터베이스 조회 실패 또는 쿼리 생성 재시도 횟수 초과"}

# --- Graph Wiring (그래프 연결) ---
workflow = StateGraph(AgentState)
# 1. 노드 추가
workflow.add_node("router", node_router)
workflow.add_node("generator", node_generator)
workflow.add_node("critic", node_critic)
workflow.add_node("executor", node_executor)
workflow.add_node("summarizer", node_summarizer)
workflow.add_node("general", node_general)
workflow.add_node("fail", node_fail)

# 2. 시작 지점 설정
workflow.add_edge(START, "router")

# 3. 라우터 조건부 분기: DB 필요 여부에 따라 분기
workflow.add_conditional_edges(
    "router",
    # 람다 함수를 통해 'mode' 값에 따라 다음 노드를 결정
    lambda s: "general" if s["mode"] == "general" else "generator",
    {"general": "general", "generator": "generator"}
)

# 4. 쿼리 생성 후 검증기로 이동
workflow.add_edge("generator", "critic")

# 5. 검증기 조건부 분기: 쿼리가 유효한지, 재시도를 했는지 판단
def critic_logic(s):
    if s["evaluation"].get("valid"): return "executor"    # 유효하면 실행
    if s["retry_count"] >= 3: return "fail"               # 3회 이상 실패 시 종료
    return "generator"                                    # 다시 생성기로 돌아가 피드백 반영

workflow.add_conditional_edges("critic", critic_logic)

# 6. 실행기 조건부 분기: DB 실행 중 에러가 났는지 판단
def exec_logic(s):
    return "fail" if s.get("error") else "summarizer" # 에러가 있다면 fail, 아니면 요약으로

workflow.add_conditional_edges("executor", exec_logic)

# 7. 최종 종료 지점 연결
workflow.add_edge("summarizer", END)
workflow.add_edge("general", END)
workflow.add_edge("fail", END)

# 8. 그래프 컴파일 및 실행 준비
app_graph = workflow.compile()