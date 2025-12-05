import os
import json
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from backend.db import execute_query, execute_sql_query
from dotenv import load_dotenv

load_dotenv()

# --- LLM Client ---
llm = ChatOpenAI(
    base_url=os.getenv("VLLM_API_BASE"),
    api_key="EMPTY",
    model=os.getenv("VLLM_MODEL_NAME"), 
    temperature=0.1
)

# --- State Definition ---
class AgentState(TypedDict):
    question: str
    graph_schema: str
    sql_schema: str
    mode: str               # "graph" | "sql" | "general"
    generated_query: str    
    query_result: str
    evaluation: Dict[str, Any]
    final_answer: str
    retry_count: int
    error: Optional[str]
    trace_log: List[str]

# --- Prompts ---
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

# --- Helper ---
def update_trace(state: AgentState, step: str, detail: str):
    log = state.get("trace_log", []) or []
    return list(log) + [f"[{step}] {detail}"]

# --- Nodes ---
def node_router(state: AgentState):
    chain = ChatPromptTemplate.from_template(ROUTER_PROMPT) | llm
    try:
        decision = chain.invoke({"question": state["question"]}).content.strip().lower()
    except:
        decision = "general"
    
    if "graph" in decision: mode = "graph"
    elif "sql" in decision: mode = "sql"
    else: mode = "general"
    
    return {"mode": mode, "trace_log": update_trace(state, "Router", f"Mode: {mode}")}

def node_generator(state: AgentState):
    mode = state["mode"]
    schema = state["graph_schema"] if mode == "graph" else state["sql_schema"]
    feedback = state.get("evaluation", {}).get("reason", "")
    
    chain = ChatPromptTemplate.from_template(GEN_PROMPT) | llm
    response = chain.invoke({
        "mode": "Cypher" if mode == "graph" else "SQL",
        "schema": schema,
        "feedback": feedback,
        "question": state["question"]
    })
    
    query = response.content.strip().replace("```cypher", "").replace("```sql", "").replace("```", "")
    return {
        "generated_query": query,
        "retry_count": state.get("retry_count", 0) + 1,
        "trace_log": update_trace(state, "Generator", f"Query: {query}")
    }

def node_critic(state: AgentState):
    mode = state["mode"]
    schema = state["graph_schema"] if mode == "graph" else state["sql_schema"]
    
    chain = ChatPromptTemplate.from_template(CRITIC_PROMPT) | llm

    valid = eval_result.get('valid')
    reason = eval_result.get('reason', '')
    detail_log = f"Valid: {valid}, Reason: {reason}"

    try:
        res = chain.invoke({"mode": mode, "schema": schema, "query": state["generated_query"]})
        content = res.content.strip()
        if content.startswith("```json"): content = content[7:-3]
        eval_result = json.loads(content)
    except:
        eval_result = {"valid": True, "reason": "Parsing failed"}
        
    return {
        "evaluation": eval_result,
        "trace_log": update_trace(state, "Critic", detail_log)
    }

def node_executor(state: AgentState):
    try:
        if state["mode"] == "graph":
            res = execute_query(state["generated_query"])
        else:
            res = execute_sql_query(state["generated_query"])
        detail_log = f"Result: {str(res)}"


        return {
            "query_result": str(res), 
            "error": None, 
            "trace_log": update_trace(state, "Executor", detail_log)
        }
    except Exception as e:
        return {
            "error": str(e), 
            "trace_log": update_trace(state, "Executor", f"Error: {e}")
        }

def node_summarizer(state: AgentState):
    chain = ChatPromptTemplate.from_template("Answer based on result.\nQ: {q}\nResult: {r}") | llm
    ans = chain.invoke({"q": state["question"], "r": state.get("query_result")})
    return {"final_answer": ans.content}

def node_general(state: AgentState):
    ans = llm.invoke(state["question"])
    return {"final_answer": ans.content}

def node_fail(state: AgentState):
    return {"final_answer": "Error: 데이터베이스 조회 실패"}

# --- Graph Wiring ---
workflow = StateGraph(AgentState)
workflow.add_node("router", node_router)
workflow.add_node("generator", node_generator)
workflow.add_node("critic", node_critic)
workflow.add_node("executor", node_executor)
workflow.add_node("summarizer", node_summarizer)
workflow.add_node("general", node_general)
workflow.add_node("fail", node_fail)

workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    lambda s: "general" if s["mode"] == "general" else "generator",
    {"general": "general", "generator": "generator"}
)

workflow.add_edge("generator", "critic")

def critic_logic(s):
    if s["evaluation"].get("valid"): return "executor"
    if s["retry_count"] >= 3: return "fail"
    return "generator"

workflow.add_conditional_edges("critic", critic_logic)

def exec_logic(s):
    return "fail" if s.get("error") else "summarizer"

workflow.add_conditional_edges("executor", exec_logic)

workflow.add_edge("summarizer", END)
workflow.add_edge("general", END)
workflow.add_edge("fail", END)

app_graph = workflow.compile()