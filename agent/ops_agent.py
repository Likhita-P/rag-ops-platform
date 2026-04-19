"""
ops_agent.py
------------
LangGraph agent for natural-language model operations.

Lets you ask in plain English:
  "How is the model performing this week?"
  "What's the hallucination rate?"
  "Trigger a retraining run"
  "What's today's spend?"
  "Show me the last eval results"

The agent decides which tool to call based on the question,
calls it, and returns a human-readable summary.

Interview angle (both interviews — existing strength + new project):
  "The agent turns MLflow metrics and Airflow into a conversational
   interface. Instead of logging into dashboards, the team asks in
   Slack: 'how did last night's retrain go?' and the agent responds
   with faithfulness score, cost, and whether it improved."
"""

import logging
import os
from typing import TypedDict, Annotated
import operator

from langchain_openai        import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools    import tool
from langgraph.graph          import StateGraph, END
from langgraph.prebuilt       import ToolNode

logger = logging.getLogger(__name__)


# ── Tools the agent can call ───────────────────────────────────────────────

@tool
def get_model_metrics() -> dict:
    """Get the latest model performance metrics from MLflow."""
    from pipelines.mlflow_tracker import get_latest_metrics
    return get_latest_metrics()


@tool
def get_today_cost() -> dict:
    """Get today's total LLM spend in USD."""
    from app.cost_tracker import get_today_spend
    return {"today_spend_usd": get_today_spend()}


@tool
def trigger_retraining() -> dict:
    """Trigger the Airflow retraining DAG manually."""
    from pipelines.drift_monitor import _trigger_airflow_dag
    _trigger_airflow_dag()
    return {"status": "Retraining DAG triggered successfully."}


@tool
def check_drift_status() -> dict:
    """Check if retrieval quality drift has been detected."""
    from pipelines.drift_monitor import check_drift
    return check_drift()


@tool
def run_eval_now() -> dict:
    """Run a RAGAS evaluation right now and return results."""
    from evals.ragas_eval import run_ragas_eval
    result = run_ragas_eval()
    return result.model_dump()


TOOLS = [get_model_metrics, get_today_cost, trigger_retraining,
         check_drift_status, run_eval_now]


# ── Agent state ────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


# ── LLM setup ─────────────────────────────────────────────────────────────

def get_llm():
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_MODEL",    "gpt-4o"),
        azure_endpoint=  os.getenv("AZURE_OPENAI_API_URL",  ""),
        api_key=         os.getenv("AZURE_OPENAI_API_KEY",  ""),
        api_version=     "2024-08-01-preview",
        temperature=0,
    ).bind_tools(TOOLS)


# ── Graph nodes ────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> AgentState:
    """LLM decides which tool to call (or responds directly)."""
    llm      = get_llm()
    system   = SystemMessage(content=(
        "You are an AI Ops assistant. You help monitor and manage the RAG chatbot. "
        "Use the available tools to answer questions about model performance, "
        "costs, drift, and to trigger retraining. Be concise and factual."
    ))
    messages = [system] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Route to tools if tool_calls present, else END."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── Build graph ────────────────────────────────────────────────────────────

def build_agent():
    tool_node = ToolNode(TOOLS)
    graph     = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Public interface ───────────────────────────────────────────────────────

_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def ask_agent(question: str) -> str:
    """
    Ask the ops agent a question in plain English.
    Returns a human-readable answer.

    Example:
        ask_agent("What is the current hallucination rate?")
        ask_agent("Trigger a retraining run")
        ask_agent("How much have we spent today?")
    """
    agent  = get_agent()
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    last   = result["messages"][-1]
    return last.content if hasattr(last, "content") else str(last)


# ── CLI for quick testing ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("AI Ops Agent — type 'quit' to exit\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if q:
            print(f"Agent: {ask_agent(q)}\n")
