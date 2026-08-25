"""Routing logic and conditional edges for LangGraph."""
from typing import Literal
from graph.state import AgentState


def route_question(state: AgentState) -> Literal["faiss_retriever", "web_searcher", "answer_generator"]:
    """
    Conditional edge from Router node:
    - 'retrieve' -> faiss_retriever
    - 'web_search' -> web_searcher
    - 'answer_from_memory' -> answer_generator
    """
    route = state.get("route", "answer_from_memory")

    if route == "retrieve":
        return "faiss_retriever"
    elif route == "web_search":
        return "web_searcher"
    else:
        return "answer_generator"


def decide_to_generate_or_fallback(state: AgentState) -> Literal["answer_generator", "web_searcher"]:
    """
    Conditional edge from Document Grader node:
    - If documents are relevant -> answer_generator
    - If documents are NOT relevant and fallback limit not exceeded -> web_searcher
    - If fallback limit exceeded -> answer_generator (best effort)
    """
    grade_verdict = state.get("grade_verdict", "relevant")
    fallback_count = state.get("fallback_count", 0)

    if grade_verdict == "not_relevant" and fallback_count < 1:
        return "web_searcher"
    return "answer_generator"
