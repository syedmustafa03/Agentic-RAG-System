"""Graph nodes for the Agentic RAG system."""
import os
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_core.documents import Document

from graph.state import AgentState, ReasoningStep
from utils.prompts import (
    ROUTER_SYSTEM_PROMPT,
    GRADER_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    MEMORY_GENERATOR_SYSTEM_PROMPT,
    RouteDecision,
    OverallGradeDecision
)
from rag.vectorstore import retrieve_documents, is_vectorstore_ready
from tools.web_search import search_web


GROQ_DEFAULT_MODEL = "llama-3.1-8b-instant"


def get_llm(model_name: str = None, temperature: float = 0.0, api_key: str = None) -> ChatGroq:
    """Initialize Groq Chat LLM (free tier)."""
    effective_api_key = api_key or os.environ.get("GROQ_API_KEY")
    effective_model = model_name or os.environ.get("GROQ_MODEL", GROQ_DEFAULT_MODEL)
    return ChatGroq(
        model=effective_model,
        temperature=temperature,
        api_key=effective_api_key
    )


def router_node(state: AgentState, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Router Node: Analyzes the question and decides whether to retrieve from FAISS,
    perform a Tavily web search, or answer directly from GPT-4o's internal memory.
    """
    question = state.get("question", "")
    steps: List[ReasoningStep] = list(state.get("steps", []))

    api_key = None
    if config and "configurable" in config:
        api_key = config["configurable"].get("groq_api_key")

    llm = get_llm(temperature=0.0, api_key=api_key)
    structured_llm = llm.with_structured_output(RouteDecision)

    try:
        decision: RouteDecision = structured_llm.invoke([
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"Question to route:\n{question}")
        ])
        route = decision.route
        reasoning = decision.reasoning
        confidence = decision.confidence
    except Exception as e:
        # Fallback to retrieve if index is available, otherwise answer from memory
        has_index = is_vectorstore_ready()
        route = "retrieve" if has_index else "answer_from_memory"
        reasoning = f"Router encountered an error ({str(e)}), defaulting to '{route}'."
        confidence = 0.5

    step_info: ReasoningStep = {
        "node": "router",
        "title": "🧠 Question Routed",
        "description": f"Decided to proceed via **{route.upper()}** (Confidence: {int(confidence * 100)}%).",
        "details": {
            "Selected Route": route,
            "Confidence": f"{int(confidence * 100)}%",
            "Reasoning": reasoning
        },
        "badge_type": "route"
    }
    steps.append(step_info)

    return {
        "route": route,
        "router_reasoning": reasoning,
        "steps": steps,
        "fallback_count": state.get("fallback_count", 0)
    }


def faiss_retriever_node(state: AgentState, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    FAISS Retriever Node: Queries the local FAISS vector store for relevant document chunks.
    """
    question = state.get("question", "")
    steps: List[ReasoningStep] = list(state.get("steps", []))

    api_key = None
    if config and "configurable" in config:
        api_key = config["configurable"].get("groq_api_key")

    documents = retrieve_documents(query=question, k=4, api_key=api_key)

    doc_count = len(documents)
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in documents]))

    step_info: ReasoningStep = {
        "node": "faiss_retriever",
        "title": "📚 FAISS Local Retrieval",
        "description": f"Retrieved {doc_count} document chunk(s) from local vector store.",
        "details": {
            "Chunks Retrieved": doc_count,
            "Sources": sources if sources else ["No documents in vectorstore"],
            "Previews": [f"[{d.metadata.get('source', 'doc')}]: {d.page_content[:150]}..." for d in documents[:3]]
        },
        "badge_type": "info"
    }
    steps.append(step_info)

    return {
        "documents": documents,
        "steps": steps
    }


def web_search_node(state: AgentState, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Tavily Web Searcher Node: Queries Tavily web search API when local documents are missing or graded irrelevant.
    """
    question = state.get("question", "")
    steps: List[ReasoningStep] = list(state.get("steps", []))
    fallback_count = state.get("fallback_count", 0) + 1

    tavily_key = None
    if config and "configurable" in config:
        tavily_key = config["configurable"].get("tavily_api_key")

    documents = search_web(query=question, max_results=3, api_key=tavily_key)
    doc_count = len(documents)
    urls = [doc.metadata.get("url") for doc in documents if doc.metadata.get("url")]

    is_fallback = state.get("grade_verdict") == "not_relevant"
    title = "🌐 Tavily Fallback Web Search" if is_fallback else "🌐 Tavily Web Search"
    desc = f"Retrieved {doc_count} web result(s) via Tavily Search API." + (" (Triggered as fallback due to irrelevant local documents)" if is_fallback else "")

    step_info: ReasoningStep = {
        "node": "web_searcher",
        "title": title,
        "description": desc,
        "details": {
            "Results Count": doc_count,
            "Source URLs": urls if urls else ["Direct Tavily Knowledge Answer"],
            "Previews": [f"[{d.metadata.get('title', 'Web')}]: {d.page_content[:150]}..." for d in documents[:3]]
        },
        "badge_type": "warning" if is_fallback else "info"
    }
    steps.append(step_info)

    return {
        "documents": documents,
        "graded_documents": documents,  # Web search documents are accepted directly to prevent infinite grading loops
        "grade_verdict": "relevant",
        "web_search_needed": False,
        "fallback_count": fallback_count,
        "steps": steps
    }


def document_grader_node(state: AgentState, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Document Grader Node: Uses GPT-4o to evaluate if the retrieved documents are relevant
    to the user's question before generating an answer.
    """
    question = state.get("question", "")
    documents = state.get("documents", [])
    steps: List[ReasoningStep] = list(state.get("steps", []))
    fallback_count = state.get("fallback_count", 0)

    # If no documents were retrieved at all, mark not_relevant
    if not documents:
        step_info: ReasoningStep = {
            "node": "document_grader",
            "title": "⚖️ Document Grader: Not Relevant (Empty)",
            "description": "No documents retrieved from local store. Routing to Web Search fallback.",
            "details": {"Verdict": "not_relevant", "Reason": "Vector store returned 0 documents"},
            "badge_type": "warning"
        }
        steps.append(step_info)
        return {
            "graded_documents": [],
            "grade_verdict": "not_relevant",
            "grade_explanation": "No documents found in FAISS vector store.",
            "web_search_needed": True,
            "steps": steps
        }

    api_key = None
    if config and "configurable" in config:
        api_key = config["configurable"].get("groq_api_key")

    llm = get_llm(temperature=0.0, api_key=api_key)
    structured_llm = llm.with_structured_output(OverallGradeDecision)

    # Prepare document texts for grading
    docs_text = "\n\n".join([
        f"--- Document {idx + 1} (Source: {doc.metadata.get('source', 'Unknown')}) ---\n{doc.page_content}"
        for idx, doc in enumerate(documents)
    ])

    try:
        grading: OverallGradeDecision = structured_llm.invoke([
            SystemMessage(content=GRADER_SYSTEM_PROMPT),
            HumanMessage(content=f"User Question: {question}\n\nRetrieved Documents:\n{docs_text}")
        ])
        verdict = grading.verdict
        explanation = grading.explanation
        relevant_indices = grading.relevant_doc_indices

        # Filter relevant documents if indices provided
        if verdict == "relevant" and relevant_indices:
            filtered_docs = [documents[i] for i in relevant_indices if i < len(documents)]
            if not filtered_docs:
                filtered_docs = documents
        elif verdict == "relevant":
            filtered_docs = documents
        else:
            filtered_docs = []

    except Exception as e:
        # Graceful fallback: accept documents
        verdict = "relevant"
        explanation = f"Grader evaluation bypassed due to error ({str(e)}). Preserving documents."
        filtered_docs = documents

    # Prevent infinite fallback if fallback_count already >= 1
    if fallback_count >= 1 and verdict == "not_relevant":
        verdict = "relevant"
        filtered_docs = documents
        explanation += " (Max fallback reached; proceeding with best-effort generation)"

    step_info: ReasoningStep = {
        "node": "document_grader",
        "title": f"⚖️ Document Grader: {verdict.upper()}",
        "description": f"Verdict: **{verdict.upper()}** — {explanation}",
        "details": {
            "Verdict": verdict,
            "Explanation": explanation,
            "Relevant Documents Kept": len(filtered_docs),
            "Original Documents Checked": len(documents)
        },
        "badge_type": "success" if verdict == "relevant" else "warning"
    }
    steps.append(step_info)

    return {
        "graded_documents": filtered_docs,
        "grade_verdict": verdict,
        "grade_explanation": explanation,
        "web_search_needed": (verdict == "not_relevant"),
        "steps": steps
    }


def answer_generator_node(state: AgentState, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Answer Generator Node: Generates the final answer using GPT-4o, grounding on retrieved
    or web context if present, or drawing directly from memory.
    """
    question = state.get("question", "")
    route = state.get("route", "answer_from_memory")
    documents = state.get("graded_documents") or state.get("documents", [])
    steps: List[ReasoningStep] = list(state.get("steps", []))

    api_key = None
    if config and "configurable" in config:
        api_key = config["configurable"].get("groq_api_key")

    llm = get_llm(temperature=0.2, api_key=api_key)

    if route == "answer_from_memory" or (not documents and route != "web_search"):
        # Answer directly from memory
        messages = [
            SystemMessage(content=MEMORY_GENERATOR_SYSTEM_PROMPT),
            HumanMessage(content=f"User Question:\n{question}")
        ]
        response = llm.invoke(messages)
        answer = response.content
        gen_type = "Llama 3.3 70B Internal Memory"
    else:
        # Context-grounded generation
        context_str = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'Unknown')} | Title: {doc.metadata.get('title', 'N/A')}]\n{doc.page_content}"
            for doc in documents
        ])
        messages = [
            SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
            HumanMessage(content=f"Context Documents:\n{context_str}\n\nUser Question:\n{question}")
        ]
        response = llm.invoke(messages)
        answer = response.content
        gen_type = "Context-Grounded Generation via Llama 3.3 70B (RAG / Web Search)"

    step_info: ReasoningStep = {
        "node": "answer_generator",
        "title": "✍️ Final Answer Generated",
        "description": f"Answer constructed using **{gen_type}**.",
        "details": {
            "Mode": gen_type,
            "Context Sources Used": len(documents) if route != "answer_from_memory" else 0
        },
        "badge_type": "success"
    }
    steps.append(step_info)

    return {
        "answer": answer,
        "steps": steps
    }
