"""Graph construction and compilation module."""
from langgraph.graph import StateGraph, START, END
from graph.state import AgentState
from graph.nodes import (
    router_node,
    faiss_retriever_node,
    web_search_node,
    document_grader_node,
    answer_generator_node
)
from graph.edges import route_question, decide_to_generate_or_fallback


def build_graph():
    """
    Construct and compile the LangGraph workflow for Agentic RAG.
    """
    workflow = StateGraph(AgentState)

    # 1. Add all graph nodes
    workflow.add_node("router", router_node)
    workflow.add_node("faiss_retriever", faiss_retriever_node)
    workflow.add_node("web_searcher", web_search_node)
    workflow.add_node("document_grader", document_grader_node)
    workflow.add_node("answer_generator", answer_generator_node)

    # 2. Add edges and conditional routing
    # START -> router
    workflow.add_edge(START, "router")

    # router -> (faiss_retriever | web_searcher | answer_generator)
    workflow.add_conditional_edges(
        "router",
        route_question,
        {
            "faiss_retriever": "faiss_retriever",
            "web_searcher": "web_searcher",
            "answer_generator": "answer_generator",
        }
    )

    # faiss_retriever -> document_grader
    workflow.add_edge("faiss_retriever", "document_grader")

    # document_grader -> (answer_generator | web_searcher)
    workflow.add_conditional_edges(
        "document_grader",
        decide_to_generate_or_fallback,
        {
            "answer_generator": "answer_generator",
            "web_searcher": "web_searcher",
        }
    )

    # web_searcher -> answer_generator
    workflow.add_edge("web_searcher", "answer_generator")

    # answer_generator -> END
    workflow.add_edge("answer_generator", END)

    # Compile the graph
    app = workflow.compile()
    return app


def get_graph_mermaid_code() -> str:
    """
    Return Mermaid flowchart code for visualization.
    """
    return """
flowchart TD
    %% Styling
    classDef startEnd fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef router fill:#312e81,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
    classDef faiss fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#f8fafc;
    classDef tavily fill:#78350f,stroke:#fbbf24,stroke-width:2px,color:#f8fafc;
    classDef grader fill:#581c87,stroke:#c084fc,stroke-width:2px,color:#f8fafc;
    classDef generator fill:#1e3a8a,stroke:#60a5fa,stroke-width:2px,color:#f8fafc;

    START([🚀 User Question]):::startEnd --> ROUTER[🧠 Router Node\nGPT-4o Route Classifier]:::router

    ROUTER -->|route = retrieve| FAISS[📚 FAISS Retriever\nLocal Vector Store]:::faiss
    ROUTER -->|route = web_search| TAVILY[🌐 Web Searcher\nTavily API Engine]:::tavily
    ROUTER -->|route = answer_from_memory| GENERATOR[✍️ Answer Generator\nGPT-4o Synthesis]:::generator

    FAISS --> GRADER[⚖️ Document Grader\nGPT-4o Relevance Evaluation]:::grader

    GRADER -->|grade = relevant| GENERATOR
    GRADER -->|grade = not_relevant - Fallback| TAVILY

    TAVILY --> GENERATOR
    GENERATOR --> END([🏁 Final Answer + Reasoning Trace]):::startEnd
"""


def get_graph_png_bytes(app=None) -> bytes:
    """
    Render graph to PNG bytes using LangGraph's draw_mermaid_png.
    """
    if app is None:
        app = build_graph()
    try:
        return app.get_graph().draw_mermaid_png()
    except Exception as e:
        print(f"Could not render PNG from LangGraph: {e}")
        return None
