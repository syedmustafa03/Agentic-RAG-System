from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict
from langchain_core.documents import Document


class ReasoningStep(TypedDict, total=False):
    node: str
    title: str
    description: str
    details: Optional[Dict[str, Any]]
    badge_type: str  # 'info', 'success', 'warning', 'route'


class AgentState(TypedDict, total=False):
    question: str
    route: str  # 'retrieve' | 'web_search' | 'answer_from_memory'
    router_reasoning: str
    documents: List[Document]
    graded_documents: List[Document]
    grade_verdict: str  # 'relevant' | 'not_relevant'
    grade_explanation: str
    web_search_needed: bool
    fallback_count: int
    answer: str
    steps: List[ReasoningStep]
