"""Prompt templates and Pydantic models for Agentic RAG."""
from pydantic import BaseModel, Field
from typing import Literal


class RouteDecision(BaseModel):
    """Route decision made by the router."""
    route: Literal["retrieve", "web_search", "answer_from_memory"] = Field(
        description=(
            "The selected action: 'retrieve' if question is related to indexed private/domain knowledge documents; "
            "'web_search' if question requires current events, real-time data, external knowledge, or fresh web info; "
            "'answer_from_memory' if it's general knowledge, coding assistance, greetings, math, or well-established facts the model is confident in answering without external docs."
        )
    )
    confidence: float = Field(
        description="Confidence score from 0.0 to 1.0 in this routing decision."
    )
    reasoning: str = Field(
        description="Brief explanation of why this route was selected."
    )


class DocumentGrade(BaseModel):
    """Grade for a single retrieved document's relevance to the user question."""
    is_relevant: bool = Field(
        description="True if the document contains keywords, semantic facts, or relevant info directly related to the user question, False otherwise."
    )
    reasoning: str = Field(
        description="Brief explanation of why the document is relevant or not relevant."
    )


class OverallGradeDecision(BaseModel):
    """Overall grading decision for all retrieved documents."""
    verdict: Literal["relevant", "not_relevant"] = Field(
        description="'relevant' if at least one retrieved document provides useful context to answer the question, 'not_relevant' if none do and fallback is required."
    )
    relevant_doc_indices: list[int] = Field(
        default_factory=list,
        description="List of 0-based indices of documents that are relevant."
    )
    explanation: str = Field(
        description="Short summary of document grading assessment."
    )


ROUTER_SYSTEM_PROMPT = """You are an expert routing assistant in an agentic RAG system.
Your job is to analyze the user's question and decide the best execution path:

1. 'retrieve': Use this if the question is about specific internal domain documents, company policies, proprietary guidelines, technical specs, uploaded project files, or detailed domain-specific notes.
2. 'web_search': Use this if the question is about recent news, current events, live dates, up-to-date documentation, external web resources, or topics unlikely to be in internal documents or static model weights.
3. 'answer_from_memory': Use this if the question is general world knowledge, standard programming concepts, logic/math problems, conversational greetings, explanations of standard concepts, or topics you are highly confident answering directly.

Analyze the question carefully and return your routing decision with clear reasoning and a confidence score."""


GRADER_SYSTEM_PROMPT = """You are an expert document relevance grader.
Your task is to evaluate retrieved documents against the user's question.
Check whether the retrieved documents contain facts, context, or semantic information useful for answering the user's question.
If the documents contain helpful or partial information to answer the question, grade as 'relevant'.
If the documents are completely off-topic, spam, irrelevant, or do not help answer the question, grade as 'not_relevant'.
Be objective and strict."""


GENERATOR_SYSTEM_PROMPT = """You are an intelligent, precise AI assistant answering questions using retrieved context.

Guidelines:
1. Provide a comprehensive, accurate, and structured answer to the user's question.
2. Use the provided context documents to ground your answer when available.
3. Clearly cite which sources or documents support your points where applicable.
4. If the retrieved context is insufficient, state what is known from the context and augment with clear, honest reasoning.
5. Format your output cleanly with markdown, bullet points, and code blocks where helpful."""


MEMORY_GENERATOR_SYSTEM_PROMPT = """You are an intelligent, articulate AI assistant.
Answer the user's question directly from your knowledge base with clear explanations, structure, and precision.
State clearly if any part requires assumptions or further context."""
