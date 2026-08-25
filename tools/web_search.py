"""Tavily web search tool wrapper for Agentic RAG."""
import os
from typing import List
from langchain_core.documents import Document
from tavily import TavilyClient


def search_web(query: str, max_results: int = 3, api_key: str = None) -> List[Document]:
    """
    Search the web using Tavily API and return results formatted as LangChain Document objects.
    """
    effective_api_key = api_key or os.environ.get("TAVILY_API_KEY")
    if not effective_api_key:
        return [
            Document(
                page_content="Tavily API Key not found. Please provide TAVILY_API_KEY in the environment or UI sidebar.",
                metadata={"source": "Tavily Search (Error)", "url": "", "title": "Missing API Key"}
            )
        ]

    try:
        client = TavilyClient(api_key=effective_api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=True
        )

        documents: List[Document] = []

        # If Tavily provided a synthesized direct answer, include it as high-priority context
        if response.get("answer"):
            documents.append(
                Document(
                    page_content=f"Tavily Summary Answer: {response['answer']}",
                    metadata={"source": "Tavily AI Answer", "url": "https://tavily.com", "title": "Tavily Synthesis"}
                )
            )

        # Process standard web search results
        for item in response.get("results", []):
            content = item.get("content", "")
            title = item.get("title", "Web Result")
            url = item.get("url", "")
            score = item.get("score", 0.0)

            doc = Document(
                page_content=f"Title: {title}\nURL: {url}\nSnippet: {content}",
                metadata={
                    "source": url or "Web Search",
                    "title": title,
                    "url": url,
                    "score": score
                }
            )
            documents.append(doc)

        if not documents:
            documents.append(
                Document(
                    page_content="No relevant web search results found for the query.",
                    metadata={"source": "Tavily Search", "url": "", "title": "No Results"}
                )
            )

        return documents

    except Exception as e:
        return [
            Document(
                page_content=f"Error executing Tavily web search: {str(e)}",
                metadata={"source": "Tavily Search (Error)", "url": "", "title": "Search Failure"}
            )
        ]
