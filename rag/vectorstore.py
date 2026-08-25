"""FAISS vector store management, embedding generation, and retrieval."""
import os
import shutil
from typing import List, Optional, Tuple
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "faiss_index")


def get_embeddings(api_key: Optional[str] = None) -> HuggingFaceEmbeddings:
    """Initialize local HuggingFace embeddings model (no API key required)."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def build_vectorstore(
    documents: List[Document],
    api_key: Optional[str] = None,
    save_path: str = INDEX_DIR
) -> FAISS:
    """
    Build a new FAISS vector store from document chunks and save to disk.
    """
    if not documents:
        raise ValueError("Cannot build vectorstore: No documents provided.")

    embeddings = get_embeddings(api_key)
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)

    # Persist index
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)
    return vectorstore


def load_vectorstore(
    api_key: Optional[str] = None,
    index_path: str = INDEX_DIR
) -> Optional[FAISS]:
    """
    Load an existing FAISS index from disk.
    """
    if not os.path.exists(os.path.join(index_path, "index.faiss")):
        return None

    try:
        embeddings = get_embeddings(api_key)
        return FAISS.load_local(
            folder_path=index_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Error loading FAISS vectorstore: {e}")
        return None


def clear_vectorstore(index_path: str = INDEX_DIR) -> bool:
    """
    Remove saved FAISS index from disk.
    """
    if os.path.exists(index_path):
        try:
            shutil.rmtree(index_path)
            return True
        except Exception as e:
            print(f"Error deleting index: {e}")
            return False
    return False


def retrieve_documents(
    query: str,
    vectorstore: Optional[FAISS] = None,
    k: int = 4,
    api_key: Optional[str] = None,
    score_threshold: Optional[float] = None
) -> List[Document]:
    """
    Retrieve top-k relevant documents from FAISS vector store.
    """
    if vectorstore is None:
        vectorstore = load_vectorstore(api_key=api_key)

    if vectorstore is None:
        return []

    try:
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
        retrieved_docs: List[Document] = []
        for doc, score in docs_and_scores:
            doc.metadata["similarity_distance"] = round(float(score), 4)
            retrieved_docs.append(doc)
        return retrieved_docs
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []


def is_vectorstore_ready(index_path: str = INDEX_DIR) -> bool:
    """Check if the FAISS index file exists."""
    return os.path.exists(os.path.join(index_path, "index.faiss"))
