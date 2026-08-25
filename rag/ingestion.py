"""Document loading, parsing, and text splitting utilities."""
import os
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents_from_directory(directory_path: str) -> List[Document]:
    """
    Load .txt, .md, .pdf files from a specified directory.
    """
    documents: List[Document] = []
    dir_path = Path(directory_path)

    if not dir_path.exists():
        return documents

    for file_path in dir_path.glob("**/*"):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            try:
                if ext in [".txt", ".md"]:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": file_path.name,
                                    "file_path": str(file_path),
                                    "file_type": ext[1:]
                                }
                            )
                        )
                elif ext == ".pdf":
                    try:
                        import pypdf
                        reader = pypdf.PdfReader(str(file_path))
                        pdf_text = ""
                        for page_num, page in enumerate(reader.pages):
                            extracted = page.extract_text()
                            if extracted:
                                pdf_text += f"\n--- Page {page_num + 1} ---\n{extracted}"
                        if pdf_text.strip():
                            documents.append(
                                Document(
                                    page_content=pdf_text,
                                    metadata={
                                        "source": file_path.name,
                                        "file_path": str(file_path),
                                        "file_type": "pdf"
                                    }
                                )
                            )
                    except Exception as e:
                        print(f"Error reading PDF {file_path}: {e}")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    return documents


def load_uploaded_files(uploaded_files) -> List[Document]:
    """
    Process Streamlit UploadedFile objects into LangChain Documents.
    """
    documents: List[Document] = []
    for uploaded_file in uploaded_files:
        name = uploaded_file.name
        ext = Path(name).suffix.lower()
        try:
            if ext in [".txt", ".md"]:
                content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                if content.strip():
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={"source": name, "file_type": ext[1:]}
                        )
                    )
            elif ext == ".pdf":
                import pypdf
                from io import BytesIO
                reader = pypdf.PdfReader(BytesIO(uploaded_file.getvalue()))
                pdf_text = ""
                for page_num, page in enumerate(reader.pages):
                    extracted = page.extract_text()
                    if extracted:
                        pdf_text += f"\n--- Page {page_num + 1} ---\n{extracted}"
                if pdf_text.strip():
                    documents.append(
                        Document(
                            page_content=pdf_text,
                            metadata={"source": name, "file_type": "pdf"}
                        )
                    )
        except Exception as e:
            print(f"Error processing uploaded file {name}: {e}")

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Split documents into semantically coherent chunks using RecursiveCharacterTextSplitter.
    """
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks
