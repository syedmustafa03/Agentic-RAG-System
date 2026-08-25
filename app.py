"""Agentic RAG System with LangGraph, FAISS, Tavily, and Streamlit.
Interactive UI featuring live reasoning traces, graph visualization, and vector store management.
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(
    page_title="Agentic RAG with LangGraph",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, high-end aesthetics
st.markdown("""
<style>
    /* Global Styling */
    .main {
        background-color: #0b0f19;
    }
    
    /* Header Card */
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 0;
    }

    /* Badges */
    .route-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-right: 8px;
    }
    .badge-faiss {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    .badge-tavily {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    .badge-memory {
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(129, 140, 248, 0.3);
    }

    /* Step Trace Box */
    .step-box {
        background: #131b2e;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .step-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 4px;
    }
    .step-desc {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 0;
    }

    /* Custom button styling */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# Imports from project modules
from graph.builder import build_graph, get_graph_mermaid_code, get_graph_png_bytes
from rag.ingestion import load_documents_from_directory, load_uploaded_files, split_documents
from rag.vectorstore import build_vectorstore, is_vectorstore_ready, clear_vectorstore, INDEX_DIR
from langchain_core.documents import Document

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph_app" not in st.session_state:
    try:
        st.session_state.graph_app = build_graph()
    except Exception as e:
        st.session_state.graph_app = None
if "vector_index_ready" not in st.session_state:
    st.session_state.vector_index_ready = is_vectorstore_ready()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.markdown("### ⚙️ System Settings")

    # API Keys Configuration
    with st.expander("🔑 API Credentials", expanded=True):
        groq_key_input = st.text_input(
            "Groq API Key",
            value=os.environ.get("GROQ_API_KEY", ""),
            type="password",
            help="Free API key from console.groq.com — powers routing, grading, and answer generation."
        )
        if groq_key_input:
            os.environ["GROQ_API_KEY"] = groq_key_input

        tavily_key_input = st.text_input(
            "Tavily API Key",
            value=os.environ.get("TAVILY_API_KEY", ""),
            type="password",
            help="Required for real-time web search fallback."
        )
        if tavily_key_input:
            os.environ["TAVILY_API_KEY"] = tavily_key_input

        # Status Indicators
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            if os.environ.get("GROQ_API_KEY"):
                st.caption("🟢 Groq Ready")
            else:
                st.caption("🔴 Groq Missing")
        with col_k2:
            if os.environ.get("TAVILY_API_KEY"):
                st.caption("🟢 Tavily Ready")
            else:
                st.caption("🟡 Tavily Optional")

    # Model Selector
    with st.expander("🤖 Groq Model", expanded=True):
        GROQ_MODELS = {
            "Llama 3 70B (Recommended)": "llama3-70b-8192",
            "Llama 3 8B (Fastest)": "llama3-8b-8192",
            "Llama 3.1 8B Instant": "llama-3.1-8b-instant",
            "Mixtral 8x7B": "mixtral-8x7b-32768",
            "Gemma2 9B": "gemma2-9b-it",
        }
        selected_model_label = st.selectbox(
            "Select Model",
            options=list(GROQ_MODELS.keys()),
            index=0,
            help="All models are available on Groq's free tier. Llama 3 70B gives the best results."
        )
        selected_model_id = GROQ_MODELS[selected_model_label]
        os.environ["GROQ_MODEL"] = selected_model_id
        st.caption(f"`{selected_model_id}`")

    st.markdown("---")

    # Vector Store Management
    st.markdown("### 📚 FAISS Vector Store")

    index_exists = is_vectorstore_ready()
    if index_exists:
        st.success("✅ FAISS Index Loaded & Active")
    else:
        st.warning("⚠️ No FAISS Index Found")

    uploaded_files = st.file_uploader(
        "Upload Custom Documents (.txt, .md, .pdf)",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        help="Add proprietary documents for local retrieval."
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚀 Ingest & Build", use_container_width=True):
            with st.spinner("Processing & embedding documents..."):
                docs_to_index = []
                
                # 1. Load from docs/ directory
                docs_dir = os.path.join(os.path.dirname(__file__), "docs")
                local_docs = load_documents_from_directory(docs_dir)
                docs_to_index.extend(local_docs)

                # 2. Load uploaded files
                if uploaded_files:
                    uploaded_docs = load_uploaded_files(uploaded_files)
                    docs_to_index.extend(uploaded_docs)

                if not docs_to_index:
                    st.error("No documents found in `docs/` folder or upload queue.")
                else:
                    chunks = split_documents(docs_to_index, chunk_size=800, chunk_overlap=150)
                    try:
                        build_vectorstore(chunks)
                        st.session_state.vector_index_ready = True
                        st.success(f"Built FAISS store with {len(chunks)} chunks from {len(docs_to_index)} documents!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Build failed: {str(e)}")

    with col_btn2:
        if st.button("🗑️ Reset Index", use_container_width=True):
            clear_vectorstore()
            st.session_state.vector_index_ready = False
            st.info("Vector index cleared.")
            st.rerun()

    st.markdown("---")
    st.markdown("### 🧭 Dynamic Routing Pathways")
    st.markdown("""
    - <span class="route-badge badge-faiss">Retrieve</span> Internal domain documents
    - <span class="route-badge badge-tavily">Web Search</span> Current news & fallback
    - <span class="route-badge badge-memory">Memory</span> Confident general knowledge
    """, unsafe_allow_html=True)


# --- MAIN HEADER ---
st.markdown("""
<div class="hero-card">
    <div class="hero-title">⚡ Autonomous Agentic RAG</div>
    <div class="hero-subtitle">
        Intelligent multi-tier LangGraph agent orchestrating <b>Local FAISS Vector Retrieval</b>, 
        <b>Llama 3.3 70B Semantic Document Grading</b>, <b>Tavily Web Search Fallback</b>, and <b>Direct Memory Synthesis</b>.
    </div>
</div>
""", unsafe_allow_html=True)


# --- TABS ---
tab_chat, tab_graph, tab_docs = st.tabs([
    "💬 Agent Chat & Live Reasoning",
    "🗺️ LangGraph Flow Visualization",
    "📂 Knowledge Base Documents"
])

# ==========================================
# TAB 1: AGENT CHAT & LIVE REASONING
# ==========================================
with tab_chat:
    # Example Quick-Prompts
    st.markdown("##### 💡 Try Example Queries")
    col_q1, col_q2, col_q3 = st.columns(3)
    
    preset_query = None
    with col_q1:
        if st.button("📚 Local FAISS Query\n'What are the Project Aurora cache specs?'", use_container_width=True):
            preset_query = "What are the core technical specifications and throughput capacity of Project Aurora distributed cache?"
    with col_q2:
        if st.button("🌐 Web Fallback Query\n'What are the latest AI breakthroughs today?'", use_container_width=True):
            preset_query = "What are the latest major artificial intelligence news and model releases this month?"
    with col_q3:
        if st.button("🧠 Memory Direct Query\n'How does Python's asyncio event loop work?'", use_container_width=True):
            preset_query = "Explain how Python's asyncio event loop and coroutines work under the hood with a brief code example."

    st.markdown("---")

    # Render Chat History
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        steps = msg.get("steps", [])
        route = msg.get("route", "")

        with st.chat_message(role):
            if role == "assistant":
                # Route badge
                if route == "retrieve":
                    st.markdown('<span class="route-badge badge-faiss">Route: FAISS Vector Retrieval</span>', unsafe_allow_html=True)
                elif route == "web_search":
                    st.markdown('<span class="route-badge badge-tavily">Route: Tavily Web Search</span>', unsafe_allow_html=True)
                elif route == "answer_from_memory":
                    st.markdown('<span class="route-badge badge-memory">Route: Llama 3.3 70B Internal Memory</span>', unsafe_allow_html=True)

                # Final Answer
                st.markdown(content)

                # Show reasoning trace in expander
                if steps:
                    with st.expander("🔍 View Agent Execution & Reasoning Trace", expanded=False):
                        for step in steps:
                            badge_type = step.get("badge_type", "info")
                            title = step.get("title", "Step")
                            desc = step.get("description", "")
                            details = step.get("details", {})

                            st.markdown(f"**{title}**")
                            st.caption(desc)
                            if details:
                                st.json(details)
                            st.markdown("---")
            else:
                st.markdown(content)

    # Chat Input Box
    user_input = st.chat_input("Ask a question to the Agentic RAG system...")
    active_prompt = preset_query or user_input

    if active_prompt:
        if not os.environ.get("GROQ_API_KEY"):
            st.error("⚠️ Please configure your Groq API Key in the left sidebar to start. Get one free at console.groq.com")
        else:
            # 1. Add user message
            st.session_state.messages.append({"role": "user", "content": active_prompt})
            with st.chat_message("user"):
                st.markdown(active_prompt)

            # 2. Run LangGraph Workflow
            with st.chat_message("assistant"):
                status_container = st.status("🧠 Agent is analyzing and executing workflow...", expanded=True)
                
                try:
                    if st.session_state.graph_app is None:
                        st.session_state.graph_app = build_graph()
                    
                    graph_app = st.session_state.graph_app

                    # Initial Agent State
                    initial_state = {
                        "question": active_prompt,
                        "steps": [],
                        "fallback_count": 0
                    }

                    # Execute graph streaming steps
                    final_state = initial_state
                    for event in graph_app.stream(initial_state):
                        for node_name, node_output in event.items():
                            if node_name == "router":
                                r = node_output.get("route", "")
                                reason = node_output.get("router_reasoning", "")
                                status_container.write(f"🧠 **Router Node**: Selected route `{r.upper()}` ({reason})")
                            elif node_name == "faiss_retriever":
                                docs = node_output.get("documents", [])
                                status_container.write(f"📚 **FAISS Retriever**: Fetched {len(docs)} document chunk(s)")
                            elif node_name == "document_grader":
                                verdict = node_output.get("grade_verdict", "")
                                expl = node_output.get("grade_explanation", "")
                                status_container.write(f"⚖️ **Document Grader**: Verdict `{verdict.upper()}` ({expl})")
                            elif node_name == "web_searcher":
                                docs = node_output.get("documents", [])
                                status_container.write(f"🌐 **Web Searcher**: Retrieved {len(docs)} web sources via Tavily")
                            elif node_name == "answer_generator":
                                status_container.write("✍️ **Answer Generator**: Formulating final synthesized response...")
                            
                            # Merge state
                            final_state.update(node_output)

                    status_container.update(label="✅ Reasoning & Generation Completed", state="complete", expanded=False)

                    final_answer = final_state.get("answer", "No answer generated.")
                    final_route = final_state.get("route", "")
                    final_steps = final_state.get("steps", [])

                    # Render Route Badge
                    if final_route == "retrieve":
                        st.markdown('<span class="route-badge badge-faiss">Route: FAISS Vector Retrieval</span>', unsafe_allow_html=True)
                    elif final_route == "web_search":
                        st.markdown('<span class="route-badge badge-tavily">Route: Tavily Web Search</span>', unsafe_allow_html=True)
                    elif final_route == "answer_from_memory":
                        st.markdown('<span class="route-badge badge-memory">Route: Llama 3.3 70B Internal Memory</span>', unsafe_allow_html=True)

                    st.markdown(final_answer)

                    # Show Reasoning Steps Trace
                    if final_steps:
                        with st.expander("🔍 View Agent Execution & Reasoning Trace", expanded=True):
                            for step in final_steps:
                                title = step.get("title", "Step")
                                desc = step.get("description", "")
                                details = step.get("details", {})

                                st.markdown(f"**{title}**")
                                st.caption(desc)
                                if details:
                                    st.json(details)
                                st.markdown("---")

                    # Save to conversation history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_answer,
                        "route": final_route,
                        "steps": final_steps
                    })

                except Exception as e:
                    status_container.update(label="❌ Error executing workflow", state="error")
                    st.error(f"Workflow execution failed: {str(e)}")


# ==========================================
# TAB 2: LANGGRAPH FLOW VISUALIZATION
# ==========================================
with tab_graph:
    st.markdown("### 🗺️ LangGraph Architecture Diagram")
    st.markdown("""
    The flowchart below visualizes the runtime graph flow. Notice the self-routing decision from the **Router Node**, 
    the **Document Grader** validation gate, and the automatic fallback path to **Tavily Web Search** when documents are graded irrelevant.
    """)

    # Render Mermaid diagram
    mermaid_chart = get_graph_mermaid_code()
    st.markdown(f"```mermaid\n{mermaid_chart}\n```")

    st.markdown("---")
    st.markdown("### 🧩 Graph Nodes & Routing Specifications")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("""
        #### 1. Router Node (`router`)
        - **Model**: Llama 3.3 70B via Groq with structured Pydantic schema (`RouteDecision`).
        - **Decisions**:
          - `retrieve`: Domain-specific or internal documents.
          - `web_search`: Up-to-date events, external facts, or search fallback.
          - `answer_from_memory`: General concepts, math, coding patterns, greetings.

        #### 2. FAISS Retriever Node (`faiss_retriever`)
        - **Embeddings**: `all-MiniLM-L6-v2` (local, free, no API key).
        - **Search**: Top-$k=4$ semantic cosine similarity retrieval.

        #### 3. Document Grader Node (`document_grader`)
        - **Model**: Llama 3.3 70B via Groq with structured schema (`OverallGradeDecision`).
        - **Logic**: Filters out hallucinated or irrelevant chunks.
        - **Edge Dispatch**:
          - `grade == 'relevant'` ➔ **Answer Generator**
          - `grade == 'not_relevant'` ➔ **Tavily Web Searcher (Fallback)**
        """)

    with col_g2:
        st.markdown("""
        #### 4. Web Searcher Node (`web_searcher`)
        - **Tool**: Tavily Search Engine with structured document parsing.
        - **Output**: Formats top web snippets, titles, and URLs into context.

        #### 5. Answer Generator Node (`answer_generator`)
        - **Model**: Llama 3.3 70B via Groq synthesis.
        - **Grounding**: Synthesizes verified context with exact citations.
        - **Memory Mode**: Operates directly for confident general knowledge.

        #### 6. Loop Guard Safeguard
        - `fallback_count` state variable prevents infinite cycles between vector search and web fallback.
        """)


# ==========================================
# TAB 3: KNOWLEDGE BASE INSPECTOR
# ==========================================
with tab_docs:
    st.markdown("### 📂 Indexed Knowledge Base Documents")
    st.markdown("These documents are stored in the local `docs/` folder and indexed into the FAISS vector store:")

    docs_dir = os.path.join(os.path.dirname(__file__), "docs")
    if os.path.exists(docs_dir):
        doc_files = [f for f in os.listdir(docs_dir) if os.path.isfile(os.path.join(docs_dir, f))]
        for doc_file in doc_files:
            file_path = os.path.join(docs_dir, doc_file)
            with st.expander(f"📄 {doc_file}", expanded=False):
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        st.markdown(f.read())
                except Exception as e:
                    st.error(f"Could not read {doc_file}: {e}")
    else:
        st.info("No `docs/` directory found.")
