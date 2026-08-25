# Enterprise AI Security & Governance Policy (v4.2)
**Classification**: Strictly Confidential - Internal Only
**Effective Date**: January 15, 2026
**Author**: Global AI Safety & Compliance Board

---

## 1. Scope & Objective
This policy governs the deployment, fine-tuning, and runtime monitoring of Generative AI systems, Large Language Models (LLMs), and Retrieval-Augmented Generation (RAG) pipelines across all engineering and customer-facing teams.

## 2. Approved Model Tiers & Deployment Constraints
- **Tier 1 (High Reliability / Customer Facing)**: GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro.
- **Tier 2 (Internal Automation / Data Analysis)**: GPT-4o-mini, Llama 3.3 70B, Mistral Large.
- **Self-Hosted Local Vector Stores**: FAISS, ChromaDB, and Qdrant are officially sanctioned for internal document retrieval. Vector stores holding Restricted Tier 3 data must utilize AES-256 encryption at rest.

## 3. Data Ingestion & Sanitation Rules
- All documents ingested into embedding pipelines must be processed through the PII Redaction Filter (`PII-Scrubber v2`).
- Customer credentials, JWT secret keys, and API tokens must never be converted into vector embeddings.
- Chunk overlap should not exceed 25% of the primary chunk size (recommended standard: 800 tokens chunk size, 150 tokens overlap).

## 4. Fallback and Safeguards in Autonomous RAG
1. **Document Grading Protocol**: Every retrieved document must undergo automated semantic grading. Any document scoring below 0.70 semantic relevance must be discarded prior to LLM synthesis.
2. **Web Search Fallback**: When internal vector retrieval fails to produce relevant documents (verdict: `not_relevant`), systems are authorized to invoke Tavily Search API with `search_depth='advanced'`.
3. **Loop Safeguard**: A maximum of 1 fallback iteration is permitted per user query to prevent cycle stalls.

## 5. Incident Escalation Contact
- AI Security Ops: `ai-security@enterprise-corp.internal`
- Slack Channel: `#ai-ops-war-room`
