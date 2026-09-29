# 🤖 RAGAgent — Agentic Retrieval-Augmented Generation

> A Streamlit-based Agentic RAG application that combines Qdrant semantic retrieval with a local **Laya decision layer** to determine whether retrieved context is sufficient. When it is not, the agent falls back to DuckDuckGo web search before using Gemini to generate the response.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40.1-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Database-red)](https://qdrant.tech/)
[![Gemini](https://img.shields.io/badge/Google-Gemini-4285F4)](https://ai.google.dev/)

## 📌 Overview

**RAGAgent** is an end-to-end Retrieval-Augmented Generation application for answering questions from user-provided knowledge sources.

You can:
- 📄 Upload one or more PDF documents.
- 🌐 Provide one or more website URLs.
- 🕷️ Optionally discover same-domain links from supplied websites.
- 🔎 Convert your data into embeddings and store them in Qdrant.
- 💬 Ask questions through a Streamlit chat interface.
- 🧠 Use **Laya locally** as the context-sufficiency decision gate before answer generation.
- 🌍 Fall back to DuckDuckGo search when indexed context is not relevant.
- 💾 Reuse a locally persisted Qdrant collection across Streamlit sessions.

The current implementation uses **Gemini** for embeddings and LLM generation and **local persistent Qdrant storage** through the Qdrant Python client.

## ✨ Key Features

### 📄 PDF Retrieval
Upload multiple PDFs and extract their text using PyPDF2.

### 🌐 Website Retrieval
Enter comma-separated website URLs. The application downloads HTML, removes script/style content, extracts readable text, and indexes it.

### 🕷️ Website Link Discovery
When **Crawl entire website(s)** is enabled, the application discovers same-domain links from supplied starting pages before processing the discovered URLs.

### 🧩 Hybrid Knowledge Base
PDF and website content can be indexed together. Each chunk keeps source metadata so retrieved information can be associated with its origin.

### 🔍 Semantic Search
Questions are converted into embeddings and compared against indexed document vectors using cosine similarity in Qdrant.

### 🧠 Agentic Routing
Before generation, the application uses **Laya** as a local decision layer to determine whether the retrieved context is sufficient to answer the question. This replaces the previous Gemini judge call and reduces LLM usage on the normal RAG path.

### 🌍 Online Search Fallback
When indexed context is judged insufficient, the application searches DuckDuckGo and uses the returned content as generation context.

### 💬 Conversational UI
The application uses Streamlit chat components and session state to maintain the visible conversation history.

## 🏗️ Architecture

```text
                    ┌──────────────────────┐
                    │        User          │
                    └──────────┬───────────┘
                               │
                    Upload PDFs / URLs
                               │
                 ┌─────────────▼─────────────┐
                 │      Ingestion Layer      │
                 │                           │
                 │ PyPDF2 / BeautifulSoup    │
                 └─────────────┬─────────────┘
                               │
                         Extracted Text
                               │
                 ┌─────────────▼─────────────┐
                 │         Chunking          │
                 │ RecursiveCharacter...     │
                 └─────────────┬─────────────┘
                               │
                         Text Chunks
                               │
                 ┌─────────────▼─────────────┐
                 │       Embeddings          │
                 │    Gemini Embeddings      │
                 └─────────────┬─────────────┘
                               │
                         Vector Data
                               │
                 ┌─────────────▼─────────────┐
                 │         Qdrant            │
                 │   Persistent Vector DB    │
                 └─────────────┬─────────────┘
                               │
                         User Question
                               │
                 ┌─────────────▼─────────────┐
                 │ Query Embedding + Search  │
                 │      Top-k Retrieval      │
                 └─────────────┬─────────────┘
                               │
                        Retrieved Context
                               │
                 ┌─────────────▼─────────────┐
                 │         Laya Gate         │
                 │ Context sufficient?       │
                 └──────────┬───────┬────────┘
                            │       │
                          YES       NO
                            │       │
                 ┌──────────▼───┐  ┌▼────────────────┐
                 │ Gemini Answer │  │ DuckDuckGo      │
                 │ from context │  │ Web Search      │
                 └───────┬───────┘  └───────┬────────┘
                         │                  │
                         │            Gemini Generation
                         │                  │
                         └─────────┬────────┘
                                   ▼
                              Final Answer
```

## 🔄 RAG Workflow

### 1. Ingestion

```text
PDF / Website
     ↓
Text extraction
     ↓
Cleaning
     ↓
Chunking
```

### 2. Indexing

```text
Text chunks
    ↓
Gemini embedding model
    ↓
3072-dimensional vectors
    ↓
Qdrant collection
```

### 3. Question Answering

```text
User question
      ↓
Question embedding
      ↓
Qdrant semantic search
      ↓
Top 3 relevant chunks
      ↓
Laya context decision
     / \\
   YES  NO
    ↓    ↓
 Gemini  DuckDuckGo
    ↓       ↓
    └─ Gemini ─┘
         ↓
    Final answer
```

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Application runtime |
| **Streamlit** | Web UI and chat interface |
| **PyPDF2** | PDF text extraction |
| **BeautifulSoup4** | HTML parsing and cleaning |
| **LangChain Text Splitters** | Document chunking |
| **Google Gemini Embeddings** | Convert text into vectors |
| **Laya** | Local context-sufficiency decision |
| **Google Gemini Flash** | Answer generation |
| **Qdrant** | Vector storage and semantic retrieval |
| **LiteLLM** | Unified LLM completion interface |
| **DuckDuckGo Search** | Web-search fallback |
| **Requests** | HTTP requests |

## 📁 Project Structure

```text
RAGAgent/
│
├── app.py                 # Main Streamlit application
├── laya_decision.py       # Local Laya context-routing layer
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Files excluded from Git
│
└── qdrant_storage/        # Generated local vector DB (ignored by Git)
```

> `qdrant_storage/` is runtime-generated local data and should normally not be committed to GitHub.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/lalitmakhansingh/RAGAgent.git
cd RAGAgent
```

### 2. Create a virtual environment

Python 3.10 is recommended for the current dependency set.

**Windows:**

```powershell
py -3.10 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

**macOS / Linux:**

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 🔑 API Key

The current implementation uses a **Google Gemini API key**.

Get a key from Google AI Studio:

👉 https://aistudio.google.com/apikey

The application asks for the key directly in the Streamlit interface:

```text
Enter your Gemini API Key:
```

### Security

Never commit API keys to GitHub.

Do not place secrets in:

```text
.env
.streamlit/secrets.toml
source code
README.md
```

## ▶️ Running the Application

Start Streamlit:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## 🧪 How to Use

### Step 1 — Enter your Gemini API key

Paste your Gemini API key into the application.

### Step 2 — Add your knowledge sources

You can either upload one or more PDFs, enter one or more website URLs, or use both together.

Example:

```text
https://example.com
```

Multiple URLs:

```text
https://example.com, https://example.org
```

### Step 3 — Optional website crawling

Enable:

```text
☑ Crawl entire website(s)
```

The application discovers same-domain links from the supplied starting pages and processes the discovered URLs.

### Step 4 — Index the data

Click:

```text
Process and Index Documents
```

The application extracts text, chunks it, generates embeddings, creates the Qdrant collection, and uploads vectors with source metadata.

### Step 5 — Ask questions

Once indexing completes, use the chat input to ask questions about your documents or websites.

## 🧠 Agentic Decision Flow\n\nThe core agentic behavior of RAGAgent is **context-aware routing**.\n\nThe system does not automatically trust the top Qdrant results. It first asks Laya whether those retrieved chunks contain enough information to answer the question without relying on outside knowledge.\n\n### Decision pipeline\n\n```text\nUser Question\n      ↓\nGemini Embedding\n      ↓\nQdrant Semantic Search\n      ↓\nTop-3 Retrieved Chunks\n      ↓\nLaya Context Gate\n      ↓\nP(context sufficient)\n     /              \\n  HIGH                LOW\n   ↓                   ↓\nGemini              DuckDuckGo\nRAG Answer              ↓\n                     Gemini\n                     Web Answer\n   \                   /\n    └──── Final Answer ────┘\n```\n\n### Why Laya?\n\nThe previous version used a Gemini generation call only to answer a binary routing question:\n\n```text\nCan the retrieved context answer the question?\n→ YES / NO\n```\n\nThe current architecture delegates that narrow decision to a local Laya model and keeps Gemini for open-ended answer generation.\n\n```text\nBefore:\nGemini Embedding → Qdrant → Gemini Judge → Gemini Answer\n\nNow:\nGemini Embedding → Qdrant → Laya → Gemini Answer\n                                  ↓\n                             DuckDuckGo\n```\n\n### Laya context gate\n\nThe application uses Laya's `noul` primitive for the decision:\n\n```text\nIs the retrieved context sufficient to answer the user's question\nwithout relying on outside knowledge?\n```\n\nLaya returns a probability for the `true` decision. RAGAgent compares that probability with the routing threshold.\n\nExample:\n\n```text\nP(sufficient) = 0.91\nthreshold     = 0.70\n→ Use retrieved context\n```\n\nFor an insufficient context:\n\n```text\nP(sufficient) = 0.32\nthreshold     = 0.70\n→ Search the web\n```\n\nLaya is loaded once with Streamlit's `st.cache_resource` and reused across Streamlit reruns. The current implementation retains a Gemini-judge fallback if Laya cannot load in the deployment environment.\n\n### Threshold configuration\n\nThe default routing threshold is:\n\n```text\nRAG_DECISION_THRESHOLD=0.70\n```\n\nA higher threshold sends more questions to web search. A lower threshold allows more queries to use retrieved context.\n\nThe `0.70` value is an initial engineering setting and should be evaluated on labelled question/context pairs before production use.\n\n### Web-search fallback\n\nWhen the indexed context is insufficient:\n\n```text\nQuestion\n   ↓\nQdrant Retrieval\n   ↓\nLaya: insufficient\n   ↓\nDuckDuckGo Search\n   ↓\nSearch Result Context\n   ↓\nGemini\n   ↓\nFinal Answer\n```\n\nThis creates a hybrid knowledge architecture:\n\n```text\nIndexed / private knowledge → Qdrant\nMissing / current knowledge → DuckDuckGo\nDecision / routing         → Laya\nAnswer generation          → Gemini\n```\n\n## 🔍 Retrieval Details

The application uses:

- **Chunk size:** 1000
- **Chunk overlap:** 200
- **Top-k retrieval:** 3
- **Vector similarity:** Cosine similarity
- **Vector size configured in Qdrant:** 3072

Conceptually:

```text
Document
   ↓
Chunks
   ↓
Embedding vectors
   ↓
Qdrant
```

Then:

```text
Question
   ↓
Question embedding
   ↓
Similarity search
   ↓
Top 3 chunks
```

## 💾 Local Qdrant Storage

This implementation uses:

```python
QdrantClient(path="qdrant_storage")
```

Therefore, it uses Qdrant's local persistent storage rather than requiring a separate Qdrant HTTP server for this version of the application.

On a later Streamlit session, the application checks whether `qdrant_storage` exists and attempts to reconnect to the `agent_rag_index` collection.

## 🧪 Example\n\nSuppose the user uploads `AI_Research.pdf` and asks:\n\n```text\nWhat is Retrieval-Augmented Generation?\n```\n\nQdrant retrieves:\n\n```text\nChunk 1 → RAG definition\nChunk 2 → retrieval process\nChunk 3 → generation process\n```\n\nLaya evaluates the retrieved context:\n\n```text\nP(sufficient) = 0.94\n```\n\nBecause this is above the threshold:\n\n```text\nRetrieved Context\n      ↓\n     Gemini\n      ↓\nGrounded Answer\n```\n\nNow ask a question that is not represented in the indexed documents:\n\n```text\nWhat is today's market price of NVIDIA?\n```\n\nIf Laya determines that the local context is insufficient:\n\n```text\nQdrant\n  ↓\nLaya\n  ↓\nInsufficient\n  ↓\nDuckDuckGo\n  ↓\nGemini\n  ↓\nAnswer\n```\n\nThis allows RAGAgent to use indexed knowledge when it is sufficient and use web search when it is not.\n\n## 🧩 Core Components

### `process_uploaded_pdfs()`

Responsible for:

```text
PDF → extracted text
```

### `extract_text_from_url()`

Responsible for:

```text
URL → HTML → cleaned text
```

### `get_embeddings()`

Responsible for:

```text
Text → Gemini embedding vector
```

### `process_and_index_documents()`

Responsible for:

```text
Documents
 → chunks
 → embeddings
 → Qdrant collection
```

### `answer_question()`

Responsible for:

```text
Question
 → retrieval
 → context assessment
 → RAG generation / web fallback
```

## ⚠️ Current Limitations

The current implementation is intentionally simple and has several areas that can be improved:

- PDF extraction is text-based and may not correctly handle scanned/image-only PDFs.
- Website extraction is basic HTML extraction rather than a production-grade crawler.
- Link discovery is limited to links found on the supplied starting pages.
- Re-indexing deletes and recreates the `agent_rag_index` collection.
- Embeddings are requested one text at a time.
- API keys are entered through the UI rather than managed through a deployment secret manager.
- The Laya routing threshold is currently an initial heuristic and should be evaluated on labelled data from the target domain.
- The web-search fallback depends on external search availability and rate limits.
- The local Qdrant directory can become large as the dataset grows.

## 🚀 Future Improvements

Possible next iterations:

### Better ingestion
- OCR support for scanned PDFs.
- Better HTML extraction.
- Sitemap-based crawling.
- Duplicate URL detection and normalization.

### Better retrieval
- Hybrid keyword + vector search.
- Metadata filtering.
- Reranking.
- MMR retrieval.
- Adjustable `top_k`.

### Better agents
- Explicit tool calling.
- Multiple specialized tools.
- Iterative agent loops.
- Structured tool outputs.
- Tool execution tracing.

### Better production architecture
- External Qdrant deployment.
- Background indexing jobs.
- Authentication.
- Rate limiting.
- Secret management.
- Observability and evaluation.

## 📌 Learning Value

This project is useful for learning the core building blocks of modern AI applications:

```text
LLM
 ↓
Embeddings
 ↓
Vector Database
 ↓
Semantic Retrieval
 ↓
Context Injection
 ↓
Grounded Generation
 ↓
Tool / Search Fallback
```

It provides a practical foundation for progressing from basic RAG to Agentic AI, tool calling, MCP, multi-agent systems, evaluation, and production AI systems.

## 👨‍💻 Author

**Lalit Makhansingh**

GitHub:

👉 https://github.com/lalitmakhansingh

Repository:

👉 https://github.com/lalitmakhansingh/RAGAgent

## 📄 License

This project is licensed under the MIT License.

## ⭐ Support

If this project helps you understand RAG and Agentic AI, consider giving the repository a ⭐ on GitHub.

---

### Built with Python, Streamlit, Qdrant & Gemini ❤️
