# 🤖 RAGAgent — Agentic Retrieval-Augmented Generation

> A Streamlit-based Agentic RAG application for asking questions over PDF documents and websites, with semantic retrieval through Qdrant, Gemini embeddings and generation, and an online-search fallback when retrieved context is not relevant.

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
- 🧠 Use an LLM-based relevance check before generating an answer from retrieved context.
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
                 │      Gemini Judge         │
                 │ Relevant context?         │
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
Gemini relevance check
     / \\
   YES  NO
    ↓    ↓
 Gemini  DuckDuckGo
    ↓    ↓
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

## 🧠 Agentic Decision Flow

The application first retrieves the most similar chunks from Qdrant.

It then asks Gemini whether the retrieved context contains relevant information.

### Relevant context found

```text
Question
   ↓
Qdrant retrieval
   ↓
Relevant context
   ↓
Gemini generation
   ↓
Answer grounded in retrieved context
```

### Relevant context not found

```text
Question
   ↓
Qdrant retrieval
   ↓
Context judged irrelevant
   ↓
DuckDuckGo search
   ↓
Search results as context
   ↓
Gemini generation
   ↓
Answer
```

If the external search path also encounters an exception, the current code contains additional Gemini fallback handling.

## 💡 Laya Decision Layer

The previous version used a Gemini generation call to answer a binary routing question:

```text
Can the retrieved context answer the user's question?
→ 1 / 0
```

The current version moves that decision to a local Laya model:

```text
Question
   ↓
Qdrant top-k retrieval
   ↓
Laya (noul)
   ↓
P(sufficient)
  /       \
YES       NO
 ↓         ↓
Gemini   DuckDuckGo
answer     ↓
         Gemini
         answer
```

Laya is loaded once with Streamlit's `st.cache_resource` and reused across requests. The routing threshold defaults to `0.70` and can be changed with:

```text
RAG_DECISION_THRESHOLD=0.70
```

The retrieved context is compacted before the Laya decision so it stays within the decision model's context window. The dedicated typed-decisions checkpoint provides a 1,024-token context window.

The Gemini generation path is intentionally unchanged: Gemini still produces the grounded answer after the router selects the RAG path or the DuckDuckGo fallback.
## 🔍 Retrieval Details

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

## 🧪 Example

Suppose you upload:

```text
AI_Research.pdf
```

and ask:

```text
What is Retrieval-Augmented Generation?
```

The system performs:

```text
Question
   ↓
Embedding
   ↓
Qdrant search
   ↓
Relevant chunks from AI_Research.pdf
   ↓
Gemini relevance decision
   ↓
Gemini answer
```

If you instead ask something that is not represented in the indexed material, the application can move to the online search fallback.

## 🧩 Core Components

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
- The relevance decision is based on an LLM response parsed as `1/0` or `yes/no`.
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
