import streamlit as st
import PyPDF2
import json
import requests
import os
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from litellm import completion
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from laya_decision import judge_context

if "client" not in st.session_state:
    st.session_state.client = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Auto-reconnect to persistent Qdrant database if it exists from a previous session
if st.session_state.client is None and os.path.exists("qdrant_storage"):
    try:
        temp_client = QdrantClient(path="qdrant_storage")
        if temp_client.collection_exists("agent_rag_index"):
            st.session_state.client = temp_client
            st.session_state.collection_name = "agent_rag_index"
    except Exception:
        pass


def get_all_urls(base_url):
    urls = set()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    try:
        response = requests.get(base_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for link in soup.find_all("a", href=True):
                url = link["href"]
                full_url = urljoin(base_url, url)
                parsed_url = urlparse(full_url)
                if parsed_url.netloc == urlparse(base_url).netloc:
                    urls.add(
                        parsed_url.scheme + "://" + parsed_url.netloc + parsed_url.path
                    )
    except Exception as e:
        st.error(f"An error occurred while crawling {base_url}: {e}")
    return urls


def extract_text_from_url(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()

            lines = (line.strip() for line in text.splitlines())

            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

            text = " ".join(chunk for chunk in chunks if chunk)

            return text
        else:
            st.warning(
                f"Failed to fetch content from {url}: Status code {response.status_code}"
            )
            return None
    except Exception as e:
        st.warning(f"Error extracting text from {url}: {e}")
        return None


def fetch_url_content(url: str) -> Optional[str]:
    try:
        return extract_text_from_url(url)
    except Exception as e:
        st.error(f"Error: Failed to fetch URL {url}. Exception: {e}")
        return None


def get_embeddings(texts, model="gemini-embedding-2", api_key=None):
    if isinstance(texts, str):
        texts = [texts]
        
    all_embeddings = []
    
    # We will use the embedContent endpoint as it is standard and supported by these models
    headers = {"Content-Type": "application/json"}
    
    for text in texts:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={api_key}"
        payload = {
            "model": f"models/{model}",
            "content": {"parts": [{"text": text}]}
        }
        
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            res_json = response.json()
            # The API returns 'embedding' dict containing a 'values' array
            all_embeddings.append({"embedding": res_json.get("embedding", {}).get("values", [])})
        else:
            # Let's try the older model if the newer one fails
            if model == "gemini-embedding-2":
                return get_embeddings(texts, model="gemini-embedding-001", api_key=api_key)
            st.error(f"Error fetching embeddings: {response.text}")
            return None
            
    return all_embeddings


def process_uploaded_pdfs(uploaded_files):
    pdf_list = []
    for uploaded_file in uploaded_files:
        content = ""
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                content += page.extract_text()
            pdf_list.append({"content": content, "filename": uploaded_file.name})
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    return pdf_list


def process_and_index_documents(
    uploaded_files, web_urls=None, chunk_size=1000, crawl_website=False
):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o-mini",
        chunk_size=chunk_size,
        chunk_overlap=200,
    )

    all_chunks = []
    doc_metadata = []

    if uploaded_files:
        all_documents = process_uploaded_pdfs(uploaded_files)
        for doc in all_documents:
            chunks = text_splitter.split_text(doc["content"])
            all_chunks.extend(chunks)
            for _ in chunks:
                doc_metadata.append(
                    {"filename": doc["filename"], "source": "pdf_dataset"}
                )

    if web_urls:
        urls = [url.strip() for url in web_urls.split(",")]

        if crawl_website:
            all_urls = set()
            progress_bar = st.progress(0)
            progress_text = st.empty()

            for i, base_url in enumerate(urls):
                progress_text.text(f"Crawling website: {base_url}")
                site_urls = get_all_urls(base_url)
                all_urls.update(site_urls)
                progress_bar.progress((i + 1) / len(urls))

            urls = list(all_urls)
            progress_text.text(f"Found {len(urls)} unique URLs")

        progress_bar = st.progress(0)
        progress_text = st.empty()

        for i, url in enumerate(urls):
            progress_text.text(f"Processing URL {i+1}/{len(urls)}: {url}")
            content = fetch_url_content(url)

            if content is not None:
                chunks = text_splitter.split_text(content)
                all_chunks.extend(chunks)
                for _ in chunks:
                    doc_metadata.append({"url": url, "source": "web_content"})

            progress_bar.progress((i + 1) / len(urls))
            time.sleep(0.5)

        progress_text.empty()
        progress_bar.empty()

    if not all_chunks:
        st.error("No content to process. Please provide valid PDFs or web URLs.")
        return None, None

    api_key = st.session_state.gemini_api_key

    with st.spinner("Generating embeddings..."):
        embeddings_objects = get_embeddings(all_chunks, api_key=api_key)
        if not embeddings_objects:
            return None, None
        embeddings = [obj["embedding"] for obj in embeddings_objects]

    if st.session_state.get("client") is not None:
        client = st.session_state.client
    else:
        client = QdrantClient(path="qdrant_storage")
        
    collection_name = "agent_rag_index"
    VECTOR_SIZE = 3072

    with st.spinner("Creating vector database..."):
        client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

        ids = list(range(len(all_chunks)))
        payload = [
            {"content": chunk, "metadata": metadata}
            for chunk, metadata in zip(all_chunks, doc_metadata)
        ]

        client.upload_collection(
            collection_name=collection_name,
            vectors=embeddings,
            payload=payload,
            ids=ids,
            batch_size=256,
        )

    st.success(
        f"Indexed {len(all_chunks)} chunks from {len(set(m['source'] for m in doc_metadata))} different sources"
    )
    return client, collection_name


def answer_question(question, client, collection_name, top_k=3):
    if not question.strip():
        st.warning("Please enter a question.")
        return

    def search(text: str):
        query_embedding = get_embeddings(text, api_key=st.session_state.gemini_api_key)[
            0
        ]["embedding"]
        return client.search(
            collection_name=collection_name, query_vector=query_embedding, limit=top_k
        )

    def format_docs(docs):
        formatted_chunks = []
        for doc in docs:
            source_info = ""
            if doc.payload["metadata"]["source"] == "pdf_dataset":
                source_info = (
                    f"\nSource: PDF file {doc.payload['metadata']['filename']}"
                )
            else:
                source_info = f"\nSource: Web article {doc.payload['metadata']['url']}"
            formatted_chunks.append(doc.payload["content"] + source_info)
        return "\n\n".join(formatted_chunks)

    with st.spinner("Searching for relevant information..."):
        results = search(question)
        context = format_docs(results)

        # Laya performs the binary routing decision locally, replacing
        # the Gemini call that was previously used only as a judge.
        try:
            decision = judge_context(question, context)
            has_answer = decision["sufficient"]

            st.caption(
                f"Laya context gate: P(sufficient)={decision['probability']:.2f} "
                f"(threshold={decision['threshold']:.2f}, confidence={decision['confidence']:.2f})"
            )
        except Exception as laya_error:
            # Preserve the original behavior if the local Laya model cannot
            # load or execute in a constrained deployment environment.
            st.warning(
                f"Laya decision layer unavailable; falling back to Gemini judge. "
                f"Reason: {laya_error}"
            )
            decision_system_prompt = """Decide whether the given question can be answered, even partially, using the given context.
            Return 1 when the context contains enough relevant information for a grounded answer.
            Return 0 when it does not.
            Context: {context}
            """

            judge_response = completion(
                model="gemini/gemini-3.6-flash",
                messages=[
                    {
                        "content": decision_system_prompt.format(context=context),
                        "role": "system",
                    },
                    {
                        "content": user_prompt.format(question=question),
                        "role": "user",
                    },
                ],
                api_key=st.session_state.gemini_api_key,
            )
            has_answer = "1" in (
                judge_response.choices[0].message.content.strip().lower()
            )

        if has_answer:
            st.info("Found relevant information in your uploaded documents/URLs!")
            response = completion(
                model="gemini/gemini-3.6-flash",
                messages=[
                    {
                        "content": system_prompt.format(context=context),
                        "role": "system",
                    },
                    {"content": user_prompt.format(question=question), "role": "user"},
                ],
                api_key=st.session_state.gemini_api_key,
            )
            return response.choices[0].message.content
        else:
            st.info("No relevant information found in documents. Searching online...")
            try:
                results = DDGS().text(question, max_results=5)
                if not results:
                    st.warning("Could not find any online sources for this query.")
                    return "Could not find an answer."
                context = "\n\n".join(doc["body"] for doc in results)
                st.info("Found online sources. Generating the response...")
                response = completion(
                    model="gemini/gemini-3.6-flash",
                    messages=[
                        {
                            "content": system_prompt.format(context=context),
                            "role": "system",
                        },
                        {"content": user_prompt.format(question=question), "role": "user"},
                    ],
                    api_key=st.session_state.gemini_api_key,
                )
                return response.choices[0].message.content
            except Exception as e:
                st.warning("The external search provider (DuckDuckGo) is temporarily rate-limited due to high traffic. Please wait a few moments before attempting another online query.")
                msgs = [
                    {
                        "content": "You are a helpful AI assistant. Answer the question relying on your own general knowledge, as external search is currently unavailable.",
                        "role": "system",
                    },
                    {"content": question, "role": "user"},
                ]
                try:
                    fallback_response = completion(
                        model="gemini/gemini-3.6-flash",
                        messages=msgs,
                        api_key=st.session_state.gemini_api_key,
                    )
                    return fallback_response.choices[0].message.content
                except Exception as api_e:
                    st.warning("The primary Gemini model is experiencing high demand. Retrying with Gemini 2.0 Flash...")
                    try:
                        fallback_response = completion(
                            model="gemini/gemini-2.0-flash",
                            messages=msgs,
                            api_key=st.session_state.gemini_api_key,
                        )
                        return fallback_response.choices[0].message.content
                    except Exception as final_e:
                        return f"Google APIs are severely congested right now. Please try again in a few moments. Details: {final_e}"


st.title("RAG System with PDF and Website Crawling Support")

api_key = st.text_input("Enter your Gemini API Key:", type="password")
if api_key:
    st.session_state.gemini_api_key = api_key

uploaded_files = st.file_uploader(
    "Upload PDF files:", accept_multiple_files=True, type=["pdf"]
)

st.subheader("Website Input")
web_urls = st.text_input(
    "Enter website URLs (comma-separated):", placeholder="https://example.com"
)
crawl_website = st.checkbox(
    "Crawl entire website(s)",
    help="Enable this to extract content from all pages of the specified website(s)",
)

if st.button("Process and Index Documents"):
    if not st.session_state.get("gemini_api_key"):
        st.error("Please enter your Gemini API key first.")
    else:
        st.session_state.client, st.session_state.collection_name = (
            process_and_index_documents(
                uploaded_files, web_urls, crawl_website=crawl_website
            )
        )

if st.session_state.client and st.session_state.collection_name:
    st.subheader("Chat Session")
    
    # Display existing chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Chat input
    if question := st.chat_input("Ask a question about the documents:"):
        # Add user question to history
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
            
        # Get and display assistant response
        with st.chat_message("assistant"):
            answer = answer_question(
                question, st.session_state.client, st.session_state.collection_name
            )
            if answer:
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
elif uploaded_files or web_urls:
    st.warning("Please process and index the documents first.")
else:
    st.info("Upload PDFs or provide web URLs to get started.")
