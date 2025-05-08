# app.py

# ─────────────────────────────────────────────────────────────────────────────
# 0) Monkey‑patch sqlite3 BEFORE importing chromadb
# ─────────────────────────────────────────────────────────────────────────────
import pysqlite3 as _sqlite3
import sys
sys.modules["sqlite3"] = _sqlite3

# ─────────────────────────────────────────────────────────────────────────────
# 1) Imports & Immediate Page Config
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
st.set_page_config(page_title="PDF Chatbot", layout="wide")  # must be first Streamlit command

import fitz                         # PyMuPDF
import cohere
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# ─────────────────────────────────────────────────────────────────────────────
# 2) Configuration & Global Setup
# ─────────────────────────────────────────────────────────────────────────────
COHERE_API_KEY = "B0n3BcGthprXNg5s4z6BmHdsD2hnH1iLcb5eeWnp"
EMBED_MODEL    = "embed-v4.0"
GEN_MODEL      = "command-r-plus-08-2024"
CHUNK_SIZE     = 500

chroma_client = chromadb.EphemeralClient(
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE
)

# ─────────────────────────────────────────────────────────────────────────────
# 3) Utility functions
# ─────────────────────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    return [" ".join(words[i:i+chunk_size])
            for i in range(0, len(words), chunk_size)]

def create_vector_store(chunks, embeddings):
    if "pdf_chunks" in [c.name for c in chroma_client.list_collections()]:
        chroma_client.delete_collection("pdf_chunks")
    col = chroma_client.create_collection(name="pdf_chunks")
    col.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return col

def get_top_chunks(collection, query_embedding, top_k=3):
    res = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return res["documents"][0]

def build_prompt(chunks, question):
    ctx = "\n".join(f"- {c}" for c in chunks)
    return f"Context:\n{ctx}\n\nQuestion:\n{question}"

def generate_answer(co, prompt):
    resp = co.generate(model=GEN_MODEL, prompt=prompt, max_tokens=600)
    return resp.generations[0].text.strip()

# ─────────────────────────────────────────────────────────────────────────────
# 4) Page Style (clean spacing + hide footer/menu)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }
        .stTextInput > div > div > input {
            height: 3rem;
            font-size: 1.1rem;
        }
        .stFileUploader > div {
            padding: 1rem 0;
        }
        .stMarkdown h3 {
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# 5) UI Layout
# ─────────────────────────────────────────────────────────────────────────────
st.title("📄 PDF Chatbot with Cohere + ChromaDB")

# Instructions and metadata
with st.sidebar:
    st.header("🛠️ How to Use")
    st.markdown("1. Upload a **PDF file**\n2. Ask a **question**\n3. View the **answer** below.")
    st.divider()
    st.metric("Chunk Size", CHUNK_SIZE)
    st.metric("Embed Model", EMBED_MODEL)
    st.metric("Gen Model", GEN_MODEL)

# Upload + Question Input - Side by side and centered
st.markdown("### 📥 Upload and Ask")
input_col1, input_col2 = st.columns([1, 2], gap="large")
with input_col1:
    uploaded_file = st.file_uploader("Upload your PDF here:", type="pdf")
with input_col2:
    user_query = st.text_input("Ask a question about the PDF:")

# ─────────────────────────────────────────────────────────────────────────────
# 6) Logic
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file and user_query:
    co_client = cohere.Client(COHERE_API_KEY)

    with st.spinner("🔍 Extracting and chunking your PDF..."):
        text   = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)

    progress = st.progress(0, text="📊 Embedding PDF chunks...")
    embs = []
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        embs.extend(co_client.embed(texts=chunks[i: i+batch_size], model=EMBED_MODEL).embeddings)
        progress.progress(min((i+batch_size)/len(chunks), 1.0))
    progress.empty()

    with st.spinner("🧠 Creating vector store..."):
        collection = create_vector_store(chunks, embs)

    with st.spinner("💡 Generating your answer..."):
        q_emb   = co_client.embed(texts=[user_query], model=EMBED_MODEL).embeddings[0]
        top_ctx = get_top_chunks(collection, q_emb)
        prompt  = build_prompt(top_ctx, user_query)
        answer  = generate_answer(co_client, prompt)

    # Final Output Section
    st.markdown("### 💬 Answer")
    st.success(answer)

    with st.expander("📚 View Retrieved Context Chunks"):
        for i, c in enumerate(top_ctx, 1):
            st.markdown(f"**Chunk {i}:**\n{c}")
