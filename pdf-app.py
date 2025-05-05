# app.py

# ─────────────────────────────────────────────────────────────────────────────
# 1) Swap out the built‑in sqlite3 for pysqlite3 (>=3.35) BEFORE chromadb import
# ─────────────────────────────────────────────────────────────────────────────
__import__("pysqlite3")       # noqa: E402
import sys                      # noqa: E402
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# ─────────────────────────────────────────────────────────────────────────────
# 2) Now it’s safe to import ChromaDB without version errors
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import fitz                         # PyMuPDF
import cohere
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# ─────────────────────────────────────────────────────────────────────────────
# 3) Cohere + model config
# ─────────────────────────────────────────────────────────────────────────────
COHERE_API_KEY = "B0n3BcGthprXNg5s4z6BmHdsD2hnH1iLcb5eeWnp"
EMBED_MODEL   = "embed-v4.0"
GEN_MODEL     = "command-r-plus-08-2024"
CHUNK_SIZE    = 500

# ─────────────────────────────────────────────────────────────────────────────
# 4) Initialize an **in‑memory** Chroma client via EphemeralClient
# ─────────────────────────────────────────────────────────────────────────────
chroma_client = chromadb.EphemeralClient(
    settings=Settings(), 
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE
)

# ─────────────────────────────────────────────────────────────────────────────
# 5) Helper functions for PDF → chunks → embeddings → retrieval → answer
# ─────────────────────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    return [" ".join(words[i:i+chunk_size])
            for i in range(0, len(words), chunk_size)]

def create_vector_store(chunks, embeddings):
    # reset existing collection
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
# 6) Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 PDF Chatbot (ChromaDB + Cohere)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
user_query    = st.text_input("Ask a question about your PDF:")

if uploaded_file and user_query:
    co_client = cohere.Client(COHERE_API_KEY)

    with st.spinner("📖 Reading & embedding PDF..."):
        text   = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)
        embs   = co_client.embed(texts=chunks, model=EMBED_MODEL).embeddings
        collection = create_vector_store(chunks, embs)

    with st.spinner("🤖 Generating answer..."):
        q_emb   = co_client.embed(texts=[user_query], model=EMBED_MODEL).embeddings[0]
        top_ctx = get_top_chunks(collection, q_emb)
        prompt  = build_prompt(top_ctx, user_query)
        answer  = generate_answer(co_client, prompt)

    st.markdown("### 💬 Answer")
    st.write(answer)

    with st.expander("📚 Context chunks used"):
        for i, c in enumerate(top_ctx, 1):
            st.markdown(f"**Chunk {i}:** {c}")
