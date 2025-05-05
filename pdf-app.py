# app.py

# ─────────────────────────────────────────────────────────────────────────────
# 1) Monkey‑patch sqlite3 BEFORE importing chromadb (unchanged from before)
# ─────────────────────────────────────────────────────────────────────────────
import pysqlite3 as _sqlite3
import sys
sys.modules["sqlite3"] = _sqlite3

# ─────────────────────────────────────────────────────────────────────────────
# 2) Imports
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
from streamlit_lottie import st_lottie, st_lottie_spinner       # for custom spinners :contentReference[oaicite:5]{index=5}
import fitz                                                     # PyMuPDF
import cohere
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# ─────────────────────────────────────────────────────────────────────────────
# 3) Configuration & Global Setup
# ─────────────────────────────────────────────────────────────────────────────
COHERE_API_KEY = "B0n3BcGthprXNg5s4z6BmHdsD2hnH1iLcb5eeWnp"  # store your key in Streamlit secrets
EMBED_MODEL    = "embed-v4.0"
GEN_MODEL      = "command-r-plus-08-2024"
CHUNK_SIZE     = 500

# In‑memory Chroma client (no on‑disk SQLite) :contentReference[oaicite:6]{index=6}
chroma_client = chromadb.EphemeralClient(
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE
)

# ─────────────────────────────────────────────────────────────────────────────
# 4) Utility functions (unchanged logic)
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
    col.add(documents=chunks, embeddings=embeddings,
            ids=[f"chunk_{i}" for i in range(len(chunks))])
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
# 5) Page‑wide CSS for polish (hide menu, add padding) :contentReference[oaicite:7]{index=7}
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .css-1d391kg {padding: 2rem;}  /* body padding */
    </style>
    """,
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# 6) Streamlit App Layout
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 PDF Chatbot (ChromaDB + Cohere)")

# Sidebar for instructions/metrics :contentReference[oaicite:8]{index=8}
with st.sidebar:
    st.header("How to use")
    st.write("1. Upload a PDF\n2. Enter your question\n3. Wait for the answer")
    st.metric("Chunksize", CHUNK_SIZE)
    st.metric("Embed model", EMBED_MODEL)

# Main upload & query area in two columns :contentReference[oaicite:9]{index=9}
col1, col2 = st.columns([1, 2], gap="large")
with col1:
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
with col2:
    user_query = st.text_input("Ask a question about your PDF:")

# ─────────────────────────────────────────────────────────────────────────────
# 7) Processing & Animations
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file and user_query:
    co_client = cohere.Client(COHERE_API_KEY)

    # 7a) File processing spinner
    with st_lottie_spinner("https://assets5.lottiefiles.com/packages/lf20_u4yrau.json", key="loading"):
        text   = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)

    # 7b) Embedding progress bar
    progress = st.progress(0, text="Embedding chunks…")
    embs    = []
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        embs.extend(co_client.embed(texts=chunks[i: i+batch_size], model=EMBED_MODEL).embeddings)
        progress.progress(min((i+batch_size)/len(chunks), 1.0))
    progress.empty()

    # 7c) Build & store vectors
    collection = create_vector_store(chunks, embs)

    # 7d) Answer generation spinner
    with st_lottie_spinner("https://assets3.lottiefiles.com/packages/lf20_jtbfg2nb.json", key="answering"):
        q_emb   = co_client.embed(texts=[user_query], model=EMBED_MODEL).embeddings[0]
        top_ctx = get_top_chunks(collection, q_emb)
        prompt  = build_prompt(top_ctx, user_query)
        answer  = generate_answer(co_client, prompt)

    # 7e) Show balloons and display answer :contentReference[oaicite:10]{index=10}
    st.balloons()
    st.markdown("### 💬 Answer")
    st.write(answer)

    with st.expander("📚 Context chunks used"):
        for i, c in enumerate(top_ctx, 1):
            st.markdown(f"**Chunk {i}:** {c}")
