import streamlit as st
import fitz  # PyMuPDF
import cohere
import chromadb
from chromadb.config import Settings

# ========== CONFIG ==========
PDF_PATH = "Demo.pdf"
COHERE_API_KEY = "your-real-api-key"
EMBED_MODEL = "embed-v4.0"
GEN_MODEL = "command-r-plus-08-2024"
CHUNK_SIZE = 500

# ========== SAFE CHROMA CLIENT ==========
chroma_client = chromadb.Client(Settings(
    chroma_api_impl="chromadb.api.local.LocalAPI",
    persist_directory=None,  # disable persistence
    allow_reset=True
))

# ========== UTILS ==========
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def create_vector_store(chunks, embeddings):
    if "pdf_chunks" in [col.name for col in chroma_client.list_collections()]:
        chroma_client.delete_collection("pdf_chunks")
    collection = chroma_client.create_collection(name="pdf_chunks")
    collection.add(documents=chunks, embeddings=embeddings, ids=[f"chunk_{i}" for i in range(len(chunks))])
    return collection

def get_top_chunks(collection, query_embedding, top_k=3):
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results['documents'][0]

def build_prompt(chunks, user_query):
    context = "\n".join(f"- {chunk}" for chunk in chunks)
    return f"Context:\n{context}\n\nQuestion:\n{user_query}"

def generate_answer(co, prompt):
    response = co.generate(model=GEN_MODEL, prompt=prompt, max_tokens=600)
    return response.generations[0].text.strip()

# ========== STREAMLIT APP ==========
st.title("📄 PDF Chat App with ChromaDB")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
user_query = st.text_input("Ask a question:")

if uploaded_file and user_query:
    co = cohere.Client(COHERE_API_KEY)

    with st.spinner("Reading and processing PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(pdf_text)
        embeddings = co.embed(texts=chunks, model=EMBED_MODEL).embeddings
        collection = create_vector_store(chunks, embeddings)

    with st.spinner("Getting answer..."):
        query_embedding = co.embed(texts=[user_query], model=EMBED_MODEL).embeddings[0]
        top_chunks = get_top_chunks(collection, query_embedding)
        prompt = build_prompt(top_chunks, user_query)
        answer = generate_answer(co, prompt)
        st.markdown("### 💬 Answer")
        st.write(answer)
