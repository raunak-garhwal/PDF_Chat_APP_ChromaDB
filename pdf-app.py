import fitz  # PyMuPDF
import cohere
import chromadb
import streamlit as st
from chromadb.config import Settings

# ========== CONFIG ==========
PDF_PATH = "Demo.pdf"
COHERE_API_KEY = "B0n3BcGthprXNg5s4z6BmHdsD2hnH1iLcb5eeWnp"
EMBED_MODEL = "embed-v4.0"
GEN_MODEL = "command-r-plus-08-2024"
CHUNK_SIZE = 500
PERSIST_DIR = "./chroma_store"

# ========== INIT ==========
chroma_client = chromadb.Client(Settings(persist_directory=PERSIST_DIR))
co = cohere.Client(COHERE_API_KEY)

# ========== FUNCTIONS ==========
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def create_vector_store(chunks, embeddings):
    if "pdf_chunks" in [col.name for col in chroma_client.list_collections()]:
        chroma_client.delete_collection("pdf_chunks")
    collection = chroma_client.create_collection(name="pdf_chunks")
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection

def get_top_chunks(collection, query_embedding, top_k=3):
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results['documents'][0]

def build_prompt(chunks, user_query):
    context = "\n".join(f"- {chunk}" for chunk in chunks)
    return f"Context:\n{context}\n\nQuestion:\n{user_query}"

def generate_answer(prompt):
    response = co.generate(model=GEN_MODEL, prompt=prompt, max_tokens=600)
    return response.generations[0].text.strip()

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 PDF Chatbot using Cohere + ChromaDB")

with st.sidebar:
    st.header("Step 1: Upload PDF")
    uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf")

# Session state for storing Chroma collection
if "collection" not in st.session_state:
    st.session_state.collection = None

# Process PDF
if uploaded_pdf is not None:
    with st.spinner("📖 Reading and processing PDF..."):
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_pdf.read())

        full_text = extract_text_from_pdf("uploaded.pdf")
        chunks = chunk_text(full_text)
        embeddings = co.embed(texts=chunks, model=EMBED_MODEL).embeddings
        collection = create_vector_store(chunks, embeddings)
        st.session_state.collection = collection
        st.success("✅ PDF processed and embedded!")

    st.subheader("Ask a question about your PDF")
    user_query = st.text_input("Your question:")

    if user_query and st.session_state.collection:
        with st.spinner("🤖 Thinking..."):
            query_embedding = co.embed(texts=[user_query], model=EMBED_MODEL).embeddings[0]
            top_chunks = get_top_chunks(st.session_state.collection, query_embedding)
            prompt = build_prompt(top_chunks, user_query)
            answer = generate_answer(prompt)
            st.markdown("### 💬 Answer")
            st.write(answer)

        with st.expander("📚 Top matched chunks (context used)"):
            for i, chunk in enumerate(top_chunks):
                st.markdown(f"**Chunk {i+1}:** {chunk}")
