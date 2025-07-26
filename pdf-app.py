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
st.set_page_config(
    page_title="AI PDF Assistant", 
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

import fitz                         # PyMuPDF
import cohere
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
import time
from datetime import datetime

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
    try:
        doc = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    return [" ".join(words[i:i+chunk_size])
            for i in range(0, len(words), chunk_size)]

def create_vector_store(chunks, embeddings):
    try:
        # Check if collection exists and delete it
        existing_collections = [c.name for c in chroma_client.list_collections()]
        if "pdf_chunks" in existing_collections:
            chroma_client.delete_collection("pdf_chunks")
        
        col = chroma_client.create_collection(name="pdf_chunks")
        col.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )
        return col
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_top_chunks(collection, query_embedding, top_k=3):
    try:
        res = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return res["documents"][0] if res["documents"] else []
    except Exception as e:
        st.error(f"Error querying vector store: {str(e)}")
        return []

def build_prompt(chunks, question):
    ctx = "\n".join(f"- {c}" for c in chunks)
    return f"Context:\n{ctx}\n\nQuestion:\n{question}"

def generate_answer(co, prompt):
    try:
        resp = co.generate(model=GEN_MODEL, prompt=prompt, max_tokens=600)
        return resp.generations[0].text.strip()
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "Sorry, I couldn't generate an answer. Please try again."

def format_file_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names)-1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"

# ─────────────────────────────────────────────────────────────────────────────
# 4) Enhanced Modern Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Hide Streamlit branding */
        #MainMenu, footer, .stDeployButton {visibility: hidden;}
        
        /* Global font */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Main container */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 1200px !important;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        }
        
        .main-header h1 {
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .main-header p {
            font-size: 1.2rem !important;
            opacity: 0.9;
            margin: 0 !important;
        }
        
        /* Card styling */
        .custom-card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            border: 1px solid #e1e5e9;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }
        
        .custom-card:hover {
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }
        
        /* Upload area styling */
        .upload-area {
            border: 2px dashed #667eea;
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            border-color: #764ba2;
            background: linear-gradient(135deg, #f0f4ff 0%, #e8f0ff 100%);
        }
        
        /* File uploader styling */
        .stFileUploader > label {
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            color: #2d3748 !important;
        }
        
        /* Input styling */
        .stTextInput > div > div > input {
            height: 3.5rem !important;
            font-size: 1.1rem !important;
            border-radius: 12px !important;
            border: 2px solid #e2e8f0 !important;
            padding: 0 1rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        }
        
        .stTextInput > label {
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            color: #2d3748 !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #f8f9ff 0%, #ffffff 100%);
        }
        
        .sidebar-metric {
            background: white;
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            border: 1px solid #e2e8f0;
        }
        
        .sidebar-metric-title {
            font-size: 0.9rem;
            color: #718096;
            font-weight: 500;
            margin-bottom: 0.25rem;
        }
        
        .sidebar-metric-value {
            font-size: 1.1rem;
            color: #2d3748;
            font-weight: 600;
        }
        
        /* Status and progress styling */
        .stAlert {
            border-radius: 12px !important;
            border: none !important;
        }
        
        .stSuccess {
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%) !important;
            color: white !important;
        }
        
        .stInfo {
            background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%) !important;
            color: white !important;
        }
        
        /* Progress bar */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background: #f7fafc !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }
        
        /* Answer box styling */
        .answer-box {
            background: linear-gradient(135deg, #e6fffa 0%, #f0fff4 100%);
            border: 1px solid #9ae6b4;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(72, 187, 120, 0.1);
        }
        
        .answer-box h3 {
            color: #22543d !important;
            margin-top: 0 !important;
            margin-bottom: 1rem !important;
        }
        
        /* Chat-like styling for Q&A */
        .question-bubble {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 20px 20px 5px 20px;
            margin: 1rem 0 0.5rem auto;
            max-width: 80%;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .answer-bubble {
            background: white;
            color: #2d3748;
            padding: 1rem 1.5rem;
            border-radius: 20px 20px 20px 5px;
            margin: 0.5rem auto 1rem 0;
            max-width: 85%;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            border: 1px solid #e2e8f0;
        }
        
        /* File info styling */
        .file-info {
            background: linear-gradient(135deg, #faf5ff 0%, #f7fafc 100%);
            border: 1px solid #d6bcfa;
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .file-info-icon {
            font-size: 1.5rem;
            color: #805ad5;
        }
        
        .file-info-text {
            flex: 1;
        }
        
        .file-name {
            font-weight: 600;
            color: #553c9a;
            margin-bottom: 0.25rem;
        }
        
        .file-details {
            font-size: 0.9rem;
            color: #718096;
        }
        
        /* Processing animation */
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .processing {
            animation: pulse 2s infinite;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 2rem !important;
            }
            .main-header p {
                font-size: 1rem !important;
            }
            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# 5) Enhanced UI Layout
# ─────────────────────────────────────────────────────────────────────────────

# Header Section
st.markdown("""
<div class="main-header">
    <h1>🤖 AI PDF Assistant</h1>
    <p>Upload your PDF and get intelligent answers instantly</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🛠️ Configuration")
    
    # Model settings in cards
    st.markdown("""
    <div class="sidebar-metric">
        <div class="sidebar-metric-title">Embedding Model</div>
        <div class="sidebar-metric-value">""" + EMBED_MODEL + """</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-metric">
        <div class="sidebar-metric-title">Language Model</div>
        <div class="sidebar-metric-value">""" + GEN_MODEL + """</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-metric">
        <div class="sidebar-metric-title">Chunk Size</div>
        <div class="sidebar-metric-value">""" + str(CHUNK_SIZE) + """ words</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📋 How to Use")
    st.markdown("""
    **Step 1:** Upload your PDF document
    
    **Step 2:** Wait for processing to complete
    
    **Step 3:** Ask any question about the content
    
    **Step 4:** Get instant AI-powered answers
    """)
    
    st.markdown("---")
    
    if "chunks" in st.session_state:
        st.markdown("### 📊 Document Stats")
        st.markdown(f"**Chunks Created:** {len(st.session_state.chunks)}")
        if "last_filename" in st.session_state:
            st.markdown(f"**Current File:** {st.session_state.last_filename}")
            
        # Add reset button
        if st.button("🔄 Process New Document", help="Clear current document and upload a new one"):
            st.session_state.clear()
            st.rerun()

# Main Content Area
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
    <div class="custom-card">
        <h3 style="margin-top: 0; color: #2d3748;">📄 Document Upload</h3>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose your PDF file",
        type="pdf",
        help="Select a PDF document to analyze"
    )
    
    if uploaded_file:
        file_size = len(uploaded_file.getvalue())
        st.markdown(f"""
        <div class="file-info">
            <div class="file-info-icon">📄</div>
            <div class="file-info-text">
                <div class="file-name">{uploaded_file.name}</div>
                <div class="file-details">Size: {format_file_size(file_size)} • Type: PDF</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="custom-card">
        <h3 style="margin-top: 0; color: #2d3748;">❓ Ask Your Question</h3>
    """, unsafe_allow_html=True)
    
    user_query = st.text_input(
        "What would you like to know?",
        placeholder="e.g., What is the main topic of this document?",
        help="Ask any question about your uploaded PDF"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6) Enhanced Processing Logic
# ─────────────────────────────────────────────────────────────────────────────

if uploaded_file:
    # Check if new file uploaded
    if "last_filename" not in st.session_state or uploaded_file.name != st.session_state.last_filename:
        st.session_state.clear()
        st.session_state.last_filename = uploaded_file.name

    # Process PDF if not already processed
    if "chunks" not in st.session_state:
        try:
            co_client = cohere.Client(COHERE_API_KEY)
            
            # Extract text
            with st.spinner("🔍 Extracting text from PDF..."):
                text = extract_text_from_pdf(uploaded_file)
                if not text.strip():
                    st.error("⚠️ Could not extract text from PDF. Please ensure the PDF contains readable text.")
                    st.stop()
                
                chunks = chunk_text(text)
                if not chunks:
                    st.error("⚠️ No content found to process. Please check your PDF file.")
                    st.stop()
                
                st.session_state.chunks = chunks
                time.sleep(1)  # Small delay for better UX
            
            st.success(f"✅ Successfully extracted text and created {len(chunks)} chunks!")
            
            # Embed chunks with progress
            st.markdown("### 🧠 Processing Document Intelligence")
            
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            embs = []
            batch_size = 50
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            for batch_idx, i in enumerate(range(0, len(chunks), batch_size)):
                current_batch = chunks[i:i+batch_size]
                progress_text.text(f"Processing batch {batch_idx + 1} of {total_batches}...")
                
                try:
                    batch_embs = co_client.embed(texts=current_batch, model=EMBED_MODEL).embeddings
                    embs.extend(batch_embs)
                except Exception as e:
                    st.error(f"Error during embedding: {str(e)}")
                    st.stop()
                
                progress = (batch_idx + 1) / total_batches
                progress_bar.progress(progress)
                
            progress_text.empty()
            progress_bar.empty()
            st.session_state.embs = embs
            
            # Create vector store
            with st.spinner("💾 Building knowledge base..."):
                collection = create_vector_store(chunks, embs)
                if collection is None:
                    st.error("⚠️ Failed to create knowledge base. Please try again.")
                    st.stop()
                st.session_state.collection = collection
                time.sleep(1)
            
            st.success("🎉 Document successfully processed! You can now ask questions.")
            
        except Exception as e:
            st.error(f"⚠️ An error occurred during processing: {str(e)}")
            st.session_state.clear()  # Clear session state on error

# Handle question answering
if uploaded_file and user_query and "collection" in st.session_state:
    try:
        co_client = cohere.Client(COHERE_API_KEY)
        
        # Create conversation-like interface
        st.markdown("---")
        st.markdown("### 💭 Conversation")
        
        # Display question
        st.markdown(f"""
        <div class="question-bubble">
            <strong>You asked:</strong><br>
            {user_query}
        </div>
        """, unsafe_allow_html=True)
        
        # Process and display answer
        with st.spinner("🤔 Thinking..."):
            q_emb = co_client.embed(texts=[user_query], model=EMBED_MODEL).embeddings[0]
            top_ctx = get_top_chunks(st.session_state.collection, q_emb)
            
            if not top_ctx:
                st.warning("⚠️ No relevant content found. Please try rephrasing your question.")
            else:
                prompt = build_prompt(top_ctx, user_query)
                answer = generate_answer(co_client, prompt)
                time.sleep(1)  # Small delay for better UX
                
                # Display answer in chat bubble
                st.markdown(f"""
                <div class="answer-bubble">
                    <strong>🤖 AI Assistant:</strong><br>
                    {answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Context sources in an expander
                with st.expander("📚 View Source Context", expanded=False):
                    st.markdown("**Relevant passages from your document:**")
                    for i, chunk in enumerate(top_ctx, 1):
                        st.markdown(f"""
                        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #667eea;">
                            <strong>Source {i}:</strong><br>
                            {chunk[:300]}{'...' if len(chunk) > 300 else ''}
                        </div>
                        """, unsafe_allow_html=True)
                        
    except Exception as e:
        st.error(f"⚠️ An error occurred while processing your question: {str(e)}")
        st.info("💡 Please try rephrasing your question or uploading the document again.")

# Help section when no file is uploaded
if not uploaded_file:
    st.markdown("---")
    st.markdown("""
    <div class="custom-card" style="text-align: center; background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);">
        <h3 style="color: #553c9a;">🚀 Get Started</h3>
        <p style="font-size: 1.1rem; color: #718096; margin-bottom: 1.5rem;">
            Upload a PDF document to begin your AI-powered document analysis experience
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📄</div>
                <div style="font-weight: 600; color: #2d3748;">Upload PDF</div>
                <div style="font-size: 0.9rem; color: #718096;">Any PDF document</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧠</div>
                <div style="font-weight: 600; color: #2d3748;">AI Processing</div>
                <div style="font-size: 0.9rem; color: #718096;">Intelligent chunking</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">💬</div>
                <div style="font-weight: 600; color: #2d3748;">Ask Questions</div>
                <div style="font-size: 0.9rem; color: #718096;">Get instant answers</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #718096; font-size: 0.9rem; margin-top: 2rem;">
    <p>Powered by Cohere AI • Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
