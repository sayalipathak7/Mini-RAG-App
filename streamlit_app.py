import streamlit as st
from data_ingest import load_and_chunk_pdfs, embed_and_store
from retriever import retrieve_top_k
from llm import query_groq

# Set page title displayed on the browser tab
st.set_page_config(page_title="Mini RAG Project")

# Main title displayed at the top of the Streamlit app
st.title("Mini RAG App with LLM Comparison")

# List of available Groq LLM models that users can select from
AVAILABLE_MODELS = [
    "llama3-8b-8192",
    "gemma2-9b-it",
    "llama-3.1-8b-instant"
]

# Cached function to load PDFs, chunk their text, embed, and store embeddings
# Caching ensures this runs once per session unless inputs or code change
@st.cache_resource(show_spinner=True)
def load_and_index_data():
    # List of PDF file paths to ingest and index
    pdf_paths = [
        "data/brain-health-basics.pdf",
        "data/Brain-health-fact-sheet.pdf",
        "data/COVID-19_Brain_Health_Handout.pdf"
    ]
    # Extract chunks of text from the PDFs
    chunks = load_and_chunk_pdfs(pdf_paths)
    # Generate embeddings for chunks and store them in vector DB
    embed_and_store(chunks)
    return "PDFs loaded and ready for querying!"

# Track whether the data has been indexed in the current session
if 'data_indexed' not in st.session_state:
    st.session_state.data_indexed = False

# If data is not yet indexed, show a button to trigger indexing
if not st.session_state.data_indexed:
    if st.button("Load and Process PDFs"):
        status = load_and_index_data()
        st.success(status)  # Show success message after indexing
        st.session_state.data_indexed = True  # Prevent re-indexing during session

# Text input for user to type their question/query
query = st.text_input("Ask your question:")

# Create two columns side-by-side to select two different models for comparison
col1, col2 = st.columns(2)
with col1:
    # Dropdown to select the first model (default is first in list)
    model1 = st.selectbox("Select First Model", AVAILABLE_MODELS, index=0)
with col2:
    # Dropdown to select the second model (default is second in list)
    model2 = st.selectbox("Select Second Model", AVAILABLE_MODELS, index=1)

# When user clicks the button and query is non-empty, run retrieval + query models
if st.button("Get Answer") and query.strip():
    # Show spinner while retrieving relevant document chunks
    with st.spinner("Retrieving relevant documents..."):
        top_chunks = retrieve_top_k(query)  # Retrieve top-k relevant text chunks
        # Join the retrieved chunks with separators to form prompt context
        context = "\n---\n".join(top_chunks)

    # Display retrieved document chunks for user reference
    st.markdown("### Retrieved Document Chunks:")
    for i, chunk in enumerate(top_chunks):
        st.write(f"Chunk {i+1}:")
        st.write(chunk)
        st.markdown("---")

    # Construct the prompt for LLM by injecting retrieved context and user question
    prompt = f"""You are a helpful assistant reading the following document excerpts:
---
{context}
---
Answer the user's question: "{query}"
"""

    # Create two columns to display responses side-by-side for easy comparison
    col1, col2 = st.columns(2)
    with col1:
        # Show spinner while querying the first model
        with st.spinner(f"Querying {model1}..."):
            response1 = query_groq(prompt, model=model1)
        # Display first model's response with a heading
        st.markdown(f"### {model1} Response:")
        st.write(response1)

    with col2:
        # Show spinner while querying the second model
        with st.spinner(f"Querying {model2}..."):
            response2 = query_groq(prompt, model=model2)
        # Display second model's response with a heading
        st.markdown(f"### {model2} Response:")
        st.write(response2)
