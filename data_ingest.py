import fitz  # PyMuPDF for reading PDFs
import chromadb  # Vector DB for storing embeddings
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer  # Embedding model
import nltk  # Natural language toolkit for tokenization
from nltk.tokenize import sent_tokenize, word_tokenize

# Download necessary tokenizers for sentence and word splitting
nltk.download('punkt')

# Load the pre-trained sentence transformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB client with telemetry disabled
# Default is in-memory storage; for persistence, configure persist_directory
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

# Create or get a collection named "rag-demo" to store documents and embeddings
collection = chroma_client.get_or_create_collection(name="rag-demo")

def nltk_sentence_word_chunk(text, min_words=200, max_words=300):
    """
    Splits input text into chunks with size roughly between min_words and max_words.
    Uses sentence tokenization to avoid splitting sentences across chunks.
    Each chunk is formed by concatenating sentences until the max_words limit is reached.
    """
    sentences = sent_tokenize(text)  # Split text into sentences
    chunks = []  # List to hold resulting chunks
    current_chunk = []  # Sentences accumulating for current chunk
    current_length = 0  # Number of words in current chunk

    for sentence in sentences:
        sentence_words = word_tokenize(sentence)  # Tokenize sentence into words
        sentence_len = len(sentence_words)  # Count words in current sentence

        # If adding this sentence exceeds max_words, close off current chunk
        if current_length + sentence_len > max_words:
            if current_chunk:
                # Join sentences in current chunk and add to chunks list
                chunks.append(' '.join(current_chunk))
            # Start a new chunk with current sentence
            current_chunk = [sentence]
            current_length = sentence_len
        else:
            # Otherwise, add sentence to current chunk and update length
            current_chunk.append(sentence)
            current_length += sentence_len

    # Add the last chunk if any sentences remain
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def load_and_chunk_pdfs(pdf_paths):
    """
    Given a list of PDF file paths, extracts text from each,
    splits the text into chunks using nltk_sentence_word_chunk,
    and returns a combined list of all chunks from all PDFs.
    """
    all_chunks = []
    for path in pdf_paths:
        print(f"\n Parsing PDF: {path}")
        doc = fitz.open(path)  # Open PDF file
        # Extract text from all pages and join into one string
        text = "\n".join([page.get_text() for page in doc])
        print(f"\nExtracted raw text (first 500 chars):\n{text[:500]}")

        chunks = nltk_sentence_word_chunk(text)  # Chunk the extracted text
        print(f"Number of chunks: {len(chunks)}")
        print(f"Sample chunk 0 (word count: {len(chunks[0].split())}):\n{chunks[0]}\n")

        all_chunks.extend(chunks)  # Collect all chunks

    return all_chunks

def embed_and_store(chunks):
    """
    Given a list of text chunks, generates embeddings for each using the
    embedding model, then stores the chunk and its embedding in ChromaDB collection.
    """
    # Generate embeddings for all chunks in batches (shows progress bar)
    embeddings = embedding_model.encode(chunks, batch_size=16, show_progress_bar=True)

    # Add each chunk and its corresponding embedding to the ChromaDB collection
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[f"chunk_{i}"],  # Unique ID for each chunk
            documents=[chunk],  # The actual text chunk
            embeddings=[embedding.tolist()]  # Convert embedding numpy array to list
        )
