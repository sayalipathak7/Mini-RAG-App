from sentence_transformers import SentenceTransformer  # For generating sentence embeddings
import chromadb  # Vector database to store and query embeddings
from chromadb.config import Settings

# Load the pre-trained embedding model (MiniLM variant)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize a Chroma client with telemetry disabled (for privacy)
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

# Get or create a collection named "rag-demo" to store chunks and embeddings
collection = chroma_client.get_or_create_collection(name="rag-demo")

def retrieve_top_k(query, k=3):
    """
    Retrieves the top-k most relevant document chunks for a given query
    by embedding the query and querying the vector database.

    Args:
        query (str): The user input query string.
        k (int): Number of top results to retrieve (default is 3).

    Returns:
        List[str]: The top-k document chunks relevant to the query.
    """

    # Fetch all stored documents, embeddings, and metadata from the collection
    data = collection.get(include=['documents', 'embeddings', 'metadatas'])

    # Print the first 3 stored embeddings and their associated document chunks for debugging
    for i in range(min(3, len(data['ids']))):
        print(f"ID: {data['ids'][i]}")
        print(f"Document chunk: {data['documents'][i][:200]}...")  # Print first 200 characters
        print(f"Embedding vector (first 10 dims): {data['embeddings'][i][:10]}")  # Preview embedding vector
        print(f"Metadata: {data['metadatas'][i] if 'metadatas' in data else 'No metadata'}")
        print('---')

    # Generate embedding vector for the input query using the same embedding model
    query_embedding = embedding_model.encode([query])[0].tolist()

    # Query the collection for top-k closest embeddings to the query embedding
    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    # Extract the list of document chunks returned by the query
    documents = results["documents"][0]

    # Print the top-k retrieved document chunks for debugging
    print(f"\n Top {k} Retrieved Chunks:\n")
    for i, doc in enumerate(documents):
        print(f"--- Chunk {i+1} ---\n{doc}\n")

    # Return the list of retrieved document chunks to the caller
    return documents
