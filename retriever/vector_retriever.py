import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ingestion.chunking import RAGConfig


# ---------------------------
# Step 1: Load existing index from ChromaDB
# ---------------------------
def load_index(cfg: RAGConfig) -> VectorStoreIndex:
    """
    Reconnect to existing ChromaDB and load the index.
    Does not re-embed anything — just connects to what is already stored.
    """
    chroma_client = chromadb.PersistentClient(path=cfg.db_path)
    collection = chroma_client.get_or_create_collection(name=cfg.collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = HuggingFaceEmbedding(model_name=cfg.embed_model_name)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    print(f"✅ Index loaded — {collection.count()} chunks available.")
    return index


# ---------------------------
# Step 2: Retrieve relevant chunks
# ---------------------------
def retrieve_chunks(query: str, index: VectorStoreIndex, cfg: RAGConfig) -> list:
    """
    Embed the query and return top-k most relevant chunks from ChromaDB.

    Args:
        query : user's natural language question
        index : loaded VectorStoreIndex
        cfg   : RAGConfig with top_k setting

    Returns:
        List of NodeWithScore objects — each has text, score and metadata
    """
    if not query or not query.strip():
        print("  ⚠️  Empty query. Skipping.")
        return []

    retriever = index.as_retriever(similarity_top_k=cfg.similarity_top_k)
    results = retriever.retrieve(query)
    return results


# ---------------------------
# Step 3: Display results
# ---------------------------
def display_results(results: list, query: str) -> None:
    """Print retrieved chunks in readable format for testing."""

    print(f"\n🔍 Query   : {query}")
    print(f"📚 Retrieved: {len(results)} chunk(s)\n")

    for i, node in enumerate(results):
        print(f"--- Chunk {i + 1} ---")
        print(f"Source     : {node.metadata.get('source', 'unknown')}")
        print(f"Chunk index: {node.metadata.get('chunk_index')} / {node.metadata.get('total_chunks')}")
        print(f"Score      : {node.score:.4f}")
        print(f"Text       : {node.text[:300]}")
        print()


# ---------------------------
# Test it directly
# ---------------------------
if __name__ == "__main__":
    cfg = RAGConfig()

    # Step 1: load index
    index = load_index(cfg)

    # Step 2: run a query
    query = "How does the 'Lubrication System' contribute to both the longevity and the temperature management of the engine?"
    results = retrieve_chunks(query, index, cfg)

    # Step 3: display
    display_results(results, query)