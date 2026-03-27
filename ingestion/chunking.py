import hashlib
import chromadb

from dataclasses import dataclass
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from collections import Counter


# ---------------------------
# Config
# ---------------------------
@dataclass
class RAGConfig:
    db_path: str = "./database"
    collection_name: str = "rag_collection"
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # embed_model_name: str = "BAAI/bge-base-en-v1.5"
    chunk_size: int = 256
    chunk_overlap: int = 40
    min_chunk_length: int = 30
    similarity_top_k: int = 5
    overlap_ratio: float = 0.15    # ← add this — 15% of actual chunk size


# ---------------------------
# Step 1: ChromaDB setup
# ---------------------------
def get_chroma_store(cfg: RAGConfig):
    chroma_client = chromadb.PersistentClient(path=cfg.db_path)
    collection = chroma_client.get_or_create_collection(
        name=cfg.collection_name
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return vector_store, collection


# ---------------------------
# Step 2: Embedding model
# ---------------------------
def get_embed_model(cfg: RAGConfig):
    return HuggingFaceEmbedding(model_name=cfg.embed_model_name)


# ---------------------------
# Step 3: Chunking
# ---------------------------
def create_nodes(text: str, source_name: str, cfg: RAGConfig) -> list[TextNode]:

    if not text or not text.strip():
        print(f"  ⚠️  '{source_name}': Empty text. Skipping.")
        return []

    # Count approximate tokens (words / 0.75 is a rough token estimate)
    approx_tokens = len(text.split()) / 0.75

    # If document is small, reduce chunk size proportionally
    if approx_tokens < cfg.chunk_size:
        actual_chunk_size = max(int(approx_tokens / 3), cfg.min_chunk_length)
        actual_overlap = max(int(actual_chunk_size * cfg.overlap_ratio), 10)
    else:
        actual_chunk_size = cfg.chunk_size
        actual_overlap = cfg.chunk_overlap

    splitter = SentenceSplitter(
        chunk_size=actual_chunk_size,
        chunk_overlap=actual_overlap,
    )

    chunks = splitter.split_text(text)
    chunks = [c for c in chunks if len(c.strip()) >= cfg.min_chunk_length]

    if not chunks:
        print(f"  ⚠️  '{source_name}': No valid chunks after filtering.")
        return []

    print(f"  ℹ️  Chunk size used: {actual_chunk_size} | Overlap: {actual_overlap}")

    nodes = []
    for i, chunk in enumerate(chunks):
        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
        node = TextNode(
            id_=f"{source_name}__chunk_{i}__{chunk_hash[:8]}",
            text=chunk,
            metadata={
                "source": source_name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_hash": chunk_hash,
            },
        )
        nodes.append(node)

    return nodes


# ---------------------------
# Step 4: Deduplication
# ---------------------------
def filter_existing_nodes(nodes: list[TextNode], collection) -> list[TextNode]:
    if collection.count() == 0:
        return nodes

    try:
        result = collection.get(ids=[n.id_ for n in nodes])
        existing_ids = set(result["ids"])
    except Exception:
        existing_ids = set()
    new_nodes = [n for n in nodes if n.id_ not in existing_ids]

    skipped = len(nodes) - len(new_nodes)
    if skipped > 0:
        print(f"  ⚠️  Skipped {skipped} duplicate chunk(s) already in DB.")

    return new_nodes


# ---------------------------
# Step 5: Build index
# ---------------------------
def build_index(
    text: str,
    source_name: str,
    cfg: RAGConfig,
    embed_model,
    vector_store,
    collection,
) -> VectorStoreIndex:

    nodes = create_nodes(text, source_name, cfg)

    if not nodes:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context, embed_model=embed_model
        )

    print(f"  📄 '{source_name}': {len(nodes)} chunks created.")

    nodes = filter_existing_nodes(nodes, collection)
    print(f"  ✅ {len(nodes)} new chunk(s) to insert.")

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if not nodes:
        print("  ℹ️  Nothing new to insert. Loading existing index.")
        return VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context, embed_model=embed_model
        )

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    return index


# ---------------------------
# Step 6: Inspect database
# ---------------------------
def inspect_database(cfg: RAGConfig) -> None:
    _, collection = get_chroma_store(cfg)
    total = collection.count()
    print(f"\n📦 ChromaDB '{cfg.collection_name}': {total} chunk(s) stored.")

    if total == 0:
        print("  (empty)")
        return

    all_data = collection.get(include=["documents", "metadatas"])

    # Per source count — taken from GPT version, genuinely useful
    source_counts = Counter(
        m.get("source", "unknown") for m in all_data["metadatas"]
    )
    print("\n📊 Chunks per source:")
    for src, count in sorted(source_counts.items()):
        print(f"   {src:<40} {count:>4} chunk(s)")

    # Sample chunks
    print("\n📋 Sample chunks (up to 3):")
    for i, (doc, meta) in enumerate(
        zip(all_data["documents"][:3], all_data["metadatas"][:3])
    ):
        print(f"\n  [{i+1}] {meta.get('source')} "
              f"(chunk {meta.get('chunk_index')}/{meta.get('total_chunks')})")
        print(f"       {doc[:160].strip()}...")