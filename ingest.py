from file_loader.file_loader import load_document
from file_loader.text_cleaner import clean_text
from ingestion.chunking import (
    RAGConfig,
    get_chroma_store,
    get_embed_model,
    build_index,
    inspect_database
)

def main():
    # --- Step 1: Load the document ---
    print("📂 Loading document...")
    doc = load_document("data_file/sample.pdf")

    if not doc["content"].strip():
        print("❌ No text extracted from document. Check your PDF.")
        return

    print(f"✅ Loaded: {doc['file_name']}")
    print(f"   Pages : {doc['metadata'].get('total_pages', 'N/A')}")

    # --- Step 2: Clean the text ---
    print("\n🧹 Cleaning text...")
    cleaned_text = clean_text(doc["content"])
    print(f"✅ Cleaning done. Characters after cleaning: {len(cleaned_text)}")

    # --- Step 3: Setup ChromaDB and embedding model ---
    print("\n⚙️  Setting up ChromaDB and embedding model...")
    cfg = RAGConfig()                               # create config first
    vector_store, collection = get_chroma_store(cfg)   # pass cfg
    embed_model = get_embed_model(cfg)                 # pass cfg
    print("✅ Setup complete.")

    # --- Step 4: Chunk, embed and store ---
    print("\n🔄 Starting ingestion...")
    index = build_index(
        text=cleaned_text,
        source_name=doc["file_name"],
        cfg=cfg,
        embed_model=embed_model,
        vector_store=vector_store,
        collection=collection,
    )

    if index is None:
        print("❌ Ingestion failed. Check errors above.")
        return

    # --- Step 5: Verify what got stored ---
    print("\n🔍 Inspecting database...")
    inspect_database(cfg)

if __name__ == "__main__":
    main()
# ```

# ---

# Fix those two things and then run it. Since ChromaDB already has 132 chunks from the previous run, you should see:
# ```
# ⚠️  Skipped 132 duplicate chunk(s) already in DB.
# ✅ 0 new chunk(s) to insert.
# ℹ️  Nothing new to insert. Loading existing index.

# 📦 ChromaDB 'rag_collection': 132 chunk(s) stored.

# 📊 Chunks per source:
#    sample.pdf                               132 chunk(s)