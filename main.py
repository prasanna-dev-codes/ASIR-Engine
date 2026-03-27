"""
main.py
───────
Entry point for the RAG pipeline.

Run this file to ask questions against your documents.

Usage:
    python main.py

Make sure you have run ingest.py at least once before running this.
"""

import time
from retriever.vector_retriever import load_index, retrieve_chunks
from generation.generator import Generator, GeneratorConfig, convert_nodes_to_chunks
from ingestion.chunking import RAGConfig


# ---------------------------
# Conversation history
# ---------------------------

class ConversationHistory:
    """
    Keeps track of previous questions and answers in the session.
    This lets Gemini understand followup questions like
    'Can you explain that in simpler terms?' or 'Tell me more about point 2.'
    """

    def __init__(self, max_turns: int = 5):
        self.history = []
        self.max_turns = max_turns   # only keep last N turns to avoid huge prompts

    def add(self, question: str, answer: str) -> None:
        self.history.append({
            "question": question,
            "answer": answer
        })
        # Keep only last max_turns conversations
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_context(self) -> str:
        """Format history as a readable string to prepend to prompts."""
        if not self.history:
            return ""

        parts = []
        for i, turn in enumerate(self.history):
            parts.append(
                f"Previous Q{i+1}: {turn['question']}\n"
                f"Previous A{i+1}: {turn['answer'][:300]}..."
            )

        return "CONVERSATION HISTORY (for context only):\n" + "\n\n".join(parts)

    def is_empty(self) -> bool:
        return len(self.history) == 0

    def clear(self) -> None:
        self.history = []
        print("  🗑️  Conversation history cleared.")


# ---------------------------
# Display helpers
# ---------------------------

def display_answer(
    question: str,
    answer: str,
    sources: list[dict],
    retrieval_time: float,
    generation_time: float,
) -> None:
    """Print answer, sources and timing stats in a clean format."""

    print("\n" + "═" * 60)
    print("  💬 ANSWER")
    print("═" * 60)
    print(answer)

    # Sources
    print("\n" + "─" * 60)
    print("  📚 SOURCES USED")
    print("─" * 60)

    if not sources:
        print("  No sources cited.")
    else:
        for i, chunk in enumerate(sources):
            print(f"\n  [{i+1}] File  : {chunk['source']}")
            print(f"       Chunk : {chunk['chunk_index']} of {chunk['total_chunks']}")
            print(f"       Score : {chunk['score']}")
            print(f"       Text  : {chunk['text'][:200]}...")

    # Timing stats
    print("\n" + "─" * 60)
    print("  ⏱️  STATS")
    print("─" * 60)
    print(f"  Retrieval time : {retrieval_time:.2f}s")
    print(f"  Generation time: {generation_time:.2f}s")
    print(f"  Total time     : {retrieval_time + generation_time:.2f}s")
    print("═" * 60)


def display_startup_banner() -> None:
    print("\n" + "═" * 60)
    print("  🤖 RAG PIPELINE")
    print("  Ask questions about your documents.")
    print("  ─────────────────────────────────")
    print("  Commands:")
    print("    'exit' or 'quit'  → stop the pipeline")
    print("    'history'         → show conversation history")
    print("    'clear'           → clear conversation history")
    print("    'sources'         → toggle showing sources on/off")
    print("    'stats'           → toggle showing timing stats on/off")
    print("═" * 60)


# ---------------------------
# Single query — full pipeline
# ---------------------------

def ask(
    question: str,
    index,
    generator: Generator,
    rag_cfg: RAGConfig,
    history: ConversationHistory,
) -> tuple[str, list[dict], float, float]:
    """
    Full pipeline for a single question.

    Flow:
        question + history
            ↓
        vector_retriever  → finds relevant chunks  [timed]
            ↓
        convert_nodes     → converts to dict format
            ↓
        generator         → sends to Gemini        [timed]
            ↓
        answer + sources + timings

    Note:
        When router is added later, replace retrieve_chunks()
        call here with router.route(question, index, rag_cfg)
        Everything else stays the same.
    """

    # Step 1 — retrieve chunks (timed)
    t0 = time.time()
    nodes = retrieve_chunks(question, index, rag_cfg)
    retrieval_time = time.time() - t0

    if not nodes:
        return "No relevant chunks found in the database.", [], retrieval_time, 0.0

    # Step 2 — convert to dict format
    chunks = convert_nodes_to_chunks(nodes)

    # Step 3 — build question with conversation context if history exists
    question_with_context = question
    if not history.is_empty():
        history_context = history.get_context()
        question_with_context = (
            f"{history_context}\n\n"
            f"CURRENT QUESTION:\n{question}"
        )

    # Step 4 — generate answer (timed)
    t0 = time.time()
    answer, sources = generator.generate(question_with_context, chunks)
    generation_time = time.time() - t0

    # Step 5 — save to history (original question, not the one with context)
    history.add(question, answer)

    return answer, sources, retrieval_time, generation_time


# ---------------------------
# Interactive loop
# ---------------------------

def interactive_mode(
    index,
    generator: Generator,
    rag_cfg: RAGConfig,
) -> None:

    display_startup_banner()

    history = ConversationHistory(max_turns=5)
    show_sources = True     # toggle with 'sources' command
    show_stats = True       # toggle with 'stats' command

    while True:
        print()
        try:
            question = input("  ❓ Your question: ").strip()
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\n  👋 Interrupted. Goodbye!")
            break

        # --- Empty input ---
        if not question:
            print("  ⚠️  Please type a question.")
            continue

        # --- Commands ---
        if question.lower() in ["exit", "quit", "q"]:
            print("\n  👋 Exiting RAG pipeline. Goodbye!")
            break

        if question.lower() == "history":
            if history.is_empty():
                print("  ℹ️  No conversation history yet.")
            else:
                print("\n  📜 CONVERSATION HISTORY")
                print("  ─" * 30)
                for i, turn in enumerate(history.history):
                    print(f"\n  Q{i+1}: {turn['question']}")
                    print(f"  A{i+1}: {turn['answer'][:200]}...")
            continue

        if question.lower() == "clear":
            history.clear()
            continue

        if question.lower() == "sources":
            show_sources = not show_sources
            state = "ON" if show_sources else "OFF"
            print(f"  ℹ️  Sources display turned {state}.")
            continue

        if question.lower() == "stats":
            show_stats = not show_stats
            state = "ON" if show_stats else "OFF"
            print(f"  ℹ️  Stats display turned {state}.")
            continue

        # --- Run pipeline ---
        try:
            answer, sources, retrieval_time, generation_time = ask(
                question, index, generator, rag_cfg, history
            )

            # Display answer always
            print("\n" + "═" * 60)
            print("  💬 ANSWER")
            print("═" * 60)
            print(answer)

            # Display sources if enabled
            if show_sources and sources:
                print("\n" + "─" * 60)
                print("  📚 SOURCES USED")
                print("─" * 60)
                for i, chunk in enumerate(sources):
                    print(f"\n  [{i+1}] File  : {chunk['source']}")
                    print(f"       Chunk : {chunk['chunk_index']} "
                          f"of {chunk['total_chunks']}")
                    print(f"       Score : {chunk['score']}")
                    print(f"       Text  : {chunk['text'][:200]}...")

            # Display stats if enabled
            if show_stats:
                print("\n" + "─" * 60)
                print("  ⏱️  STATS")
                print("─" * 60)
                print(f"  Retrieval : {retrieval_time:.2f}s")
                print(f"  Generation: {generation_time:.2f}s")
                print(f"  Total     : {retrieval_time + generation_time:.2f}s")

            print("═" * 60)

        except Exception as e:
            # Show error but do NOT exit — let user ask another question
            print(f"\n  ❌ Error: {str(e)}")
            print("  ℹ️  You can try a different question or type 'exit' to quit.")


# ---------------------------
# Main
# ---------------------------

def main():
    print("🚀 Starting RAG pipeline...")
    print("─" * 60)

    # Load configs
    rag_cfg = RAGConfig()
    gen_cfg = GeneratorConfig()

    # Load ChromaDB index
    print("\n📦 Loading index from ChromaDB...")
    try:
        index = load_index(rag_cfg)
    except Exception as e:
        print(f"❌ Failed to load index: {e}")
        print("   Make sure you have run ingest.py first.")
        return

    # Setup Gemini
    print("\n🤖 Setting up Gemini...")
    try:
        generator = Generator(gen_cfg)
    except ValueError as e:
        print(e)
        return

    # Start interactive loop
    interactive_mode(index, generator, rag_cfg)


if __name__ == "__main__":
    main()
# ```

# ---

# ## What improved and why

# **Conversation history** — the biggest addition. Now if you ask "What is ReTreever?" and follow up with "How is it different from other methods?", Gemini sees the previous question and answer as context. Without this, every question is completely isolated and followup questions do not work properly.

# **Timing stats** — shows exactly how long retrieval took vs generation. Useful for your project demo and for comparing retrievers later. You will clearly see that keyword retrieval is faster than vector retrieval for example.

# **Toggle commands** — `sources` and `stats` can be turned on and off during the session. Useful when you want clean output without source noise during a demo.

# **History command** — type `history` to see all previous questions and answers in the current session.

# **Clear command** — type `clear` to reset conversation history when switching topics.

# **Ctrl+C handling** — if you press Ctrl+C to stop the program it exits gracefully instead of showing a Python traceback error.

# **Error handling without exit** — if Gemini fails on one question the program does not crash. It shows the error and lets you ask another question. Previously any error would kill the whole session.

# **Router placeholder comment** — inside `ask()` there is a clear comment showing exactly where the router goes when you build it. You will replace one line and everything else stays the same.

# ---

# ## Session will now look like this
# ```
# 🚀 Starting RAG pipeline...

# ════════════════════════════════════════════════════════════
#   🤖 RAG PIPELINE
#   Ask questions about your documents.
#   ─────────────────────────────────
#   Commands:
#     'exit' or 'quit'  → stop the pipeline
#     'history'         → show conversation history
#     'clear'           → clear conversation history
#     'sources'         → toggle showing sources on/off
#     'stats'           → toggle showing timing stats on/off
# ════════════════════════════════════════════════════════════

#   ❓ Your question: What is ReTreever?

#   💬 ANSWER
# ════════════════════════════════════════════════════════════
# ReTreever is a tree-based method...

#   ⏱️  STATS
# ────────────────────────────────────────────────────────────
#   Retrieval : 0.18s
#   Generation: 2.34s
#   Total     : 2.52s

#   ❓ Your question: How is it different from other methods?

#   💬 ANSWER  ← Gemini now remembers the previous question
# ════════════════════════════════════════════════════════════
# Unlike other hierarchical methods which rely on LLM calls...