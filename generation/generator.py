"""
generator.py
Connects retrieved chunks to Gemini and generates a final answer.

Flow:
    vector_retriever.py  → returns NodeWithScore objects
    convert_nodes()      → converts to clean dict format
    Generator.generate() → filters, builds prompt, calls Gemini
    returns answer + sources
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()   # loads GEMINI_API_KEY from .env file automatically


# ---------------------------
# Config
# ---------------------------
@dataclass
class GeneratorConfig:
    api_key: str = ""
    model_name: str = "gemini-2.5-flash"
    max_output_tokens: int = 1024
    temperature: float = 0.2
    min_score_threshold: float = 0.35   # lowered slightly for MiniLM scores
    min_chunks_required: int = 1

    def __post_init__(self):
        # Load from .env if not provided directly
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY", "")


# ---------------------------
# Convert NodeWithScore → dict
# This bridges vector_retriever output to generator input
# ---------------------------
def convert_nodes_to_chunks(nodes: list) -> list[dict]:
    """
    Converts LlamaIndex NodeWithScore objects from vector_retriever
    into plain dicts that generator can work with.

    Args:
        nodes: list of NodeWithScore from retriever.retrieve()

    Returns:
        list of dicts with keys: text, score, source, chunk_index
    """
    chunks = []
    for node in nodes:
        chunks.append({
            "text":        node.node.text,
            "score":       round(node.score, 4),
            "source":      node.node.metadata.get("source", "unknown"),
            "chunk_index": node.node.metadata.get("chunk_index", -1),
            "total_chunks": node.node.metadata.get("total_chunks", -1),
        })
    return chunks


# ---------------------------
# Score filter
# ---------------------------
def filter_chunks_by_score(chunks: list[dict], min_score: float) -> list[dict]:
    """Remove chunks below minimum similarity score."""
    filtered = [c for c in chunks if c.get("score", 0.0) >= min_score]

    removed = len(chunks) - len(filtered)
    if removed > 0:
        print(f"  ⚠️  Filtered out {removed} chunk(s) below score {min_score}.")

    return filtered


# ---------------------------
# Prompt builder
# ---------------------------
def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Builds the full prompt sent to Gemini.
    Numbered chunks make it easy to trace which chunk the answer came from.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Chunk {i} | Source: {chunk.get('source')} | "
            f"Score: {chunk.get('score', 0):.2f}]\n"
            f"{chunk.get('text', '').strip()}"
        )

    context_block = "\n\n".join(context_parts)

    prompt = f"""You are a precise document assistant. Answer questions using ONLY the context provided below.

STRICT RULES:
- Use ONLY the information from the context below to answer.
- If the context does not contain enough information say exactly: "I could not find enough information in the document to answer this."
- Do NOT use your own general knowledge.
- Do NOT make up or guess any facts.
- Keep your answer clear and concise.
- At the end of your answer mention which chunk number(s) you used.

─────────────────────────────
CONTEXT FROM DOCUMENT:
─────────────────────────────
{context_block}

─────────────────────────────
QUESTION:
─────────────────────────────
{question}

─────────────────────────────
ANSWER:
─────────────────────────────"""

    return prompt


# ---------------------------
# Generator class
# ---------------------------
class Generator:
    """
    Takes retrieved chunks and a question, returns Gemini's answer.

    Usage in main.py:
        from generation.generator import Generator, GeneratorConfig, convert_nodes_to_chunks

        gen = Generator()

        # convert retriever output to chunk dicts
        chunks = convert_nodes_to_chunks(retrieved_nodes)

        # generate answer
        answer, sources = gen.generate(question, chunks)
        print(answer)
    """

    def __init__(self, cfg: GeneratorConfig = None):
        self.cfg = cfg or GeneratorConfig()
        self._setup_gemini()

    def _setup_gemini(self):
        if not self.cfg.api_key:
            raise ValueError(
                "\n❌ Gemini API key not found.\n"
                "   Add GEMINI_API_KEY=your_key to your .env file at project root.\n"
            )

        genai.configure(api_key=self.cfg.api_key)

        self.model = genai.GenerativeModel(
            model_name=self.cfg.model_name,
            generation_config=genai.GenerationConfig(
                max_output_tokens=self.cfg.max_output_tokens,
                temperature=self.cfg.temperature,
            )
        )
        print(f"  ✅ Gemini connected — model: {self.cfg.model_name}")

    def generate(
        self,
        question: str,
        chunks: list[dict],
    ) -> tuple[str, list[dict]]:
        """
        Generate an answer from question and retrieved chunks.

        Args:
            question : user's natural language question
            chunks   : list of dicts from convert_nodes_to_chunks()

        Returns:
            answer  : Gemini's answer as string
            sources : list of chunks that were actually used
        """

        print(f"\n{'─'*50}")
        print(f"  🔍 Question : {question}")
        print(f"  📦 Chunks received: {len(chunks)}")

        # Step 1 — filter low score chunks
        filtered = filter_chunks_by_score(chunks, self.cfg.min_score_threshold)

        # Step 2 — check if enough chunks remain
        if len(filtered) < self.cfg.min_chunks_required:
            msg = "I could not find relevant information in the document to answer this question."
            print("  ❌ Not enough relevant chunks. Returning fallback.")
            return msg, []

        print(f"  ✅ {len(filtered)} chunk(s) passed score filter.")

        # Step 3 — build prompt
        prompt = build_prompt(question, filtered)

        # Step 4 — call Gemini
        print(f"  🤖 Sending to Gemini ({self.cfg.model_name})...")

        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            print(f"  ✅ Answer received ({len(answer)} characters).")
            return answer, filtered      # return answer AND sources used

        except Exception as e:
            error_msg = f"❌ Gemini API error: {str(e)}"
            print(f"  {error_msg}")
            return error_msg, []

    def show_prompt(self, question: str, chunks: list[dict]) -> None:
        """Print full prompt for debugging bad answers."""
        filtered = filter_chunks_by_score(chunks, self.cfg.min_score_threshold)
        prompt = build_prompt(question, filtered)
        print("\n" + "═"*60)
        print("DEBUG — FULL PROMPT SENT TO GEMINI:")
        print("═"*60)
        print(prompt)
        print("═"*60 + "\n")


# ---------------------------
# Test generator standalone
# ---------------------------
if __name__ == "__main__":
    from retriever.vector_retriever import load_index, retrieve_chunks
    from ingestion.chunking import RAGConfig

    cfg = RAGConfig()
    index = load_index(cfg)

    question = "What is ReTreever and how does it organize documents?"

    # Get real chunks from your actual ChromaDB
    nodes = retrieve_chunks(question, index, cfg)

    # Convert to dict format
    chunks = convert_nodes_to_chunks(nodes)

    # Generate answer
    gen = Generator()
    answer, sources = gen.generate(question, chunks)

    print("\n" + "═"*50)
    print("FINAL ANSWER:")
    print("═"*50)
    print(answer)

    print("\n" + "═"*50)
    print("SOURCES USED:")
    print("═"*50)
    for i, chunk in enumerate(sources):
        print(f"\n[{i+1}] Source: {chunk['source']} | "
              f"Chunk: {chunk['chunk_index']} | "
              f"Score: {chunk['score']}")
        print(f"     {chunk['text'][:150]}...")
# ```

# ---

# ## What changed and why

# | What | Original | Improved |
# |---|---|---|
# | Node conversion | Missing — would crash in main.py | `convert_nodes_to_chunks()` added |
# | API key loading | Complex `field(default_factory=lambda...)` | Simple `__post_init__` + dotenv |
# | Model name | `gemini-1.5-flash` (old) | `gemini-2.0-flash` (current) |
# | Return value | Only answer string | Answer + sources used |
# | Score threshold | 0.40 (too high for MiniLM) | 0.35 (fits MiniLM score range) |
# | Test in `__main__` | Fake hardcoded chunks | Uses real chunks from your ChromaDB |
# | Config separation | Separate from RAGConfig | `__post_init__` keeps it clean |

# ---

# ## Before running — add this to your `.env` file

# Create a `.env` file at your project root if you do not have one:
# ```
# GEMINI_API_KEY=your_actual_key_here
# ```

# And install dotenv:
# ```
# pip install python-dotenv
# ```

# Also add to `requirements.txt`:
# ```
# python-dotenv
# google-generativeai
# ```

# ---

# ## Also create `generation/` folder with `__init__.py`

# Your folder structure needs this:
# ```
# generation/
# ├── __init__.py      ← empty file
# └── generator.py     ← the code above
# ```

# Once you have done all that run:
# ```
# python -m generation.generator