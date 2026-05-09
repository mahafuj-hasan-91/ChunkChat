"""
rag/chunker.py
──────────────
Handles splitting large documents into smaller, overlapping chunks.

RAG Concept: "Text Chunking"
  An LLM has a limited "context window" — it can only read a fixed number
  of tokens at once (e.g., 8,192 tokens for llama3-8b-8192). A 50-page PDF
  might have 50,000+ tokens — far too large to fit in one prompt.

  Solution: Split the document into small, overlapping chunks.
  - chunk_size=500  → Each chunk ≈ 500 characters (~100–125 tokens)
  - chunk_overlap=50 → Consecutive chunks share 50 characters of text,
                        so information at chunk boundaries isn't lost.

  During retrieval, we only send the top-3 most relevant chunks to the LLM,
  keeping the prompt small while still providing focused context.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[str]:
    """
    Split raw text into overlapping chunks using LangChain's splitter.

    Uses RecursiveCharacterTextSplitter which tries to split on natural
    boundaries in this priority order:
      1. Double newlines (paragraph breaks)  → "\n\n"
      2. Single newlines (line breaks)       → "\n"
      3. Spaces (word boundaries)            → " "
      4. Individual characters               → ""

    This preserves semantic coherence better than splitting every N chars.

    Args:
        text (str): The full extracted text from the PDF.
        chunk_size (int): Maximum characters per chunk. Default: 500.
                          Larger = more context per chunk but fewer results fit
                          in LLM prompt. Smaller = more precise retrieval.
        chunk_overlap (int): Characters shared between consecutive chunks.
                             Default: 50. Prevents losing context at boundaries.

    Returns:
        List[str]: A list of text chunks ready for embedding.

    Example:
        >>> chunks = chunk_text("Long document text...", chunk_size=500)
        >>> print(f"Created {len(chunks)} chunks")
    """
    if not text or not text.strip():
        print("[Chunker] Warning: Empty text received. Returning empty list.")
        return []

    # RecursiveCharacterTextSplitter is the recommended default in LangChain
    # for most RAG use cases. It's smarter than a simple character splitter.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # length_function defines how "size" is measured — here, characters
        length_function=len,
        # separators tried in order (most preferred → least preferred)
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(text)

    # Filter out any chunks that are just whitespace after splitting
    chunks = [c.strip() for c in chunks if c.strip()]

    print(f"[Chunker] Split into {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")

    return chunks


def get_chunk_stats(chunks: List[str]) -> dict:
    """
    Compute statistics about the generated chunks for display in the UI.

    Useful for debugging and for showing users how their document
    was processed.

    Args:
        chunks (List[str]): List of text chunks.

    Returns:
        dict: Stats including count, avg_length, min_length, max_length.
    """
    if not chunks:
        return {"count": 0, "avg_length": 0, "min_length": 0, "max_length": 0}

    lengths = [len(c) for c in chunks]

    return {
        "count": len(chunks),
        "avg_length": round(sum(lengths) / len(lengths)),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }
