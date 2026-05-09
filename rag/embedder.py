"""
rag/embedder.py
───────────────
Generates dense vector embeddings using Sentence Transformers.

RAG Concept: "Embeddings"
  An embedding is a numerical representation of text — a list of floating-
  point numbers (a "vector") that captures the semantic meaning of the text.

  The key property: texts with SIMILAR MEANING have vectors that are CLOSE
  to each other in high-dimensional space.

  Example:
    "What is machine learning?" → [0.12, -0.45, 0.88, ..., 0.03]  (384 dims)
    "Explain ML to me"          → [0.11, -0.43, 0.86, ..., 0.04]  ← very close!
    "I like pizza"              → [-0.72, 0.31, -0.15, ..., 0.91] ← far away

  We use "all-MiniLM-L6-v2":
    - 384-dimensional embeddings
    - Only 80MB model size — downloads once, runs locally
    - Fast inference (no GPU needed)
    - Excellent quality for semantic search tasks
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# ─── Model Configuration ───────────────────────────────────────────────────
# This model downloads automatically on first run (~80MB)
# Stored in ~/.cache/huggingface/hub/
MODEL_NAME = "all-MiniLM-L6-v2"

# Singleton: load the model once and reuse across the app lifecycle
_model: SentenceTransformer = None


def get_embedding_model() -> SentenceTransformer:
    """
    Return the Sentence Transformer model (lazy-loaded singleton).

    Loading a transformer model takes ~1-3 seconds. We load it once
    and cache it in the module-level `_model` variable so subsequent
    calls are instant.

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    global _model
    if _model is None:
        print(f"[Embedder] Loading model: {MODEL_NAME} (first load only)...")
        _model = SentenceTransformer(MODEL_NAME)
        print(f"[Embedder] Model loaded. Embedding dimension: "
              f"{_model.get_sentence_embedding_dimension()}")
    return _model


def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings.

    Processes texts in batches for memory efficiency. For large documents
    (hundreds of chunks), batching prevents out-of-memory errors.

    Args:
        texts (List[str]): List of text strings to embed.
                           These are your document chunks.
        batch_size (int): Number of texts to process at once. Default: 32.
                          Lower this if you run out of memory.

    Returns:
        List[List[float]]: A list of embedding vectors, one per input text.
                           Each vector has 384 dimensions for all-MiniLM-L6-v2.

    Example:
        >>> embeddings = embed_texts(["Hello world", "Goodbye world"])
        >>> print(len(embeddings[0]))  # 384
    """
    if not texts:
        return []

    model = get_embedding_model()

    print(f"[Embedder] Generating embeddings for {len(texts)} texts "
          f"(batch_size={batch_size})...")

    # encode() returns a numpy array of shape (n_texts, embedding_dim)
    # convert_to_tensor=False keeps them as numpy arrays (easier to serialize)
    embeddings_np = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,  # show progress only for large batches
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalization → cosine similarity = dot product
    )

    # Convert numpy array → Python list of lists (JSON-serializable, ChromaDB-compatible)
    embeddings = embeddings_np.tolist()

    print(f"[Embedder] Done. Each vector has {len(embeddings[0])} dimensions.")
    return embeddings


def embed_query(query: str) -> List[float]:
    """
    Generate an embedding for a single query string.

    IMPORTANT: We must use the SAME model for both document chunks AND
    queries. If you used Model A to store chunks, you must use Model A
    to embed the query — otherwise the similarity scores are meaningless
    (you'd be comparing apples to oranges in vector space).

    Args:
        query (str): The user's question text.

    Returns:
        List[float]: A single embedding vector (384 dimensions).
    """
    if not query or not query.strip():
        raise ValueError("Query text cannot be empty.")

    model = get_embedding_model()

    # For a single string, encode() still returns a 2D array, so we take [0]
    embedding_np = model.encode(
        [query.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True  # must match the normalization used for chunks
    )

    return embedding_np[0].tolist()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    RAG Concept: "Cosine Similarity"
      Measures the angle between two vectors, ranging from -1 to 1.
      - Score = 1.0  → identical meaning
      - Score = 0.7+ → very similar
      - Score = 0.0  → unrelated
      - Score = -1.0 → opposite meaning (rare in practice)

      NOTE: Since we use normalize_embeddings=True above, the vectors
      are already unit-length, so cosine_similarity = dot product.
      ChromaDB handles this internally — this function is just for reference.

    Args:
        vec1 (List[float]): First embedding vector.
        vec2 (List[float]): Second embedding vector.

    Returns:
        float: Cosine similarity score between -1 and 1.
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    # Dot product of unit vectors = cosine similarity
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
