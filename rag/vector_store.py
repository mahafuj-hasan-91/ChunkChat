"""
rag/vector_store.py
───────────────────
Handles all ChromaDB operations: storing, querying, and managing embeddings.

RAG Concept: "Vector Database"
  A vector database stores embeddings and enables fast "similarity search":
  given a query vector, find the N stored vectors that are most similar to it.

  ChromaDB is an open-source, locally-running vector database. It stores:
    - The embedding vectors themselves
    - The original text (as metadata/documents)
    - An ID for each entry

  We persist ChromaDB to disk so re-uploading the same PDF doesn't require
  re-embedding all chunks (expensive). The database lives in ./chroma_db/.

  How ChromaDB similarity search works:
    1. You provide a query embedding vector
    2. ChromaDB computes cosine similarity against ALL stored vectors
    3. Returns the top-N most similar ones (with their text and scores)
    This happens in milliseconds even for thousands of chunks.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import hashlib
import os


# ─── Constants ───────────────────────────────────────────────────────────────
CHROMA_DB_PATH = "./chroma_db"        # local persistence directory
COLLECTION_NAME = "rag_documents"     # our collection inside ChromaDB


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Initialize and return a persistent ChromaDB client.

    PersistentClient saves data to disk at CHROMA_DB_PATH so embeddings
    survive app restarts. Without persistence, you'd have to re-embed
    every time you restart the Streamlit app.

    Returns:
        chromadb.PersistentClient: A connected ChromaDB client instance.
    """
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)  # disable usage tracking
    )
    return client


def get_or_create_collection(client: chromadb.PersistentClient) -> Any:
    """
    Get an existing ChromaDB collection or create a new one.

    A "collection" is like a table in a relational database — it holds
    a set of related embeddings. We use one collection for all documents.

    We use "cosine" distance metric because our embeddings are normalized
    (unit vectors), making cosine similarity the optimal choice.

    Args:
        client (chromadb.PersistentClient): Active ChromaDB client.

    Returns:
        chromadb.Collection: The collection to store/query embeddings in.
    """
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        # "cosine" distance = 1 - cosine_similarity
        # ChromaDB returns the LOWEST distance = MOST SIMILAR
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def store_embeddings(
    chunks: List[str],
    embeddings: List[List[float]],
    doc_name: str,
    collection: Any
) -> int:
    """
    Store text chunks and their embeddings into ChromaDB.

    Each entry in ChromaDB requires three things:
      - id:        unique string identifier (we derive from doc_name + chunk index)
      - embedding: the vector (list of floats)
      - document:  the raw text (stored so we can retrieve it later)
      - metadata:  optional dict with extra info (source file, chunk index)

    We use a hash of doc_name to namespace IDs — this way uploading a new
    PDF doesn't conflict with a previously stored document's IDs.

    Args:
        chunks (List[str]): List of text chunks from the document.
        embeddings (List[List[float]]): Corresponding embeddings (same order).
        doc_name (str): Name of the source PDF file.
        collection: ChromaDB collection to store into.

    Returns:
        int: Number of chunks successfully stored.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings."
        )

    # Create a short hash from the doc name for ID namespacing
    # This lets us store multiple documents without ID collisions
    doc_hash = hashlib.md5(doc_name.encode()).hexdigest()[:8]

    ids = [f"{doc_hash}_chunk_{i}" for i in range(len(chunks))]

    # Metadata stored alongside each chunk — useful for source attribution
    metadatas = [
        {
            "source": doc_name,
            "chunk_index": i,
            "chunk_total": len(chunks)
        }
        for i in range(len(chunks))
    ]

    print(f"[VectorStore] Storing {len(chunks)} chunks for '{doc_name}'...")

    # ChromaDB's upsert = insert or update if ID already exists
    # This allows re-uploading the same PDF without duplicate entries
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )

    print(f"[VectorStore] Stored {len(chunks)} chunks. "
          f"Total in collection: {collection.count()}")

    return len(chunks)


def query_similar_chunks(
    query_embedding: List[float],
    collection: Any,
    n_results: int = 3
) -> Dict[str, Any]:
    """
    Retrieve the top-N most similar chunks to a query embedding.

    RAG Concept: "Semantic Retrieval"
      Instead of keyword matching (like grep), we match by MEANING.
      The query embedding is compared against all stored chunk embeddings
      using cosine similarity. The closest chunks (lowest distance) are
      returned regardless of exact word overlap.

      Example: Query "What causes inflation?" can retrieve a chunk that
      says "Rising prices result from excess money supply" — even though
      the words don't match, the meanings are similar.

    Args:
        query_embedding (List[float]): The embedded user question vector.
        collection: ChromaDB collection to search in.
        n_results (int): Number of similar chunks to retrieve. Default: 3.
                         Sending top-3 chunks keeps the LLM prompt concise
                         while providing enough context.

    Returns:
        Dict containing:
          - 'documents': List of retrieved chunk texts
          - 'metadatas': List of metadata dicts (source, chunk_index)
          - 'distances': List of cosine distances (lower = more similar)
    """
    if collection.count() == 0:
        print("[VectorStore] Warning: Collection is empty. No results.")
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    # Ensure we don't request more results than available
    actual_n = min(n_results, collection.count())

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=actual_n,
        # include both the stored text and metadata in results
        include=["documents", "metadatas", "distances"]
    )

    # Log similarity scores for debugging
    if results["distances"] and results["distances"][0]:
        scores = results["distances"][0]
        print(f"[VectorStore] Retrieved {len(scores)} chunks. "
              f"Cosine distances: {[round(s, 3) for s in scores]}")
        # Note: distance = 1 - similarity, so lower distance = better match

    return results


def clear_collection(collection: Any) -> None:
    """
    Delete all documents from the collection.

    Used when a user uploads a new PDF and wants a fresh start,
    or for testing purposes.

    Args:
        collection: The ChromaDB collection to clear.
    """
    count = collection.count()
    if count == 0:
        print("[VectorStore] Collection already empty.")
        return

    # Get all IDs and delete them
    all_ids = collection.get()["ids"]
    if all_ids:
        collection.delete(ids=all_ids)
        print(f"[VectorStore] Cleared {count} chunks from collection.")


def get_collection_info(collection: Any) -> Dict[str, Any]:
    """
    Return summary information about the current collection state.

    Useful for displaying status in the UI (e.g., "32 chunks stored
    from resume.pdf").

    Args:
        collection: The ChromaDB collection to inspect.

    Returns:
        dict: Info with 'count' and 'sources' (unique document names).
    """
    count = collection.count()
    if count == 0:
        return {"count": 0, "sources": []}

    # Retrieve metadata for all stored chunks
    all_data = collection.get(include=["metadatas"])
    sources = list({
        m.get("source", "Unknown")
        for m in all_data["metadatas"]
        if m
    })

    return {"count": count, "sources": sources}
