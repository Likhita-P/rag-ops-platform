"""
app/vector_store.py
-------------------
Pinecone vector store integration.
Handles upsert, retrieval, and deletion of document chunks.
"""

import os
import logging
import hashlib
from typing import List, Tuple

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-ops-index")
PINECONE_REGION     = os.getenv("PINECONE_REGION", "us-east-1")
EMBED_MODEL_NAME    = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM       = 384

# ── Singletons ────────────────────────────────────────────────────────────
_pc      = None
_index   = None
_embedder = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def _get_index():
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
        existing = [i.name for i in _pc.list_indexes()]
        if PINECONE_INDEX_NAME not in existing:
            logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
            _pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
            )
        _index = _pc.Index(PINECONE_INDEX_NAME)
        logger.info(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
    return _index


# ── Public API ────────────────────────────────────────────────────────────

def upsert_chunks(chunks: List[str], source: str = "upload") -> int:
    """
    Embed and upsert text chunks into Pinecone.
    Returns number of vectors upserted.
    """
    if not chunks:
        return 0

    embedder = _get_embedder()
    index    = _get_index()

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()
        chunk_id  = hashlib.md5(f"{source}_{i}_{chunk[:50]}".encode()).hexdigest()
        vectors.append({
            "id":     chunk_id,
            "values": embedding,
            "metadata": {
                "text":   chunk,
                "source": source,
                "chunk_idx": i,
            },
        })

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size])

    logger.info(f"Upserted {len(vectors)} chunks from '{source}' into Pinecone")
    return len(vectors)


def retrieve_chunks(
    question: str,
    top_k: int = 3,
    min_score: float = 0.75,
) -> Tuple[List[str], List[float]]:
    """
    Embed the question and retrieve top-k matching chunks from Pinecone.
    Returns (chunks, scores) filtered by min_score.
    """
    embedder  = _get_embedder()
    index     = _get_index()

    query_vec = embedder.encode(question).tolist()
    results   = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
    )

    chunks = []
    scores = []
    for match in results.matches:
        if match.score >= min_score:
            chunks.append(match.metadata.get("text", ""))
            scores.append(match.score)

    logger.info(
        f"Retrieved {len(chunks)}/{top_k} chunks above min_score={min_score} "
        f"for question: '{question[:60]}...'"
    )
    return chunks, scores


def delete_all_chunks() -> None:
    """Delete all vectors from the index (called before re-uploading a PDF)."""
    try:
        index = _get_index()
        index.delete(delete_all=True)
        logger.info("Deleted all vectors from Pinecone index")
    except Exception as e:
        logger.warning(f"Could not delete all chunks: {e}")