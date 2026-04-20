"""
app/pdf_processor.py
--------------------
PDF ingestion and chunking only.
Embeddings handled by vector_store.py (Cohere + Pinecone).
"""

import logging
import fitz  # PyMuPDF
from typing import List

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract raw text from PDF bytes."""
    doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = " ".join([page.get_text() for page in doc]).strip()
    if not text:
        raise ValueError("No extractable text found in PDF.")
    return text


def chunk_text(
    text:       str,
    max_tokens: int = 300,
    overlap:    int = 50,
) -> List[str]:
    """
    Overlap chunking — prevents mid-fact splits.
    overlap=50 carries ~2-3 sentences across chunk boundaries.
    """
    sentences     = text.replace("\n", " ").split(". ")
    chunks        = []
    current_words = []

    for sentence in sentences:
        words = sentence.split()
        if len(current_words) + len(words) < max_tokens:
            current_words.extend(words)
        else:
            if current_words:
                chunks.append(" ".join(current_words).strip())
            current_words = current_words[-overlap:] + words

    if current_words:
        chunks.append(" ".join(current_words).strip())

    chunks = [c for c in chunks if len(c.split()) >= 20]
    logger.info(f"Chunked into {len(chunks)} chunks (overlap={overlap})")
    return chunks