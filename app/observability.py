"""
app/observability.py
--------------------
Langfuse tracing for every RAG request.
Tracks latency, cost, grounding, and confidence per session.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────
LANGFUSE_PUBLIC_KEY  = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY  = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST        = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# ── Singleton ─────────────────────────────────────────────────────────────
_langfuse = None


def _get_langfuse():
    global _langfuse
    if _langfuse is None:
        if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
            logger.warning("Langfuse keys not set — tracing disabled")
            return None
        try:
            from langfuse import Langfuse
            _langfuse = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST,
            )
            logger.info("Langfuse client initialised")
        except Exception as e:
            logger.warning(f"Langfuse init failed: {e}")
            return None
    return _langfuse


# ── Public API ────────────────────────────────────────────────────────────

def trace_request(
    question:       str,
    answer:         str,
    session_id:     Optional[str],
    latency_ms:     float,
    input_tokens:   int,
    output_tokens:  int,
    cost_usd:       float,
    is_grounded:    bool,
    used_fallback:  bool,
    prompt_version: str,
    top_score:      float,
    confidence:     str,
) -> None:
    """
    Send a trace to Langfuse for every RAG request.
    Silently skips if Langfuse is not configured.
    """
    lf = _get_langfuse()
    if lf is None:
        return

    try:
        trace = lf.trace(
            name="rag-request",
            session_id=session_id or "anon",
            input=question,
            output=answer,
            metadata={
                "prompt_version":  prompt_version,
                "confidence":      confidence,
                "is_grounded":     is_grounded,
                "used_fallback":   used_fallback,
                "top_score":       top_score,
                "latency_ms":      round(latency_ms, 2),
                "input_tokens":    input_tokens,
                "output_tokens":   output_tokens,
                "cost_usd":        round(cost_usd, 6),
            },
        )
        logger.debug(f"Langfuse trace created: {trace.id}")
    except Exception as e:
        # Never let observability break the main request
        logger.warning(f"Langfuse trace failed: {e}")