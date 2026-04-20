"""
logger.py
---------
Structured JSON request logging for every LLM call.

Every request logs: question, latency, tokens, model, prompt version,
confidence, whether fallback was used, and the top retrieval score.

This is your observability layer — without this you can't debug
why a bad answer happened or spot performance regressions.

Interview angle (ACKO Round 2 — they check if you add logging naturally):
  "I always add structured logging to LLM services so I can answer:
   which questions are falling back most? Which prompt version performs
   better? What's the p95 latency? You can't improve what you can't see."
"""

import logging
import json
import os
from datetime import datetime, timezone
from pythonjsonlogger import jsonlogger

LOG_DIR  = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.path.join(LOG_DIR, "/tmp/logs")


def setup_logging():
    """Call once at app startup to configure JSON logging."""
    os.makedirs(LOG_DIR, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Console handler — human-readable
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))

    # File handler — JSON for machine parsing
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    ))

    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)


def log_request(
    question:        str,
    answer:          str,
    latency_ms:      float,
    input_tokens:    int,
    output_tokens:   int,
    cost_usd:        float,
    confidence:      str,
    is_grounded:     bool,
    used_fallback:   bool,
    fallback_reason: str,
    prompt_version:  str,
    top_score:       float,
    session_id:      str = None,
):
    """
    Emit a structured log line for every request.
    Parsed by dashboards, alerts, and the ops agent.
    """
    logger = logging.getLogger("request")
    logger.info(
        "llm_request",
        extra={
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "session_id":      session_id,
            "question_len":    len(question),
            "answer_len":      len(answer),
            "latency_ms":      round(latency_ms, 2),
            "input_tokens":    input_tokens,
            "output_tokens":   output_tokens,
            "cost_usd":        round(cost_usd, 6),
            "confidence":      confidence,
            "is_grounded":     is_grounded,
            "used_fallback":   used_fallback,
            "fallback_reason": fallback_reason,
            "prompt_version":  prompt_version,
            "top_retrieval_score": round(top_score, 4),
        },
    )
