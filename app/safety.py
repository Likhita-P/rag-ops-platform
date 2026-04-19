"""
safety.py
---------
Input validation and output safety checks.

Two responsibilities:
  1. Prompt injection detection — catch attempts to hijack the system prompt
  2. Output grounding check — verify the answer is traceable to the context

Interview angle (ACKO JD differentiator):
  "Security and abuse prevention instincts applied to AI: prompt injection,
   output filtering, rate limiting."
  This file directly addresses that bullet point.
"""

import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ── Prompt injection patterns ──────────────────────────────────────────────
# These are common jailbreak / injection attempts
INJECTION_PATTERNS = [
    r"ignore (all |previous |above |prior )?instructions",
    r"forget (everything|all|your instructions)",
    r"you are now",
    r"act as (a |an )?(?!assistant)",
    r"pretend (you are|to be)",
    r"jailbreak",
    r"disregard (the |all |your )?system",
    r"new personality",
    r"repeat after me",
    r"reveal (your |the )?system prompt",
    r"what (are|were) your instructions",
    r"</?(system|prompt|instruction)>",   # XML injection
]

INJECTION_RE = re.compile(
    "|".join(INJECTION_PATTERNS),
    re.IGNORECASE,
)


def check_for_injection(question: str) -> Tuple[bool, str]:
    """
    Returns (is_safe, reason).
    is_safe=False means the input should be blocked.
    """
    if INJECTION_RE.search(question):
        logger.warning(f"Prompt injection detected: {question[:100]}")
        return False, "Input contains disallowed patterns."
    if len(question) > 2000:
        return False, "Question exceeds maximum length of 2000 characters."
    return True, ""


# ── Output grounding check ─────────────────────────────────────────────────

def check_grounding(answer: str, context_chunks: list) -> bool:
    """
    Heuristic grounding check.

    A response is considered grounded if meaningful content from
    the answer overlaps with the retrieved chunks.

    This is intentionally lightweight — a full grounding check
    would use an LLM-as-judge pattern, but that adds cost and latency.
    For a demo/project this heuristic catches obvious hallucinations.

    Interview angle:
      "We use a lightweight token overlap check as a first pass.
       For high-stakes queries in production we'd add an LLM-as-judge
       step, but we gate that behind a confidence threshold to control cost."
    """
    if not context_chunks:
        return False

    if any(phrase in answer.lower() for phrase in [
        "i could not find",
        "not in the document",
        "insufficient information",
        "i don't know",
        "not mentioned",
    ]):
        # Model correctly admitted it doesn't know — that's grounded behaviour
        return True

    # Token overlap heuristic
    answer_tokens  = set(answer.lower().split())
    context_tokens = set(" ".join(context_chunks).lower().split())
    overlap        = answer_tokens & context_tokens

    # Remove stopwords from overlap count
    stopwords = {"the", "a", "an", "is", "it", "in", "of", "to", "and",
                 "or", "that", "this", "with", "for", "on", "are", "be"}
    meaningful_overlap = overlap - stopwords

    # At least 15% of answer tokens should appear in context
    threshold = max(3, int(len(answer_tokens) * 0.15))
    is_grounded = len(meaningful_overlap) >= threshold

    if not is_grounded:
        logger.warning(
            f"Low grounding score: {len(meaningful_overlap)}/{threshold} "
            f"meaningful tokens overlap. Answer may be hallucinated."
        )
    return is_grounded
