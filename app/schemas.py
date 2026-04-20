"""
schemas.py
----------
All Pydantic request/response models for the AI Ops Platform.
Every API endpoint uses these — no raw dicts anywhere.

"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


# ── Enums ──────────────────────────────────────────────────────────────────

class ConfidenceLevel(str, Enum):
    HIGH   = "high"    # similarity score >= 0.85
    MEDIUM = "medium"  # similarity score 0.75 – 0.84
    LOW    = "low"     # similarity score < 0.75 → fallback triggered


class FallbackReason(str, Enum):
    LLM_TIMEOUT      = "llm_timeout"
    LLM_RATE_LIMIT   = "llm_rate_limit"
    LOW_CONFIDENCE   = "low_confidence"
    NO_CONTEXT_FOUND = "no_context_found"
    SAFETY_BLOCKED   = "safety_blocked"
    NONE             = "none"


# ── Requests ───────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    """
    Incoming question from the user.
    session_id enables multi-turn conversation tracking.
    """
    question:        str            = Field(..., min_length=3, max_length=2000)
    session_id:      Optional[str]  = Field(None,  description="For multi-turn context")
    top_k:           int            = Field(3,     ge=1, le=10)
    min_score:       float          = Field(0.75,  ge=0.0, le=1.0,
                                           description="Min cosine similarity to include chunk")
    prompt_version:  str            = Field("v1",  description="Which prompt version to use")


class PDFUploadResponse(BaseModel):
    """Returned after a PDF is uploaded and processed."""
    filename:     str
    num_chunks:   int
    status:       str
    message:      str


# ── Responses ──────────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A single chunk retrieved from the vector store."""
    text:       str
    score:      float
    chunk_idx:  int


class AnswerResponse(BaseModel):
    """
    Full response returned to the user.

    is_grounded:  True if answer is traceable to retrieved chunks.
                  False means the LLM may have hallucinated — flag to user.
    used_fallback: True if rules-based path was used instead of LLM.

    Interview angle (ACKO Round 3 system design):
      "Every response carries a confidence score and grounding flag so
       downstream systems can decide whether to surface the answer
       directly or route to a human."
    """
    answer:           str
    confidence:       ConfidenceLevel
    is_grounded:      bool
    used_fallback:    bool
    fallback_reason:  FallbackReason
    retrieved_chunks: List[RetrievedChunk]
    prompt_version:   str
    latency_ms:       float
    input_tokens:     int
    output_tokens:    int
    estimated_cost_usd: float
    session_id:       Optional[str]


# ── Cost tracking ──────────────────────────────────────────────────────────

class CostRecord(BaseModel):
    """Persisted to cost_log.jsonl for budget tracking."""
    session_id:         Optional[str]
    question_preview:   str          = Field(..., max_length=100)
    input_tokens:       int
    output_tokens:      int
    model:              str
    estimated_cost_usd: float
    timestamp:          str


# ── RAGAS Eval ─────────────────────────────────────────────────────────────

class EvalSample(BaseModel):
    """One question/answer/context triple for RAGAS evaluation."""
    question:   str
    answer:     str
    contexts:   List[str]
    ground_truth: Optional[str] = None


class EvalResult(BaseModel):
    """Output of a RAGAS eval run."""
    faithfulness:       float
    answer_relevancy:   float
    context_recall:     Optional[float]
    hallucination_rate: float          # derived: 1 - faithfulness
    num_samples:        int
    run_id:             str            # MLflow run ID


# ── MLflow metrics ─────────────────────────────────────────────────────────

class RetrievalMetrics(BaseModel):
    """Logged to MLflow after every request."""
    avg_similarity_score: float
    top_score:            float
    chunks_above_threshold: int
    total_chunks_retrieved: int
    query_latency_ms:     float


# ── SHAP ───────────────────────────────────────────────────────────────────

class ShapExplanation(BaseModel):
    """Returned by /explain endpoint."""
    question:      str
    top_features:  List[dict]   # [{"token": "coverage", "shap_value": 0.42}, ...]
    base_value:    float
    explanation:   str          # human-readable summary
