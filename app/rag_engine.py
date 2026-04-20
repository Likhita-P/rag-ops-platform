"""
app/rag_engine.py
-----------------
RAG pipeline using LangChain + Pinecone + Langfuse.
"""
import time
import logging

from app.llm_client  import call_llm_with_context, estimate_cost
from app.vector_store     import retrieve_chunks
from app.prompt_store     import get_system_prompt
from app.cost_tracker     import build_cost_record, log_cost, is_budget_exceeded
from app.logger           import log_request
from app.safety           import check_for_injection, check_grounding
from app.observability    import trace_request
from app.schemas          import (
    QuestionRequest, AnswerResponse, RetrievedChunk,
    ConfidenceLevel, FallbackReason,
)

logger = logging.getLogger(__name__)


def determine_confidence(scores):
    if not scores: return ConfidenceLevel.LOW
    top = max(scores)
    if top >= 0.85: return ConfidenceLevel.HIGH
    elif top >= 0.75: return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def answer_question(req: QuestionRequest) -> AnswerResponse:
    start = time.time()

    # 1. Safety check
    is_safe, reason = check_for_injection(req.question)
    if not is_safe:
        return AnswerResponse(
            answer=f"Request blocked: {reason}",
            confidence=ConfidenceLevel.LOW,
            is_grounded=False, used_fallback=True,
            fallback_reason=FallbackReason.SAFETY_BLOCKED,
            retrieved_chunks=[], prompt_version=req.prompt_version,
            latency_ms=0, input_tokens=0, output_tokens=0,
            estimated_cost_usd=0.0, session_id=req.session_id,
        )

    # 2. Budget check
    if is_budget_exceeded():
        return AnswerResponse(
            answer="Daily budget exceeded.",
            confidence=ConfidenceLevel.LOW,
            is_grounded=False, used_fallback=True,
            fallback_reason=FallbackReason.LOW_CONFIDENCE,
            retrieved_chunks=[], prompt_version=req.prompt_version,
            latency_ms=0, input_tokens=0, output_tokens=0,
            estimated_cost_usd=0.0, session_id=req.session_id,
        )

    # 3. Retrieve from Pinecone
    selected_chunks, scores = retrieve_chunks(
        question=req.question,
        top_k=req.top_k,
        min_score=req.min_score,
    )

    if not selected_chunks:
        return AnswerResponse(
            answer="I could not find relevant information in the uploaded document.",
            confidence=ConfidenceLevel.LOW,
            is_grounded=True, used_fallback=False,
            fallback_reason=FallbackReason.NO_CONTEXT_FOUND,
            retrieved_chunks=[], prompt_version=req.prompt_version,
            latency_ms=(time.time()-start)*1000,
            input_tokens=0, output_tokens=0,
            estimated_cost_usd=0.0, session_id=req.session_id,
        )

    context       = "\n\n".join(selected_chunks)
    confidence    = determine_confidence(scores)
    system_prompt = get_system_prompt(req.prompt_version)

    # 4. LangChain LLM call
    answer, input_toks, output_toks, used_fallback, fallback_reason = \
        call_llm_with_context(
            context=context,
            question=req.question,
            system_prompt=system_prompt,
        )

    # 5. Grounding check
    is_grounded = check_grounding(answer, selected_chunks)
    latency_ms  = (time.time() - start) * 1000
    cost_usd    = estimate_cost(input_toks, output_toks)

    # 6. Langfuse trace
    trace_request(
        question=req.question, answer=answer,
        session_id=req.session_id, latency_ms=latency_ms,
        input_tokens=input_toks, output_tokens=output_toks,
        cost_usd=cost_usd, is_grounded=is_grounded,
        used_fallback=used_fallback, prompt_version=req.prompt_version,
        top_score=max(scores) if scores else 0.0,
        confidence=confidence.value,
    )

    # 7. Log cost + request
    log_cost(build_cost_record(
        session_id=req.session_id or "anon",
        question=req.question,
        input_tokens=input_toks, output_tokens=output_toks,
        model="gpt-4o", cost_usd=cost_usd,
    ))
    log_request(
        question=req.question, answer=answer,
        latency_ms=latency_ms, input_tokens=input_toks,
        output_tokens=output_toks, cost_usd=cost_usd,
        confidence=confidence.value, is_grounded=is_grounded,
        used_fallback=used_fallback, fallback_reason=fallback_reason.value,
        prompt_version=req.prompt_version,
        top_score=max(scores) if scores else 0.0,
        session_id=req.session_id,
    )

    # 8. Return response
    retrieved = [
        RetrievedChunk(text=c, score=s, chunk_idx=i)
        for i, (c, s) in enumerate(zip(selected_chunks, scores))
    ]
    return AnswerResponse(
        answer=answer, confidence=confidence,
        is_grounded=is_grounded, used_fallback=used_fallback,
        fallback_reason=fallback_reason, retrieved_chunks=retrieved,
        prompt_version=req.prompt_version,
        latency_ms=round(latency_ms, 2),
        input_tokens=input_toks, output_tokens=output_toks,
        estimated_cost_usd=round(cost_usd, 6),
        session_id=req.session_id,
    )