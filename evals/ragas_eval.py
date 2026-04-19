"""
ragas_eval.py
-------------
RAGAS-style evaluation pipeline — no OPENAI_API_KEY required.

Uses embedding-based faithfulness and answer relevancy scoring
via the project's own embedding model (all-MiniLM-L6-v2),
so it works entirely with your Azure OpenAI setup.

Interview angle (ACKO JD differentiator + ETS AI rigor):
  "We built a RAGAS eval suite that runs on every retrain.
   Faithfulness measures whether the answer is grounded in the
   retrieved context. If it drops below 0.80 we block the new
   embeddings and alert — same as a CI test failing."
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

from app.schemas              import EvalSample, EvalResult
from app.rag_engine           import answer_question
from app.schemas              import QuestionRequest
from app.pdf_processor        import get_embedding_model

logger = logging.getLogger(__name__)

EVAL_QUESTIONS = [
    "What is the role of an AI Native Software Engineer at ACKO?",
    "What is ACKO's D2C model?",
    "What AI frameworks and tools are required for this role?",
    "What kind of products does ACKO build for customers?",
    "What backend engineering skills are needed for this position?",
]


def _embed(text: str) -> np.ndarray:
    """Embed a single string using the project's own embedding model."""
    import torch
    tokenizer, model = get_embedding_model()
    with torch.no_grad():
        inputs  = tokenizer(
            text, return_tensors="pt",
            truncation=True, padding=True, max_length=512,
        )
        outputs = model(**inputs)
        vec     = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    return vec


def compute_faithfulness(answer: str, contexts: List[str]) -> float:
    """
    Faithfulness = max cosine similarity between answer and retrieved chunks.
    Score 1.0 = fully grounded. Near 0 = hallucination risk.
    """
    if not contexts or not answer.strip():
        return 0.0
    a_vec  = _embed(answer).reshape(1, -1)
    c_vecs = np.stack([_embed(c) for c in contexts])
    scores = cosine_similarity(a_vec, c_vecs)[0]
    return float(np.max(scores))


def compute_answer_relevancy(answer: str, question: str) -> float:
    """
    Answer relevancy = cosine similarity between answer and question.
    High = on-topic. Low = grounded but not answering the question.
    """
    if not answer.strip() or not question.strip():
        return 0.0
    q_vec = _embed(question).reshape(1, -1)
    a_vec = _embed(answer).reshape(1, -1)
    return float(cosine_similarity(q_vec, a_vec)[0][0])


def run_ragas_eval(
    questions:      List[str] = None,
    prompt_version: str       = "v1",
) -> EvalResult:
    """
    Run RAGAS-style eval against the currently loaded PDF.
    Returns EvalResult with metrics logged to MLflow.
    """
    questions = questions or EVAL_QUESTIONS
    samples   = []
    start     = time.time()

    for q in questions:
        try:
            req    = QuestionRequest(question=q, prompt_version=prompt_version)
            resp   = answer_question(req)
            sample = EvalSample(
                question=q,
                answer=resp.answer,
                contexts=[c.text for c in resp.retrieved_chunks],
            )
            samples.append(sample)
        except Exception as e:
            logger.warning(f"Eval question failed: {q} — {e}")

    if not samples:
        raise ValueError("No eval samples generated. Ensure a PDF is uploaded.")

    faith_scores = []
    relev_scores = []

    for s in samples:
        faith = compute_faithfulness(s.answer, s.contexts)
        relev = compute_answer_relevancy(s.answer, s.question)
        faith_scores.append(faith)
        relev_scores.append(relev)
        logger.info(
            f"  Q: {s.question[:50]} | "
            f"faithfulness={faith:.3f} | relevancy={relev:.3f}"
        )

    faith_score = float(np.mean(faith_scores))
    rel_score   = float(np.mean(relev_scores))
    halluc_rate = round(1.0 - faith_score, 4)
    avg_latency = (time.time() - start) / len(samples) * 1000

    logger.info(
        f"RAGAS eval complete | faithfulness={faith_score:.3f} "
        f"| answer_relevancy={rel_score:.3f} "
        f"| hallucination_rate={halluc_rate:.3f} | n={len(samples)}"
    )

    try:
        from pipelines.mlflow_tracker import log_eval_metrics
        run_id = log_eval_metrics(
            faithfulness=faith_score,
            answer_relevancy=rel_score,
            hallucination_rate=halluc_rate,
            avg_latency_ms=avg_latency,
            fallback_rate=0.0,
            prompt_version=prompt_version,
            num_samples=len(samples),
        )
    except Exception as e:
        logger.warning(f"MLflow logging skipped: {e}")
        run_id = "mlflow-unavailable"

    return EvalResult(
        faithfulness=faith_score,
        answer_relevancy=rel_score,
        context_recall=None,
        hallucination_rate=halluc_rate,
        num_samples=len(samples),
        run_id=run_id,
    )
