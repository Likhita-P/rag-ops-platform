"""
main.py
-------
FastAPI application entry point.
Mounts all routes and runs startup tasks.

Run with:
    uvicorn app.main:app --reload --port 8000

Endpoints:
    POST /upload      — upload + process a PDF
    POST /ask         — ask a question
    GET  /health      — health check
    GET  /cost        — today's spend
    GET  /prompts     — list prompt versions
    POST /explain     — SHAP explanation for a prediction
    GET  /eval        — run RAGAS eval on sample questions
    GET  /metrics     — MLflow metrics summary
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.logger       import setup_logging
from app.prompt_store import create_prompt_files, list_versions
from app.cost_tracker import get_today_spend
from app.schemas      import QuestionRequest, AnswerResponse, PDFUploadResponse
from app.rag_engine   import answer_question
from app.pdf_processor import (
    extract_text_from_pdf, chunk_text,
    compute_embeddings, save_chunks_and_embeddings,
)

logger = logging.getLogger(__name__)


# ── Startup / shutdown ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    create_prompt_files()          # creates prompts/v1.json, v2.json if missing
    import os; os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    logger.info("AI Ops Platform started")
    yield
    logger.info("AI Ops Platform shutting down")


# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Ops Intelligence Platform",
    description=(
        "Production-grade RAG chatbot with MLOps, drift monitoring, "
        "cost tracking, prompt versioning, and LangGraph agent."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/upload", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF. Extracts text, chunks it, computes embeddings,
    and saves to disk for subsequent /ask calls.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes  = await file.read()

    try:
        text       = extract_text_from_pdf(pdf_bytes)
        chunks     = chunk_text(text)
        embeddings = compute_embeddings(chunks)
        save_chunks_and_embeddings(chunks, embeddings)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PDFUploadResponse(
        filename=file.filename,
        num_chunks=len(chunks),
        status="success",
        message=f"Processed {len(chunks)} chunks. Ready to answer questions.",
    )


@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    """
    Ask a question against the uploaded PDF.
    Returns a typed response with confidence score, grounding flag,
    cost, and whether a fallback was used.
    """
    try:
        return answer_question(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error in /ask: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.get("/cost")
def cost_summary():
    """Today's total LLM spend."""
    return {
        "today_spend_usd": round(get_today_spend(), 4),
        "currency": "USD",
    }


@app.get("/prompts")
def prompt_versions():
    """List available prompt versions."""
    return {"versions": list_versions()}


@app.get("/metrics")
def metrics_summary():
    """
    Pull latest model metrics from MLflow.
    Wired to mlflow_tracker.py.
    """
    try:
        from pipelines.mlflow_tracker import get_latest_metrics
        return get_latest_metrics()
    except Exception as e:
        return {"error": str(e), "hint": "Run a RAGAS eval first to populate metrics."}


@app.post("/eval")
def run_eval(prompt_version: str = "v1"):
    """
    Run RAGAS evaluation on sample questions.
    Computes faithfulness, answer_relevancy, BLEU, ROUGE-L.
    Logs results to MLflow.

    ETS JD: Evaluation & experimentation — design offline metrics
    to compare models and approaches.
    """
    try:
        from evals.ragas_eval import run_ragas_eval
        result = run_ragas_eval(prompt_version=prompt_version)
        return result.model_dump()
    except Exception as e:
        logger.exception(f"Eval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
def explain_response(req: QuestionRequest):
    """
    SHAP token-importance explanation for a RAG response.

    ETS JD: "Familiarity with tools for model explainability (SHAP, LIME)"
    ACKO JD: Audit trail for AI decisions — explainability in financial services.
    """
    try:
        from monitoring.drift_monitor import explain_with_shap
        from app.schemas import ShapExplanation

        # Get the answer first
        response = answer_question(req)

        context = " ".join([c.text for c in response.retrieved_chunks])
        shap_result = explain_with_shap(
            question=req.question,
            context=context,
            answer=response.answer,
        )
        return ShapExplanation(**shap_result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Explain failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drift")
def check_drift():
    """
    Detect metric and embedding drift from production request logs.

    ETS JD: "Monitoring & maintenance: Set up monitoring for drift,
    performance, latency, cost, and failures. Retrain or roll back
    models as data and business conditions change."
    """
    try:
        from monitoring.drift_monitor import detect_metric_drift
        return detect_metric_drift()
    except Exception as e:
        logger.exception(f"Drift check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/finetune")
def finetune_embedding(epochs: int = 3, batch_size: int = 8):
    """
    Fine-tune the embedding model on domain-specific data.
    Uses LoRA/PEFT for parameter-efficient training.
    Logs to MLflow. Evaluates with TensorFlow.

    ETS JD: "Model development: Train, tune, and validate models"
           "Expert in TensorFlow, PyTorch, HuggingFace Transformers"
           "MLOps: versioning of data/models, reproducible training"
    """
    try:
        from pipelines.fine_tune import run_fine_tuning
        result = run_fine_tuning(epochs=epochs, batch_size=batch_size)
        return result
    except Exception as e:
        logger.exception(f"Fine-tuning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compare-prompts")
def compare_prompts(v1: str = "v1", v2: str = "v2"):
    """
    Compare two prompt versions across all their MLflow eval runs.
    Returns mean metrics per version and a promotion recommendation.

    ACKO JD: Prompt versioning, A/B testing, eval pipelines.
    ETS JD: Evaluation & experimentation — A/B tests and offline metrics.
    """
    try:
        from pipelines.mlflow_tracker import compare_prompt_versions
        return compare_prompt_versions(v1=v1, v2=v2)
    except Exception as e:
        logger.exception(f"Prompt comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
