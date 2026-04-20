import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.logger       import setup_logging
from app.prompt_store import create_prompt_files, list_versions
from app.cost_tracker import get_today_spend
from app.schemas      import QuestionRequest, AnswerResponse, PDFUploadResponse
from app.rag_engine   import answer_question
from app.pdf_processor import extract_text_from_pdf, chunk_text
from app.vector_store  import upsert_chunks, delete_all_chunks

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    create_prompt_files()
    os.makedirs("/tmp/logs", exist_ok=True)
    logger.info("RAG Ops Platform started")
    yield

app = FastAPI(
    title="RAG Ops Intelligence Platform",
    description="Production RAG chatbot — LangChain + LangGraph + Pinecone + Langfuse",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Serve static UI
static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/")
def root():
    if not os.path.exists(static_path):
        return {"error": f"static_path not found: {static_path}"}
    index = os.path.join(static_path, "index.html")
    if not os.path.exists(index):
        return {"error": f"index.html not found at: {index}"}
    return FileResponse(index)

@app.get("/debug")
def debug():
    return {
        "static_path": static_path,
        "exists": os.path.exists(static_path),
        "index_exists": os.path.exists(os.path.join(static_path, "index.html")),
        "app_dir": os.path.dirname(__file__),
        "cwd": os.getcwd(),
    }
    
@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}

@app.post("/upload", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")
    pdf_bytes = await file.read()
    try:
        text   = extract_text_from_pdf(pdf_bytes)
        chunks = chunk_text(text)
        delete_all_chunks()
        upsert_chunks(chunks, source=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return PDFUploadResponse(
        filename=file.filename, num_chunks=len(chunks),
        status="success",
        message=f"Processed {len(chunks)} chunks. Ready to answer questions.",
    )

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    try:
        return answer_question(req)
    except Exception as e:
        logger.exception(f"Error in /ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent")
def agent_query(req: QuestionRequest):
    try:
        from agent.ops_agent import run_agent
        answer = run_agent(req.question)
        return {"question": req.question, "answer": answer}
    except Exception as e:
        logger.exception(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cost")
def cost_summary():
    return {"today_spend_usd": round(get_today_spend(), 4), "currency": "USD"}

@app.get("/prompts")
def prompt_versions():
    return {"versions": list_versions()}

@app.get("/metrics")
def metrics_summary():
    try:
        from pipelines.mlflow_tracker import get_latest_metrics
        return get_latest_metrics()
    except Exception as e:
        return {"error": str(e)}

@app.post("/eval")
def run_eval(prompt_version: str = "v1"):
    try:
        from evals.ragas_eval import run_ragas_eval
        result = run_ragas_eval(prompt_version=prompt_version)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
def explain_response(req: QuestionRequest):
    try:
        from monitoring.drift_monitor import explain_with_shap
        response = answer_question(req)
        context  = " ".join([c.text for c in response.retrieved_chunks])
        return explain_with_shap(req.question, context, response.answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drift")
def check_drift():
    try:
        from monitoring.drift_monitor import detect_metric_drift
        return detect_metric_drift()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/finetune")
def finetune(epochs: int = 3, batch_size: int = 8):
    try:
        from pipelines.fine_tune import run_fine_tuning
        return run_fine_tuning(epochs=epochs, batch_size=batch_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/compare-prompts")
def compare_prompts(v1: str = "v1", v2: str = "v2"):
    try:
        from pipelines.mlflow_tracker import compare_prompt_versions
        return compare_prompt_versions(v1=v1, v2=v2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))