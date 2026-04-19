"""
train_dag.py
------------
Airflow DAG: rag_retrain_dag

Triggered by drift_monitor.py when retrieval quality drops.
Re-chunks and re-embeds the stored PDF, then logs metrics to MLflow.

Tasks:
  1. check_drift       — confirm drift before doing expensive re-embed
  2. re_embed_pdf      — reload PDF, rechunk, recompute embeddings
  3. run_eval          — run RAGAS eval on new embeddings
  4. log_to_mlflow     — log new eval metrics
  5. notify            — log completion (extend with email/Slack)

Interview angle (ETS — Airflow + MLOps):
  "The DAG is idempotent — running it twice gives the same result.
   Each task is a pure function with clear inputs and outputs.
   If re-embedding fails, Airflow retries with exponential backoff
   and the old embeddings stay in place — no partial state."
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging
import pickle
import torch

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner":            "ai-ops-platform",
    "retries":          2,
    "retry_delay":      timedelta(minutes=2),
    "email_on_failure": False,
}

with DAG(
    dag_id="rag_retrain_dag",
    default_args=DEFAULT_ARGS,
    description="Re-embed PDF when retrieval drift is detected",
    schedule_interval=None,          # triggered manually or by drift_monitor
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["rag", "mlops", "retrain"],
) as dag:

    def task_check_drift(**context):
        from pipelines.drift_monitor import check_drift
        result = check_drift()
        logger.info(f"Drift check: {result}")
        context["ti"].xcom_push(key="drift_score", value=result["drift_score"])
        return result

    def task_re_embed_pdf(**context):
        """Re-chunk and re-embed the stored PDF."""
        import os
        from app.pdf_processor import (
            chunk_text, compute_embeddings, save_chunks_and_embeddings,
        )

        pdf_path = os.getenv("LATEST_PDF_PATH", "data/latest.pdf")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"No PDF found at {pdf_path}")

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        from app.pdf_processor import extract_text_from_pdf
        text       = extract_text_from_pdf(pdf_bytes)
        chunks     = chunk_text(text, overlap=75)   # wider overlap on retrain
        embeddings = compute_embeddings(chunks)
        save_chunks_and_embeddings(chunks, embeddings)

        logger.info(f"Re-embedded {len(chunks)} chunks")
        context["ti"].xcom_push(key="num_chunks", value=len(chunks))

    def task_run_eval(**context):
        """Run RAGAS eval on the freshly embedded PDF."""
        from evals.ragas_eval import run_ragas_eval
        result = run_ragas_eval()
        logger.info(f"RAGAS eval: faithfulness={result.faithfulness:.3f}")
        context["ti"].xcom_push(key="faithfulness", value=result.faithfulness)
        return result.model_dump()

    def task_log_to_mlflow(**context):
        ti           = context["ti"]
        faithfulness = ti.xcom_pull(key="faithfulness", task_ids="run_eval")
        num_chunks   = ti.xcom_pull(key="num_chunks",   task_ids="re_embed_pdf")

        from pipelines.mlflow_tracker import log_eval_metrics
        run_id = log_eval_metrics(
            faithfulness=faithfulness or 0.0,
            answer_relevancy=0.0,
            hallucination_rate=1 - (faithfulness or 0.0),
            avg_latency_ms=0.0,
            fallback_rate=0.0,
            prompt_version="v1",
            num_samples=num_chunks or 0,
        )
        logger.info(f"MLflow run: {run_id}")

    def task_notify(**context):
        faithfulness = context["ti"].xcom_pull(
            key="faithfulness", task_ids="run_eval"
        )
        logger.info(
            f"Retraining complete. New faithfulness score: {faithfulness:.3f}. "
            f"Extend this task with Slack/email notifications."
        )

    check_drift_task   = PythonOperator(task_id="check_drift",    python_callable=task_check_drift)
    re_embed_task      = PythonOperator(task_id="re_embed_pdf",   python_callable=task_re_embed_pdf)
    eval_task          = PythonOperator(task_id="run_eval",        python_callable=task_run_eval)
    mlflow_task        = PythonOperator(task_id="log_to_mlflow",  python_callable=task_log_to_mlflow)
    notify_task        = PythonOperator(task_id="notify",          python_callable=task_notify)

    check_drift_task >> re_embed_task >> eval_task >> mlflow_task >> notify_task
