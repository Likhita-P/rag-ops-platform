"""
drift_monitor.py
----------------
Detects drift in RAG retrieval quality using Evidently AI.

What "drift" means here:
  The distribution of cosine similarity scores shifts downward —
  meaning the PDF content has changed but the embeddings haven't
  been updated. Questions that used to score 0.85 now score 0.60.

When drift is detected, this module triggers the Airflow retraining DAG
to re-chunk and re-embed the PDF automatically.

Interview angle (ETS — drift monitoring + retraining):
  "We track the distribution of retrieval scores over a rolling
   window. When Evidently detects the mean score has dropped by
   more than 10%, it triggers the Airflow DAG to re-embed. This
   is how we keep the chatbot accurate as PDFs get updated."
"""

import json
import os
import logging
import requests
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnDriftMetric

logger          = logging.getLogger(__name__)
COST_LOG        = os.getenv("COST_LOG_PATH",  "logs/cost_log.jsonl")
AIRFLOW_API     = os.getenv("AIRFLOW_API_URL", "http://localhost:8080/api/v1")
AIRFLOW_DAG_ID  = "rag_retrain_dag"
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.10"))


def load_score_history(log_path: str = "logs/requests.jsonl") -> pd.DataFrame:
    """Load retrieval scores from request logs."""
    records = []
    if not os.path.exists(log_path):
        return pd.DataFrame()
    with open(log_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if "top_retrieval_score" in rec:
                    records.append({
                        "similarity_score": rec["top_retrieval_score"],
                        "timestamp":        rec.get("timestamp", ""),
                    })
            except Exception:
                continue
    return pd.DataFrame(records)


def check_drift() -> dict:
    """
    Compare recent retrieval scores vs baseline.
    Returns drift report summary.
    """
    df = load_score_history()
    if len(df) < 20:
        return {"status": "insufficient_data", "drift_detected": False}

    midpoint  = len(df) // 2
    reference = df.iloc[:midpoint][["similarity_score"]]
    current   = df.iloc[midpoint:][["similarity_score"]]

    report = Report(metrics=[ColumnDriftMetric(column_name="similarity_score")])
    report.run(reference_data=reference, current_data=current)
    result = report.as_dict()

    drift_score   = result["metrics"][0]["result"]["drift_score"]
    drift_detected = drift_score > DRIFT_THRESHOLD

    logger.info(f"Drift score: {drift_score:.4f} | Detected: {drift_detected}")

    if drift_detected:
        logger.warning("Drift detected — triggering retraining DAG")
        _trigger_airflow_dag()

    return {
        "drift_score":    drift_score,
        "drift_detected": drift_detected,
        "threshold":      DRIFT_THRESHOLD,
        "num_samples":    len(df),
    }


def _trigger_airflow_dag():
    """Trigger the Airflow retraining DAG via REST API."""
    try:
        url      = f"{AIRFLOW_API}/dags/{AIRFLOW_DAG_ID}/dagRuns"
        response = requests.post(
            url,
            json={"conf": {"triggered_by": "drift_monitor"}},
            auth=("airflow", "airflow"),
            timeout=10,
        )
        response.raise_for_status()
        logger.info(f"Airflow DAG {AIRFLOW_DAG_ID} triggered successfully")
    except Exception as e:
        logger.error(f"Failed to trigger Airflow DAG: {e}")
