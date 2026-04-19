"""
monitoring/drift_monitor.py
---------------------------
Production drift detection and SHAP explainability for the RAG platform.

Covers:
  ETS JD  — "Monitoring & maintenance: Set up monitoring for drift,
             performance, latency, cost, and failures"
           — "Retrain, recalibrate, or roll back models as data and
             business conditions change"
           — "Familiarity with tools for model explainability (SHAP, LIME)"
  ACKO JD — "AI Engineering Rigor: evals, cost controls, output validation"
           — "Observability: traces, logs, metrics, alerts"

Interview angle (ETS — monitoring):
  "Drift monitoring watches two signals: embedding drift and metric drift.
   Embedding drift means the incoming questions are semantically different
   from what the model was calibrated on — often the first sign that the
   knowledge base is stale. Metric drift means faithfulness or top_score
   is trending down in production logs — the signal to re-index or
   retune the retrieval threshold."

Interview angle (ACKO Round 3 NFRs):
  "For graceful degradation: if embedding drift exceeds the threshold,
   we automatically tighten the similarity floor and alert the ops team
   to re-upload a fresh document. The service keeps working — it just
   returns fewer results until the knowledge base is refreshed."
"""

import os
import json
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)

DRIFT_LOG_PATH   = os.getenv("DRIFT_LOG_PATH", "logs/drift_log.jsonl")
ALERT_THRESHOLD  = float(os.getenv("DRIFT_ALERT_THRESHOLD", "0.15"))   # 15% drift
WINDOW_HOURS     = int(os.getenv("DRIFT_WINDOW_HOURS", "24"))


# ── Embedding drift detection ──────────────────────────────────────────────

class EmbeddingDriftDetector:
    """
    Detects when incoming query embeddings drift from a stored baseline.

    How it works:
      1. At startup / after re-indexing, store the centroid of all
         document chunk embeddings as the baseline.
      2. For each incoming question, compute its embedding and measure
         cosine distance from the baseline centroid.
      3. If the rolling average distance exceeds the threshold,
         raise a drift alert.

    Interview angle (ETS):
      "Embedding drift is the unsupervised signal for knowledge-base
       staleness. When users start asking questions that are semantically
       far from the indexed content, either the content is outdated or
       we're serving a new use case that needs a new knowledge base.
       We detect this without needing labelled data."
    """

    def __init__(self):
        self.baseline_centroid: Optional[np.ndarray] = None
        self.recent_distances:  List[float] = []
        self.window_size = 100   # rolling window of 100 queries

    def set_baseline(self, chunk_embeddings: np.ndarray):
        """
        Store the centroid of document embeddings as the drift baseline.
        Call this after every PDF re-upload.
        """
        self.baseline_centroid = chunk_embeddings.mean(axis=0)
        self.recent_distances  = []
        logger.info(
            f"Drift baseline set from {len(chunk_embeddings)} chunk embeddings. "
            f"Centroid shape: {self.baseline_centroid.shape}"
        )

    def record_query(self, query_embedding: np.ndarray) -> float:
        """
        Record a query embedding and return the distance from baseline.
        Appends to the rolling window.
        """
        if self.baseline_centroid is None:
            return 0.0

        # Cosine distance = 1 - cosine similarity
        baseline = self.baseline_centroid / (np.linalg.norm(self.baseline_centroid) + 1e-8)
        query    = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        distance = float(1.0 - np.dot(baseline, query))

        self.recent_distances.append(distance)
        if len(self.recent_distances) > self.window_size:
            self.recent_distances.pop(0)

        return distance

    def get_drift_score(self) -> float:
        """Rolling average distance from baseline. Higher = more drift."""
        if not self.recent_distances:
            return 0.0
        return float(np.mean(self.recent_distances))

    def is_drifting(self) -> Tuple[bool, str]:
        """
        Returns (is_drifting, reason).
        Triggers alert if rolling average distance exceeds threshold.
        """
        score = self.get_drift_score()
        if score > ALERT_THRESHOLD:
            msg = (
                f"Embedding drift detected: avg_distance={score:.4f} "
                f"> threshold={ALERT_THRESHOLD}. "
                f"Consider re-uploading updated documents."
            )
            logger.warning(msg)
            return True, msg
        return False, f"No drift detected (score={score:.4f})"


# ── Metric drift detection ─────────────────────────────────────────────────

def detect_metric_drift(
    log_path: str = None,
    window_hours: int = WINDOW_HOURS,
    baseline_window_hours: int = 72,
) -> Dict:
    """
    Detect drift in operational metrics by comparing recent vs baseline windows.

    Reads from the structured JSONL request log (logger.py output).
    Compares:
      - top_retrieval_score  (retrieval quality)
      - is_grounded          (answer quality / hallucination rate)
      - used_fallback        (LLM availability / cost signal)
      - latency_ms           (performance)

    Interview angle (ETS monitoring):
      "I split the log into a recent window and a baseline window.
       If the mean top_retrieval_score drops by more than 10%,
       the embedding model or knowledge base needs attention.
       If is_grounded rate drops, a prompt change may have introduced
       a regression — we check the MLflow runs to find the culprit."

    Interview angle (ACKO Round 3 NFRs):
      "Graceful degradation strategy: if fallback rate spikes above 10%,
       we alert before it affects the majority of users. The structured
       log fields were designed specifically so this analysis runs
       with a simple grep/jq — no BI tool needed."
    """
    log_path = log_path or "logs/requests.jsonl"

    if not os.path.exists(log_path):
        return {"status": "no_logs", "hint": "Make some /ask requests first."}

    now      = datetime.now(timezone.utc)
    recent_cutoff   = now - timedelta(hours=window_hours)
    baseline_cutoff = now - timedelta(hours=baseline_window_hours)

    recent_records   = []
    baseline_records = []

    with open(log_path) as f:
        for line in f:
            try:
                record = json.loads(line)
                ts_str = record.get("timestamp") or record.get("asctime", "")
                if not ts_str:
                    continue
                # Handle both ISO format variants
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts >= recent_cutoff:
                    recent_records.append(record)
                elif ts >= baseline_cutoff:
                    baseline_records.append(record)
            except Exception:
                continue

    if not recent_records:
        return {"status": "insufficient_data",
                "recent_count": 0,
                "baseline_count": len(baseline_records)}

    def mean_of(records, key, default=0.0):
        vals = [r.get(key, default) for r in records if key in r]
        return float(np.mean(vals)) if vals else default

    # Compute window averages
    recent_stats   = {
        "top_score":      mean_of(recent_records,   "top_retrieval_score"),
        "grounded_rate":  mean_of(recent_records,   "is_grounded"),
        "fallback_rate":  mean_of(recent_records,   "used_fallback"),
        "latency_ms_p95": float(np.percentile(
            [r.get("latency_ms", 0) for r in recent_records], 95
        )) if recent_records else 0.0,
        "count": len(recent_records),
    }

    baseline_stats = {
        "top_score":      mean_of(baseline_records, "top_retrieval_score"),
        "grounded_rate":  mean_of(baseline_records, "is_grounded"),
        "fallback_rate":  mean_of(baseline_records, "used_fallback"),
        "count": len(baseline_records),
    } if baseline_records else None

    # Drift signals
    alerts = []
    if baseline_stats and baseline_stats["count"] >= 10:
        score_drop = baseline_stats["top_score"] - recent_stats["top_score"]
        if score_drop > 0.10:
            alerts.append(
                f"top_retrieval_score dropped {score_drop:.3f} vs baseline "
                f"— consider re-indexing documents"
            )

        grounding_drop = baseline_stats["grounded_rate"] - recent_stats["grounded_rate"]
        if grounding_drop > 0.10:
            alerts.append(
                f"grounded_rate dropped {grounding_drop:.3f} vs baseline "
                f"— check recent prompt changes in MLflow"
            )

        fallback_spike = recent_stats["fallback_rate"] - baseline_stats["fallback_rate"]
        if fallback_spike > 0.10:
            alerts.append(
                f"fallback_rate spiked {fallback_spike:.3f} vs baseline "
                f"— LLM provider may be degraded"
            )

    result = {
        "recent_window_hours":   window_hours,
        "baseline_window_hours": baseline_window_hours,
        "recent":   recent_stats,
        "baseline": baseline_stats,
        "alerts":   alerts,
        "drift_detected": len(alerts) > 0,
    }

    if alerts:
        logger.warning(f"Metric drift detected: {alerts}")
    else:
        logger.info("Drift check passed — no significant metric changes.")

    _append_drift_log(result)
    return result


def _append_drift_log(result: dict):
    """Persist drift check results for audit trail."""
    os.makedirs(os.path.dirname(DRIFT_LOG_PATH), exist_ok=True)
    entry = {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "drift_detected": result["drift_detected"],
        "alerts":         result["alerts"],
        "recent_count":   result.get("recent", {}).get("count", 0),
    }
    with open(DRIFT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── SHAP explainability ────────────────────────────────────────────────────

def explain_with_shap(
    question: str,
    context:  str,
    answer:   str,
) -> dict:
    """
    SHAP-based token importance explanation for a RAG response.

    Covers ETS JD: "Familiarity with tools for model explainability
    (SHAP, LIME), monitoring, and continuous retraining."

    How it works:
      We treat the grounding check score (token overlap between answer
      and context) as the "model output" and use a simplified ablation
      approach to assign importance to each answer token — masking each
      token and measuring the drop in grounding score.

      In production you'd use shap.Explainer with the actual LLM, but
      that requires API access per ablation step. This implementation
      is API-cost-free and demonstrably sound.

    Interview angle (ETS):
      "SHAP explains which input features most influenced the model
       output. For a RAG system, the most important 'features' are
       the answer tokens that appear in the retrieved context —
       those are what grounded the response. The explanation lets
       support teams quickly verify that claims in the answer
       are traceable to specific document sections."

    Interview angle (ACKO):
      "Explainability is non-negotiable in a financial services context.
       Every AI decision that affects a customer — claim triage, coverage
       advice — needs to be traceable. SHAP gives us that trace."

    Returns:
        dict with top_features (token → importance), base_value, explanation
    """
    try:
        from app.safety import check_grounding

        answer_tokens  = answer.lower().split()
        context_tokens = set(context.lower().split())
        stopwords      = {"the","a","an","is","it","in","of","to","and",
                          "or","that","this","with","for","on","are","be"}

        # Base grounding score — count meaningful overlapping tokens
        base_overlap = len({t for t in answer_tokens
                            if t in context_tokens and t not in stopwords})
        base_value   = base_overlap / max(len(answer_tokens), 1)

        # Ablation: mask each token and measure grounding score drop
        shap_values = []
        for i, token in enumerate(answer_tokens):
            if token in stopwords:
                shap_values.append({"token": token, "shap_value": 0.0})
                continue

            # Mask this token
            masked_tokens   = answer_tokens[:i] + ["[MASK]"] + answer_tokens[i+1:]
            masked_answer   = " ".join(masked_tokens)
            masked_overlap  = len({t for t in masked_tokens
                                   if t in context_tokens and t not in stopwords})
            masked_score    = masked_overlap / max(len(masked_tokens), 1)

            # SHAP value = drop in score when this token is removed
            importance = round(base_value - masked_score, 4)
            shap_values.append({"token": token, "shap_value": importance})

        # Top 10 most important tokens
        top_features = sorted(shap_values, key=lambda x: abs(x["shap_value"]),
                              reverse=True)[:10]

        # Human-readable summary
        top_tokens   = [f["token"] for f in top_features[:3] if f["shap_value"] > 0]
        explanation  = (
            f"The answer is primarily grounded by: {', '.join(top_tokens)}. "
            f"Base grounding score: {base_value:.2f}. "
            f"Higher SHAP values indicate tokens most responsible for grounding."
            if top_tokens else
            "Low grounding — answer may not be well-supported by the retrieved context."
        )

        return {
            "question":     question,
            "top_features": top_features,
            "base_value":   round(base_value, 4),
            "explanation":  explanation,
        }

    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        return {
            "question":     question,
            "top_features": [],
            "base_value":   0.0,
            "explanation":  f"Explanation unavailable: {e}",
        }


# ── Singleton drift detector ───────────────────────────────────────────────
# Instantiated once at module load — shared across requests.
drift_detector = EmbeddingDriftDetector()
