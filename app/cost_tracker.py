"""
cost_tracker.py
---------------
Tracks token usage and cost per request.
Appends to cost_log.jsonl for easy analysis.

Interview angle (ACKO Round 3 NFRs):
  "In production we set a per-request cost cap and a daily budget.
   If we're trending toward the daily limit, we switch to a cheaper
   model or increase the similarity threshold to reduce context size."
"""

import json
import os
import logging
from datetime import datetime, timezone
from app.schemas import CostRecord

logger       = logging.getLogger(__name__)
COST_LOG     = os.getenv("COST_LOG_PATH", "/tmp/logs/cost_log.jsonl")
DAILY_BUDGET = float(os.getenv("DAILY_BUDGET_USD", "10.0"))


def log_cost(record: CostRecord):
    """Append a cost record to the JSONL log file."""
    os.makedirs(os.path.dirname(COST_LOG), exist_ok=True)
    with open(COST_LOG, "a") as f:
        f.write(record.model_dump_json() + "\n")


def get_today_spend() -> float:
    """Sum today's costs from the log file."""
    if not os.path.exists(COST_LOG):
        return 0.0
    today   = datetime.now(timezone.utc).date().isoformat()
    total   = 0.0
    with open(COST_LOG, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("timestamp", "").startswith(today):
                    total += rec.get("estimated_cost_usd", 0.0)
            except json.JSONDecodeError:
                continue
    return total


def is_budget_exceeded() -> bool:
    spend = get_today_spend()
    if spend >= DAILY_BUDGET:
        logger.warning(f"Daily budget ${DAILY_BUDGET} exceeded. Spend: ${spend:.4f}")
        return True
    return False


def build_cost_record(
    session_id:     str,
    question:       str,
    input_tokens:   int,
    output_tokens:  int,
    model:          str,
    cost_usd:       float,
) -> CostRecord:
    return CostRecord(
        session_id=session_id,
        question_preview=question[:100],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        estimated_cost_usd=cost_usd,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
