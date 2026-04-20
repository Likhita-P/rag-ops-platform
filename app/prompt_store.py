"""
prompt_store.py
---------------
Versioned prompt registry.

Prompts are stored as JSON files in prompts/ folder.
Each version is immutable once deployed — you add new versions,
never edit old ones. This enables rollback in one line.

Interview angle (ACKO Round 3 system design):
  "We version prompts the same way we version code. If v2 increases
   hallucination rate as measured by RAGAS, we roll back to v1 in
   one config change without touching application code."
"""

import json
import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)

PROMPTS_DIR = os.getenv("PROMPTS_DIR", "/tmp/prompts")


def load_prompt(version: str = "v1") -> Dict[str, str]:
    """
    Load a prompt version from prompts/{version}.json.
    Returns dict with keys: system_prompt, version, description.
    """
    path = os.path.join(PROMPTS_DIR, f"{version}.json")
    if not os.path.exists(path):
        logger.warning(f"Prompt version {version} not found, falling back to v1")
        path = os.path.join(PROMPTS_DIR, "v1.json")

    with open(path, "r") as f:
        return json.load(f)


def get_system_prompt(version: str = "v1") -> str:
    return load_prompt(version)["system_prompt"]


def list_versions() -> list:
    """List all available prompt versions."""
    if not os.path.exists(PROMPTS_DIR):
        return []
    return [
        f.replace(".json", "")
        for f in os.listdir(PROMPTS_DIR)
        if f.endswith(".json")
    ]


def create_prompt_files():
    """
    Creates the prompts/ directory and default prompt versions.
    Call once on startup if prompts/ doesn't exist.
    """
    os.makedirs(PROMPTS_DIR, exist_ok=True)

    v1 = {
        "version":     "v1",
        "description": "Base grounding prompt — instructs model to stay in context",
        "system_prompt": (
            "You are a helpful assistant that answers questions based strictly "
            "on the provided document context.\n\n"
            "Rules:\n"
            "1. Only use information present in the context below.\n"
            "2. If the answer is not in the context, respond exactly with: "
            "'I could not find this information in the provided document.'\n"
            "3. Never guess, infer, or use outside knowledge.\n"
            "4. Quote the relevant section when possible.\n"
            "5. Be concise and factual."
        ),
    }

    v2 = {
        "version":     "v2",
        "description": "v2 — adds confidence instruction and structured output hint",
        "system_prompt": (
            "You are a precise document Q&A assistant.\n\n"
            "Rules:\n"
            "1. Answer using ONLY the provided context.\n"
            "2. If unsure or context is insufficient, say: "
            "'Insufficient information in document to answer this.'\n"
            "3. Never hallucinate facts, numbers, or dates not in the context.\n"
            "4. If the answer involves a number or date, quote the exact text.\n"
            "5. Keep answers under 150 words unless the question requires detail."
        ),
    }

    for prompt in [v1, v2]:
        fpath = os.path.join(PROMPTS_DIR, f"{prompt['version']}.json")
        if not os.path.exists(fpath):
            with open(fpath, "w") as f:
                json.dump(prompt, f, indent=2)
            logger.info(f"Created prompt file: {fpath}")
