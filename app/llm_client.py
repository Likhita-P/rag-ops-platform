"""
llm_client.py
-------------
Production-hardened LLM client wrapping your existing Azure OpenAI call.
"""

import os
import logging
from typing import Tuple

import httpx
import tiktoken
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from app.schemas import FallbackReason

logger = logging.getLogger(__name__)

# ── Config — hardcoded like your original chatbot ──────────────────────────
API_KEY = "56530256cc8e464db3d2283b3ccbf49f"
API_URL = "https://open-ai-help-system-local.openai.azure.com/openai/deployments/gpt4o_11_2024/chat/completions?api-version=2024-08-01-preview"
MODEL   = "gpt-4o"
HEADERS = {"api-key": API_KEY, "Content-Type": "application/json"}
TIMEOUT = 30.0
MAX_COST_USD = 0.05

COST_PER_1K_INPUT  = 0.005
COST_PER_1K_OUTPUT = 0.015


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens  / 1000 * COST_PER_1K_INPUT +
        output_tokens / 1000 * COST_PER_1K_OUTPUT
    )


def rules_based_answer(question: str, context: str) -> str:
    q_lower = question.lower()

    if any(kw in q_lower for kw in ["maximum", "limit", "cap"]):
        for line in context.split("\n"):
            if any(kw in line.lower() for kw in ["maximum", "limit", "$", "usd"]):
                return f"[Fallback] Based on document: {line.strip()}"

    if any(kw in q_lower for kw in ["covered", "coverage", "eligible"]):
        for line in context.split("\n"):
            if "covered" in line.lower() or "eligible" in line.lower():
                return f"[Fallback] Based on document: {line.strip()}"

    if any(kw in q_lower for kw in ["name", "who", "candidate"]):
        for line in context.split("\n"):
            line = line.strip()
            if line and len(line.split()) <= 5 and line[0].isupper():
                return f"[Fallback] Based on document: {line}"

    return (
        "[Fallback] The AI service is temporarily unavailable. "
        "Please refer to the document directly for your answer."
    )


class RateLimitError(Exception):
    pass

class LLMTimeoutError(Exception):
    pass

class LLMEmptyResponseError(Exception):
    pass


@retry(
    retry=retry_if_exception_type((RateLimitError, LLMTimeoutError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(2),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _call_llm(messages: list, max_tokens: int = 512) -> Tuple[str, int, int]:
    payload = {
        "messages":    messages,
        "temperature": 0.2,
        "max_tokens":  max_tokens,
    }

    try:
        with httpx.Client(verify=False, timeout=TIMEOUT) as client:
            response = client.post(API_URL, headers=HEADERS, json=payload)

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 30))
                raise RateLimitError(f"Rate limited. Retry after {retry_after}s")

            if response.status_code == 401:
                raise PermissionError("Invalid API key.")

            response.raise_for_status()
            data = response.json()

            choices = data.get("choices", [])
            if not choices or not choices[0].get("message", {}).get("content"):
                raise LLMEmptyResponseError("LLM returned empty content")

            answer      = choices[0]["message"]["content"].strip()
            usage       = data.get("usage", {})
            input_toks  = usage.get("prompt_tokens",     0)
            output_toks = usage.get("completion_tokens", 0)
            return answer, input_toks, output_toks

    except httpx.TimeoutException:
        raise LLMTimeoutError(f"LLM call timed out after {TIMEOUT}s")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise RateLimitError("Rate limited")
        raise LLMTimeoutError(f"HTTP error: {e}")
    except httpx.HTTPError as e:
        raise LLMTimeoutError(f"Connection error: {e}")


def call_llm_with_context(
    context:       str,
    question:      str,
    system_prompt: str,
    max_tokens:    int = 512,
) -> Tuple[str, int, int, bool, FallbackReason]:

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    input_text   = system_prompt + context + question
    input_tokens = count_tokens(input_text)
    pre_cost     = estimate_cost(input_tokens, max_tokens)

    if pre_cost > MAX_COST_USD:
        logger.warning(f"Cost estimate ${pre_cost:.4f} exceeds cap ${MAX_COST_USD}.")
        return (
            rules_based_answer(question, context),
            input_tokens, 0, True, FallbackReason.LOW_CONFIDENCE,
        )

    try:
        answer, input_toks, output_toks = _call_llm(messages, max_tokens)

        if len(answer.strip()) < 5:
            raise LLMEmptyResponseError("Answer too short")

        return answer, input_toks, output_toks, False, FallbackReason.NONE

    except RateLimitError:
        logger.error("Rate limit exhausted after retries.")
        return (
            rules_based_answer(question, context),
            input_tokens, 0, True, FallbackReason.LLM_RATE_LIMIT,
        )

    except LLMTimeoutError:
        logger.error("LLM timed out after retries.")
        return (
            rules_based_answer(question, context),
            input_tokens, 0, True, FallbackReason.LLM_TIMEOUT,
        )

    except LLMEmptyResponseError:
        logger.error("LLM returned empty response.")
        return (
            rules_based_answer(question, context),
            input_tokens, 0, True, FallbackReason.LLM_TIMEOUT,
        )

    except Exception as e:
        logger.exception(f"Unexpected LLM error: {e}")
        return (
            rules_based_answer(question, context),
            input_tokens, 0, True, FallbackReason.LLM_TIMEOUT,
        )
