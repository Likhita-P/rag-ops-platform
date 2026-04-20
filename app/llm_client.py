"""
app/llm_client.py
-----------------
LangChain-based LLM client using Azure OpenAI.
Replaces raw httpx calls with LangChain AzureChatOpenAI.
"""

import os
import logging
from typing import Tuple

import tiktoken
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.schemas import FallbackReason

logger = logging.getLogger(__name__)

# Config from environment variables — no hardcoded keys
AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt4o_11_2024")
MAX_COST_USD             = float(os.getenv("MAX_COST_PER_REQUEST", "0.05"))

COST_PER_1K_INPUT  = 0.005
COST_PER_1K_OUTPUT = 0.015


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens  / 1000 * COST_PER_1K_INPUT +
        output_tokens / 1000 * COST_PER_1K_OUTPUT
    )


def count_tokens(text: str) -> int:
    try:
        enc = tiktoken.encoding_for_model("gpt-4o")
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def rules_based_answer(question: str, context: str) -> str:
    """Fallback when LLM is unavailable or cost cap exceeded."""
    q_lower = question.lower()
    for line in context.split("\n"):
        if any(kw in q_lower for kw in ["maximum", "limit", "cap"]):
            if any(kw in line.lower() for kw in ["maximum", "limit", "$"]):
                return f"[Fallback] Based on document: {line.strip()}"
        if any(kw in q_lower for kw in ["covered", "coverage"]):
            if "covered" in line.lower():
                return f"[Fallback] Based on document: {line.strip()}"
    return (
        "[Fallback] The AI service is temporarily unavailable. "
        "Please refer to the document directly."
    )


def get_llm() -> AzureChatOpenAI:
    """
    Returns LangChain AzureChatOpenAI instance.
    Interview angle: swapping providers = one line change.
    """
    return AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0.2,
        max_tokens=512,
        timeout=30,
        max_retries=2,
    )


def call_llm_with_context(
    context:       str,
    question:      str,
    system_prompt: str,
    max_tokens:    int = 512,
) -> Tuple[str, int, int, bool, FallbackReason]:
    """
    Call Azure OpenAI via LangChain LCEL chain:
    prompt | llm | output_parser
    """
    input_text   = system_prompt + context + question
    input_tokens = count_tokens(input_text)
    pre_cost     = estimate_cost(input_tokens, max_tokens)

    if pre_cost > MAX_COST_USD:
        logger.warning(f"Cost ${pre_cost:.4f} exceeds cap ${MAX_COST_USD}")
        return (
            rules_based_answer(question, context),
            input_tokens, 0, True, FallbackReason.LOW_CONFIDENCE,
        )

    try:
        llm = get_llm()

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])

        # LCEL chain
        chain  = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        if not answer or len(answer.strip()) < 5:
            raise ValueError("Empty response")

        output_tokens = count_tokens(answer)
        return answer, input_tokens, output_tokens, False, FallbackReason.NONE

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return (
            rules_based_answer(question, context),
            input_tokens, 0, True, FallbackReason.LLM_TIMEOUT,
        )