"""Local Laya decision layer for RAG routing.

Laya replaces the generative LLM call that was previously used only to
decide whether retrieved context was sufficient.

The model is loaded once per Streamlit process via cache_resource and the
decision returns a calibrated P(true) score through the `noul` primitive.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import streamlit as st
import laya


DEFAULT_THRESHOLD = 0.70
MAX_CONTEXT_CHARS = 3000
# typed-decisions provides a larger 1024-token decision context than the
# base English checkpoint and is a better fit for retrieved RAG context.
LAYA_MODEL = "convaiinnovations/laya-typed-decisions"


@st.cache_resource(show_spinner=False)
def get_laya_agent():
    """Load the local Laya model once and reuse it across Streamlit reruns."""
    return laya.load(LAYA_MODEL)


def get_context_threshold() -> float:
    """Read the routing threshold from the environment."""
    try:
        threshold = float(
            os.getenv("RAG_DECISION_THRESHOLD", str(DEFAULT_THRESHOLD))
        )
    except ValueError:
        threshold = DEFAULT_THRESHOLD

    return min(max(threshold, 0.0), 1.0)


def _compact_context(context: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Keep each retrieved chunk represented while staying inside Laya's context window."""
    chunks = [chunk.strip() for chunk in context.split("\n\n") if chunk.strip()]
    if not chunks:
        return context[:max_chars]

    per_chunk = max(600, max_chars // len(chunks))
    compacted = []

    for chunk in chunks:
        if len(chunk) <= per_chunk:
            compacted.append(chunk)
        else:
            compacted.append(chunk[:per_chunk].rstrip() + " ...")

    return "\n\n".join(compacted)[:max_chars]


def judge_context(
    question: str,
    context: str,
) -> Dict[str, Any]:
    """Return Laya's probability that the retrieved context is sufficient."""
    agent = get_laya_agent()

    compacted_context = _compact_context(context)

    # Keep the state explicit and structured so the model can distinguish
    # the user's question from the retrieved evidence.
    state = {
        "question": question,
        "context": compacted_context,
    }

    questions = {
        "context_sufficient": {
            "type": "noul",
            "instructions": (
                "Can the retrieved context provide enough factual information "
                "to answer the user's question without relying on outside "
                "knowledge? Return true when the context is sufficient for "
                "a grounded answer; return false when it is insufficient."
            ),
            "criteria": {
                "false": "The retrieved context is insufficient to answer the question.",
                "true": "The retrieved context is sufficient to answer the question.",
            },
        }
    }

    result = agent.predict(state, questions)
    answer = result["answers"]["context_sufficient"]

    probability = float(answer["noul"])
    confidence = float(answer.get("confidence", 0.0))
    threshold = get_context_threshold()

    return {
        "sufficient": probability >= threshold,
        "probability": probability,
        "confidence": confidence,
        "threshold": threshold,
        "raw": result,
    }
