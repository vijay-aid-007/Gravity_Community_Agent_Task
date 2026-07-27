"""
Structured classification of a ContentItem using Groq's OpenAI-compatible
chat completions endpoint. Output is validated against a Pydantic model
so a malformed LLM response fails loudly rather than silently corrupting
downstream branching (escalate vs. draft).
"""
from __future__ import annotations

import json
import logging
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from config import settings
from classification.prompts import CLASSIFICATION_SYSTEM_PROMPT, CLASSIFICATION_USER_TEMPLATE
from ingestion.normalize import ContentItem

logger = logging.getLogger(__name__)

Category = Literal[
    "general_question",
    "praise",
    "minor_complaint",
    "refund_demand",
    "legal_threat",
    "angry_customer",
    "spam",
]


class ClassificationResult(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    category: Category
    sensitivity_flag: bool
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


def _get_client():
    from groq import Groq

    if not settings.groq.api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at console.groq.com "
            "and add it to your .env file."
        )
    return Groq(api_key=settings.groq.api_key)


def classify(item: ContentItem) -> ClassificationResult:
    """Classify a single ContentItem, retrying on transient/format errors."""
    client = _get_client()
    cfg = settings.groq

    user_prompt = CLASSIFICATION_USER_TEMPLATE.format(
        platform=item.platform.value, author=item.author, text=item.text
    )

    models_to_try = [cfg.classification_model, cfg.fallback_model]
    last_error: Exception | None = None

    for model in models_to_try:
        for attempt in range(cfg.max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=cfg.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                raw = response.choices[0].message.content
                data = json.loads(raw)
                result = ClassificationResult(**data)

                # Hard safety net: never let a mis-classified escalation
                # category slip through with sensitivity_flag=False.
                if result.category in settings.pipeline.escalation_categories:
                    result.sensitivity_flag = True

                return result

            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                logger.warning(
                    "Classification parse failure on model=%s attempt=%d: %s",
                    model, attempt + 1, e,
                )
                continue
            except Exception as e:  # rate limit / network — try fallback model
                last_error = e
                logger.warning("Classification call failed on model=%s: %s", model, e)
                break

    # Fail-safe default: if the LLM never returns valid structured output,
    # escalate rather than silently auto-drafting a reply to unknown content.
    logger.error("All classification attempts failed (%s) — defaulting to escalation.", last_error)
    return ClassificationResult(
        sentiment="neutral",
        category="angry_customer",
        sensitivity_flag=True,
        reasoning="Classification failed; routed to human review as a safe default.",
        confidence=0.0,
    )
