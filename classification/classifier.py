"""
Structured classification of a ContentItem using Groq's OpenAI-compatible
chat completions endpoint. Output is validated against a Pydantic model
so a malformed LLM response fails loudly rather than silently corrupting
downstream branching (escalate vs. draft).
"""
from __future__ import annotations

import json
import logging
from typing import Dict, Literal, Optional

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
    model_used: Optional[str] = None


def _get_client():
    """Return an authenticated Groq client, raising clearly if the key is missing."""
    from groq import Groq

    if not settings.groq.api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at console.groq.com "
            "and add it to your .env file."
        )
    return Groq(api_key=settings.groq.api_key)


def _force_sensitivity(result: ClassificationResult, model: str) -> ClassificationResult:
    """Stamp model_used and force sensitivity_flag=True for escalation categories."""
    updates: Dict[str, object] = {"model_used": model}
    if result.category in settings.pipeline.escalation_categories and not result.sensitivity_flag:
        updates["sensitivity_flag"] = True
    return result.model_copy(update=updates)


def classify(item: ContentItem) -> ClassificationResult:
    """Classify a single ContentItem, retrying on transient or format errors."""
    if item.is_empty:
        logger.warning("Empty text for item %s - defaulting to general_question.", item.external_id)
        return ClassificationResult(
            sentiment="neutral",
            category="general_question",
            sensitivity_flag=False,
            reasoning="Empty content - no classification possible.",
            confidence=0.0,
            model_used="none",
        )

    client = _get_client()
    cfg = settings.groq

    user_prompt = CLASSIFICATION_USER_TEMPLATE.format(
        platform=item.platform.value, author=item.author, text=item.text
    )

    models_to_try = [cfg.classification_model, cfg.fallback_model]
    last_error: Optional[Exception] = None

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
                raw = (response.choices[0].message.content or "").strip()
                data = json.loads(raw)
                classified: ClassificationResult = _force_sensitivity(
                    ClassificationResult(**data), model
                )
                logger.debug(
                    "Classified %s/%s -> category=%s sensitivity=%s confidence=%.2f model=%s",
                    item.platform.value, item.external_id,
                    classified.category, classified.sensitivity_flag,
                    classified.confidence, model,
                )
                return classified
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "Classification parse failure on model=%s attempt=%d: %s",
                    model, attempt + 1, exc,
                )
                continue
            except Exception as exc:
                last_error = exc
                logger.warning("Classification call failed on model=%s: %s", model, exc)
                break

    logger.error("All classification attempts failed (%s) - defaulting to escalation.", last_error)
    return ClassificationResult(
        sentiment="neutral",
        category="angry_customer",
        sensitivity_flag=True,
        reasoning="Classification failed; routed to human review as a safe default.",
        confidence=0.0,
        model_used="fallback",
    )
