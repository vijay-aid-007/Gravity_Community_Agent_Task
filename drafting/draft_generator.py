"""
Generates an on-brand draft reply for non-sensitive content items.
Retrieves relevant brand-voice guidance first (RAG), then asks the
LLM to draft strictly within that guidance.
"""
from __future__ import annotations

import logging

from config import settings
from classification.classifier import ClassificationResult
from ingestion.normalize import ContentItem
from retrieval.brand_voice_store import get_store

logger = logging.getLogger(__name__)

DRAFT_SYSTEM_PROMPT = """You are drafting a public reply on behalf of a brand's
community team. Follow the brand voice guidance provided EXACTLY — do not
invent tone, promises, or policy not present in the guidance. Keep the reply
under 3 sentences. Never mention that you are an AI. Never promise refunds,
compensation, or policy exceptions."""

DRAFT_USER_TEMPLATE = """Brand voice guidance (retrieved for this situation):
---
{voice_context}
---

Incoming {platform} comment from {author}:
\"\"\"{text}\"\"\"

Classification: category={category}, sentiment={sentiment}

Write only the reply text. No preamble, no quotation marks around it."""


def generate_draft(item: ContentItem, classification: ClassificationResult) -> str:
    from groq import Groq

    if not settings.groq.api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    store = get_store()
    voice_chunks = store.retrieve(item.text, top_k=settings.retrieval.top_k)
    voice_context = (
        "\n\n".join(c.text for c in voice_chunks)
        or "(no specific guidance matched — default to warm, direct, brief tone)"
    )

    client = Groq(api_key=settings.groq.api_key)
    user_prompt = DRAFT_USER_TEMPLATE.format(
        voice_context=voice_context,
        platform=item.platform.value,
        author=item.author,
        text=item.text,
        category=classification.category,
        sentiment=classification.sentiment,
    )

    last_error: Exception | None = None
    for attempt in range(settings.groq.max_retries):
        try:
            response = client.chat.completions.create(
                model=settings.groq.drafting_model,
                temperature=settings.groq.temperature,
                messages=[
                    {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            draft = (response.choices[0].message.content or "").strip()
            if not draft:
                raise ValueError("LLM returned empty draft.")
            logger.info("Drafted reply for %s item %s", item.platform.value, item.external_id)
            logger.debug("Draft content: %s", draft)
            return draft
        except Exception as e:
            last_error = e
            logger.warning("Draft attempt %d failed: %s", attempt + 1, e)

    raise RuntimeError(f"Draft generation failed after {settings.groq.max_retries} attempts: {last_error}")
