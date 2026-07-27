"""
Sends sensitive items (refund demands, legal threats, angry customers)
straight to the team's Slack channel instead of drafting a public reply.
"""
from __future__ import annotations

import logging

from config import settings
from classification.classifier import ClassificationResult
from ingestion.normalize import ContentItem

logger = logging.getLogger(__name__)

_CATEGORY_EMOJI = {
    "refund_demand": "💸",
    "legal_threat": "⚖️",
    "angry_customer": "😡",
    "minor_complaint": "⚠️",
    "spam": "🚫",
    "general_question": "❓",
    "praise": "⭐",
}


def _format_escalation_blocks(item: ContentItem, classification: ClassificationResult) -> list:
    emoji = _CATEGORY_EMOJI.get(classification.category, "🚨")
    text_preview = item.text[:500] + ("…" if len(item.text) > 500 else "")
    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{emoji} Escalation: {classification.category}"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Platform:*\n{item.platform.value}"},
                {"type": "mrkdwn", "text": f"*Author:*\n{item.author}"},
                {"type": "mrkdwn", "text": f"*Sentiment:*\n{classification.sentiment}"},
                {"type": "mrkdwn", "text": f"*Confidence:*\n{classification.confidence:.2f}"},
            ],
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Text:*\n>{text_preview}"},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Why flagged:* {classification.reasoning}"},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"<{item.url}|View original post>"},
        },
    ]


def escalate(item: ContentItem, classification: ClassificationResult) -> None:
    try:
        from slack_sdk import WebClient
        from slack_sdk.errors import SlackApiError
    except ImportError as e:
        raise RuntimeError("slack_sdk is not installed. Run: pip install slack_sdk") from e

    cfg = settings.slack
    if not cfg.bot_token:
        logger.warning(
            "Slack bot token missing — escalation not sent (logged only): [%s] %s",
            classification.category, item.text[:80],
        )
        return

    client = WebClient(token=cfg.bot_token)
    try:
        client.chat_postMessage(
            channel=cfg.escalation_channel,
            blocks=_format_escalation_blocks(item, classification),
            text=f"Escalation: {classification.category} on {item.platform.value} by {item.author}",
        )
        logger.info(
            "Escalated %s item %s to %s (category=%s)",
            item.platform.value, item.external_id, cfg.escalation_channel, classification.category,
        )
    except SlackApiError as e:
        logger.error("Slack escalation post failed: %s", e.response["error"])
        raise
