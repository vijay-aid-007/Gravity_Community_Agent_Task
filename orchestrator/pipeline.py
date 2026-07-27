"""
End-to-end orchestration:

  ingest (Reddit + YouTube + X) -> normalize -> classify
    -> if sensitive: escalate to Slack (no draft generated)
    -> else: retrieve brand-voice context -> draft reply -> queue for approval

Run directly: python -m orchestrator.pipeline
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from classification.classifier import classify, ClassificationResult
from drafting.draft_generator import generate_draft
from escalation.slack_notifier import escalate
from approval.approval_queue import enqueue, is_already_pending
from ingestion import reddit_ingest, youtube_ingest, x_ingest
from ingestion.normalize import ContentItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    item: ContentItem
    classification: ClassificationResult
    action: str  # "escalated" | "queued_for_approval" | "dropped_spam" | "skipped_empty" | "skipped_duplicate"
    draft: Optional[str] = None

    @property
    def summary(self) -> str:
        return (
            f"{self.item.platform.value}/{self.item.external_id} "
            f"-> {self.action} "
            f"[{self.classification.category}, conf={self.classification.confidence:.2f}]"
        )


def ingest_all() -> List[ContentItem]:
    items: List[ContentItem] = []
    for fetch_fn, label in (
        (reddit_ingest.fetch_recent_mentions, "Reddit"),
        (youtube_ingest.fetch_recent_comments, "YouTube"),
        (x_ingest.fetch_recent_mentions, "X"),
    ):
        try:
            fetched = fetch_fn()
            items.extend(fetched)
            logger.info("Ingested %d items from %s", len(fetched), label)
        except Exception as e:
            logger.error("Ingestion failed for %s: %s", label, e)
    logger.info("Total items ingested: %d", len(items))
    return items


def process_item(item: ContentItem) -> ProcessingResult:
    if item.is_empty:
        logger.info("Skipping empty item %s/%s", item.platform.value, item.external_id)
        return ProcessingResult(
            item=item,
            classification=ClassificationResult(
                sentiment="neutral", category="general_question",
                sensitivity_flag=False, reasoning="Empty content.", confidence=0.0,
            ),
            action="skipped_empty",
        )

    if is_already_pending(item.platform.value, item.external_id):
        logger.info("Skipping duplicate pending item %s/%s", item.platform.value, item.external_id)
        return ProcessingResult(
            item=item,
            classification=ClassificationResult(
                sentiment="neutral", category="general_question",
                sensitivity_flag=False, reasoning="Already pending approval.", confidence=0.0,
            ),
            action="skipped_duplicate",
        )

    classification = classify(item)
    logger.info(
        "Classified %s/%s: category=%s sensitivity=%s confidence=%.2f",
        item.platform.value, item.external_id,
        classification.category, classification.sensitivity_flag, classification.confidence,
    )

    if classification.sensitivity_flag:
        escalate(item, classification)
        return ProcessingResult(item=item, classification=classification, action="escalated")

    if classification.category == "spam" and classification.confidence >= 0.6:
        logger.info("Dropping spam item %s/%s", item.platform.value, item.external_id)
        return ProcessingResult(item=item, classification=classification, action="dropped_spam")

    draft = generate_draft(item, classification)
    enqueue(item, draft)
    return ProcessingResult(item=item, classification=classification, action="queued_for_approval", draft=draft)


def run_pipeline() -> List[ProcessingResult]:
    items = ingest_all()
    results = [process_item(item) for item in items]

    summary: dict = {}
    for r in results:
        summary[r.action] = summary.get(r.action, 0) + 1
        logger.debug(r.summary)

    logger.info("Pipeline run complete: %s", summary)
    return results


if __name__ == "__main__":
    run_pipeline()
