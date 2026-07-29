"""
Reddit ingestion.

Design decision: Reddit's "Responsible Builder Policy" (introduced late 2025)
closed self-service OAuth registration — new API access now requires manual
approval through Reddit's ticket form, with reported wait times ranging from
days to indefinite silence, and small non-commercial projects frequently
rejected outright. Waiting on that approval is incompatible with this
assessment's timeline.

Like X, this module supports two modes:

  - "mock" (default): reads simulated comments from tests/sample_inputs.json,
    so the full classify -> escalate/draft -> approve flow can be built,
    tested, and demoed end-to-end without an approved Reddit app.
  - "live": uses PRAW once an approved client_id/client_secret pair is
    available. Swapping modes is a one-line config change (REDDIT_MODE=live
    in .env) with no changes needed anywhere else in the pipeline, since
    both paths return ContentItem.

In the Activepieces build, live mode maps 1:1 to the native Reddit piece's
"New Comment" trigger once Reddit approves the app.
"""
from __future__ import annotations

import json
import logging
from typing import List

from config import settings
from ingestion.normalize import ContentItem, Platform

logger = logging.getLogger(__name__)


def _fetch_mock() -> List[ContentItem]:
    path = settings.reddit.mock_data_path
    if not path.exists():
        logger.warning("Mock Reddit data file not found at %s", path)
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = [
        ContentItem.from_dict(entry)
        for entry in data
        if entry.get("platform") == Platform.REDDIT.value
    ]
    logger.info("Loaded %d mock Reddit comments", len(items))
    return items


def _fetch_live() -> List[ContentItem]:
    try:
        import praw
    except ImportError as e:
        raise RuntimeError("praw is not installed. Run: pip install praw") from e

    cfg = settings.reddit
    if not cfg.client_id or not cfg.client_secret:
        logger.warning("Reddit credentials missing — returning empty result set.")
        return []

    reddit = praw.Reddit(
        client_id=cfg.client_id,
        client_secret=cfg.client_secret,
        user_agent=cfg.user_agent,
    )

    items: List[ContentItem] = []
    subreddit = reddit.subreddit(cfg.subreddit)
    for comment in subreddit.comments(limit=cfg.poll_limit):
        items.append(
            ContentItem(
                platform=Platform.REDDIT,
                external_id=comment.id,
                author=str(comment.author) if comment.author else "[deleted]",
                text=comment.body,
                url=f"https://reddit.com{comment.permalink}",
                created_at=str(comment.created_utc),
            )
        )
    logger.info("Fetched %d live Reddit comments", len(items))
    return items


def fetch_recent_mentions() -> List[ContentItem]:
    if settings.reddit.mode == "live":
        return _fetch_live()
    return _fetch_mock()