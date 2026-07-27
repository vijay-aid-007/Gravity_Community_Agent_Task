"""
Reddit ingestion using PRAW (official-style Reddit API wrapper).
Pulls the newest comments/mentions from a configured subreddit.

In the Activepieces build, this maps 1:1 to the native Reddit piece's
"New Comment" trigger — this module exists so the exact filtering and
normalization logic can be tested locally before wiring it into a
no-code trigger.
"""
from __future__ import annotations

import logging
from typing import List

from config import settings
from ingestion.normalize import ContentItem, Platform

logger = logging.getLogger(__name__)


def fetch_recent_mentions() -> List[ContentItem]:
    """Fetch the most recent comments from the configured subreddit."""
    try:
        import praw
    except ImportError as e:
        raise RuntimeError(
            "praw is not installed. Run: pip install praw"
        ) from e

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
    logger.info("Fetched %d Reddit comments", len(items))
    return items
