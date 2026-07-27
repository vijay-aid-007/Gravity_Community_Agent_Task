"""
YouTube ingestion via the YouTube Data API v3 (free, key-based).
Pulls recent top-level comments across the channel's uploads.

Maps to Activepieces' native YouTube piece in the final flow; kept
here as a pure-Python module so the pagination/normalization logic
is unit-testable without the builder UI.
"""
from __future__ import annotations

import logging
from typing import List

from config import settings
from ingestion.normalize import ContentItem, Platform

logger = logging.getLogger(__name__)


def fetch_recent_comments() -> List[ContentItem]:
    try:
        from googleapiclient.discovery import build
    except ImportError as e:
        raise RuntimeError(
            "google-api-python-client is not installed. "
            "Run: pip install google-api-python-client"
        ) from e

    cfg = settings.youtube
    if not cfg.api_key or not cfg.channel_id:
        logger.warning("YouTube credentials missing — returning empty result set.")
        return []

    youtube = build("youtube", "v3", developerKey=cfg.api_key)

    items: List[ContentItem] = []
    request = youtube.commentThreads().list(
        part="snippet",
        allThreadsRelatedToChannelId=cfg.channel_id,
        maxResults=cfg.poll_limit,
        order="time",
        textFormat="plainText",
    )
    response = request.execute()

    for thread in response.get("items", []):
        top = thread["snippet"]["topLevelComment"]["snippet"]
        comment_id = thread["snippet"]["topLevelComment"]["id"]
        video_id = thread["snippet"]["videoId"]
        items.append(
            ContentItem(
                platform=Platform.YOUTUBE,
                external_id=comment_id,
                author=top.get("authorDisplayName", "unknown"),
                text=top.get("textDisplay", ""),
                url=f"https://www.youtube.com/watch?v={video_id}&lc={comment_id}",
                created_at=top.get("publishedAt", ""),
            )
        )
    logger.info("Fetched %d YouTube comments", len(items))
    return items
