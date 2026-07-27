"""
X (Twitter) ingestion.

Design decision: X's API requires a paid tier for mention/search access.
Rather than block the whole pipeline on procuring paid API access during
a timed assessment, this module supports two modes:

  - "mock" (default): reads simulated mentions from tests/sample_inputs.json,
    so the full classify -> escalate/draft -> approve flow can be built,
    tested, and demoed end-to-end without spending money.
  - "live": uses the X API v2 recent-search endpoint once a bearer token
    with the appropriate access level is available. Swapping modes is a
    one-line config change (X_MODE=live in .env) with no changes needed
    anywhere else in the pipeline, since both paths return ContentItem.
"""
from __future__ import annotations

import json
import logging
from typing import List

from config import settings
from ingestion.normalize import ContentItem, Platform

logger = logging.getLogger(__name__)


def _fetch_mock() -> List[ContentItem]:
    path = settings.x.mock_data_path
    if not path.exists():
        logger.warning("Mock X data file not found at %s", path)
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = [
        ContentItem.from_dict(entry)
        for entry in data
        if entry.get("platform") == Platform.X.value
    ]
    logger.info("Loaded %d mock X mentions", len(items))
    return items


def _fetch_live() -> List[ContentItem]:
    import requests

    cfg = settings.x
    if not cfg.bearer_token:
        logger.warning("X bearer token missing — returning empty result set.")
        return []

    headers = {"Authorization": f"Bearer {cfg.bearer_token}"}
    params = {
        "query": "@YourBrandHandle",
        "max_results": 25,
        "tweet.fields": "created_at,author_id",
    }
    resp = requests.get(
        "https://api.x.com/2/tweets/search/recent",
        headers=headers,
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()

    items: List[ContentItem] = []
    for tweet in payload.get("data", []):
        items.append(
            ContentItem(
                platform=Platform.X,
                external_id=tweet["id"],
                author=tweet.get("author_id", "unknown"),
                text=tweet.get("text", ""),
                url=f"https://x.com/i/web/status/{tweet['id']}",
                created_at=tweet.get("created_at", ""),
            )
        )
    logger.info("Fetched %d live X mentions", len(items))
    return items


def fetch_recent_mentions() -> List[ContentItem]:
    if settings.x.mode == "live":
        return _fetch_live()
    return _fetch_mock()
