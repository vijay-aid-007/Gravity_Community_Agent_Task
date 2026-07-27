"""
Every generated draft is queued for human approval before it is ever
posted back to Reddit/YouTube/X. This module persists queue state to
a local SQLite file (agent_state.sqlite3) and posts a notification to
a Slack approval channel with a link back to the original mention.

Approval itself (the actual click/decision) happens wherever your team
already works — Slack thread reply, or a lightweight admin view — and
is recorded via `approve()` / `reject()`. Posting the approved reply
back to the source platform is a stub here (`post_approved_reply`)
since it's a straightforward platform-API call once approved.
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import settings
from ingestion.normalize import ContentItem

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS approval_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    platform TEXT NOT NULL,
    external_id TEXT NOT NULL,
    author TEXT NOT NULL,
    original_text TEXT NOT NULL,
    url TEXT NOT NULL,
    draft_reply TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | approved | rejected | posted
    created_at TEXT NOT NULL,
    decided_at TEXT
);
"""


@contextmanager
def _connect():
    db_path: Path = settings.pipeline.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


@dataclass
class QueueEntry:
    id: int
    platform: str
    external_id: str
    author: str
    original_text: str
    url: str
    draft_reply: str
    status: str
    created_at: str
    decided_at: Optional[str]


def enqueue(item: ContentItem, draft_reply: str) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO approval_queue
               (platform, external_id, author, original_text, url, draft_reply, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                item.platform.value,
                item.external_id,
                item.author,
                item.text,
                item.url,
                draft_reply,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        entry_id = cur.lastrowid
    _notify_slack(entry_id, item, draft_reply)
    logger.info("Queued draft reply id=%d for %s item %s", entry_id, item.platform.value, item.external_id)
    return entry_id


def _notify_slack(entry_id: int, item: ContentItem, draft_reply: str) -> None:
    try:
        from slack_sdk import WebClient
        from slack_sdk.errors import SlackApiError
    except ImportError:
        logger.warning("slack_sdk not installed — skipping approval notification.")
        return

    cfg = settings.slack
    if not cfg.bot_token:
        logger.warning("Slack bot token missing — approval notification skipped.")
        return

    client = WebClient(token=cfg.bot_token)
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": f"📝 Draft reply pending approval (#{entry_id})"}},
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*Platform:*\n{item.platform.value}"},
            {"type": "mrkdwn", "text": f"*Author:*\n{item.author}"},
        ]},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Original:*\n>{item.text}"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Draft reply:*\n>{draft_reply}"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"<{item.url}|View original post>"}},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": f"React ✅ to approve or ❌ to reject queue id {entry_id}"}]},
    ]
    try:
        client.chat_postMessage(
            channel=cfg.approval_channel,
            blocks=blocks,
            text=f"Draft reply pending approval (#{entry_id})",
        )
    except SlackApiError as e:
        logger.error("Slack approval notification failed: %s", e.response["error"])


def approve(entry_id: int) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE approval_queue SET status='approved', decided_at=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), entry_id),
        )
    logger.info("Approved queue id=%d", entry_id)


def reject(entry_id: int) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE approval_queue SET status='rejected', decided_at=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), entry_id),
        )
    logger.info("Rejected queue id=%d", entry_id)


def get_pending() -> list[QueueEntry]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM approval_queue WHERE status='pending' ORDER BY created_at"
        ).fetchall()
        cols = [d[0] for d in conn.execute("SELECT * FROM approval_queue LIMIT 0").description]
    return [QueueEntry(**dict(zip(cols, row))) for row in rows]
