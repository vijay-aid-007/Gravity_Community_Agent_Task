"""
Every generated draft is queued for human approval before it is ever
posted back to Reddit/YouTube/X. Persists state to SQLite and notifies
a Slack approval channel.
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

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
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    decided_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_platform_external ON approval_queue(platform, external_id, status);
"""


@contextmanager
def _connect():
    db_path: Path = settings.pipeline.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for stmt in _SCHEMA.strip().split(";"):
            if stmt.strip():
                conn.execute(stmt)
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


def _row_to_entry(row: sqlite3.Row) -> QueueEntry:  # type: ignore[type-arg]
    return QueueEntry(**dict(row))


def is_already_pending(platform: str, external_id: str) -> bool:
    """Return True if this item already has a pending draft — prevents duplicate queue entries."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM approval_queue WHERE platform=? AND external_id=? AND status='pending'",
            (platform, external_id),
        ).fetchone()
    return row is not None


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

    preview_text = item.text[:300] + ("…" if len(item.text) > 300 else "")
    preview_draft = draft_reply[:300] + ("…" if len(draft_reply) > 300 else "")

    client = WebClient(token=cfg.bot_token)
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": f"📝 Draft reply pending approval (#{entry_id})"}},
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*Platform:*\n{item.platform.value}"},
            {"type": "mrkdwn", "text": f"*Author:*\n{item.author}"},
        ]},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Original:*\n>{preview_text}"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Draft reply:*\n>{preview_draft}"}},
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


def get_pending() -> List[QueueEntry]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM approval_queue WHERE status='pending' ORDER BY created_at"
        ).fetchall()
    return [_row_to_entry(r) for r in rows]


def get_by_id(entry_id: int) -> Optional[QueueEntry]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM approval_queue WHERE id=?", (entry_id,)).fetchone()
    return _row_to_entry(row) if row else None


def get_all(status: Optional[str] = None) -> List[QueueEntry]:
    with _connect() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM approval_queue WHERE status=? ORDER BY created_at", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM approval_queue ORDER BY created_at").fetchall()
    return [_row_to_entry(r) for r in rows]


def get_stats() -> dict:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) as count FROM approval_queue GROUP BY status"
        ).fetchall()
    return {row["status"]: row["count"] for row in rows}
