"""
Every platform (Reddit, YouTube, X) has a different raw payload shape.
This module defines the single normalized schema the rest of the
pipeline operates on, so classification/RAG/escalation never need to
know which platform a mention came from.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum


class Platform(str, Enum):
    REDDIT = "reddit"
    YOUTUBE = "youtube"
    X = "x"


@dataclass
class ContentItem:
    platform: Platform
    external_id: str          # platform-native comment/mention id
    author: str
    text: str
    url: str
    created_at: str           # ISO8601
    fetched_at: str = ""

    def __post_init__(self):
        if not self.fetched_at:
            self.fetched_at = datetime.now(timezone.utc).isoformat()

    @property
    def is_empty(self) -> bool:
        return not self.text or not self.text.strip()

    @property
    def text_length(self) -> int:
        return len(self.text.strip())

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"ContentItem({self.platform.value}/{self.external_id} by {self.author!r}: {preview!r})"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["platform"] = self.platform.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "ContentItem":
        d = dict(d)
        d["platform"] = Platform(d["platform"])
        return ContentItem(**d)
