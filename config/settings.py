"""
Central configuration for the Community Management Agent.
All values are loaded from environment variables (.env) so no secrets
are ever hard-coded. Copy .env.example to .env and fill in real values.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class GroqConfig:
    api_key: str = os.getenv("GROQ_API_KEY", "").strip()  # strip accidental whitespace
    base_url: str = "https://api.groq.com/openai/v1"
    # Primary model: best quality/reasoning on Groq's free tier.
    classification_model: str = os.getenv("GROQ_CLASSIFY_MODEL", "llama-3.3-70b-versatile")
    drafting_model: str = os.getenv("GROQ_DRAFT_MODEL", "llama-3.3-70b-versatile")
    # Fallback: higher free-tier request budget, used if the primary is rate-limited.
    fallback_model: str = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")
    temperature: float = 0.2  # low temperature -> grounded, consistent classifications
    max_retries: int = 3


@dataclass(frozen=True)
class RedditConfig:
    client_id: str = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret: str = os.getenv("REDDIT_CLIENT_SECRET", "")
    user_agent: str = os.getenv("REDDIT_USER_AGENT", "gravity-community-agent/0.1")
    subreddit: str = os.getenv("REDDIT_SUBREDDIT", "YourBrandSubreddit")
    poll_limit: int = int(os.getenv("REDDIT_POLL_LIMIT", "25"))


@dataclass(frozen=True)
class YouTubeConfig:
    api_key: str = os.getenv("YOUTUBE_API_KEY", "")
    channel_id: str = os.getenv("YOUTUBE_CHANNEL_ID", "")
    poll_limit: int = int(os.getenv("YOUTUBE_POLL_LIMIT", "25"))


@dataclass(frozen=True)
class XConfig:
    """
    X/Twitter's API requires a paid tier for search/mention access.
    To avoid blocking the pipeline on a paid subscription during this
    assessment, ingestion runs in MOCK mode by default, reading
    simulated mentions from tests/sample_inputs.json. Flip to live mode
    by setting X_MODE=live and supplying a bearer token once available.
    """
    mode: str = os.getenv("X_MODE", "mock")  # "mock" | "live"
    bearer_token: str = os.getenv("X_BEARER_TOKEN", "")
    mock_data_path: Path = BASE_DIR / "tests" / "sample_inputs.json"


@dataclass(frozen=True)
class SlackConfig:
    bot_token: str = os.getenv("SLACK_BOT_TOKEN", "")
    escalation_channel: str = os.getenv("SLACK_ESCALATION_CHANNEL", "#community-escalations")
    approval_channel: str = os.getenv("SLACK_APPROVAL_CHANNEL", "#reply-approvals")


@dataclass(frozen=True)
class RetrievalConfig:
    brand_voice_path: Path = BASE_DIR / "config" / "brand_voice.md"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    top_k: int = 3
    use_reranker: bool = True
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


@dataclass(frozen=True)
class PipelineConfig:
    # Sensitivity categories that must always be escalated, never auto-drafted.
    escalation_categories: tuple = (
        "refund_demand",
        "legal_threat",
        "angry_customer",
    )
    spam_confidence_floor: float = 0.6
    db_path: Path = BASE_DIR / "data" / "agent_state.sqlite3"


groq = GroqConfig()
reddit = RedditConfig()
youtube = YouTubeConfig()
x = XConfig()
slack = SlackConfig()
retrieval = RetrievalConfig()
pipeline = PipelineConfig()
