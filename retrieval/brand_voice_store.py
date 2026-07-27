"""
Lightweight RAG store for brand-voice retrieval.

Pipeline: chunk brand_voice.md -> embed with a local sentence-transformer
-> FAISS index for fast top-k retrieval -> optional cross-encoder reranker
to reorder candidates by actual relevance to the incoming comment before
handing the top snippet(s) to the draft generator.

Everything here runs locally/free — no embedding API costs, which matters
given the assessment's Groq-only LLM budget.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, List, Optional, cast

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class VoiceChunk:
    text: str
    section: str


def _chunk_markdown(raw_text: str) -> List[VoiceChunk]:
    """Split brand_voice.md into section-level chunks on '## ' headers."""
    chunks: List[VoiceChunk] = []
    current_section = "General"
    buffer: List[str] = []

    def flush() -> None:
        if buffer:
            chunks.append(VoiceChunk(text="\n".join(buffer).strip(), section=current_section))
            buffer.clear()

    for line in raw_text.splitlines():
        header_match = re.match(r"^##\s+(.*)", line)
        if header_match:
            flush()
            current_section = header_match.group(1).strip()
        else:
            buffer.append(line)
    flush()
    return [c for c in chunks if c.text]


def _keyword_score(query: str, chunk: VoiceChunk) -> int:
    """Return word-overlap count between query and chunk — used as fallback retrieval score."""
    query_words = set(re.findall(r"\w+", query.lower()))
    chunk_words = set(re.findall(r"\w+", chunk.text.lower()))
    return len(query_words & chunk_words)


class BrandVoiceStore:
    def __init__(self) -> None:
        self._chunks: List[VoiceChunk] = []
        self._index: Optional[Any] = None
        self._embedder: Optional[Any] = None
        self._reranker: Optional[Any] = None
        self._loaded: bool = False
        self._use_embeddings: bool = False

    def build(self) -> None:
        """Load brand_voice.md, chunk it, and build the FAISS index if available."""
        path = settings.retrieval.brand_voice_path
        raw_text = path.read_text(encoding="utf-8")
        self._chunks = _chunk_markdown(raw_text)

        try:
            import faiss  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
            from sentence_transformers import SentenceTransformer, CrossEncoder  # noqa: PLC0415

            self._embedder = SentenceTransformer(settings.retrieval.embedding_model)
            if settings.retrieval.use_reranker:
                self._reranker = CrossEncoder(settings.retrieval.reranker_model)

            embeddings = cast(Any, self._embedder).encode(
                [c.text for c in self._chunks], convert_to_numpy=True, normalize_embeddings=True
            )
            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            cast(Any, self._index).add(embeddings.astype(np.float32))
            self._use_embeddings = True
            logger.info("Brand voice store built with FAISS (%d chunks).", len(self._chunks))
        except ImportError:
            logger.warning(
                "sentence_transformers not available — using keyword fallback for brand voice retrieval. "
                "Install sentence-transformers for full RAG quality."
            )
            self._use_embeddings = False

        self._loaded = True

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[VoiceChunk]:
        """Return the most relevant brand-voice chunks for a given comment."""
        if not self._loaded:
            self.build()

        k = top_k or settings.retrieval.top_k

        if self._use_embeddings and self._embedder is not None and self._index is not None:
            import numpy as np  # noqa: PLC0415

            embedder = cast(Any, self._embedder)
            index = cast(Any, self._index)
            recall_k = min(len(self._chunks), max(k * 3, k))
            query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            _, idxs = index.search(query_vec.astype(np.float32), recall_k)
            candidates = [self._chunks[i] for i in idxs[0] if i != -1]

            if self._reranker is not None and len(candidates) > 1:
                reranker = cast(Any, self._reranker)
                pairs = [[query, c.text] for c in candidates]
                rerank_scores = reranker.predict(pairs)
                ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
                candidates = [c for c, _ in ranked]

            return candidates[:k]

        # Keyword fallback — used when sentence_transformers is not installed.
        scored = sorted(self._chunks, key=lambda c: _keyword_score(query, c), reverse=True)
        return scored[:k]


# Module-level singleton — index build is expensive, so it happens only once.
_store: Optional[BrandVoiceStore] = None


def get_store() -> BrandVoiceStore:
    """Return the shared BrandVoiceStore instance, building it on first call."""
    global _store
    if _store is None:
        _store = BrandVoiceStore()
    return _store
