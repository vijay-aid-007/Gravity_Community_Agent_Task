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
from typing import List

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

    def flush():
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


class BrandVoiceStore:
    def __init__(self):
        self._chunks: List[VoiceChunk] = []
        self._index = None
        self._embedder = None
        self._reranker = None
        self._loaded = False

    def _lazy_load_models(self):
        if self._embedder is not None:
            return
        from sentence_transformers import SentenceTransformer, CrossEncoder

        self._embedder = SentenceTransformer(settings.retrieval.embedding_model)
        if settings.retrieval.use_reranker:
            self._reranker = CrossEncoder(settings.retrieval.reranker_model)

    def build(self):
        """Load brand_voice.md, chunk it, and build the FAISS index."""
        import faiss
        import numpy as np

        self._lazy_load_models()

        path = settings.retrieval.brand_voice_path
        raw_text = path.read_text(encoding="utf-8")
        self._chunks = _chunk_markdown(raw_text)

        embeddings = self._embedder.encode(
            [c.text for c in self._chunks], convert_to_numpy=True, normalize_embeddings=True
        )
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
        self._index.add(embeddings.astype(np.float32))
        self._loaded = True
        logger.info("Brand voice store built with %d chunks", len(self._chunks))

    def retrieve(self, query: str, top_k: int | None = None) -> List[VoiceChunk]:
        """Return the most relevant brand-voice chunks for a given comment,
        reranked by a cross-encoder for precision over the initial FAISS recall set."""
        import numpy as np

        if not self._loaded:
            self.build()

        top_k = top_k or settings.retrieval.top_k
        recall_k = min(len(self._chunks), max(top_k * 3, top_k))

        query_vec = self._embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, idxs = self._index.search(query_vec.astype(np.float32), recall_k)
        candidates = [self._chunks[i] for i in idxs[0] if i != -1]

        if not candidates:
            return []

        if self._reranker is not None and len(candidates) > 1:
            pairs = [[query, c.text] for c in candidates]
            rerank_scores = self._reranker.predict(pairs)
            ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
            candidates = [c for c, _ in ranked]

        return candidates[:top_k]


# Module-level singleton so the (relatively expensive) index build happens once.
_store: BrandVoiceStore | None = None


def get_store() -> BrandVoiceStore:
    global _store
    if _store is None:
        _store = BrandVoiceStore()
    return _store
