"""
Tests for the core branching logic. LLM calls are monkeypatched so this
suite runs with zero API keys, zero cost, and zero network access.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.normalize import ContentItem, Platform
from ingestion import x_ingest
from classification.classifier import ClassificationResult
import orchestrator.pipeline as pipeline_mod


SAMPLE_PATH = Path(__file__).parent / "sample_inputs.json"


def load_samples() -> list:
    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        return [ContentItem.from_dict(d) for d in json.load(f)]


def _make_item(
    external_id: str = "t1",
    text: str = "hello",
    platform: Platform = Platform.REDDIT,
) -> ContentItem:
    return ContentItem(
        platform=platform,
        external_id=external_id,
        author="tester",
        text=text,
        url="https://example.com",
        created_at="2026-07-24T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# ContentItem
# ---------------------------------------------------------------------------

def test_content_item_roundtrip() -> None:
    samples = load_samples()
    assert len(samples) == 8
    assert samples[0].platform == Platform.REDDIT


def test_content_item_properties() -> None:
    item = _make_item(text="  hello world  ")
    assert not item.is_empty
    assert item.text_length == 11
    empty = _make_item(text="   ")
    assert empty.is_empty


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def test_x_mock_ingestion_filters_by_platform() -> None:
    items = x_ingest._fetch_mock()
    assert all(i.platform == Platform.X for i in items)
    assert len(items) == 3


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected_category",
    [
        ("I want a full refund right now or I'm disputing the charge.", "refund_demand"),
        ("My lawyer will hear about this, reporting you to consumer court.", "legal_threat"),
    ],
)
def test_escalation_categories_force_sensitivity_flag(
    monkeypatch: pytest.MonkeyPatch, text: str, expected_category: str
) -> None:
    """Even if the LLM omits sensitivity_flag=True, the classifier must force it for escalation categories."""
    from classification import classifier as classifier_mod

    fake_result = ClassificationResult(
        sentiment="negative",
        category=expected_category,  # type: ignore[arg-type]
        sensitivity_flag=False,       # intentionally wrong — tests the safety net
        reasoning="test",
        confidence=0.9,
    )

    class FakeCompletions:
        def create(self, **kwargs: object) -> object:
            class Msg:
                content = fake_result.model_dump_json()
            class Choice:
                message = Msg()
            class Resp:
                choices = [Choice()]
            return Resp()

    class FakeClient:
        class chat:
            completions = FakeCompletions()

    monkeypatch.setattr(classifier_mod, "_get_client", lambda: FakeClient())

    result = classifier_mod.classify(_make_item(text=text))
    assert result.sensitivity_flag is True
    assert result.category == expected_category
    assert result.model_used is not None
    assert len(result.model_used) > 0


def test_classify_empty_text_returns_safe_default() -> None:
    from classification import classifier as classifier_mod

    result = classifier_mod.classify(_make_item(text="   "))
    assert result.sensitivity_flag is False
    assert result.category == "general_question"
    assert result.model_used is not None
    assert result.model_used == "none"


# ---------------------------------------------------------------------------
# Pipeline routing
# ---------------------------------------------------------------------------

def test_pipeline_routes_sensitive_items_to_escalation(monkeypatch: pytest.MonkeyPatch) -> None:
    escalated_calls: list = []
    queued_calls: list = []

    def fake_classify(item: ContentItem) -> ClassificationResult:
        is_sensitive = item.external_id in {"r2", "y2", "x2"}
        return ClassificationResult(
            sentiment="negative" if is_sensitive else "neutral",
            category="refund_demand" if is_sensitive else "general_question",  # type: ignore[arg-type]
            sensitivity_flag=is_sensitive,
            reasoning="test",
            confidence=0.95,
        )

    monkeypatch.setattr(pipeline_mod, "classify", fake_classify)
    monkeypatch.setattr(pipeline_mod, "escalate", lambda item, c: escalated_calls.append(item.external_id))
    monkeypatch.setattr(pipeline_mod, "generate_draft", lambda item, c: "draft text")
    monkeypatch.setattr(pipeline_mod, "enqueue", lambda item, draft: queued_calls.append(item.external_id))
    monkeypatch.setattr(pipeline_mod, "is_already_pending", lambda p, e: False)

    samples = load_samples()
    results = [pipeline_mod.process_item(item) for item in samples]

    assert "r2" in escalated_calls
    assert "y2" in escalated_calls
    assert "x2" in escalated_calls
    assert len(escalated_calls) == 3
    assert len(queued_calls) == len(samples) - 3
    assert all(r.action in {"escalated", "queued_for_approval"} for r in results)


def test_pipeline_drops_spam(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_mod, "is_already_pending", lambda p, e: False)
    monkeypatch.setattr(
        pipeline_mod, "classify",
        lambda item: ClassificationResult(
            sentiment="neutral",
            category="spam",  # type: ignore[arg-type]
            sensitivity_flag=False,
            reasoning="spam",
            confidence=0.9,
        ),
    )
    result = pipeline_mod.process_item(_make_item(text="Buy crypto now!!!"))
    assert result.action == "dropped_spam"


def test_pipeline_skips_empty_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_mod, "is_already_pending", lambda p, e: False)
    result = pipeline_mod.process_item(_make_item(text="   "))
    assert result.action == "skipped_empty"


def test_pipeline_skips_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_mod, "is_already_pending", lambda p, e: True)
    result = pipeline_mod.process_item(_make_item(text="hello"))
    assert result.action == "skipped_duplicate"
