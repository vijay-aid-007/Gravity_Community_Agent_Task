"""
Tests for the core branching logic. LLM calls are monkeypatched so this
suite runs with zero API keys / zero cost / zero network access — useful
both for CI and for quickly proving the decision logic before wiring the
equivalent branches into the Activepieces builder.
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


def load_samples():
    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        return [ContentItem.from_dict(d) for d in json.load(f)]


def test_content_item_roundtrip():
    samples = load_samples()
    assert len(samples) == 8
    assert samples[0].platform == Platform.REDDIT


def test_x_mock_ingestion_filters_by_platform():
    items = x_ingest._fetch_mock()
    assert all(i.platform == Platform.X for i in items)
    assert len(items) == 3


@pytest.mark.parametrize(
    "text,expected_category",
    [
        ("I want a full refund right now or I'm disputing the charge.", "refund_demand"),
        ("My lawyer will hear about this, reporting you to consumer court.", "legal_threat"),
    ],
)
def test_escalation_categories_force_sensitivity_flag(monkeypatch, text, expected_category):
    """Even if the LLM forgets to set sensitivity_flag, the classifier must
    force it to True for known escalation categories (safety net)."""
    from classification import classifier as classifier_mod

    fake_result = ClassificationResult(
        sentiment="negative",
        category=expected_category,
        sensitivity_flag=False,  # intentionally wrong, to test the safety net
        reasoning="test",
        confidence=0.9,
    )

    class FakeCompletions:
        def create(self, **kwargs):
            class Msg:
                content = fake_result.model_dump_json()
            class Choice:
                message = Msg()
            class Resp:
                choices = [Choice()]
            return Resp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(classifier_mod, "_get_client", lambda: FakeClient())

    item = ContentItem(
        platform=Platform.REDDIT, external_id="t1", author="tester",
        text=text, url="https://example.com", created_at="2026-07-24T00:00:00Z",
    )
    result = classifier_mod.classify(item)
    assert result.sensitivity_flag is True
    assert result.category == expected_category


def test_pipeline_routes_sensitive_items_to_escalation(monkeypatch):
    escalated_calls = []
    queued_calls = []

    def fake_classify(item):
        is_sensitive = item.external_id in {"r2", "y2", "x2"}
        return ClassificationResult(
            sentiment="negative" if is_sensitive else "neutral",
            category="refund_demand" if is_sensitive else "general_question",
            sensitivity_flag=is_sensitive,
            reasoning="test",
            confidence=0.95,
        )

    monkeypatch.setattr(pipeline_mod, "classify", fake_classify)
    monkeypatch.setattr(pipeline_mod, "escalate", lambda item, c: escalated_calls.append(item.external_id))
    monkeypatch.setattr(pipeline_mod, "generate_draft", lambda item, c: "draft text")
    monkeypatch.setattr(pipeline_mod, "enqueue", lambda item, draft: queued_calls.append(item.external_id))

    samples = load_samples()
    results = [pipeline_mod.process_item(item) for item in samples]

    assert "r2" in escalated_calls
    assert "y2" in escalated_calls
    assert "x2" in escalated_calls
    assert len(escalated_calls) == 3
    assert len(queued_calls) == len(samples) - 3
    assert all(r.action in {"escalated", "queued_for_approval"} for r in results)
