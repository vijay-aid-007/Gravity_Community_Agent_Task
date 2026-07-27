# Gravity AI Assessment — Community Management Agent

Reference implementation for: *"Watch our mentions across Reddit, YouTube, and X,
draft replies in our voice, and flag anything sensitive to the team."*

## Why this exists

The actual submission is an `agent.json` **exported from the Activepieces builder
UI** — this repo is not that file. It exists to lock down the exact prompts,
classification schema, RAG logic, and escalation rules with real, tested code
*before* clicking through the no-code builder, so the Activepieces build is a
direct translation rather than trial-and-error design-while-building.

## Architecture

```
config/            Environment-driven settings + brand_voice.md (RAG source doc)
ingestion/         Reddit (PRAW), YouTube (Data API v3), X (mock/live toggle)
classification/    Groq LLM call -> structured JSON -> Pydantic validation
retrieval/          FAISS + sentence-transformer embeddings + cross-encoder rerank
drafting/           RAG-grounded reply generation, brand-voice constrained
escalation/         Slack notification for sensitive items (no draft generated)
approval/           SQLite-backed approval queue + Slack approval notification
orchestrator/       Ties it all into one pipeline: ingest -> classify -> branch
tests/              Sample data (all categories x all platforms) + pytest suite
```

## Data flow

```
Reddit ─┐
YouTube ├─> normalize -> classify (LLM) ─┬─ sensitive? ──> Slack escalation (STOP)
X (mock)┘                                 └─ not sensitive ─> RAG retrieve
                                                              -> draft reply
                                                              -> approval queue
                                                              -> Slack notify reviewer
```

## Design decisions worth knowing for the interview / write-up

1. **X/Twitter mock mode**: X's API requires a paid tier for search/mentions.
   Rather than block the assessment on procuring paid access, `ingestion/x_ingest.py`
   defaults to reading simulated mentions from `tests/sample_inputs.json`.
   Flipping `X_MODE=live` in `.env` with a bearer token switches to the real
   endpoint with zero changes anywhere else in the pipeline — both paths
   return the same `ContentItem` shape.

2. **Fail-safe classification default**: if the LLM call fails or returns
   malformed JSON after retries, the pipeline defaults to **escalation**, not
   silent auto-reply. An unnecessary Slack ping costs seconds; an unescalated
   legal threat costs real trust.

3. **Escalation categories always force `sensitivity_flag=True`** in code,
   even if the LLM forgets to set it — a deterministic safety net on top of
   a probabilistic classifier (tested in `test_escalation_categories_force_sensitivity_flag`).

4. **RAG over bare prompting for drafts**: brand voice guidance is retrieved
   per-comment (top-k, cross-encoder reranked) rather than stuffed wholesale
   into every prompt — cheaper, more consistent, and scales as the brand
   voice doc grows.

5. **Groq free tier**: `llama-3.3-70b-versatile` for classification/drafting
   quality, `llama-3.1-8b-instant` as an automatic fallback if the primary
   model is rate-limited — the classifier already tries both.

## Running it locally

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in whichever keys you have; missing ones degrade gracefully
python -m orchestrator.pipeline
```

## Running the tests (no API keys required)

```bash
pytest tests/ -v
```

All classification/escalation/routing logic is tested with mocked LLM
responses — proves the decision logic is correct independent of any live
API call.

## Translating this into the Activepieces build

| This module | Activepieces equivalent |
|---|---|
| `ingestion/reddit_ingest.py` | Native **Reddit** piece, "New Comment" trigger |
| `ingestion/youtube_ingest.py` | Native **YouTube** piece, comment trigger |
| `ingestion/x_ingest.py` (mock) | **Schedule/Webhook trigger** reading a Google Sheet or Table seeded with `tests/sample_inputs.json` rows, standing in for live X mentions |
| `ingestion/normalize.py` | A **Code piece** (JS) or simple field-mapping step after each trigger, writing into one shared **Table** |
| `classification/*` | **AI piece** (Claude/OpenAI-compatible, point at Groq's endpoint) with the same system prompt, JSON mode |
| Branch on `sensitivity_flag` | **Condition/Branch** step |
| `escalation/slack_notifier.py` | **Slack** piece, "Send Message" action with Block Kit |
| `retrieval/brand_voice_store.py` | Either a **Table** of pre-chunked brand voice snippets with simple keyword filtering, or a **Code piece** running the same FAISS logic |
| `drafting/draft_generator.py` | Second **AI piece** call, same prompt template |
| `approval/approval_queue.py` | **Table** row insert + **Slack** "Send Message" for the approval ping; a **Human Input / approval** piece if available handles the approve/reject step natively |

Build it in this order — ingestion first, then classification + branch (this
is the part most likely to be scored on decision quality), then drafting/RAG,
then approval. Get a working baseline submitted early; the assessment allows
unlimited resubmissions until the deadline.
