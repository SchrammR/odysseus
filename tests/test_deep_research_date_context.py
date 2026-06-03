"""Regression tests for issue #1341 — deep research used the model's
training-cutoff year (e.g. "best Python tutorials 2025") because the
query-generation and planning prompts never told the LLM the current date.

The chat/agent path already injects "Today is ..." (src/agent_loop.py); deep
research had no equivalent. These tests pin that the current year now reaches
the LLM at both the planning and query-generation steps, without needing a live
LLM or DB.
"""
import asyncio
from datetime import datetime

from src.deep_research import (
    DeepResearcher,
    CATEGORY_PROMPTS,
    current_date_context,
    RESEARCH_PLAN_PROMPT,
)


def _this_year() -> str:
    return datetime.now().astimezone().strftime("%Y")


def test_current_date_context_names_the_real_year():
    ctx = current_date_context()
    assert _this_year() in ctx
    # It must actively steer the model away from training-data years.
    assert "training data" in ctx.lower()


def test_generate_queries_prompt_carries_the_current_year():
    # Build without the heavy __init__; _generate_queries only needs these.
    r = DeepResearcher.__new__(DeepResearcher)
    r.research_plan = ""
    r.queries_used = set()

    seen = {}

    async def _fake_llm(messages, **kwargs):
        seen["prompt"] = messages[0]["content"]
        return '["python tutorials", "python guides"]'

    r._llm = _fake_llm

    queries = asyncio.run(r._generate_queries("best python tutorials", "", 1))

    assert queries  # sanity: the JSON array parsed
    # The fix: the real current year is in the prompt the LLM actually sees.
    assert _this_year() in seen["prompt"]


def test_plan_prompt_carries_the_current_year():
    r = DeepResearcher.__new__(DeepResearcher)

    seen = {}

    async def _fake_llm(messages, **kwargs):
        seen["prompt"] = messages[0]["content"]
        return "{}"

    r._llm = _fake_llm

    asyncio.run(r._create_plan("what changed this year"))

    assert _this_year() in seen["prompt"]
    # The base template itself stays year-agnostic; the year comes from the
    # prepended context, proving the wiring (not a hard-coded prompt edit).
    assert _this_year() not in RESEARCH_PLAN_PROMPT


def test_scientific_category_promotes_scholarly_sources():
    prompt = CATEGORY_PROMPTS["scientific"].lower()
    assert "peer-reviewed" in prompt
    assert "preprint" in prompt
    assert "evidence base" in prompt
    assert "doi" in prompt


def test_scientific_query_prompt_targets_scholarly_indexes():
    r = DeepResearcher.__new__(DeepResearcher)
    r.research_plan = ""
    r.queries_used = set()
    r.category = "scientific"

    seen = {}

    async def _fake_llm(messages, **kwargs):
        seen["prompt"] = messages[0]["content"]
        return '["site:arxiv.org graph neural network reproducibility", "IEEE Xplore graph neural network benchmark"]'

    r._llm = _fake_llm

    queries = asyncio.run(r._generate_queries("graph neural network reproducibility", "", 1))

    assert queries
    prompt = seen["prompt"].lower()
    assert "arxiv" in prompt
    assert "google scholar" in prompt
    assert "dl.acm.org" in prompt
    assert "ieeexplore.ieee.org" in prompt
    assert "medium" in prompt
    assert "contextual only" in prompt


def test_scientific_arxiv_search_parses_atom_feed(monkeypatch):
    import httpx

    captured = {}

    class _Response:
        text = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/2601.01234v1</id>
            <published>2026-01-03T00:00:00Z</published>
            <title> Embodied Agents in Driving Contexts </title>
            <summary> A controlled simulator study of embodiment and empathy. </summary>
            <link href="http://arxiv.org/abs/2601.01234v1" rel="alternate" type="text/html"/>
            <link title="pdf" href="http://arxiv.org/pdf/2601.01234v1" rel="related" type="application/pdf"/>
          </entry>
        </feed>"""

        def raise_for_status(self):
            return None

    def _fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(httpx, "get", _fake_get)

    results = DeepResearcher._search_arxiv(
        "site:arxiv.org IEEE Xplore embodied empathy driving",
        count=3,
    )

    assert captured["url"] == "https://export.arxiv.org/api/query"
    assert "site:" not in captured["params"]["search_query"]
    assert results == [
        {
            "title": "Embodied Agents in Driving Contexts",
            "url": "http://arxiv.org/abs/2601.01234v1",
            "snippet": (
                "arXiv preprint, published 2026-01-03. "
                "A controlled simulator study of embodiment and empathy."
            ),
        }
    ]
