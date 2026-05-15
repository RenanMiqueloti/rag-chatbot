"""Smoke tests for rag-chatbot.

Validates the repo's structure and importability without external API calls.
The pipeline modules carry heavy ML deps (langgraph, qdrant, langchain
providers); guarded with ``pytest.importorskip`` so a partial install still
runs the structural assertions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_repo_layout() -> None:
    assert (ROOT / "api.py").is_file()
    assert (ROOT / "app.py").is_file()
    assert (ROOT / "gradio_app.py").is_file()
    assert (ROOT / "Dockerfile").is_file()
    assert (ROOT / "Dockerfile.api").is_file()
    assert (ROOT / "requirements.txt").is_file()
    assert (ROOT / ".env.example").is_file()
    assert (ROOT / "evals" / "__init__.py").is_file()
    assert (ROOT / "data" / "sample_docs.txt").is_file()
    assert (ROOT / "data" / "example.md").is_file()


def test_readme_present_and_branded() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "rag-chatbot" in readme.lower()
    assert "qdrant" in readme.lower() or "Qdrant" in readme


def test_app_module_imports() -> None:
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_qdrant")
    import app

    assert hasattr(app, "build_rag_graph")
    assert callable(app.build_rag_graph)
    assert hasattr(app, "reciprocal_rank_fusion")
    assert callable(app.reciprocal_rank_fusion)
    assert hasattr(app, "load_documents_from_files")
    assert callable(app.load_documents_from_files)


def test_gradio_app_imports() -> None:
    pytest.importorskip("gradio")
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_qdrant")
    import gradio_app

    assert hasattr(gradio_app, "demo")


def test_load_documents_from_files_txt(tmp_path: Path) -> None:
    pytest.importorskip("langchain_community")
    import app

    sample = tmp_path / "sample.txt"
    sample.write_text("Texto de exemplo para teste.", encoding="utf-8")

    docs = app.load_documents_from_files([sample])
    assert len(docs) == 1
    assert "Texto de exemplo" in docs[0].page_content


def test_build_retrievers_accepts_documents() -> None:
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_qdrant")
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("langchain_huggingface")
    from langchain_core.documents import Document

    import app

    documents = [
        Document(page_content="A LangChain é um framework para LLMs."),
        Document(page_content="FAISS é uma biblioteca de busca vetorial."),
    ]
    semantic, bm25, chunks = app.build_retrievers(documents, collection_name="test")
    assert hasattr(semantic, "invoke")
    assert hasattr(bm25, "invoke")
    assert isinstance(chunks, list) and len(chunks) >= 1


def test_reciprocal_rank_fusion_basic() -> None:
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document

    import app

    docs_a = [Document(page_content=f"doc-{i}") for i in range(3)]
    docs_b = [Document(page_content=f"doc-{i}") for i in [2, 0, 4]]
    fused = app.reciprocal_rank_fusion([docs_a, docs_b], k=60)

    contents = [d.page_content for d in fused]
    # docs that appear in both lists rank above docs that appear in only one
    assert contents.index("doc-0") < contents.index("doc-1")
    assert contents.index("doc-2") < contents.index("doc-1")


def test_api_health_endpoint() -> None:
    """Health endpoint responds even when the heavy pipeline has not initialized.

    Uses ``TestClient`` without the ``with`` context manager so the FastAPI
    lifespan (which builds the full RAG graph and connects to Qdrant) is
    skipped — the goal is to validate routing, not infra.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import api

    client = TestClient(api.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "ok"


def test_evals_dataset_loads() -> None:
    import json

    dataset_path = ROOT / "evals" / "dataset.json"
    with dataset_path.open(encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list | dict)


def test_rag_state_has_sources_struct() -> None:
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_qdrant")
    import app

    annotations = app.RAGState.__annotations__
    assert "sources_struct" in annotations


def test_extract_citation_ids_unique_and_ordered() -> None:
    pytest.importorskip("langchain_qdrant")
    import app

    text = "Primeiro ponto [1]. Segundo ponto [2]. Reforço o primeiro [1] e cito [3]."
    assert app.extract_citation_ids(text) == [1, 2, 3]
    assert app.extract_citation_ids("Sem citações aqui.") == []
    assert app.extract_citation_ids("[7][7][7]") == [7]


def test_respond_handles_none_state() -> None:
    import asyncio

    pytest.importorskip("gradio")
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_qdrant")
    import gradio_app

    async def consume() -> list:
        out = []
        async for chunk in gradio_app.respond("oi", [], None):
            out.append(chunk)
        return out

    results = asyncio.run(consume())
    assert results, "respond deveria yield ao menos uma vez"
    history, _, _ = results[-1]
    assert any("documento" in turn["content"].lower() for turn in history)


def test_is_rate_limit_detects_common_signatures() -> None:
    pytest.importorskip("gradio")
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_qdrant")
    import gradio_app

    class RateLimitError(Exception):
        pass

    class HTTPError(Exception):
        pass

    assert gradio_app._is_rate_limit(RateLimitError("over quota"))
    assert gradio_app._is_rate_limit(HTTPError("HTTP 429 Too Many Requests"))
    assert not gradio_app._is_rate_limit(ValueError("just a value error"))


def test_daily_request_budget_consumes_and_refuses() -> None:
    from rate_limits import DailyRequestBudget

    b = DailyRequestBudget(cap=2)
    assert b.remaining() == 2
    assert b.try_consume() is True
    assert b.try_consume() is True
    assert b.try_consume() is False
    assert b.remaining() == 0


def test_daily_request_budget_disabled_when_cap_zero() -> None:
    from rate_limits import DailyRequestBudget

    b = DailyRequestBudget(cap=0)
    assert b.remaining() == -1
    for _ in range(1000):
        assert b.try_consume() is True


def test_daily_request_budget_resets_after_midnight(monkeypatch) -> None:
    from datetime import UTC, datetime, timedelta

    import rate_limits

    fixed_now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    class _Clock:
        current = fixed_now

        @classmethod
        def now(cls, tz=None):
            return cls.current

    monkeypatch.setattr(rate_limits, "datetime", _Clock)

    b = rate_limits.DailyRequestBudget(cap=1)
    assert b.try_consume() is True
    assert b.try_consume() is False

    _Clock.current = fixed_now + timedelta(days=1, hours=1)
    assert b.try_consume() is True


def test_api_query_rejects_too_long_payload() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("slowapi")
    from fastapi.testclient import TestClient

    import api

    client = TestClient(api.app)
    long_query = "a" * (api.MAX_QUERY_CHARS + 1)
    resp = client.post("/query", json={"query": long_query})
    assert resp.status_code == 422


def test_api_query_rejects_empty_payload() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("slowapi")
    from fastapi.testclient import TestClient

    import api

    client = TestClient(api.app)
    resp = client.post("/query", json={"query": ""})
    assert resp.status_code == 422


def test_api_rate_limiter_returns_429_after_cap(monkeypatch) -> None:
    """Slowapi 429 acionado quando IP excede o limite por minuto."""
    pytest.importorskip("fastapi")
    pytest.importorskip("slowapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    monkeypatch.setenv("RATE_LIMIT_PER_HOUR", "1000")
    monkeypatch.setenv("DAILY_REQUEST_CAP", "0")

    import importlib

    import api as api_module

    api_module = importlib.reload(api_module)
    client = TestClient(api_module.app)

    statuses = [client.post("/query", json={"query": "ola"}).status_code for _ in range(4)]
    # com pipeline não inicializado, primeiras passam pelo limiter e batem em 503;
    # depois do cap, slowapi devolve 429 antes de chegar no handler
    assert 429 in statuses
    assert statuses.count(429) >= 2


def test_api_daily_budget_returns_429(monkeypatch) -> None:
    """Cap diário esgotado vira 429 com mensagem amigável."""
    pytest.importorskip("fastapi")
    pytest.importorskip("slowapi")
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_qdrant")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1000")
    monkeypatch.setenv("RATE_LIMIT_PER_HOUR", "10000")
    monkeypatch.setenv("DAILY_REQUEST_CAP", "1")

    import importlib

    import api as api_module

    api_module = importlib.reload(api_module)

    class _StubGraph:
        async def ainvoke(self, state):
            return {"answer": "ok", "sources_struct": []}

    api_module._rag_graph = _StubGraph()  # bypass lifespan

    client = TestClient(api_module.app)
    first = client.post("/query", json={"query": "ola"})
    second = client.post("/query", json={"query": "ola"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert (
        "diária" in second.json()["detail"].lower() or "diaria" in second.json()["detail"].lower()
    )


def test_api_upstream_429_returns_friendly_message(monkeypatch) -> None:
    """Erro de rate limit do provider vira 429 com mensagem clara, não 500."""
    pytest.importorskip("fastapi")
    pytest.importorskip("slowapi")
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_qdrant")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1000")
    monkeypatch.setenv("RATE_LIMIT_PER_HOUR", "10000")
    monkeypatch.setenv("DAILY_REQUEST_CAP", "0")

    import importlib

    import api as api_module

    api_module = importlib.reload(api_module)

    class _RateLimitedGraph:
        async def ainvoke(self, state):
            raise RuntimeError("HTTP 429 Too Many Requests from provider")

    api_module._rag_graph = _RateLimitedGraph()

    client = TestClient(api_module.app)
    resp = client.post("/query", json={"query": "ola"})
    assert resp.status_code == 429
    assert "provider" in resp.json()["detail"].lower()
