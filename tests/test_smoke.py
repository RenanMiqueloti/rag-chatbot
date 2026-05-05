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
    assert (ROOT / "requirements.txt").is_file()
    assert (ROOT / ".env.example").is_file()
    assert (ROOT / "evals" / "__init__.py").is_file()
    assert (ROOT / "data" / "sample_docs.txt").is_file()


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
