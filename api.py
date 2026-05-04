"""FastAPI — endpoint de streaming para o pipeline RAG.

Uso:
    uvicorn api:app --reload

Endpoints:
    POST /query   — resposta completa (JSON)
    POST /stream  — streaming palavra a palavra (text/plain)
    GET  /health  — health check

Provider LLM: defina LLM_PROVIDER=anthropic|openai no .env (padrão: openai).
Observabilidade: defina LANGCHAIN_TRACING_V2=true + LANGSMITH_API_KEY no .env.
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

# Instanciado em startup para evitar cold-start no primeiro request
_rag_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Indexa o corpus e compila o grafo na inicialização do servidor."""
    global _rag_graph

    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    required_key = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"

    if not os.getenv(required_key):
        raise RuntimeError(
            f"{required_key} não definida. Configure o arquivo .env "
            f"(LLM_PROVIDER={provider})"
        )

    from app import build_rag_graph  # noqa: PLC0415

    _rag_graph = await asyncio.to_thread(build_rag_graph)
    yield


app = FastAPI(
    title="RAG Chatbot API",
    description="Production RAG pipeline: hybrid retrieval → re-ranking → generation",
    version="2.1.0",
    lifespan=lifespan,
)


def _get_graph():
    if _rag_graph is None:
        raise HTTPException(status_code=503, detail="RAG pipeline não inicializado")
    return _rag_graph


# ── Request/Response models ───────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "pipeline": "ready" if _rag_graph else "initializing",
        "provider": os.getenv("LLM_PROVIDER", "openai"),
        "tracing": os.getenv("LANGCHAIN_TRACING_V2", "false"),
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Resposta completa em JSON."""
    rag = _get_graph()
    result = await asyncio.to_thread(
        rag.invoke,
        {
            "query": request.query,
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": "",
        },
    )
    sources = [d.page_content[:120] + "..." for d in result["reranked_docs"]]
    return QueryResponse(answer=result["answer"], sources=sources)


@app.post("/stream")
async def stream_endpoint(request: QueryRequest) -> StreamingResponse:
    """Streaming da resposta palavra a palavra."""

    async def token_generator():
        rag = _get_graph()
        result = await asyncio.to_thread(
            rag.invoke,
            {
                "query": request.query,
                "retrieved_docs": [],
                "reranked_docs": [],
                "answer": "",
            },
        )
        for word in result["answer"].split():
            yield word + " "
            await asyncio.sleep(0.015)

    return StreamingResponse(token_generator(), media_type="text/plain")
