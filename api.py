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
    if provider == "anthropic":
        required_key = "ANTHROPIC_API_KEY"
    elif provider == "groq":
        required_key = "GROQ_API_KEY"
    else:
        required_key = "OPENAI_API_KEY"

    if not os.getenv(required_key):
        raise RuntimeError(
            f"{required_key} não definida. Configure o arquivo .env (LLM_PROVIDER={provider})"
        )

    from app import build_rag_graph, load_documents_from_files

    documents = await asyncio.to_thread(load_documents_from_files, ["data/sample_docs.txt"])
    _rag_graph = await asyncio.to_thread(build_rag_graph, documents)
    yield


app = FastAPI(
    title="RAG Chatbot API",
    description="Pipeline RAG: hybrid retrieval → re-ranking → generation",
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


class Source(BaseModel):
    id: int
    snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


# ── Endpoints ─────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "pipeline": "ready" if _rag_graph else "initializing",
        "provider": os.getenv("LLM_PROVIDER", "openai"),
        "tracing": os.getenv("LANGCHAIN_TRACING_V2", "false"),
    }


def _initial_state(query: str) -> dict:
    return {
        "query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "sources_struct": [],
        "answer": "",
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Resposta completa em JSON, com fontes numeradas."""
    rag = _get_graph()
    result = await asyncio.to_thread(rag.invoke, _initial_state(request.query))
    sources = [Source(**s) for s in result.get("sources_struct", [])]
    return QueryResponse(answer=result["answer"], sources=sources)


@app.post("/stream")
async def stream_endpoint(request: QueryRequest) -> StreamingResponse:
    """Streaming token-a-token via LangGraph astream_events."""
    rag = _get_graph()

    async def token_generator():
        async for ev in rag.astream_events(_initial_state(request.query), version="v2"):
            if ev["event"] != "on_chat_model_stream":
                continue
            chunk = ev["data"].get("chunk")
            token = getattr(chunk, "content", "") if chunk is not None else ""
            if token:
                yield token

    return StreamingResponse(token_generator(), media_type="text/plain")
