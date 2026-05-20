"""FastAPI — endpoint de streaming para o pipeline RAG.

Uso:
    uvicorn api:app --reload --proxy-headers --forwarded-allow-ips="*"

Endpoints:
    POST /query   — resposta completa (JSON)
    POST /stream  — streaming palavra a palavra (text/plain)
    GET  /health  — health check (inclui quota diária restante)

Provider LLM: defina LLM_PROVIDER=anthropic|openai|groq no .env (padrão: openai).
Observabilidade: defina LANGCHAIN_TRACING_V2=true + LANGSMITH_API_KEY no .env.

Rate limiting (todos opcionais via env):
    RATE_LIMIT_PER_MINUTE / RATE_LIMIT_PER_HOUR — slowapi por IP
    DAILY_REQUEST_CAP — circuit breaker global com reset em meia-noite UTC
    MAX_QUERY_CHARS — tamanho máximo da query (Pydantic, retorna 422)
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from rate_limits import DAILY_CAP_MSG, RATE_LIMIT_MSG, DailyRequestBudget, is_rate_limit

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


RATE_LIMIT_PER_MINUTE = _int_env("RATE_LIMIT_PER_MINUTE", 10)
RATE_LIMIT_PER_HOUR = _int_env("RATE_LIMIT_PER_HOUR", 100)
DAILY_REQUEST_CAP = _int_env("DAILY_REQUEST_CAP", 80)
MAX_QUERY_CHARS = _int_env("MAX_QUERY_CHARS", 2000)
LLM_REQUEST_TIMEOUT_SECONDS = _int_env("LLM_REQUEST_TIMEOUT_SECONDS", 60)

TIMEOUT_MSG = "Tempo limite excedido aguardando o provider LLM. Tente de novo."

_LIMITS = [f"{RATE_LIMIT_PER_MINUTE}/minute", f"{RATE_LIMIT_PER_HOUR}/hour"]
limiter = Limiter(key_func=get_remote_address, default_limits=_LIMITS)
budget = DailyRequestBudget(cap=DAILY_REQUEST_CAP)

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

    documents = await asyncio.to_thread(
        load_documents_from_files, ["data/sample_docs.txt", "data/example.md"]
    )
    _rag_graph = await asyncio.to_thread(build_rag_graph, documents)
    yield


app = FastAPI(
    title="RAG Chatbot API",
    description="Pipeline RAG: hybrid retrieval → re-ranking → generation",
    version="2.2.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "detail": f"Limite de requisições excedido: {exc.detail}. Tente novamente em breve."
        },
    )


def _get_graph():
    if _rag_graph is None:
        raise HTTPException(status_code=503, detail="RAG pipeline não inicializado")
    return _rag_graph


def _enforce_daily_budget() -> None:
    if not budget.try_consume():
        raise HTTPException(status_code=429, detail=DAILY_CAP_MSG)


# ── Request/Response models ───────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)


class Source(BaseModel):
    id: int
    snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


# ── Endpoints ─────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> JSONResponse:
    body = {
        "status": "ok" if _rag_graph else "initializing",
        "pipeline": "ready" if _rag_graph else "initializing",
        "provider": os.getenv("LLM_PROVIDER", "openai"),
        "tracing": os.getenv("LANGCHAIN_TRACING_V2", "false"),
        "daily_remaining": budget.remaining(),
    }
    status_code = 200 if _rag_graph else 503
    return JSONResponse(status_code=status_code, content=body)


def _initial_state(query: str) -> dict:
    return {
        "query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "sources_struct": [],
        "answer": "",
    }


@app.post("/query", response_model=QueryResponse)
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute")
async def query_endpoint(request: Request, body: QueryRequest) -> QueryResponse:
    """Resposta completa em JSON, com fontes numeradas."""
    rag = _get_graph()
    _enforce_daily_budget()
    try:
        coro = rag.ainvoke(_initial_state(body.query))
        if LLM_REQUEST_TIMEOUT_SECONDS > 0:
            result = await asyncio.wait_for(coro, timeout=LLM_REQUEST_TIMEOUT_SECONDS)
        else:
            result = await coro
    except TimeoutError as exc:
        logger.warning("query timeout after %ss", LLM_REQUEST_TIMEOUT_SECONDS)
        raise HTTPException(status_code=504, detail=TIMEOUT_MSG) from exc
    except Exception as exc:
        if is_rate_limit(exc):
            logger.warning("upstream rate limit on /query: %s", exc)
            raise HTTPException(status_code=429, detail=RATE_LIMIT_MSG) from exc
        logger.exception("query failed")
        raise
    sources = [Source(**s) for s in result.get("sources_struct", [])]
    return QueryResponse(answer=result["answer"], sources=sources)


@app.post("/stream")
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute")
async def stream_endpoint(request: Request, body: QueryRequest) -> StreamingResponse:
    """Streaming token-a-token via LangGraph astream_events.

    Sem timeout global no iterator — o cancelamento natural por desconexão do
    cliente é o que limita streams penduradas. ``LLM_REQUEST_TIMEOUT_SECONDS``
    só se aplica a ``/query`` por causa disso.
    """
    rag = _get_graph()
    _enforce_daily_budget()

    async def token_generator():
        try:
            async for ev in rag.astream_events(_initial_state(body.query), version="v2"):
                if ev["event"] != "on_chat_model_stream":
                    continue
                chunk = ev["data"].get("chunk")
                token = getattr(chunk, "content", "") if chunk is not None else ""
                if token:
                    yield token
        except Exception as exc:
            if is_rate_limit(exc):
                logger.warning("upstream rate limit on /stream: %s", exc)
                yield f"\n[{RATE_LIMIT_MSG}]"
                return
            logger.exception("stream failed")
            raise

    return StreamingResponse(token_generator(), media_type="text/plain")
