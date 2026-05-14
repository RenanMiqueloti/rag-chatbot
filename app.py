"""RAG Chatbot com LangGraph, hybrid retrieval e re-ranking.

Pipeline LangGraph:
  retrieve (BM25 + Semantic → RRF fusion)
      → rerank (cross-encoder via FlashRank)
          → generate (Claude, GPT-4o-mini ou Llama 3.3 via Groq com contexto)

Banco vetorial: Qdrant in-memory (sem servidor extra — troque por
QdrantClient(url="http://localhost:6333") para um deploy real).

Retrieval híbrido: Reciprocal Rank Fusion de BM25 e embeddings semânticos
para cobertura de vocabulário exato + semântica.

Observabilidade: LangSmith ativo quando LANGCHAIN_TRACING_V2=true no .env.

Provider LLM: defina LLM_PROVIDER=anthropic|groq (padrão: openai).
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

load_dotenv()

logger = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[(\d+)\]")

# ── LLM factory ───────────────────────────────────────────────────────────


def build_llm(provider: str | None = None) -> BaseChatModel:
    """Retorna o LLM configurado pelo env var LLM_PROVIDER.

    Providers suportados:
      - openai    → gpt-4o-mini  (requer OPENAI_API_KEY)
      - anthropic → claude-3-5-haiku-20241022  (requer ANTHROPIC_API_KEY)
      - groq      → llama-3.3-70b-versatile  (requer GROQ_API_KEY; free tier rate-limited)

    Padrão: openai.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model="claude-3-5-haiku-20241022",  # type: ignore[call-arg]
            temperature=0.2,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


# ── State ─────────────────────────────────────────────────────────────────


class RAGState(TypedDict):
    query: str
    retrieved_docs: list[Document]
    reranked_docs: list[Document]
    sources_struct: list[dict]
    answer: str


def extract_citation_ids(text: str) -> list[int]:
    """Extrai IDs de citações ``[N]`` na ordem de aparição, sem duplicar."""
    seen: list[int] = []
    for match in CITATION_RE.finditer(text):
        n = int(match.group(1))
        if n not in seen:
            seen.append(n)
    return seen


# ── Document loading ──────────────────────────────────────────────────────


def load_documents_from_files(paths: list[str | Path]) -> list[Document]:
    """Lê .txt, .md e .pdf, retorna Documents prontos pra split.

    Args:
        paths: Caminhos para os arquivos. Extensões suportadas: .txt, .md, .pdf.

    Returns:
        Lista de Documents prontos pra ``build_retrievers``.
    """
    from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

    documents: list[Document] = []
    for raw in paths:
        path = Path(raw)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            documents.extend(PyMuPDFLoader(str(path)).load())
        elif suffix in {".txt", ".md"}:
            documents.extend(TextLoader(str(path), encoding="utf-8").load())
        else:
            raise ValueError(f"Tipo de arquivo não suportado: {suffix} ({path})")
    return documents


# ── Indexing ──────────────────────────────────────────────────────────────


def build_retrievers(
    documents: list[Document],
    collection_name: str = "docs",
) -> tuple[object, BM25Retriever]:
    """Constrói o retriever semântico (Qdrant) e o BM25 a partir dos documentos.

    Args:
        documents: Documentos já carregados (use ``load_documents_from_files``).
        collection_name: Nome da coleção Qdrant — útil pra isolar sessões.

    Returns:
        Tupla (semantic_retriever, bm25_retriever).
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Local embeddings keep the demo free of per-query API cost.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = QdrantVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        location=":memory:",
        collection_name=collection_name,
    )
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 6

    return semantic_retriever, bm25


# ── Hybrid fusion via RRF ─────────────────────────────────────────────────


def reciprocal_rank_fusion(
    results_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """Funde múltiplas listas de resultados via Reciprocal Rank Fusion.

    Args:
        results_lists: Lista de listas de documentos ranqueados.
        k: Constante de suavização RRF (padrão 60, recomendado pela literatura).

    Returns:
        Lista fundida ordenada por score decrescente.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for results in results_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + k)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k_] for k_ in sorted_keys]


# ── Re-ranking ────────────────────────────────────────────────────────────


def rerank(query: str, docs: list[Document], top_n: int = 3) -> list[tuple[Document, float]]:
    """Re-rank documentos com cross-encoder (FlashRank), com fallback gracioso.

    Retorna pares ``(Document, score)``. Quando FlashRank não está disponível
    cai num fallback que passa os top_n por score de fusão com score 0.0.
    """
    try:
        from flashrank import Ranker, RerankRequest  # type: ignore[import]

        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp")
        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)
        out: list[tuple[Document, float]] = []
        for r in results[:top_n]:
            idx = r["id"]
            if idx < len(docs):
                out.append((docs[idx], float(r.get("score", 0.0))))
        return out
    except ImportError:
        return [(d, 0.0) for d in docs[:top_n]]


# ── LangGraph nodes ───────────────────────────────────────────────────────


def make_retrieve_node(semantic_retriever: object, bm25_retriever: BM25Retriever):
    """Cria o nó de retrieval híbrido: BM25 + semântico fundidos via RRF."""

    def retrieve(state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        query = state["query"]
        semantic = semantic_retriever.invoke(query)  # type: ignore[union-attr]
        keyword = bm25_retriever.invoke(query)
        fused = reciprocal_rank_fusion([semantic, keyword])
        dt_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "retrieve query=%r semantic=%d bm25=%d fused=%d latency_ms=%.1f",
            query[:80],
            len(semantic),
            len(keyword),
            len(fused),
            dt_ms,
        )
        return {**state, "retrieved_docs": fused}

    return retrieve


def rerank_node(state: RAGState) -> RAGState:
    """Nó de re-ranking: cross-encoder sobre os candidatos fundidos."""
    t0 = time.perf_counter()
    query = state["query"]
    in_docs = state["retrieved_docs"]
    pairs = rerank(query, in_docs)
    reranked = [d for d, _ in pairs]
    sources_struct = [
        {"id": i + 1, "snippet": d.page_content[:180], "score": s} for i, (d, s) in enumerate(pairs)
    ]
    dt_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "rerank query=%r in=%d out=%d latency_ms=%.1f",
        query[:80],
        len(in_docs),
        len(reranked),
        dt_ms,
    )
    return {**state, "reranked_docs": reranked, "sources_struct": sources_struct}


def make_generate_node(llm: BaseChatModel):
    """Cria o nó de geração: prompt com contexto re-rankeado e numerado → LLM."""

    def generate(state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        query = state["query"]
        numbered = "\n\n".join(
            f"[{i + 1}] {d.page_content}" for i, d in enumerate(state["reranked_docs"])
        )
        prompt = (
            "Use ONLY the context below to answer the question.\n"
            "Cite each claim with [N] referencing the source document.\n"
            "If the answer is not in the context, say you don't know — "
            "without inventing citations.\n\n"
            f"Context:\n{numbered}\n\n"
            f"Question: {query}"
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        dt_ms = (time.perf_counter() - t0) * 1000
        answer = response.content
        logger.info(
            "generate query=%r docs=%d answer_chars=%d latency_ms=%.1f",
            query[:80],
            len(state["reranked_docs"]),
            len(answer) if isinstance(answer, str) else 0,
            dt_ms,
        )
        return {**state, "answer": answer}

    return generate


# ── Graph factory ─────────────────────────────────────────────────────────


def build_rag_graph(documents: list[Document], collection_name: str = "docs"):
    """Constrói e compila o grafo LangGraph do pipeline RAG.

    Args:
        documents: Documentos já carregados (use ``load_documents_from_files``).
        collection_name: Nome da coleção Qdrant — útil pra isolar sessões.

    Returns:
        Grafo compilado pronto para invoke/stream.
    """
    semantic, bm25 = build_retrievers(documents, collection_name=collection_name)
    llm = build_llm()

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", make_retrieve_node(semantic, bm25))
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", make_generate_node(llm))

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ── CLI ───────────────────────────────────────────────────────────────────


def _required_api_key(provider: str) -> str:
    if provider == "anthropic":
        return "ANTHROPIC_API_KEY"
    if provider == "groq":
        return "GROQ_API_KEY"
    return "OPENAI_API_KEY"


def _provider_label(provider: str) -> str:
    if provider == "anthropic":
        return "Claude (Anthropic)"
    if provider == "groq":
        return "Llama 3.3 70B (Groq)"
    return "GPT-4o-mini (OpenAI)"


def main() -> None:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    required_key = _required_api_key(provider)

    if not os.getenv(required_key):
        print(f"⚠️  Defina {required_key} no arquivo .env")
        return

    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

    print("🔎 Indexando corpus (Qdrant in-memory + BM25)...")
    print(f"Provider: {_provider_label(provider)}")
    if tracing:
        project = os.getenv("LANGSMITH_PROJECT", "rag-chatbot")
        print(f"📊 LangSmith tracing ativo — projeto: {project}")

    documents = load_documents_from_files(["data/sample_docs.txt"])
    rag = build_rag_graph(documents)
    print("✅ RAG chatbot pronto! Digite 'sair' para encerrar.\n")

    while True:
        query = input("Você: ").strip()
        if not query or query.lower() in {"sair", "exit", "quit"}:
            break

        initial_state: RAGState = {
            "query": query,
            "retrieved_docs": [],
            "reranked_docs": [],
            "sources_struct": [],
            "answer": "",
        }
        result = rag.invoke(initial_state)
        print(f"Bot: {result['answer']}\n")


if __name__ == "__main__":
    main()
