"""RAG Chatbot — produção-grade com LangGraph, hybrid retrieval e re-ranking.

Pipeline LangGraph:
  retrieve (BM25 + Semantic → RRF fusion)
      → rerank (cross-encoder via FlashRank)
          → generate (gpt-4o-mini com contexto)

Banco vetorial: Qdrant in-memory (sem servidor extra — troque por
QdrantClient(url="http://localhost:6333") para ambiente de produção).

Retrieval híbrido: Reciprocal Rank Fusion de BM25 e embeddings semânticos
para cobertura de vocabulário exato + semântica.
"""
from __future__ import annotations

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# ── State ─────────────────────────────────────────────────────────────────


class RAGState(TypedDict):
    query: str
    retrieved_docs: list[Document]
    reranked_docs: list[Document]
    answer: str


# ── Indexing ──────────────────────────────────────────────────────────────


def build_retrievers(
    path: str = "data/sample_docs.txt",
) -> tuple[object, BM25Retriever]:
    """Constrói o retriever semântico (Qdrant) e o BM25 a partir do corpus.

    Args:
        path: Caminho para o arquivo de texto a indexar.

    Returns:
        Tupla (semantic_retriever, bm25_retriever).
    """
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Qdrant in-memory — substitua por QdrantClient(url=...) em produção
    client = QdrantClient(":memory:")
    embeddings = OpenAIEmbeddings()

    vectorstore = QdrantVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        client=client,
        collection_name="docs",
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


def rerank(query: str, docs: list[Document], top_n: int = 3) -> list[Document]:
    """Re-rank documentos com cross-encoder (FlashRank), com fallback gracioso.

    Args:
        query: Query do usuário.
        docs: Documentos candidatos a re-rankear.
        top_n: Número de documentos a retornar.

    Returns:
        Top N documentos re-rankeados.
    """
    try:
        from flashrank import RerankRequest, Ranker  # type: ignore[import]

        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp")
        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)
        reranked_ids = [r["id"] for r in results[:top_n]]
        return [docs[i] for i in reranked_ids if i < len(docs)]
    except ImportError:
        # FlashRank não instalado — passa os top_n por score de fusão
        return docs[:top_n]


# ── LangGraph nodes ───────────────────────────────────────────────────────


def make_retrieve_node(semantic_retriever: object, bm25_retriever: BM25Retriever):
    """Cria o nó de retrieval híbrido: BM25 + semântico fundidos via RRF."""

    def retrieve(state: RAGState) -> RAGState:
        query = state["query"]
        semantic = semantic_retriever.invoke(query)  # type: ignore[union-attr]
        keyword = bm25_retriever.invoke(query)
        fused = reciprocal_rank_fusion([semantic, keyword])
        return {**state, "retrieved_docs": fused}

    return retrieve


def rerank_node(state: RAGState) -> RAGState:
    """Nó de re-ranking: cross-encoder sobre os candidatos fundidos."""
    reranked = rerank(state["query"], state["retrieved_docs"])
    return {**state, "reranked_docs": reranked}


def make_generate_node(llm: ChatOpenAI):
    """Cria o nó de geração: prompt com contexto re-rankeado → LLM."""

    def generate(state: RAGState) -> RAGState:
        context = "\n\n".join(d.page_content for d in state["reranked_docs"])
        prompt = (
            "Use ONLY the context below to answer the question.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {state['query']}"
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        return {**state, "answer": response.content}

    return generate


# ── Graph factory ─────────────────────────────────────────────────────────


def build_rag_graph(data_path: str = "data/sample_docs.txt"):
    """Constrói e compila o grafo LangGraph do pipeline RAG.

    Args:
        data_path: Caminho para o corpus de documentos.

    Returns:
        Grafo compilado pronto para invoke/stream.
    """
    semantic, bm25 = build_retrievers(data_path)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

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


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Defina OPENAI_API_KEY no arquivo .env")
        return

    print("🔎 Indexando corpus (Qdrant in-memory + BM25)...")
    rag = build_rag_graph()
    print("🤖 RAG chatbot pronto! Digite 'sair' para encerrar.\n")

    while True:
        query = input("Você: ").strip()
        if not query or query.lower() in {"sair", "exit", "quit"}:
            break

        initial_state: RAGState = {
            "query": query,
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": "",
        }
        result = rag.invoke(initial_state)
        print(f"Bot: {result['answer']}\n")


if __name__ == "__main__":
    main()
