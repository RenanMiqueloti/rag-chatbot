"""Demo Gradio do rag-chatbot — pensada para Hugging Face Spaces.

Visitante sobe documentos (.txt, .md, .pdf), o pipeline indexa em uma coleção
Qdrant in-memory exclusiva da sessão e responde perguntas a partir desse
contexto. Documentos não são persistidos entre sessões nem entre restarts.

Embeddings rodam localmente (sentence-transformers/all-MiniLM-L6-v2) e o LLM
sai pelo provider configurado via ``LLM_PROVIDER`` (recomendado: groq para
custo zero por query no free tier).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from uuid import uuid4

import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import build_rag_graph, load_documents_from_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


MAX_FILE_SIZE_MB = 5
MAX_FILES_PER_SESSION = 3
MAX_QUERIES_PER_SESSION = 30

QUOTA_QUERIES_MSG = "Limite da demo atingido para esta sessão. Recarregue a página pra recomeçar."
RATE_LIMIT_MSG = "Limite de requisições do provider atingido. Tente em ~1 minuto."
NO_STATE_MSG = "Envie documentos primeiro."


def _is_rate_limit(exc: BaseException) -> bool:
    s = f"{type(exc).__name__} {exc!s}".lower()
    return "ratelimit" in s or "rate_limit" in s or "429" in s


def _new_session_state(graph) -> dict:
    return {
        "id": uuid4().hex[:8],
        "graph": graph,
        "queries": 0,
        "files_uploaded": 0,
    }


def index_files(files, current_state):
    """Valida limites, carrega documentos e compila o grafo RAG da sessão."""
    if not files:
        return None, "Envie ao menos um arquivo (.txt, .md ou .pdf)."

    if len(files) > MAX_FILES_PER_SESSION:
        return current_state, (
            f"Máximo de {MAX_FILES_PER_SESSION} arquivos por sessão (enviou {len(files)})."
        )

    for f in files:
        try:
            size_mb = os.path.getsize(f.name) / (1024 * 1024)
        except OSError as exc:
            return current_state, f"Erro lendo {Path(f.name).name}: {exc}"
        if size_mb > MAX_FILE_SIZE_MB:
            return current_state, (
                f"{Path(f.name).name} excede {MAX_FILE_SIZE_MB} MB ({size_mb:.1f} MB)."
            )

    try:
        paths = [f.name for f in files]
        documents = load_documents_from_files(paths)
    except Exception as exc:
        return current_state, f"Erro ao ler arquivos: {exc}"

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    n_chunks = len(splitter.split_documents(documents))

    try:
        session_id = uuid4().hex[:8]
        graph = build_rag_graph(documents, collection_name=f"docs_{session_id}")
    except Exception as exc:
        return current_state, f"Erro ao indexar: {exc}"

    state = _new_session_state(graph)
    state["files_uploaded"] = len(files)
    logger.info(
        "session=%s indexed files=%d chunks=%d",
        state["id"],
        len(files),
        n_chunks,
    )
    status = f"Indexados {n_chunks} chunks de {len(documents)} documento(s). Pronto pra perguntas."
    return state, status


def _render_sources(sources_struct: list[dict]) -> str:
    if not sources_struct:
        return "_Faça uma pergunta para ver as fontes citadas._"
    lines = ["**Fontes recuperadas:**", ""]
    for s in sources_struct:
        lines.append(f"**[{s['id']}]** (score {s['score']:.3f})")
        lines.append(f"> {s['snippet']}")
        lines.append("")
    return "\n".join(lines)


async def respond(message: str, history: list[dict], state):
    """Async generator: streama a resposta token a token e atualiza fontes."""
    history = list(history or [])

    if not message or not message.strip():
        yield history, "", gr.update()
        return

    user_turn = {"role": "user", "content": message}

    if state is None:
        history.append(user_turn)
        history.append({"role": "assistant", "content": NO_STATE_MSG})
        yield history, "", gr.update()
        return

    if state.get("queries", 0) >= MAX_QUERIES_PER_SESSION:
        history.append(user_turn)
        history.append({"role": "assistant", "content": QUOTA_QUERIES_MSG})
        yield history, "", gr.update()
        return

    state["queries"] = state.get("queries", 0) + 1
    history.append(user_turn)
    history.append({"role": "assistant", "content": ""})
    yield history, "", gr.update()

    initial = {
        "query": message,
        "retrieved_docs": [],
        "reranked_docs": [],
        "sources_struct": [],
        "answer": "",
    }

    buffer: list[str] = []
    sources_struct: list[dict] = []

    try:
        async for ev in state["graph"].astream_events(initial, version="v2"):
            name = ev.get("event")
            if name == "on_chat_model_stream":
                chunk = ev["data"].get("chunk")
                token = getattr(chunk, "content", "") if chunk is not None else ""
                if token:
                    buffer.append(token)
                    history[-1]["content"] = "".join(buffer)
                    yield history, "", gr.update()
            elif name == "on_chain_end" and ev.get("name") == "rerank":
                output = ev.get("data", {}).get("output") or {}
                sources_struct = output.get("sources_struct", []) or sources_struct
    except Exception as exc:
        logger.exception("session=%s respond failed", state.get("id"))
        msg = RATE_LIMIT_MSG if _is_rate_limit(exc) else f"Erro ao gerar resposta: {exc}"
        history[-1]["content"] = msg
        yield history, "", gr.update()
        return

    yield history, "", _render_sources(sources_struct)


with gr.Blocks(title="rag-chatbot — demo") as demo:
    gr.Markdown(
        f"""
        # rag-chatbot — demo

        Pipeline RAG com **LangGraph + Qdrant + BM25 + RRF + cross-encoder rerank**.
        Suba `.txt`, `.md` ou `.pdf` e pergunte sobre o conteúdo.

        > Demo efêmera: documentos vivem só na sessão e somem em qualquer restart.
        > Limites: {MAX_FILES_PER_SESSION} arquivos (≤ {MAX_FILE_SIZE_MB} MB cada),
        > {MAX_QUERIES_PER_SESSION} perguntas por sessão.
        """
    )

    state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(
                file_count="multiple",
                file_types=[".txt", ".md", ".pdf"],
                label="Documentos",
            )
            status = gr.Textbox(
                label="Status",
                interactive=False,
                value="Envie documentos para começar.",
            )

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=420, label="Conversa")
            msg = gr.Textbox(label="Pergunta", placeholder="O que você quer saber?")
            send = gr.Button("Enviar", variant="primary")

            gr.Examples(
                examples=[
                    "O que é LangChain?",
                    "Para que serve FAISS?",
                    "Como funciona RAG?",
                ],
                inputs=[msg],
                label="Perguntas-âncora (sample_docs.txt)",
            )

    with gr.Accordion("Fontes", open=False):
        sources_md = gr.Markdown("_Faça uma pergunta para ver as fontes citadas._")

    files.change(index_files, inputs=[files, state], outputs=[state, status])
    send.click(respond, inputs=[msg, chatbot, state], outputs=[chatbot, msg, sources_md])
    msg.submit(respond, inputs=[msg, chatbot, state], outputs=[chatbot, msg, sources_md])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
