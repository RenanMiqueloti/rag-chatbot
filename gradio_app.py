"""Demo Gradio do rag-chatbot — pensada para Hugging Face Spaces.

Visitante sobe documentos (.txt, .md, .pdf), o pipeline indexa em uma coleção
Qdrant in-memory exclusiva da sessão e responde perguntas a partir desse
contexto. Documentos não são persistidos entre sessões nem entre restarts.

Embeddings rodam localmente (sentence-transformers/all-MiniLM-L6-v2) e o LLM
sai pelo provider configurado via ``LLM_PROVIDER`` (recomendado: groq para
custo zero por query no free tier).
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from uuid import uuid4

import gradio as gr

from app import (
    EmptyDocumentError,
    EncodingError,
    ProtectedDocumentError,
    build_rag_graph,
    load_documents_from_files,
)
from rate_limits import RATE_LIMIT_MSG as _SHARED_RATE_LIMIT_MSG
from rate_limits import is_rate_limit as _shared_is_rate_limit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


MAX_FILE_SIZE_MB = 5
MAX_TOTAL_SIZE_MB = 15  # soma máxima cumulativa por batch (3 arquivos × 5 MB)
MAX_FILES_PER_SESSION = 3
MAX_QUERIES_PER_SESSION = 30
MAX_INDEX_OPERATIONS_PER_SESSION = 3

QUOTA_QUERIES_MSG = "Limite da demo atingido para esta sessão. Recarregue a página pra recomeçar."
QUOTA_INDEX_MSG = (
    f"Limite de {MAX_INDEX_OPERATIONS_PER_SESSION} indexações por sessão atingido. "
    "Recarregue a página pra recomeçar."
)
RATE_LIMIT_MSG = _SHARED_RATE_LIMIT_MSG
NO_STATE_MSG = "Envie documentos primeiro."


THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Menlo", "monospace"],
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="1px",
    block_radius="14px",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
)

CSS = """
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
#hero { padding: 4px 0 16px; }
#hero h1 { margin: 0 0 4px; font-weight: 600; letter-spacing: -0.01em; font-size: 1.5rem; }
#hero p { margin: 0; color: var(--body-text-color-subdued); }
.message { padding: 14px 16px !important; line-height: 1.6 !important; }
"""

HERO_HTML = """
<div id="hero">
  <h1>rag-chatbot · demo</h1>
  <p>Pipeline RAG com LangGraph + Qdrant + BM25 + RRF + cross-encoder rerank.</p>
</div>
"""

CHAT_PLACEHOLDER = (
    "Suba documentos no painel ao lado e pergunte sobre o conteúdo.\n\n"
    "Sem corpus próprio? `data/example.md` no repo é um primer curto sobre RAG."
)

_CHIP_PALETTE = {
    "neutral": ("#e0e7ff", "#3730a3"),
    "ready": ("#dcfce7", "#15803d"),
    "active": ("#fef3c7", "#a16207"),
    "error": ("#fee2e2", "#b91c1c"),
}


def _status_chip(text: str, kind: str = "neutral") -> str:
    bg, fg = _CHIP_PALETTE.get(kind, _CHIP_PALETTE["neutral"])
    return (
        f'<div style="display:inline-flex;align-items:center;gap:8px;'
        f"padding:8px 14px;border-radius:999px;background:{bg};color:{fg};"
        f'font-weight:500;font-size:14px;">{text}</div>'
    )


INITIAL_STATUS = _status_chip("Envie documentos para começar.", "neutral")


def _is_rate_limit(exc: BaseException) -> bool:
    return _shared_is_rate_limit(exc)


def index_files(files, current_state):
    """Valida limites, carrega documentos e compila o grafo RAG da sessão.

    Generator que faz yield em cada etapa pra UI mostrar progresso:
    validação → leitura → indexação → pronto. Cada yield atualiza o status
    chip no Gradio.

    Estado preservado entre uploads: o contador de queries não é zerado,
    pra evitar que um usuário burle o limite por re-upload. Total de
    indexações é capado por ``MAX_INDEX_OPERATIONS_PER_SESSION``.
    """
    if not files:
        yield current_state, _status_chip("Envie ao menos um arquivo (.txt, .md ou .pdf).", "error")
        return

    uploads_done = (current_state or {}).get("uploads", 0)
    if uploads_done >= MAX_INDEX_OPERATIONS_PER_SESSION:
        yield current_state, _status_chip(QUOTA_INDEX_MSG, "error")
        return

    if len(files) > MAX_FILES_PER_SESSION:
        yield (
            current_state,
            _status_chip(
                f"Máximo de {MAX_FILES_PER_SESSION} arquivos por sessão (enviou {len(files)}).",
                "error",
            ),
        )
        return

    total_mb = 0.0
    for f in files:
        try:
            size_mb = os.path.getsize(f.name) / (1024 * 1024)
        except OSError as exc:
            yield current_state, _status_chip(f"Erro lendo {Path(f.name).name}: {exc}", "error")
            return
        if size_mb > MAX_FILE_SIZE_MB:
            yield (
                current_state,
                _status_chip(
                    f"{Path(f.name).name} excede {MAX_FILE_SIZE_MB} MB ({size_mb:.1f} MB).",
                    "error",
                ),
            )
            return
        total_mb += size_mb

    if total_mb > MAX_TOTAL_SIZE_MB:
        yield (
            current_state,
            _status_chip(
                f"Soma dos arquivos ({total_mb:.1f} MB) excede o limite de {MAX_TOTAL_SIZE_MB} MB.",
                "error",
            ),
        )
        return

    yield current_state, _status_chip(f"Lendo {len(files)} arquivo(s)…", "active")

    try:
        paths = [f.name for f in files]
        documents = load_documents_from_files(paths)
    except (EmptyDocumentError, ProtectedDocumentError, EncodingError) as exc:
        yield current_state, _status_chip(str(exc), "error")
        return
    except Exception as exc:
        yield current_state, _status_chip(f"Erro ao ler arquivos: {exc}", "error")
        return

    yield (
        current_state,
        _status_chip(f"Indexando {len(documents)} página(s)/documento(s)…", "active"),
    )

    if current_state and current_state.get("id"):
        session_id = current_state["id"]
    else:
        session_id = uuid4().hex[:8]
    next_upload = uploads_done + 1
    collection_name = f"docs_{session_id}_{next_upload}"

    try:
        # ``qdrant_url=""`` força in-memory mesmo se ``QDRANT_URL`` estiver no env
        # (a demo é efêmera; não escreve num Qdrant compartilhado).
        graph = build_rag_graph(documents, collection_name=collection_name, qdrant_url="")
    except Exception as exc:
        yield current_state, _status_chip(f"Erro ao indexar: {exc}", "error")
        return

    if current_state is None:
        state = {"id": session_id, "graph": graph, "queries": 0, "uploads": 1}
    else:
        # Drop the previous graph eagerly so its Qdrant in-memory data can be freed.
        current_state["graph"] = graph
        current_state["uploads"] = next_upload
        state = current_state
        gc.collect()

    logger.info(
        "session=%s indexed files=%d documents=%d upload=%d",
        state["id"],
        len(files),
        len(documents),
        next_upload,
    )
    yield (
        state,
        _status_chip(
            f"Pronto · {len(documents)} doc(s) · {next_upload}/{MAX_INDEX_OPERATIONS_PER_SESSION} uploads",
            "ready",
        ),
    )


def _render_sources(sources_struct: list[dict]) -> str:
    if not sources_struct:
        return "_Faça uma pergunta para ver as fontes citadas._"
    lines: list[str] = []
    for s in sources_struct:
        meta_bits = [f"`{s.get('source', 'unknown')}`"]
        if s.get("page") is not None:
            meta_bits.append(f"página {s['page']}")
        meta_bits.append(f"score {s['score']:.3f}")
        lines.append(f"**[{s['id']}]** · {' · '.join(meta_bits)}")
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


with gr.Blocks(title="rag-chatbot — demo", analytics_enabled=False) as demo:
    gr.HTML(HERO_HTML)

    state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(
                file_count="multiple",
                file_types=[".txt", ".md", ".pdf"],
                label="Documentos",
            )
            status = gr.HTML(value=INITIAL_STATUS)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=460,
                label="Conversa",
                placeholder=CHAT_PLACEHOLDER,
            )
            msg = gr.Textbox(label="Pergunta", placeholder="O que você quer saber?")
            send = gr.Button("Enviar", variant="primary")

            gr.Examples(
                examples=[
                    "Faça um resumo dos documentos.",
                    "Quais são os principais conceitos abordados?",
                    "Liste os tópicos cobertos.",
                ],
                inputs=[msg],
                label="Exemplos",
            )

            with gr.Accordion("Fontes recuperadas", open=True):
                sources_md = gr.Markdown("_Faça uma pergunta para ver as fontes citadas._")

    files.change(index_files, inputs=[files, state], outputs=[state, status], api_name=False)
    send.click(
        respond, inputs=[msg, chatbot, state], outputs=[chatbot, msg, sources_md], api_name=False
    )
    msg.submit(
        respond, inputs=[msg, chatbot, state], outputs=[chatbot, msg, sources_md], api_name=False
    )


if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=THEME,
        css=CSS,
        footer_links=["gradio"],
    )
