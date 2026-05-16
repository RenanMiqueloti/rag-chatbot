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
import time
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


# Alerta se um Space público ficou com LangSmith ligado. Não bloqueia —
# pode haver razão legítima — mas força visibilidade no log porque cada
# trace persiste indefinidamente em smith.langchain.com com query +
# chunks + resposta de todos os visitantes.
if os.getenv("SPACE_ID") and os.getenv("LANGCHAIN_TRACING_V2", "").lower() in {
    "1",
    "true",
    "yes",
}:
    logger.warning(
        "LangSmith tracing ATIVO em Space público — conteúdo de queries, "
        "chunks e respostas vai persistir externamente em smith.langchain.com. "
        "Confirme se isso é desejado; pra desativar, remova LANGCHAIN_TRACING_V2 "
        "das Variables do Space."
    )


MAX_FILE_SIZE_MB = 5
MAX_TOTAL_SIZE_MB = 15  # soma máxima cumulativa por batch (3 arquivos × 5 MB)
MAX_FILES_PER_SESSION = 3
MAX_QUERIES_PER_SESSION = int(os.getenv("MAX_QUERIES_PER_SESSION", "15"))
MAX_INDEX_OPERATIONS_PER_SESSION = 3
QUERY_COOLDOWN_SECONDS = float(os.getenv("QUERY_COOLDOWN_SECONDS", "3"))
SESSION_IDLE_TIMEOUT_SECONDS = int(os.getenv("SESSION_IDLE_TIMEOUT_SECONDS", "1800"))

QUOTA_QUERIES_MSG = "Limite da demo atingido para esta sessão. Recarregue a página pra recomeçar."
QUOTA_INDEX_MSG = (
    f"Limite de {MAX_INDEX_OPERATIONS_PER_SESSION} indexações por sessão atingido. "
    "Recarregue a página pra recomeçar."
)
COOLDOWN_MSG = (
    f"Aguarde {QUERY_COOLDOWN_SECONDS:.0f} segundos entre perguntas. Tenta de novo daqui a pouco."
)
SESSION_EXPIRED_MSG = (
    "Sessão expirou por inatividade. Recarregue a página e suba os documentos de novo."
)
RATE_LIMIT_MSG = _SHARED_RATE_LIMIT_MSG
NO_STATE_MSG = "Envie documentos primeiro."


THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Menlo", "monospace"],
).set(
    block_border_width="1px",
    block_radius="12px",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
)

CSS = """
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; padding-top: 28px !important; }
#hero { display:flex; align-items:flex-start; gap:14px; padding: 0 0 22px; }
#hero .mark { font-family: var(--font-mono); font-weight:700; color: var(--primary-600); font-size: 1.5rem; line-height: 1.2; letter-spacing: -0.02em; }
#hero .copy h1 { margin:0; font-weight:600; letter-spacing:-0.018em; font-size:1.55rem; line-height:1.2; }
#hero .copy p { margin:6px 0 0; color: var(--body-text-color-subdued); font-size: 0.95rem; line-height:1.45; }
.status-chip { display:inline-flex; align-items:center; gap:8px; padding:6px 12px; border-radius:999px; font-weight:500; font-size:13px; }
.status-chip.active::before { content:""; width:6px; height:6px; border-radius:50%; background:currentColor; animation: pulse 1.2s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:.35 } }
.message { padding: 12px 16px !important; line-height: 1.6 !important; }
footer { display: none !important; }
"""

HERO_HTML = """
<div id="hero">
  <div class="mark">[1]</div>
  <div class="copy">
    <h1>rag-chatbot</h1>
    <p>Pergunte sobre seu documento — cada resposta cita as fontes.</p>
  </div>
</div>
"""

CHAT_PLACEHOLDER = "Sem conversa ainda. Suba um arquivo e pergunte."

_CHIP_PALETTE = {
    "neutral": ("rgba(99,102,241,.10)", "var(--primary-600)"),
    "ready": ("rgba(34,197,94,.12)", "#15803d"),
    "active": ("rgba(234,179,8,.14)", "#a16207"),
    "error": ("rgba(239,68,68,.12)", "#b91c1c"),
}


def _status_chip(text: str, kind: str = "neutral") -> str:
    bg, fg = _CHIP_PALETTE.get(kind, _CHIP_PALETTE["neutral"])
    cls = f"status-chip {kind}"
    return f'<div class="{cls}" style="background:{bg};color:{fg};">{text}</div>'


INITIAL_STATUS = _status_chip("Envie um documento.", "neutral")


def _is_rate_limit(exc: BaseException) -> bool:
    return _shared_is_rate_limit(exc)


def _cleanup_upload_files(paths: list[str]) -> None:
    """Apaga arquivos temporários do Gradio após indexação (best-effort).

    Gradio salva uploads em ``/tmp/gradio/<uuid>/<filename>`` e só limpa
    no próximo restart do container. Como o conteúdo já vive em RAM
    após ``load_documents_from_files``, a cópia em disco é redundante e
    aumenta a superfície de leak (file-serving do Gradio, dump de /tmp,
    visitante adivinhando o uuid). Apagar imediatamente fecha essa
    janela. Falhas são silenciosas — não queremos que erro de cleanup
    apareça pro usuário.
    """
    import contextlib

    for p in paths:
        with contextlib.suppress(OSError):
            Path(p).unlink(missing_ok=True)


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

    paths = [f.name for f in files]
    try:
        documents = load_documents_from_files(paths)
    except (EmptyDocumentError, ProtectedDocumentError, EncodingError) as exc:
        _cleanup_upload_files(paths)
        yield current_state, _status_chip(str(exc), "error")
        return
    except Exception as exc:
        _cleanup_upload_files(paths)
        yield current_state, _status_chip(f"Erro ao ler arquivos: {exc}", "error")
        return
    # Conteúdo já está em RAM como Documents. Apaga os arquivos temporários
    # do Gradio (/tmp/gradio/<uuid>/<filename>) imediatamente — fecha a
    # janela em que file-serving do Gradio ou dump de /tmp poderia ler
    # docs brutos de outros visitantes.
    _cleanup_upload_files(paths)

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

    now = time.monotonic()
    if current_state is None:
        state = {
            "id": session_id,
            "graph": graph,
            "queries": 0,
            "uploads": 1,
            "last_activity_at": now,
            "last_query_at": None,
        }
    else:
        # Drop the previous graph eagerly so its Qdrant in-memory data can be freed.
        current_state["graph"] = graph
        current_state["uploads"] = next_upload
        current_state["last_activity_at"] = now
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
        meta_bits.append(f"score {s['score']:.3f}" if s["score"] > 0 else "sem rerank")
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

    now = time.monotonic()
    last_activity = state.get("last_activity_at")
    if last_activity is not None and now - last_activity > SESSION_IDLE_TIMEOUT_SECONDS:
        state["graph"] = None
        history.append(user_turn)
        history.append({"role": "assistant", "content": SESSION_EXPIRED_MSG})
        yield history, "", gr.update()
        return

    last_query = state.get("last_query_at")
    if last_query is not None and now - last_query < QUERY_COOLDOWN_SECONDS:
        history.append(user_turn)
        history.append({"role": "assistant", "content": COOLDOWN_MSG})
        yield history, "", gr.update()
        return

    if state.get("queries", 0) >= MAX_QUERIES_PER_SESSION:
        history.append(user_turn)
        history.append({"role": "assistant", "content": QUOTA_QUERIES_MSG})
        yield history, "", gr.update()
        return

    state["queries"] = state.get("queries", 0) + 1
    state["last_query_at"] = now
    state["last_activity_at"] = now
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


with gr.Blocks(title="rag-chatbot", analytics_enabled=False, fill_width=False) as demo:
    gr.HTML(HERO_HTML)

    state = gr.State(value=None)

    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=240):
            files = gr.File(
                file_count="multiple",
                file_types=[".txt", ".md", ".pdf"],
                label="Documentos",
                show_label=False,
            )
            status = gr.HTML(value=INITIAL_STATUS)

        with gr.Column(scale=5, min_width=420):
            chatbot = gr.Chatbot(
                height=460,
                show_label=False,
                placeholder=CHAT_PLACEHOLDER,
            )
            with gr.Row():
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="Pergunte algo sobre o documento…",
                    container=False,
                    scale=8,
                )
                send = gr.Button("Enviar", variant="primary", scale=1, min_width=100)

            gr.Examples(
                examples=[
                    "Faça um resumo do documento.",
                    "Quais os principais conceitos?",
                    "Liste as principais conclusões.",
                ],
                inputs=[msg],
                label="",
                api_name=False,
            )

            with gr.Accordion("Fontes", open=False):
                sources_md = gr.Markdown("_Faça uma pergunta para ver as fontes citadas._")

    files.change(index_files, inputs=[files, state], outputs=[state, status], api_name=False)
    send.click(
        respond, inputs=[msg, chatbot, state], outputs=[chatbot, msg, sources_md], api_name=False
    )
    msg.submit(
        respond, inputs=[msg, chatbot, state], outputs=[chatbot, msg, sources_md], api_name=False
    )


# ── Casual abuse defenses ────────────────────────────────────────────────
#
# Camadas:
#   1. Bloqueio de IPs em Tor exit list pública (snapshot na startup).
#   2. Rate limit por IP nos endpoints reais do Gradio (/gradio_api/*).
# Não defende contra atacante motivado (proxies residenciais, IPv6, restart
# do container reseta contador). Defende contra abuso casual de scripts
# rasteiros e tabs anônimas em rajada.

TOR_EXIT_LIST_URL = os.getenv("TOR_EXIT_LIST_URL", "https://check.torproject.org/torbulkexitlist")
TOR_EXIT_LIST_FALLBACK = Path(__file__).resolve().parent / "tor_exit_nodes.txt"
GRADIO_RATE_LIMIT_PER_MINUTE = int(os.getenv("GRADIO_RATE_LIMIT_PER_MINUTE", "30"))
GRADIO_RATE_LIMIT_PER_HOUR = int(os.getenv("GRADIO_RATE_LIMIT_PER_HOUR", "300"))


def _parse_tor_list(text: str) -> set[str]:
    return {line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")}


def _fetch_tor_exit_nodes(url: str = TOR_EXIT_LIST_URL, timeout: float = 5.0) -> set[str]:
    """Baixa a lista pública de Tor exit nodes. Se a rede falhar, cai pra
    snapshot local (``tor_exit_nodes.txt``). Se nem snapshot existir, devolve
    set vazio e segue sem bloqueio Tor.
    """
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        nodes = _parse_tor_list(text)
        logger.info("tor exit list carregada via rede: %d IPs", len(nodes))
        return nodes
    except Exception as exc:
        logger.warning("tor exit list fetch falhou: %s — tentando snapshot local", exc)
    try:
        nodes = _parse_tor_list(TOR_EXIT_LIST_FALLBACK.read_text(encoding="utf-8"))
        logger.info("tor exit list carregada via snapshot: %d IPs", len(nodes))
        return nodes
    except FileNotFoundError:
        logger.warning("snapshot %s ausente — bloqueio Tor desativado", TOR_EXIT_LIST_FALLBACK.name)
        return set()


class _IPRateLimiter:
    """Rate limiter in-memory por IP, janela deslizante. Stdlib only."""

    def __init__(self, per_minute: int, per_hour: int):
        from collections import defaultdict, deque
        from threading import Lock

        self.per_minute = per_minute
        self.per_hour = per_hour
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = Lock()

    def try_consume(self, ip: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hist = self._hits[ip]
            cutoff_hour = now - 3600
            while hist and hist[0] < cutoff_hour:
                hist.popleft()
            recent_minute = sum(1 for t in hist if t > now - 60)
            if recent_minute >= self.per_minute or len(hist) >= self.per_hour:
                return False
            hist.append(now)
            return True


def _client_ip(request) -> str:
    """IP do cliente. Prioriza X-Forwarded-For (HF Spaces, proxy reverso)."""
    fwd = request.headers.get("x-forwarded-for") or request.headers.get("x-real-ip")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _build_app():
    """Monta o FastAPI com defesas + Gradio."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    tor_nodes = _fetch_tor_exit_nodes()
    limiter = _IPRateLimiter(GRADIO_RATE_LIMIT_PER_MINUTE, GRADIO_RATE_LIMIT_PER_HOUR)
    app = FastAPI()

    @app.middleware("http")
    async def _defenses(request: Request, call_next):
        ip = _client_ip(request)
        if ip in tor_nodes:
            logger.info("blocked tor exit ip=%s path=%s", ip, request.url.path)
            return JSONResponse({"detail": "Acesso indisponível neste IP."}, status_code=403)
        if request.url.path.startswith("/gradio_api") and not limiter.try_consume(ip):
            logger.info("rate limited ip=%s path=%s", ip, request.url.path)
            return JSONResponse(
                {"detail": "Muitas requisições. Aguarde alguns minutos."},
                status_code=429,
            )
        return await call_next(request)

    return gr.mount_gradio_app(
        app,
        demo,
        path="/",
        theme=THEME,
        css=CSS,
        footer_links=["gradio"],
        # Gradio por default whitelista /tmp/gradio no file-serving — é
        # onde os uploads de visitantes acabam. Bloqueia explicitamente
        # pra ninguém ler doc bruto de outra sessão via /file=<path>.
        blocked_paths=["/tmp/gradio"],
    )


if __name__ == "__main__":
    import uvicorn

    # HF Spaces (Docker SDK) injeta GRADIO_SERVER_PORT=8000 em runtime,
    # sobrepondo o ENV do Dockerfile. Pra manter a porta alinhada com o
    # ``app_port`` do README YAML (7860) e o EXPOSE do Dockerfile,
    # hardcoda. Pra rodar local em outra porta, edita esta linha.
    uvicorn.run(_build_app(), host="0.0.0.0", port=7860, log_level="info")
