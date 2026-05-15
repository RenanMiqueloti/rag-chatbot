---
title: rag-chatbot
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: docker
app_file: Dockerfile.spaces
app_port: 7860
pinned: false
short_description: Demo de RAG com upload de documentos e citações
---

# rag-chatbot

![CI](https://github.com/RenanMiqueloti/rag-chatbot/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)

Pipeline RAG com LangGraph, Qdrant, hybrid retrieval (BM25 + dense + RRF), re-ranking via cross-encoder e tracing via LangSmith. Suporta Claude (Anthropic), GPT-4o-mini (OpenAI) e Llama 3.3 70B (Groq).

> Demo local autocontido — troque `QdrantClient(":memory:")` por `QdrantClient(url=...)` para um deploy real.

---

## Arquitetura

```mermaid
graph LR
    A([Query]) --> B
    B["retrieve\nBM25 + Semantic\nRRF Fusion"] --> C
    C["rerank\nCross-encoder\nFlashRank"] --> D
    D["generate\nClaude / GPT-4o-mini\ncom contexto"] --> E([Answer])

    style B fill:#2d3748,color:#e2e8f0
    style C fill:#2d3748,color:#e2e8f0
    style D fill:#2d3748,color:#e2e8f0
```

| Nó | O que faz | Por que importa |
|---|---|---|
| **retrieve** | BM25 + semantic → Reciprocal Rank Fusion | Cobre tanto vocabulário exato (siglas, IDs) quanto similaridade semântica |
| **rerank** | Cross-encoder FlashRank, fallback gracioso | Reordena candidatos com contexto da query — menos alucinação |
| **generate** | Prompt grounded + Claude ou GPT-4o-mini | Responde apenas com o que está no contexto recuperado |

---

## Stack

- **LangGraph** 0.4+ — orquestração do pipeline como grafo de estado
- **Qdrant** (in-memory) — banco vetorial; substitua por instância dedicada num deploy real
- **BM25** via `rank-bm25` — retrieval por vocabulário exato
- **Reciprocal Rank Fusion** — fusão dos dois rankings sem parâmetros extras
- **FlashRank** (opcional) — cross-encoder leve para re-ranking local
- **FastAPI** — endpoint REST + streaming
- **LangSmith** — tracing nativo LangGraph (node-by-node state diffs)
- **Claude (Anthropic) / GPT-4o-mini / Llama 3.3 70B (Groq)** — provider configurável via `LLM_PROVIDER` env var
- **LLM-as-judge evals** — avaliação automática de relevance, faithfulness, completeness

---

## Estrutura

```
rag-chatbot/
├── app.py             # Pipeline LangGraph: retrieve → rerank → generate
├── api.py             # FastAPI: POST /query, POST /stream, GET /health
├── evals/
│   ├── evaluate.py    # Harness de evals com LLM-as-judge
│   └── dataset.json   # Dataset de perguntas para regressão
├── data/
│   ├── sample_docs.txt
│   └── example.md     # primer sobre RAG — bom corpus de partida pra demo
├── .env.example
├── requirements.txt
└── LICENSE
```

---

## Quick start

```bash
git clone https://github.com/RenanMiqueloti/rag-chatbot.git
cd rag-chatbot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # configure LLM_PROVIDER, API keys e LangSmith
```

**CLI:**
```bash
python app.py
```

**API REST:**
```bash
uvicorn api:app --reload
# POST http://localhost:8000/query  {"query": "..."}
# POST http://localhost:8000/stream {"query": "..."}
```

**Evals:**
```bash
python -m evals.evaluate
```

---

## Demo online

Demo Gradio rodando em Hugging Face Spaces — URL será publicada após o deploy.

Limitações da demo:

- 3 arquivos por sessão, até 5 MB cada (`.txt`, `.md`, `.pdf`)
- 30 perguntas por sessão (acumulado — não reseta ao re-upload)
- 3 indexações por sessão
- Documentos não são persistidos entre sessões ou restarts do Space

Sem corpus próprio? `data/example.md` neste repo é um primer curto sobre RAG e
serve como ponto de partida — baixe e suba na demo.

---

## Providers LLM

Configure `LLM_PROVIDER` no `.env`:

| Provider | Modelo | Env var necessária |
|---|---|---|
| `openai` (padrão) | gpt-4o-mini | `OPENAI_API_KEY` |
| `anthropic` | claude-3-5-haiku-20241022 | `ANTHROPIC_API_KEY` |
| `groq` | llama-3.3-70b-versatile | `GROQ_API_KEY` (free tier, rate-limited) |

---

## Observabilidade — LangSmith

Configure no `.env`:

```env
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=rag-chatbot
```

Com tracing ativo, cada execução do pipeline registra no LangSmith:
- Inputs e outputs de cada nó (retrieve → rerank → generate)
- Documentos recuperados e re-rankeados
- Prompt final enviado ao LLM
- Latência por nó

---

## Re-ranking (FlashRank)

`flashrank` já vem em `requirements.txt`. Se você remover, o nó `rerank` cai
num fallback que apenas trunca os top-3 do RRF — o pipeline continua funcionando,
mas sem cross-encoder reordenando.

---

## Qdrant servidor (deploy real)

Sem `QDRANT_URL` definido, o pipeline cai em in-memory (sem persistência).
Pra apontar para um Qdrant rodando, configure no `.env`:

```env
QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=...   # se a instância exigir auth
```

A função `build_retrievers(documents, qdrant_url=...)` também aceita override
explícito (`""` força in-memory mesmo com env definido — usado pela demo Gradio).

---

## Deploy via Docker Compose

O repositório inclui `Dockerfile` + `docker-compose.yml` pra subir a API junto com um Qdrant dedicado.

### Subir o stack

```bash
cp .env.example .env       # preencha a key do provider escolhido
docker compose up -d --build
```

### Serviços

| Serviço | URL | Volume |
|---|---|---|
| API (FastAPI) | http://localhost:8000 | — |
| Qdrant (REST + gRPC) | http://localhost:6333 / :6334 | `qdrant_data` |

A API tem `HEALTHCHECK` em `GET /health` (intervalo 30s, timeout 5s, 3 retries).

### Encerrar e limpar volumes

```bash
docker compose down            # mantém volumes (estado persiste)
docker compose down -v         # apaga volumes (reset completo)
```

### Rodar só o serviço de API (sem Qdrant dedicado)

```bash
docker build -t rag-chatbot:local .
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  rag-chatbot:local
```

Nesse modo o pipeline cai no `QdrantClient(":memory:")` e funciona standalone.

### Demo Gradio (Hugging Face Spaces)

A imagem `Dockerfile.spaces` empacota o `gradio_app.py` para rodar no Hugging Face Spaces (Docker SDK, porta 7860). Local:

```bash
docker build -f Dockerfile.spaces -t rag-chatbot:spaces .
docker run --rm -p 7860:7860 \
  -e LLM_PROVIDER=groq \
  -e GROQ_API_KEY=$GROQ_API_KEY \
  rag-chatbot:spaces
```

Sem Docker:

```bash
pip install -r requirements.txt
python gradio_app.py
```

---

## Design decisions

**Por que LangGraph e não LCEL puro?**
O grafo de estado torna cada etapa auditável e substituível independentemente. Com LCEL puro, trocar o nó de re-ranking exigiria reescrever a chain. Com LangGraph, é um `add_node` + `add_edge`.

**Por que Qdrant e não FAISS?**
FAISS não tem servidor, não tem filtros, não escala horizontalmente. Qdrant resolve os três. O modo in-memory mantém a DX de desenvolvimento sem dependência externa.

**Por que BM25 + semântico?**
Modelos de embedding não capturam vocabulário exato (siglas, nomes próprios, IDs). BM25 captura. A fusão via RRF cobre os dois casos sem tuning de pesos.

**Por que LangSmith e não logging manual?**
LangSmith tem integração nativa com LangGraph: cada nó do grafo vira um span rastreado automaticamente, com state diffs e latência por nó, sem instrumentação extra no código.

**Por que LLM-as-judge?**
Métricas clássicas como ROUGE e BLEU não capturam faithfulness (resposta grounded no contexto recuperado). LLM-as-judge com prompts estruturados serve como aproximação razoável quando não há ground truth de fact-checking.
