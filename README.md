# rag-chatbot

Chatbot de linha de comando com fluxo de RAG construido com LangChain, FAISS e modelos da OpenAI. O projeto indexa um arquivo de texto local, gera embeddings e responde perguntas com base no contexto recuperado.

## O que o projeto faz

- carrega um documento local em `data/sample_docs.txt`
- divide o texto em chunks
- gera embeddings com OpenAI
- monta um indice vetorial FAISS
- responde perguntas usando `RetrievalQA`

## Stack

- Python
- LangChain
- OpenAI
- FAISS
- python-dotenv

## Estrutura

```text
.
|-- app.py
|-- requirements.txt
|-- .gitignore
`-- data/
    `-- sample_docs.txt
```

## Pre-requisitos

- Python 3.10+
- chave `OPENAI_API_KEY` em um arquivo `.env`

## Como executar

```bash
git clone https://github.com/RenanMiqueloti/rag-chatbot.git
cd rag-chatbot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Crie um arquivo `.env` com:

```env
OPENAI_API_KEY=sua_chave_aqui
```

Depois rode:

```bash
python app.py
```

## Como usar

1. O script constroi o indice vetorial ao iniciar.
2. Digite perguntas no terminal.
3. Use `sair`, `exit` ou `quit` para encerrar.

## Observacoes

- o documento indexado por padrao e `data/sample_docs.txt`
- o projeto e propositalmente pequeno e serve como exemplo didatico de RAG em CLI
