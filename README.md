# rag-chatbot

Chatbot CLI com RAG usando LangChain 0.3+ (LCEL), FAISS e OpenAI.

Indexa um arquivo `.txt` local, gera embeddings e responde perguntas usando apenas o contexto recuperado — sem `RetrievalQA` deprecated.

## Pipeline

```
TextLoader → RecursiveCharacterTextSplitter → OpenAIEmbeddings (text-embedding-3-small)
→ FAISS → retriever (top-3) → ChatPromptTemplate → gpt-4o-mini → StrOutputParser
```

## Stack

- Python 3.10+
- LangChain 0.3+ (LCEL)
- OpenAI (`text-embedding-3-small` + `gpt-4o-mini`)
- FAISS (índice vetorial local, CPU)

## Como executar

```bash
git clone https://github.com/RenanMiqueloti/rag-chatbot.git
cd rag-chatbot
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Crie um `.env`:

```env
OPENAI_API_KEY=sk-...
```

```bash
python app.py
```

## Tradeoffs e decisões

- **LCEL sobre `RetrievalQA`**: `RetrievalQA.from_chain_type` foi substituído pelo pipeline `retriever | prompt | llm | parser` (LCEL) para alinhar com LangChain 0.3+ e eliminar warnings de deprecação.
- **`text-embedding-3-small`**: melhor custo-benefício que `ada-002` para coleções pequenas.
- **FAISS vs. Weaviate/pgvector**: FAISS é suficiente para demonstração local sem servidor. Em produção com grandes volumes, migrar para pgvector ou Weaviate.
- **CLI vs. interface web**: mantido como CLI propositalmente — foco no pipeline, não na UI.
