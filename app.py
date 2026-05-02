"""RAG chatbot — LCEL chain com LangChain 0.3+, FAISS e OpenAI.

Substitui RetrievalQA.from_chain_type pelo padrão atual:
    retriever | prompt | llm | StrOutputParser (LCEL chain).
"""
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

_PROMPT = ChatPromptTemplate.from_template(
    "Você é um assistente direto e preciso. "
    "Responda apenas com base no contexto abaixo. "
    "Se a resposta não estiver no contexto, diga que não sabe.\n\n"
    "Contexto:\n{context}\n\n"
    "Pergunta: {question}"
)


def build_chain(doc_path: str = "data/sample_docs.txt"):
    """Constrói a LCEL RAG chain a partir do documento informado.

    Pipeline:
        TextLoader → RecursiveCharacterTextSplitter → OpenAIEmbeddings
        → FAISS → retriever → prompt → ChatOpenAI → StrOutputParser.

    Args:
        doc_path: Caminho para o arquivo .txt a indexar.

    Returns:
        LCEL chain invocável com ``{"question": str}``.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se OPENAI_API_KEY não estiver definida.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY não encontrada. "
            "Crie um arquivo .env com OPENAI_API_KEY=sk-..."
        )
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Documento não encontrado: {doc_path!r}")

    loader = TextLoader(doc_path, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | _PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def main() -> None:
    print("🔎 Construindo índice vetorial...")
    try:
        chain = build_chain()
    except (ValueError, FileNotFoundError) as exc:
        print(f"❌ {exc}")
        return

    print("🤖 Chatbot RAG pronto! Digite 'sair' para encerrar.\n")
    while True:
        query = input("Você: ").strip()
        if not query:
            continue
        if query.lower() in {"sair", "exit", "quit"}:
            print("Encerrando.")
            break
        try:
            answer = chain.invoke(query)
        except Exception as exc:  # noqa: BLE001
            answer = f"Erro ao responder: {exc}"
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    main()
