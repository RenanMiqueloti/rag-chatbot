import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente (OPENAI_API_KEY)
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

def build_vectorstore(path="data/sample_docs.txt"):
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    return vs

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Defina OPENAI_API_KEY no arquivo .env")
        return

    print("🔎 Construindo índice vetorial...")
    vs = build_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    print("🤖 Chatbot RAG pronto! Digite 'sair' para encerrar.")
    while True:
        query = input("Você: ").strip()
        if query.lower() in {"sair", "exit", "quit"}:
            break
        try:
            answer = qa.run(query)
        except Exception as e:
            answer = f"Erro ao responder: {e}"
        print("Bot:", answer)

if __name__ == "__main__":
    main()
