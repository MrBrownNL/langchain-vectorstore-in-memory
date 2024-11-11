import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

load_dotenv()

if __name__ == "__main__":
    print("hi")

    faiss_index = "faiss_index_react"
    embeddings = OllamaEmbeddings(model=os.environ.get("OLLAMA_EMBEDDINGS_MODEL"))

    if not os.path.exists(f"{faiss_index}/index.faiss"):
        print("Processing PDF...")
        pdf_path = "2210.03629v3.pdf"
        loader = PyPDFLoader(file_path=pdf_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=30, separator="\n"
        )
        docs = text_splitter.split_documents(documents=documents)

        # here is a good point to debug to find out if the document sizes in docs are around the chunk_size

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(faiss_index)
    else:
        print("Using vectorstore from disk.")

    new_vectorstore = FAISS.load_local(
        faiss_index, embeddings, allow_dangerous_deserialization=True
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        ChatOllama(model=os.environ.get("OLLAMA_MODEL")), retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combine_docs_chain
    )

    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    print(res["answer"])
