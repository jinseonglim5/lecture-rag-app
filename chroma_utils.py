from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

def save_to_chroma(text: str, doc_id: str):
    from langchain.schema import Document
    doc = Document(page_content=text, metadata={"doc_id": doc_id})
    vectordb.add_documents([doc])
    vectordb.persist()

def search_chroma(query: str):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    results = retriever.get_relevant_documents(query)
    return [r.page_content for r in results]
