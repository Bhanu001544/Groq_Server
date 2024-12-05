import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
try:
    from langchain_chroma import Chroma
except:
    from langchain.vectorstores import Chroma

from langchain.embeddings import SentenceTransformerEmbeddings


def retrieve(document):

    # load the document
    loader = PyPDFLoader(document)
    documents = loader.load()

    # split the document to chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # create the database if it doesnt exist and add the texts
    itech_db = "itech_vectorStore"

    # create the embedding function. This is a model that vectorizes the chunks
    embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    if os.path.exists(itech_db): #check whether the vector database exists
        db = Chroma(persist_directory=itech_db,
                            embedding_function=embedding_function)
        db.add_documents(
            texts,
            embedding = embedding_function
        )
    else:
        db = Chroma.from_documents(
            texts,
            embedding=embedding_function,
            persist_directory=itech_db
        )
    retriever = db.as_retriever()
    return retriever

