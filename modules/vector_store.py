import os
import shutil
from langchain_community.vectorstores import FAISS

def create_faiss_vector_store(chunks, embeddings, vector_index_path):
    """
    Create and save a FAISS vector store from document chunks.

    Args:
        chunks: List of LangChain Document chunks
        embeddings: Embedding model instance
        vector_index_path: Path to save the FAISS index

    Returns:
        FAISS vector store object
    """
    parent = os.path.dirname(vector_index_path)
    if not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    if os.path.exists(vector_index_path):
        shutil.rmtree(vector_index_path)

    os.makedirs(vector_index_path, exist_ok=True)

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(vector_index_path)

    return vector_store
