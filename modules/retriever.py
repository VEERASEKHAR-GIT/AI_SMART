def create_faiss_retriever(vector_store):
    """
    Creates a FAISS retriever with fixed parameters:
    - search_type = 'similarity'
    - k = 4 (top 4 matches)
    
    Args:
        vector_store: FAISS vector store object

    Returns:
        LangChain retriever instance
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
#     return vector_store.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": 2, "score_threshold": 0.7}
# )
