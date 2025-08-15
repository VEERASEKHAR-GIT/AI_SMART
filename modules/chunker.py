from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents_to_chunks(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks for embedding or search.
    
    Args:
        documents: List of LangChain Document objects
        chunk_size: Max characters/tokens per chunk
        chunk_overlap: Overlap between chunks to preserve context
    
    Returns:
        List of Document chunks
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)
