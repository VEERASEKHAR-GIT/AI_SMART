import os
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)

# def load_document(file_path: str):
#     """
#     Load a document based on its file extension.
#     Supports: PDF, DOCX/DOC, TXT
#     Returns: List of Document objects from LangChain
#     """
#     file_extension = os.path.splitext(file_path)[1].lower()

#     if file_extension == '.txt':
#         loader = TextLoader(file_path)
#     elif file_extension == '.pdf':
#         loader = PyMuPDFLoader(file_path)
#     elif file_extension in ['.docx', '.doc']:
#         loader = UnstructuredWordDocumentLoader(file_path)
#     else:
#         raise ValueError(f"Unsupported file type: {file_extension}")

#     return loader.load()
import os
import pandas as pd
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain.schema import Document

def load_document(file_path: str):
    """
    Load a document based on its file extension.
    Supports: PDF, DOCX/DOC, TXT, CSV
    Returns: List of Document objects from LangChain
    """
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.txt':
        loader = TextLoader(file_path)
        return loader.load()
    elif file_extension == '.pdf':
        loader = PyMuPDFLoader(file_path)
        return loader.load()
    elif file_extension in ['.docx', '.doc']:
        loader = UnstructuredWordDocumentLoader(file_path)
        return loader.load()
    elif file_extension == '.csv':
        return load_csv_as_documents(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def load_csv_as_documents(file_path: str):
    """
    Convert CSV file to LangChain Document objects.
    Each row becomes a separate document for better searchability.
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Convert to documents
        documents = []
        
        # Create a summary document with column info
        summary_content = f"""
        CSV File Summary:
        - Total rows: {len(df)}
        - Columns: {', '.join(df.columns.tolist())}
        - Sample data preview:
        {df.head(3).to_string()}
        """
        
        summary_doc = Document(
            page_content=summary_content,
            metadata={
                "source": file_path,
                "type": "csv_summary",
                "row_count": len(df),
                "columns": df.columns.tolist()
            }
        )
        documents.append(summary_doc)
        
        # Convert each row to a document
        for idx, row in df.iterrows():
            # Create readable content for each row
            row_content = f"Row {idx + 1}:\n"
            for col, value in row.items():
                row_content += f"- {col}: {value}\n"
            
            row_doc = Document(
                page_content=row_content,
                metadata={
                    "source": file_path,
                    "type": "csv_row",
                    "row_number": idx + 1,
                    "columns": df.columns.tolist()
                }
            )
            documents.append(row_doc)
        
        return documents
        
    except Exception as e:
        # If CSV loading fails, create an error document
        error_doc = Document(
            page_content=f"Error loading CSV file: {str(e)}",
            metadata={"source": file_path, "type": "csv_error"}
        )
        return [error_doc]