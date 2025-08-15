import os
import streamlit as st
from langchain_openai import AzureOpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def initialize_azure_embeddings(model_name):
    """Initialize Azure OpenAI Embeddings with given model name."""
    api_key = st.secrets.get("AZURE_OPENAI_API_KEY")
    endpoint = st.secrets.get("AZURE_OPENAI_ENDPOINT")
    version = st.secrets.get("AZURE_OPENAI_API_VERSION")

    if not all([api_key, endpoint, version, model_name]):
        st.error("Missing one or more Azure OpenAI environment variables.")
        return None

    return AzureOpenAIEmbeddings(
        azure_deployment=model_name,
        openai_api_version=version,
        azure_endpoint=endpoint,
        api_key=api_key
    )

def initialize_google_embeddings(model_name):
    """Initialize Google Gemini Embeddings with given model name."""
    api_key = st.secrets.get("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = api_key

    if not all([model_name, api_key]):
        st.error("Missing one or more Google Gemini environment variables.")
        return None

    return GoogleGenerativeAIEmbeddings(model=model_name, api_key=api_key)

def get_embeddings(llm_provider: str, model_name: str):
    """
    Select and initialize embeddings based on provider and model name.
    Args:
        llm_provider: 'Azure' or 'Google'
        model_name: embedding model string
    """
    provider = llm_provider.lower()
    if provider == "azure":
        return initialize_azure_embeddings(model_name)
    elif provider == "google":
        return initialize_google_embeddings(model_name)
    else:
        st.error(f"Unsupported LLM provider: {llm_provider}")
        return None
