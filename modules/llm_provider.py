import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI

def initialize_google_llm(model_name):
    """Initialize Google Gemini LLM with given model name."""
    api_key = st.secrets.get("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = api_key
    temperature = float(st.secrets.get("TEMPERATURE", 0.7))
    top_p = float(st.secrets.get("TOP_P", 1.0))
    max_tokens = int(st.secrets.get("MAX_TOKENS", 1024))

    if not all([api_key, model_name]):
        st.error("Missing one or more Google Gemini environment variables.")
        return None

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens
    )

def initialize_azure_llm(model_name):
    """Initialize Azure OpenAI LLM with given model name."""
    api_key = st.secrets.get("AZURE_OPENAI_API_KEY")
    endpoint = st.secrets.get("AZURE_OPENAI_ENDPOINT")
    version = st.secrets.get("AZURE_OPENAI_API_VERSION")
    temperature = float(st.secrets.get("TEMPERATURE", 0.7))
    top_p = float(st.secrets.get("TOP_P", 1.0))
    max_tokens = int(st.secrets.get("MAX_TOKENS", 1024))

    if not all([api_key, endpoint, version, model_name]):
        st.error("Missing one or more Azure OpenAI environment variables.")
        return None

    return AzureChatOpenAI(
        openai_api_version=version,
        azure_deployment=model_name,
        azure_endpoint=endpoint,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )

def get_llm_model(llm_provider: str, model_name: str):
    """
    Select and initialize LLM based on provider and model name.
    Args:
        llm_provider: 'Azure' or 'Google'
        model_name: llm model string
    """
    provider = llm_provider.lower()
    if provider == "azure":
        return initialize_azure_llm(model_name)
    elif provider == "google":
        return initialize_google_llm(model_name)
    else:
        st.error(f"Unsupported LLM provider: {llm_provider}")
        return None
